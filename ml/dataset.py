"""Feature frames -> standardized LSTM sequences.

Two feature sources share one schema (config.FEATURE_COLUMNS + `target`):
    - historical : offline 5-min eth-alpha dataset (ml/historical.build_features)
    - parquet    : the live feature store written by processing/feature_join.py

`target` is always the NEXT window's volatility. A sequence of L windows ending
at row t predicts target[t] (= volatility at t+1).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import config


def load_historical_frame(symbol: str = None) -> pd.DataFrame:
    """Offline training source — merged 5-min historical features."""
    from ml.historical import build_features
    return build_features(symbol)


def load_parquet_frame(features_dir: str = None) -> pd.DataFrame:
    """Live feature store written by feature_join.py, with `target` appended."""
    features_dir = features_dir or config.FEATURES_DIR
    parts = glob.glob(os.path.join(features_dir, "*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No parquet files found in {features_dir!r}. "
            "Run the streaming jobs first (or use --source historical)."
        )
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df = df.dropna(subset=["window_start"]).sort_values("window_start")
    df = df.drop_duplicates(subset=["window_start"], keep="last").reset_index(drop=True)

    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df[config.FEATURE_COLUMNS] = df[config.FEATURE_COLUMNS].fillna(0.0).astype(float)

    df["target"] = df[config.TARGET_COLUMN].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def load_feature_frame(source: str = "historical") -> pd.DataFrame:
    if source == "historical":
        return load_historical_frame()
    if source == "parquet":
        return load_parquet_frame()
    raise ValueError(f"unknown source {source!r} (use 'historical' or 'parquet')")


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the [N, F] model-input matrix, applying log1p to heavy-tailed
    features. Used by both training and inference so the transform is identical.
    """
    feats = df[config.FEATURE_COLUMNS].astype(np.float64).copy()
    for col in config.LOG_FEATURES:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    return feats.to_numpy(dtype=np.float32)


def make_sequences(df: pd.DataFrame, seq_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X is [N, L, F], y is [N] next-window volatility.

    Requires a `target` column (already = next window's volatility).
    """
    seq_len = seq_len or config.SEQUENCE_LENGTH
    if "target" not in df.columns:
        raise ValueError("frame is missing the 'target' column")

    feats = feature_matrix(df)
    target = df["target"].to_numpy(dtype=np.float32)

    X, y = [], []
    # Sequence i = rows [i, i+L); label = target of the last row in the sequence.
    for i in range(len(df) - seq_len + 1):
        X.append(feats[i : i + seq_len])
        y.append(target[i + seq_len - 1])
    if not X:
        raise ValueError(
            f"Not enough rows ({len(df)}) for a sequence of length {seq_len}."
        )
    return np.stack(X), np.asarray(y, dtype=np.float32)


@dataclass
class Standardizer:
    """Per-feature mean/std scaler, persisted alongside the model."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        # Accumulate stats in float64 to avoid drift over millions of elements.
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
        )
