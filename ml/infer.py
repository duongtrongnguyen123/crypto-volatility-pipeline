"""Load the trained LSTM and predict next-window volatility.

Loads the most recent SEQUENCE_LENGTH windows and prints the predicted
volatility for the upcoming 5-minute window. Serves from the live Parquet
feature store by default; falls back to the historical dataset if the live
store is empty (e.g. before the streaming pipeline has produced data).

Run:
    python -m ml.infer                  # live store, fallback to historical
    python -m ml.infer --source historical
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import config
from ml.dataset import (
    Standardizer,
    feature_matrix,
    load_historical_frame,
    load_parquet_frame,
)
from ml.model import LSTMVolatility


def load_model(path: str = None):
    path = path or config.MODEL_PATH
    ckpt = torch.load(path, map_location="cpu")
    model = LSTMVolatility(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = Standardizer.from_dict(ckpt["scaler"])
    return model, scaler, ckpt


def _load_serving_frame(source: str):
    if source == "historical":
        return load_historical_frame(), "window_start_hist"
    try:
        df = load_parquet_frame()
        if len(df) >= config.SEQUENCE_LENGTH:
            return df, "window_start"
        print("[infer] live store too small; falling back to historical")
    except FileNotFoundError:
        print("[infer] live store empty; falling back to historical")
    return load_historical_frame(), "window_start_hist"


def predict_latest(source: str = "parquet") -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler, ckpt = load_model()
    model.to(device)

    seq_len = ckpt["sequence_length"]
    feature_columns = ckpt["feature_columns"]

    df, ts_col = _load_serving_frame(source)
    if len(df) < seq_len:
        raise ValueError(f"Need at least {seq_len} windows, only have {len(df)}.")

    window = feature_matrix(df[feature_columns].tail(seq_len))[-seq_len:]
    window = scaler.transform(window)
    x = torch.from_numpy(window).unsqueeze(0).to(device)  # [1, L, F]

    with torch.no_grad():
        pred = float(model(x).cpu().item())

    last_ts = df.index[-1] if ts_col == "window_start_hist" else df[ts_col].iloc[-1]
    print(f"[infer] last window: {last_ts}")
    print(f"[infer] predicted next-window volatility: {pred:.6f}")
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["parquet", "historical"], default="parquet")
    args = ap.parse_args()
    predict_latest(args.source)
