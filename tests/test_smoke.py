"""Fast smoke tests for the ML core — no GPU, no Kafka, no network.

Validates the invariants the rest of the pipeline depends on:
    - chronological_split partitions are ordered, disjoint, and cover [0, n)
    - make_sequences produces the right shapes and label alignment
    - Standardizer round-trips through its dict form
    - LSTMVolatility runs and reconstructs exactly from .hparams
    - baseline predictions align 1:1 with the model's test targets

Run:
    python -m pytest tests/ -q
    # or without pytest:
    python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow `python tests/test_smoke.py` from anywhere by putting the repo root first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.dataset import (
    Standardizer,
    chronological_split,
    feature_matrix,
    make_sequences,
)


def _toy_frame(n: int = 200) -> pd.DataFrame:
    """A deterministic frame with all feature columns + a target column."""
    rng = np.random.default_rng(0)
    data = {col: rng.random(n) + 1.0 for col in config.FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    df["target"] = df[config.TARGET_COLUMN].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def test_chronological_split_is_a_partition():
    train_idx, val_idx, test_idx = chronological_split(100, val_frac=0.15, test_frac=0.15)
    combined = list(train_idx) + list(val_idx) + list(test_idx)
    assert combined == list(range(100))                  # ordered + complete
    assert len(test_idx) == 15 and len(val_idx) == 15    # fractions honored
    assert max(train_idx) < min(val_idx) < min(test_idx)  # strictly chronological


def test_make_sequences_shapes_and_alignment():
    df = _toy_frame(200)
    seq_len = 24
    X, y = make_sequences(df, seq_len=seq_len)
    assert X.shape == (len(df) - seq_len + 1, seq_len, len(config.FEATURE_COLUMNS))
    assert y.shape == (X.shape[0],)
    # Label of sequence i is the target of the sequence's last row.
    assert np.isclose(y[0], df["target"].to_numpy()[seq_len - 1])
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_feature_matrix_log1p_applied():
    df = _toy_frame(50)
    fm = feature_matrix(df)
    # log-compressed columns must differ from their raw values.
    for col in config.LOG_FEATURES:
        j = config.FEATURE_COLUMNS.index(col)
        raw = df[col].to_numpy()
        assert np.allclose(fm[:, j], np.log1p(np.clip(raw, 0, None)), atol=1e-5)


def test_standardizer_roundtrip():
    df = _toy_frame(200)
    X, _ = make_sequences(df)
    s = Standardizer.fit(X)
    s2 = Standardizer.from_dict(s.to_dict())
    assert np.allclose(s.mean, s2.mean) and np.allclose(s.std, s2.std)
    # Standardized features are ~zero-mean / unit-std (constant cols excluded).
    flat = s.transform(X).reshape(-1, X.shape[-1]).astype(np.float64)
    nonconst = s.std > 0
    assert np.abs(flat.mean(0)[nonconst]).max() < 1e-3


def test_model_forward_and_hparam_roundtrip():
    import torch

    from ml.model import LSTMVolatility

    model = LSTMVolatility(input_size=len(config.FEATURE_COLUMNS))
    x = torch.randn(4, config.SEQUENCE_LENGTH, len(config.FEATURE_COLUMNS))
    y = model(x)
    assert tuple(y.shape) == (4,)
    rebuilt = LSTMVolatility(**model.hparams)
    rebuilt.load_state_dict(model.state_dict())  # architectures must match exactly
    assert tuple(rebuilt(x).shape) == (4,)


def test_baseline_alignment():
    from ml.baselines import baseline_predictions

    df = _toy_frame(200)
    seq_len = 24
    X, y = make_sequences(df, seq_len=seq_len)
    _, _, test_idx = chronological_split(len(X), 0.15, 0.15)
    preds = baseline_predictions(df, test_idx, seq_len)
    assert "persistence" in preds
    for name, p in preds.items():
        assert p.shape == (len(test_idx),), f"{name} misaligned: {p.shape}"
        assert np.isfinite(p).all()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
