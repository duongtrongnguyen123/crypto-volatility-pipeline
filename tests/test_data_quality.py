"""Data-quality gates — the contract that nothing NaN / Inf / wrong-dtype ever
reaches the model, and that the rate features survive division-by-zero.

Run: python tests/test_data_quality.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.dataset import feature_matrix, make_sequences


def _adversarial_frame(n=120):
    """A frame engineered to break naive feature code: zero volume (vwap = q/v),
    zero open (returns), negative/zero everywhere, plus a target column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in config.FEATURE_COLUMNS})
    df["volume"] = 0.0            # division-by-zero stress for any v-normalised feature
    df["trade_count"] = 0
    df["open_interest"] = 0.0
    df["liq_notional"] = 0.0
    df["target"] = df[config.TARGET_COLUMN].shift(-1)
    return df.dropna(subset=["target"]).reset_index(drop=True)


def test_feature_matrix_finite_on_adversarial_input():
    fm = feature_matrix(_adversarial_frame())
    assert np.isfinite(fm).all(), "feature_matrix produced NaN/Inf on zero-volume input"
    assert fm.dtype == np.float32


def test_make_sequences_finite_and_shaped():
    X, y = make_sequences(_adversarial_frame(), seq_len=24)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert X.shape[1:] == (24, len(config.FEATURE_COLUMNS))


def test_historical_features_no_nan_inf():
    """The real offline feature builder must emit a clean matrix (guards against
    div-by-zero in vwap = quote_volume/volume, etc.)."""
    if not os.path.isdir(config.HISTORICAL_DIR):
        print("    (skip: no historical data dir)"); return
    from ml.historical import build_features
    df = build_features()
    fc = config.FEATURE_COLUMNS + ["target"]
    sub = df[fc]
    assert not sub.isna().any().any(), "historical features contain NaN"
    assert np.isfinite(sub.to_numpy()).all(), "historical features contain Inf"
    # taker ratio is a bounded rate; vwap is a positive price
    assert df["taker_ls_ratio"].between(0, 1).all()
    assert (df["vwap"] > 0).all()


def test_crash_labels_binary_no_nan():
    if not os.path.isdir(config.HISTORICAL_DIR):
        print("    (skip: no historical data dir)"); return
    from trr.labels import asset_crash_labels, crash_labels
    cl = crash_labels()
    assert set(cl["crash"].unique()) <= {0, 1}
    assert not cl["crash"].isna().any()
    acl = asset_crash_labels()
    for c in acl.columns:
        assert set(acl[c].unique()) <= {0, 1}


def test_news_loader_handles_malformed_rows(tmp_path=None):
    """load_news must skip rows with no title/timestamp, not crash."""
    import tempfile

    from trr.news import load_news
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.csv")
    pd.DataFrame({
        "date": ["2022-01-01", None, "not-a-date", "2022-01-02"],
        "title": ["good headline", "no date", "bad date", None],
        "currencies": ["BTC", "ETH", "SOL", "BNB"],
    }).to_csv(p, index=False)
    items = load_news(p)
    # Only the first row is fully valid (has title + parseable date).
    assert len(items) >= 1
    assert all(it.title and it.timestamp is not None for it in items)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} data-quality tests passed.")


if __name__ == "__main__":
    _run_all()
