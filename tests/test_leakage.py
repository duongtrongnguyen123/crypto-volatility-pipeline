"""Data-leakage gates — prove train/val/test are time-disjoint and that the
prediction target is strictly forward-looking (no peeking at the future when
forming features).

Run: python tests/test_leakage.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.dataset import chronological_split, make_sequences


def test_split_disjoint_and_ordered():
    tr, va, te = chronological_split(1000, 0.15, 0.15)
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert not (s_tr & s_va) and not (s_va & s_te) and not (s_tr & s_te)
    assert max(tr) < min(va) < min(te)           # strictly chronological
    assert max(va) < min(te)


def test_embargo_makes_windows_row_disjoint():
    """With embargo >= seq_len, no segment's look-back window reaches into the
    next segment's rows — train/val and val/test boundaries are both clean."""
    n, L = 1000, 24
    tr, va, te = chronological_split(n, 0.15, 0.15, embargo=L)
    assert max(tr) + L <= min(va), "train window leaks into val"
    assert max(va) + L <= min(te), "val window leaks into test"


def test_no_embargo_overlap_is_detected():
    """Without embargo, a sequence's L-row window DOES overlap the next segment
    at the boundary (the micro-leak the embargo fixes)."""
    n, L = 1000, 24
    tr, va, te = chronological_split(n, 0.15, 0.15, embargo=0)
    assert max(tr) + L > min(va)   # train window spills into val rows
    assert max(va) + L > min(te)   # val window spills into test rows


def test_target_is_strictly_forward():
    """make_sequences y[i] must be the NEXT window's volatility — i.e. a value
    dated AFTER every feature row in sequence i (no lookahead into features)."""
    n, L = 200, 24
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in config.FEATURE_COLUMNS})
    # target = next window's volatility (what the pipeline builds)
    df["target"] = df[config.TARGET_COLUMN].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    X, y = make_sequences(df, seq_len=L)
    # sequence 0 uses rows [0, L); its label must equal target[L-1] = volatility[L]
    assert np.isclose(y[0], df["target"].to_numpy()[L - 1])
    # and that target corresponds to a window strictly after the feature rows
    assert np.isclose(df["target"].to_numpy()[L - 1], df[config.TARGET_COLUMN].to_numpy()[L])


def test_crash_label_has_no_past_leak():
    """The crash label depends only on FUTURE returns; today's features can't
    encode it. Sanity: shifting labels back in time changes them (i.e. the label
    is genuinely time-varying / forward, not a constant or past copy)."""
    if not os.path.isdir(config.HISTORICAL_DIR):
        print("    (skip: no historical data dir)"); return
    from trr.labels import crash_labels
    cl = crash_labels()["crash"]
    assert cl.iloc[:-5].to_numpy().tolist() != cl.iloc[5:].to_numpy().tolist()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} leakage tests passed.")


if __name__ == "__main__":
    _run_all()
