"""Naive next-window volatility baselines.

These predictors require no learning; they extrapolate the next window's
volatility directly from the observed volatility history. They exist so the
LSTM in ml/evaluate.py is compared against an honest reference rather than only
against itself: a model that cannot beat "tomorrow looks like today" has not
earned its complexity.

Index alignment (the part that's easy to get subtly wrong)
----------------------------------------------------------
ml.dataset.make_sequences builds, for each i in range(len(df) - L + 1):
    X[i] = feature rows [i, i+L)         (the L windows fed to the model)
    y[i] = df["target"][i+L-1]
         = volatility at row (i+L)       (the NEXT window after the sequence)

So sequence i looks at observed volatilities at rows i .. i+L-1 and must predict
the volatility at row i+L. Every baseline below is therefore a function of the
observed volatility *inside the sequence window* (rows i .. i+L-1) only — it
never peeks at row i+L (that would be leakage).

The "most recent observed volatility" for sequence i is the volatility at row
i+L-1 (the last window in the sequence). That is the persistence prediction and
also the reference point used for directional accuracy in ml/evaluate.py.

`baseline_predictions(df, test_idx, seq_len)` returns each baseline's predictions
aligned 1:1 with the model's y_test (i.e. with X[test_idx] / y[test_idx]).
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

import config


def observed_volatility(df: pd.DataFrame) -> np.ndarray:
    """The raw, un-transformed volatility column as float32 [N]."""
    return df[config.TARGET_COLUMN].to_numpy(dtype=np.float32)


def _sequence_windows(vol: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack the in-sequence volatility windows: returns [M, seq_len].

    Row i holds vol[i : i+seq_len] — exactly the volatilities the model sees in
    sequence i. M = len(vol) - seq_len + 1 matches make_sequences' sequence count.
    """
    n_seq = len(vol) - seq_len + 1
    if n_seq <= 0:
        raise ValueError(
            f"not enough rows ({len(vol)}) for a sequence of length {seq_len}"
        )
    # Sliding window view, materialized to a plain contiguous array.
    idx = np.arange(seq_len)[None, :] + np.arange(n_seq)[:, None]
    return vol[idx]


def persistence(windows: np.ndarray) -> np.ndarray:
    """Predict next volatility = the most recent observed volatility.

    For each sequence that is the last column (row i+L-1 of the original frame).
    """
    return windows[:, -1].astype(np.float32)


def rolling_mean(windows: np.ndarray, k: int) -> np.ndarray:
    """Mean of the last `k` observed volatilities in each sequence."""
    k = min(k, windows.shape[1])
    return windows[:, -k:].mean(axis=1).astype(np.float32)


def ewma(windows: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average over each sequence window.

    Weights decay toward older windows: the most recent window gets weight
    proportional to 1, the previous to (1-alpha), etc. We use the standard
    finite-window EWMA (not the recursive seed) so every prediction is a clean
    normalized weighted average of exactly the in-sequence volatilities.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    seq_len = windows.shape[1]
    # Most recent step (last column) is age 0 -> weight (1-alpha)^0 = 1.
    ages = np.arange(seq_len - 1, -1, -1, dtype=np.float64)  # [L-1, ..., 1, 0]
    weights = (1.0 - alpha) ** ages
    weights /= weights.sum()
    return (windows.astype(np.float64) @ weights).astype(np.float32)


def baseline_predictions(
    df: pd.DataFrame,
    test_idx: Iterable[int],
    seq_len: int,
    rolling_ks: tuple[int, ...] = (6, 12),
    ewma_alpha: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Compute every baseline's predictions aligned 1:1 with the model's y_test.

    Args:
        df:        the feature frame already sliced the same way the model saw it
                   (e.g. df.iloc[-max_rows:] when --max-rows was used at train time).
        test_idx:  the SEQUENCE indices of the held-out test set, i.e. the same
                   range returned by chronological_split(len(X), ...). These index
                   into the array of all sequences, so test_idx[j] selects window
                   row test_idx[j] of `_sequence_windows`.
        seq_len:   the sequence length the model was trained with.

    Returns:
        dict mapping baseline name -> predictions [len(test_idx)] float32, each
        aligned exactly with y[test_idx].
    """
    vol = observed_volatility(df)
    windows = _sequence_windows(vol, seq_len)  # [num_sequences, seq_len]

    test_idx = np.asarray(list(test_idx), dtype=np.int64)
    if test_idx.size and test_idx.max() >= len(windows):
        raise ValueError(
            f"test_idx max {int(test_idx.max())} out of range for "
            f"{len(windows)} sequences"
        )
    test_windows = windows[test_idx]  # [num_test, seq_len]

    preds: Dict[str, np.ndarray] = {
        "persistence": persistence(test_windows),
        "ewma": ewma(test_windows, ewma_alpha),
    }
    for k in rolling_ks:
        preds[f"rolling_mean({k})"] = rolling_mean(test_windows, k)
    return preds


def last_observed_volatility(
    df: pd.DataFrame, test_idx: Iterable[int], seq_len: int
) -> np.ndarray:
    """The most-recent observed volatility for each test sequence (row i+L-1).

    This is the reference point for directional accuracy: did next-window
    volatility rise or fall relative to this value? It equals the persistence
    prediction, exposed separately for clarity at the call site.
    """
    vol = observed_volatility(df)
    windows = _sequence_windows(vol, seq_len)
    test_idx = np.asarray(list(test_idx), dtype=np.int64)
    return windows[test_idx][:, -1].astype(np.float32)


if __name__ == "__main__":
    # Smoke test: build sequences the same way the model does, run every baseline
    # on a held-out slice, and confirm shapes line up with y_test.
    from ml.dataset import chronological_split, load_feature_frame, make_sequences

    df = load_feature_frame("historical")
    df = df.iloc[-20000:].reset_index(drop=True)  # fast slice, mirrors --max-rows
    X, y = make_sequences(df, config.SEQUENCE_LENGTH)
    _, _, test_idx = chronological_split(len(X), 0.15, 0.15)
    y_test = y[np.asarray(list(test_idx))]

    preds = baseline_predictions(df, test_idx, config.SEQUENCE_LENGTH)
    last_obs = last_observed_volatility(df, test_idx, config.SEQUENCE_LENGTH)

    print(f"frame rows={len(df)}  sequences={len(X)}  test={len(y_test)}")
    print(f"y_test shape={y_test.shape}")
    for name, p in preds.items():
        assert p.shape == y_test.shape, f"{name} {p.shape} != {y_test.shape}"
        assert np.isfinite(p).all(), f"{name} has non-finite values"
        print(f"  {name:18s} shape={p.shape}  mean={p.mean():.6e}")
    assert last_obs.shape == y_test.shape
    # persistence prediction must equal the last observed volatility by definition.
    assert np.allclose(preds["persistence"], last_obs)
    print(f"  {'last_observed':18s} shape={last_obs.shape}  (== persistence: OK)")
    print("OK")
