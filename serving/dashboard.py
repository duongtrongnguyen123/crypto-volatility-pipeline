"""Streamlit dashboard for the LOCAL crash-prediction serving tier.

Renders three panels:
    1. Crash-risk timeline   — crash probability over time.
    2. Portfolio equity      — buy&hold vs the de-risk strategy
                               (serving.paper_trader.simulate).
    3. Latest headlines/impacts — most recent crash-signal rows.

Data source: the live Parquet crash-signal store written by
processing/consumer_trr.py. When that store is absent (e.g. before the streaming
pipeline has produced anything) the dashboard falls back to a deterministic
SYNTHETIC series so it always renders.

Run locally:
    streamlit run serving/dashboard.py

`streamlit` is imported lazily inside `main()` so this module is importable for
testing/CI on a box without streamlit installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from serving.api import read_latest_signal
from serving.paper_trader import simulate

try:  # pragma: no cover - exercised only when the live store exists
    from serving.api import CRASH_SIGNAL_DIR
except Exception:  # pragma: no cover
    CRASH_SIGNAL_DIR = "./data/features/crash_signal"

import glob
import os


def load_signal_timeline() -> tuple[pd.DataFrame, bool]:
    """Load the crash-signal timeline from Parquet, or a synthetic fallback.

    Returns (frame, is_live) where frame has a DatetimeIndex and a `crash_prob`
    column (plus `crash_risk`, `n_edges`, `assets_hit` when live).
    """
    if os.path.isdir(CRASH_SIGNAL_DIR):
        files = glob.glob(os.path.join(CRASH_SIGNAL_DIR, "**", "*.parquet"),
                          recursive=True)
        if files:
            try:
                df = pd.concat((pd.read_parquet(f) for f in files),
                               ignore_index=True)
                if not df.empty and "window_start" in df.columns:
                    df["window_start"] = pd.to_datetime(df["window_start"])
                    df = df.sort_values("window_start").set_index("window_start")
                    df = df.rename(columns={"crash_risk": "crash_prob"})
                    return df, True
            except Exception:
                pass
    return _synthetic_timeline(), False


def _synthetic_timeline() -> pd.DataFrame:
    """Deterministic synthetic crash-risk timeline with one crash spike, so the
    dashboard renders meaningfully without any live data."""
    rng = np.random.default_rng(7)
    n = 60
    idx = pd.date_range("2026-05-01", periods=n, freq="D")
    base = 0.12 + 0.04 * rng.standard_normal(n)
    # A crash episode in the middle: risk ramps up then fades.
    spike = np.zeros(n)
    spike[28:34] = np.array([0.3, 0.55, 0.8, 0.7, 0.5, 0.35])
    crash_prob = np.clip(base + spike, 0.0, 1.0)
    return pd.DataFrame({
        "crash_prob": crash_prob,
        "n_edges": rng.integers(1, 12, n),
        "assets_hit": rng.integers(1, 6, n),
    }, index=idx)


def _synthetic_returns(index: pd.DatetimeIndex, crash_prob: pd.Series) -> pd.Series:
    """Forward returns that go sharply negative during the high-risk window, so
    de-risking visibly helps — used only for the synthetic fallback panel."""
    rng = np.random.default_rng(11)
    rets = 0.002 + 0.01 * rng.standard_normal(len(index))
    rets = rets - 0.08 * (crash_prob.values >= 0.5)  # crashes when risk is high
    return pd.Series(rets, index=index)


def main() -> None:  # pragma: no cover - requires streamlit + a browser
    import streamlit as st

    st.set_page_config(page_title="Crypto Crash Risk", layout="wide")
    st.title("Crypto Crash-Prediction — Live Serving")

    df, is_live = load_signal_timeline()
    st.caption(
        ("LIVE data from the crash-signal Parquet store."
         if is_live else
         "No live data yet — showing a SYNTHETIC fallback so the dashboard "
         "renders. Start processing/consumer_trr.py to populate the store.")
    )

    # Panel 1 — crash-risk timeline.
    st.subheader("Crash-risk timeline")
    st.line_chart(df["crash_prob"])

    # Panel 2 — portfolio equity: buy&hold vs de-risk.
    st.subheader("Portfolio equity — buy & hold vs de-risk")
    latest = read_latest_signal()
    if is_live and "n_neg" in df.columns:
        # No realised forward returns in the live store; approximate with a
        # placeholder of zeros so the curve is flat but honest.
        fwd = pd.Series(0.0, index=df.index)
    else:
        fwd = _synthetic_returns(df.index, df["crash_prob"])
    result = simulate(df["crash_prob"], fwd, threshold=0.5)
    equity = pd.DataFrame({
        "de_risk": result["equity_curve"],
        "buy_hold": result["buy_hold"]["equity_curve"],
    })
    st.line_chart(equity)
    col1, col2, col3 = st.columns(3)
    col1.metric("De-risk total return", f"{result['total_return']:.1%}")
    col2.metric("Buy&hold total return",
                f"{result['buy_hold']['total_return']:.1%}")
    col3.metric("Max-drawdown improvement",
                f"{result['drawdown_improvement']:.1%}")

    # Panel 3 — latest headlines / impacts.
    st.subheader("Latest crash-signal")
    if latest is not None:
        st.json(latest)
    else:
        st.info("No live crash-signal rows yet.")
    st.dataframe(df.tail(15))


if __name__ == "__main__":  # pragma: no cover
    main()
