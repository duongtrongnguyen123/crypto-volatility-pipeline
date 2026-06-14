"""Portfolio crash labels from price data — the TRR ground truth.

Builds an equal-weight crypto portfolio from the daily closes of the PORTFOLIO
assets and labels each day as a "crash" if the portfolio's forward return over
the next `horizon` days breaches `-threshold` (a sharp drawdown). This mirrors
the crash-detection target of arXiv:2410.17266, adapted to crypto.

Crash events are intentionally imbalanced (rare) — hence AUROC for evaluation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
from trr.schema import PORTFOLIO, SYMBOLS

# A day is a "crash" if the equal-weight portfolio falls more than this over the
# forward window. 8% over 3 days is a severe multi-asset crypto drawdown.
DEFAULT_THRESHOLD = 0.08
DEFAULT_HORIZON = 3  # days


def _load_daily_close(symbol: str, hist_dir: str) -> pd.Series:
    path = os.path.join(hist_dir, f"{symbol}_5min_long.csv")
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.set_index("timestamp")["close"].sort_index()
    # Last close of each calendar day.
    return s.resample("1D").last().dropna()


def build_portfolio(hist_dir: str = None) -> pd.DataFrame:
    """Equal-weight daily portfolio level + per-asset daily returns."""
    hist_dir = hist_dir or config.HISTORICAL_DIR
    closes = {}
    for ticker in PORTFOLIO:
        closes[ticker] = _load_daily_close(SYMBOLS[ticker], hist_dir)
    px = pd.DataFrame(closes).dropna(how="all").sort_index()
    px = px.ffill().dropna()

    rets = px.pct_change().fillna(0.0)
    # Equal-weight portfolio daily return -> cumulative level.
    port_ret = rets.mean(axis=1)
    port_level = (1.0 + port_ret).cumprod()

    out = rets.add_suffix("_ret")
    out["portfolio_ret"] = port_ret
    out["portfolio_level"] = port_level
    return out


def crash_labels(
    hist_dir: str = None,
    threshold: float = DEFAULT_THRESHOLD,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Return a frame indexed by day with a `crash` 0/1 column and the forward
    return used to derive it.
    """
    port = build_portfolio(hist_dir)
    level = port["portfolio_level"]

    # Forward return over the next `horizon` days: min level ahead / today - 1.
    fwd_min = level.shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    # Simpler, robust forward drawdown: lowest close within the next horizon days.
    fwd_low = (
        level.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    )
    fwd_ret = fwd_low / level - 1.0

    port = port.copy()
    port["fwd_ret"] = fwd_ret
    port["crash"] = (fwd_ret <= -threshold).astype("Int64")
    # Drop the tail where the forward window is undefined.
    port = port.dropna(subset=["fwd_ret"]).copy()
    port["crash"] = port["crash"].astype(int)
    return port


if __name__ == "__main__":
    df = crash_labels()
    n = len(df)
    n_crash = int(df["crash"].sum())
    print(f"[labels] {n} days {df.index.min().date()} -> {df.index.max().date()}")
    print(f"[labels] crash days: {n_crash} ({100 * n_crash / n:.1f}%)  "
          f"threshold={DEFAULT_THRESHOLD:.0%} horizon={DEFAULT_HORIZON}d")
    # Show the worst forward-drawdown days — should surface known crypto crashes.
    worst = df.nsmallest(8, "fwd_ret")[["fwd_ret", "crash"]]
    print("[labels] worst forward drawdowns:")
    for ts, row in worst.iterrows():
        print(f"    {ts.date()}  fwd_ret={row['fwd_ret']:+.1%}  crash={int(row['crash'])}")
