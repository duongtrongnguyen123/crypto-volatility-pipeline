"""Generic DAILY price loader + labels — lets TRR run on any asset class
(equities, FX, ...), not just the crypto 5-min CSVs. Used for the stock port.

Reads one CSV per ticker: `{price_dir}/{TICKER}.csv` with columns date, close.
Builds an equal-weight portfolio and forward-looking crash / direction labels —
mirroring trr.labels but asset-class-agnostic and at daily resolution.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def build_portfolio_daily(price_dir: str, tickers: list[str]) -> pd.DataFrame:
    closes = {}
    for t in tickers:
        p = os.path.join(price_dir, f"{t}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        closes[t] = df.set_index("date")["close"].sort_index()
    px = pd.DataFrame(closes).sort_index().ffill().dropna()
    rets = px.pct_change().fillna(0.0)
    out = rets.add_suffix("_ret")
    out["portfolio_ret"] = rets.mean(axis=1)
    out["portfolio_level"] = (1.0 + out["portfolio_ret"]).cumprod()
    return out


def crash_labels_daily(price_dir: str, tickers: list[str],
                       threshold: float = 0.06, horizon: int = 3) -> pd.DataFrame:
    """Equal-weight portfolio crash: forward `horizon`-day low breaches -threshold.
    Stocks are less volatile than crypto, so the default threshold is 6%.
    """
    port = build_portfolio_daily(price_dir, tickers)
    lvl = port["portfolio_level"]
    fwd_low = lvl.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    fwd_ret = fwd_low / lvl - 1.0
    port = port.copy()
    port["fwd_ret"] = fwd_ret
    port["crash"] = (fwd_ret <= -threshold).astype("Int64")
    port = port.dropna(subset=["fwd_ret"]).copy()
    port["crash"] = port["crash"].astype(int)
    return port


def direction_labels_daily(price_dir: str, tickers: list[str],
                           horizon: int = 1) -> pd.DataFrame:
    """Next-`horizon`-day portfolio direction: up=1 if forward return > 0."""
    port = build_portfolio_daily(price_dir, tickers)
    fwd = port["portfolio_level"].shift(-horizon) / port["portfolio_level"] - 1.0
    port = port.copy()
    port["fwd_ret"] = fwd
    port["up"] = (fwd > 0).astype("Int64")
    port = port.dropna(subset=["fwd_ret"]).copy()
    port["up"] = port["up"].astype(int)
    return port


if __name__ == "__main__":
    TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
    pd_ = "data/stockdata/prices"
    cl = crash_labels_daily(pd_, TICKERS)
    dl = direction_labels_daily(pd_, TICKERS)
    print(f"[stocks] {len(cl)} days {cl.index.min()}..{cl.index.max()}")
    print(f"[stocks] crashes (6%/3d): {int(cl['crash'].sum())} ({cl['crash'].mean():.1%})")
    print(f"[stocks] up-days: {dl['up'].mean():.1%}")
    worst = cl.nsmallest(5, "fwd_ret")[["fwd_ret", "crash"]]
    print("[stocks] worst forward drawdowns (expect COVID Feb-Mar 2020):")
    for d, r in worst.iterrows():
        print(f"    {d}  {r['fwd_ret']:+.1%}")
