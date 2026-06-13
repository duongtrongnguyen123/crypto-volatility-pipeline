"""Offline feature builder — merge the historical 5-minute eth-alpha dataset
into the canonical 11-feature matrix used to train the volatility LSTM.

Source files (per symbol, in HISTORICAL_DIR):
    {SYM}_5min_long.csv      OHLCV: open, high, low, close, volume, quote_volume,
                             n_trades, taker_buy_volume
    {SYM}_metrics_full.csv   futures: sum_open_interest (and long/short ratios)
    {SYM}_funding.csv        perpetual funding rate (sparse, ~8h cadence)
    {SYM}_bookdepth_5min.csv order-book depth at +/- 0.2% .. 5% (starts 2023)
    {SYM}_liquidations_5min.csv  liquidation qty/notional (ETH only -> BTC uses
                             ETH as a market-wide liquidation-stress proxy)

The output columns exactly match config.FEATURE_COLUMNS plus `target`
(= next window's volatility) and the `timestamp` index.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config

# Liquidations only exist for ETH in the dataset; use it as a market-wide proxy
# for symbols (e.g. BTC) that lack their own file.
_LIQ_PROXY_SYMBOL = "ETHUSDT"


def _path(symbol: str, suffix: str) -> str:
    return os.path.join(config.HISTORICAL_DIR, f"{symbol}{suffix}")


def _load_ohlcv(symbol: str) -> pd.DataFrame:
    p = _path(symbol, "_5min_long.csv")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    out = pd.DataFrame(index=df.index)
    vol = df["volume"].replace(0, np.nan)
    out["vwap"] = (df["quote_volume"] / vol).ffill()
    out["price_return"] = (df["close"] - df["open"]) / df["open"]
    out["volume"] = df["volume"]
    out["trade_count"] = df["n_trades"]
    out["volatility"] = (df["high"] - df["low"]) / df["open"]
    out["taker_ls_ratio"] = (df["taker_buy_volume"] / vol).clip(0, 1).fillna(0.5)
    return out


def _reindex(series: pd.Series, index: pd.Index, how: str) -> pd.Series:
    """Align an auxiliary series onto the base 5-min index."""
    series = series[~series.index.duplicated(keep="last")].sort_index()
    aligned = series.reindex(index.union(series.index)).sort_index()
    if how == "ffill":
        aligned = aligned.ffill()
    aligned = aligned.reindex(index)
    if how == "zero":
        aligned = aligned.fillna(0.0)
    return aligned


def _load_open_interest(symbol: str, index: pd.Index) -> pd.Series:
    p = _path(symbol, "_metrics_full.csv")
    if not os.path.exists(p):
        return pd.Series(0.0, index=index)
    df = pd.read_csv(p, usecols=["create_time", "sum_open_interest"])
    df["create_time"] = pd.to_datetime(df["create_time"])
    s = df.set_index("create_time")["sum_open_interest"]
    return _reindex(s, index, how="ffill").fillna(0.0)


def _load_funding(symbol: str, index: pd.Index) -> pd.Series:
    p = _path(symbol, "_funding.csv")
    if not os.path.exists(p):
        return pd.Series(0.0, index=index)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("5min")
    s = df.set_index("timestamp")["funding_rate"]
    # Funding is set every ~8h; forward-fill across the 5-min grid.
    return _reindex(s, index, how="ffill").fillna(0.0)


def _load_book_depth(symbol: str, index: pd.Index) -> pd.Series:
    p = _path(symbol, "_bookdepth_5min.csv")
    if not os.path.exists(p):
        return pd.Series(0.0, index=index)
    df = pd.read_csv(p)
    df["bar_5m"] = pd.to_datetime(df["bar_5m"])
    df = df.set_index("bar_5m")
    # Near-touch notional within ~1% on both sides.
    cols = [c for c in ["depth_-1.0pct", "depth_1.0pct"] if c in df.columns]
    depth = df[cols].sum(axis=1, min_count=1)
    return _reindex(depth, index, how="zero")


def _load_liquidations(symbol: str, index: pd.Index) -> pd.Series:
    p = _path(symbol, "_liquidations_5min.csv")
    if not os.path.exists(p):
        p = _path(_LIQ_PROXY_SYMBOL, "_liquidations_5min.csv")
        if not os.path.exists(p):
            return pd.Series(0.0, index=index)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    notional = df["liq_buy_notional"].fillna(0) + df["liq_sell_notional"].fillna(0)
    return _reindex(notional, index, how="zero")


def build_features(symbol: str = None) -> pd.DataFrame:
    """Return a DataFrame with config.FEATURE_COLUMNS + `target`, indexed by time."""
    symbol = symbol or config.SYMBOL

    base = _load_ohlcv(symbol)
    idx = base.index

    base["open_interest"] = _load_open_interest(symbol, idx)
    base["funding_rate"] = _load_funding(symbol, idx)
    base["book_depth"] = _load_book_depth(symbol, idx)
    base["liq_notional"] = _load_liquidations(symbol, idx)
    # No historical news -> neutral sentiment. Live pipeline supplies real values.
    base["sentiment_score"] = 0.0

    base = base[config.FEATURE_COLUMNS].copy()

    # Clean: replace inf, drop rows still missing core OHLCV-derived features.
    base = base.replace([np.inf, -np.inf], np.nan)
    core = ["vwap", "price_return", "volume", "trade_count", "volatility"]
    base = base.dropna(subset=core)
    base = base.fillna(0.0)

    # Target = NEXT window's volatility; drop the final row (no future label).
    base["target"] = base[config.TARGET_COLUMN].shift(-1)
    base = base.dropna(subset=["target"])
    return base


if __name__ == "__main__":
    df = build_features()
    print(f"[historical] {len(df)} rows  {df.index.min()} -> {df.index.max()}")
    print(df[config.FEATURE_COLUMNS + ["target"]].describe().T[["mean", "std", "min", "max"]])
