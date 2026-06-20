"""Feature engineering for the TRR meta-learner.

Assembles a per-day feature table by merging:
  - the zero-shot LLM signal   (crash_prob, n_news, n_edges) from the kernel
    prediction CSVs (kaggle/out_<tag>/crash/trr_predictions.csv), and
  - price-derived TECHNICAL features (trailing returns, realised vol, drawdown)
    computed causally from the equal-weight portfolio prices.

Target = label_true (the forward 3-day crash label already in the CSVs).

This lets us ask the honest question: does TRAINING a model that combines the
LLM signal with cheap technicals beat (a) the zero-shot LLM alone and (b) a
news-volume baseline — and does the LLM add anything over technicals alone?

All technical features use only trailing data (no lookahead). Source eras are
labelled so we can do a true out-of-time / cross-source split.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from trr.prices import build_portfolio_daily

TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]

# tag -> (predictions csv, price dir, era)
SHARDS_2016_2020 = [f"s{i}" for i in range(1, 10)]   # analyst-ratings news
SHARDS_2021_2023 = [f"f{i}" for i in range(1, 6)]     # FNSPID news


def _pred_path(tag: str) -> str:
    return f"kaggle/out_{tag}/crash/trr_predictions.csv"


def _technical(price_dir: str) -> pd.DataFrame:
    """Causal trailing technical features of the equal-weight portfolio."""
    p = build_portfolio_daily(price_dir, TICKERS)
    r = p["portfolio_ret"]
    lvl = p["portfolio_level"]
    out = pd.DataFrame(index=p.index)
    out["ret_1d"] = r
    out["ret_5d"] = lvl / lvl.shift(5) - 1.0
    out["ret_10d"] = lvl / lvl.shift(10) - 1.0
    out["vol_10d"] = r.rolling(10).std()
    out["vol_20d"] = r.rolling(20).std()
    out["downside_5d"] = r.clip(upper=0).rolling(5).sum()
    out["dd_from_high_20d"] = lvl / lvl.rolling(20, min_periods=1).max() - 1.0
    return out


def build_dataset() -> pd.DataFrame:
    """Pooled per-day feature table across all available shards."""
    tech_2016 = _technical("data/fnspid/prices")  # 2016-2023 prices cover both eras
    rows = []
    for tag, era in ([(t, "2016_2020") for t in SHARDS_2016_2020]
                     + [(t, "2021_2023") for t in SHARDS_2021_2023]):
        path = _pred_path(tag)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, index_col=0)
        d.index = pd.to_datetime(d.index).date
        d = d[["crash_prob", "n_news", "n_edges", "label_true"]].copy()
        d["era"] = era
        d["shard"] = tag
        rows.append(d)
    llm = pd.concat(rows)
    llm = llm[~llm.index.duplicated(keep="first")].sort_index()

    tech = tech_2016.copy()
    tech.index = pd.to_datetime(tech.index).date
    df = llm.join(tech, how="left")
    df = df.dropna(subset=["vol_20d"])  # drop warm-up days with no trailing window
    # log news volume (heavy-tailed)
    df["log_news"] = np.log1p(df["n_news"])
    df["log_edges"] = np.log1p(df["n_edges"])
    return df


FEATURES_FULL = ["crash_prob", "log_news", "log_edges", "ret_1d", "ret_5d",
                 "ret_10d", "vol_10d", "vol_20d", "downside_5d", "dd_from_high_20d"]
FEATURES_TECH = ["ret_1d", "ret_5d", "ret_10d", "vol_10d", "vol_20d",
                 "downside_5d", "dd_from_high_20d", "log_news", "log_edges"]


if __name__ == "__main__":
    df = build_dataset()
    print(f"[features] {len(df)} days  {df.index.min()}..{df.index.max()}")
    print(f"[features] crashes={int(df['label_true'].sum())} ({df['label_true'].mean():.1%})")
    print(f"[features] by era:\n{df.groupby('era')['label_true'].agg(['size','sum','mean'])}")
    print(f"[features] columns: {FEATURES_FULL}")
    print(df[FEATURES_FULL + ['label_true']].describe().round(3).T.to_string())
