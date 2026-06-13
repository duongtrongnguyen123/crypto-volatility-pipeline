"""Generate a synthetic live feature store (parquet) matching the 11-feature
schema produced by processing/feature_join.py.

Useful for exercising the `--source parquet` ML path without a running stream.
The primary training path uses the real historical dataset (ml/historical.py);
this is only a stand-in for the *live* store.

Run:
    python -m scripts.generate_sample_features --rows 3000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2026-01-01 00:00:00")
    window_start = pd.date_range(start, periods=rows, freq="5min")

    base_price = 60000.0
    returns = rng.normal(0, 0.0015, size=rows)
    price = base_price * np.exp(np.cumsum(returns))

    # GARCH-like volatility clustering.
    vol = np.empty(rows)
    vol[0] = 0.002
    for i in range(1, rows):
        vol[i] = abs(0.9 * vol[i - 1] + rng.normal(0, 0.0005))

    df = pd.DataFrame(
        {
            "window_start": window_start,
            "window_end": window_start + pd.Timedelta(minutes=5),
            "vwap": price,
            "price_return": returns,
            "volume": rng.gamma(2.0, 100.0, size=rows),
            "trade_count": rng.integers(2000, 20000, size=rows),
            "volatility": vol,
            "sentiment_score": np.clip(rng.normal(0, 0.4, size=rows), -1, 1),
            "open_interest": 80000 + np.cumsum(rng.normal(0, 200, size=rows)),
            "funding_rate": rng.normal(0.0001, 0.00008, size=rows),
            "taker_ls_ratio": np.clip(rng.normal(0.5, 0.08, size=rows), 0, 1),
            "book_depth": rng.gamma(3.0, 7e7, size=rows),
            "liq_notional": rng.gamma(0.3, 5e6, size=rows),
        }
    )
    return df[["window_start", "window_end"] + config.FEATURE_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    df = generate(args.rows, args.seed)
    out = os.path.join(config.FEATURES_DIR, "sample_features.parquet")
    df.to_parquet(out, index=False)
    print(f"[sample] wrote {len(df)} rows ({len(config.FEATURE_COLUMNS)} features) -> {out}")


if __name__ == "__main__":
    main()
