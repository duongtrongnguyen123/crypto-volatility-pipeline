"""Build the RAG-selected stock news file from the 2016-2023 FNSPID corpus.

This is the LOCAL half of the decoupled design: RAG retrieval (CPU) runs here,
producing the bounded per-day slices the offline 32B kernel reasons over. For
every trading day in the window we pull that day's full pool from the corpus
index and keep only the k most crash/portfolio-relevant items (select_relevant).

The output (date,title,assets,source) is a drop-in replacement for
data/stockdata/stocknews.csv — point any Kaggle stock kernel's dataset at it and
the backtest runs over the full 4.5M-article corpus, while the LLM still reads
only k items/day.

Usage:
    python -m scripts.build_corpus_news \
        --start 2016-01-01 --end 2023-12-31 --k 20 \
        --out data/stockdata/stocknews_corpus.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from trr.corpus import CorpusIndex
from trr.select import select_salient

STOCK_TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
# FNSPID tags Google mostly as GOOG; treat it as GOOGL for the portfolio.
PSET = {t.upper() for t in STOCK_TICKERS} | {"GOOG"}


def trading_days(price_csv: str, start: str, end: str) -> list[str]:
    """Trading days (YYYY-MM-DD) from a price file, within [start, end]."""
    df = pd.read_csv(price_csv)
    col = df.columns[0]
    days = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return [d for d in days if start <= d <= end]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--price", default="data/ohlcv/AAPL.csv")
    ap.add_argument("--db", default="data/fnspid_corpus/news.db")
    ap.add_argument("--out", default="data/stockdata/stocknews_corpus.csv")
    args = ap.parse_args()

    idx = CorpusIndex(args.db)
    days = trading_days(args.price, args.start, args.end)
    print(f"[corpus-news] {len(days)} trading days {args.start}..{args.end}, k={args.k} "
          f"(PORTFOLIO-filtered: {sorted(PSET)})")

    rows, pool_total, pf_total, t0 = [], 0, 0, time.time()
    for i, d in enumerate(days, 1):
        pool = idx.day(d)
        pool_total += len(pool)
        # PORTFOLIO FILTER: keep only news tagged with a portfolio ticker, and
        # normalise GOOG -> GOOGL so the pipeline's portfolio matching sees it.
        pf = []
        for it in pool:
            if {a.upper() for a in it.assets} & PSET:
                it.assets = ["GOOGL" if a.upper() in ("GOOG", "GOOGL") else a.upper()
                             for a in it.assets]
                pf.append(it)
        pf_total += len(pf)
        # Salience-rank WITHIN the portfolio pool (no fixed crash query, so calm
        # days surface calm headlines — the discrimination the all-ticker
        # crash-query selection destroyed).
        sel = select_salient(pf, args.k, STOCK_TICKERS) if pf else []
        for it in sel:
            rows.append({"date": d, "title": it.title,
                         "assets": ",".join(it.assets), "source": it.source or "fnspid"})
        if i % 200 == 0:
            dt = time.time() - t0
            print(f"  [{i}/{len(days)}] kept={len(rows)} pf_pool={pf_total} "
                  f"(avg pf {pf_total/i:.1f}/day)  {dt:.0f}s  eta {dt/i*(len(days)-i):.0f}s",
                  flush=True)

    pd.DataFrame(rows, columns=["date", "title", "assets", "source"]).to_csv(
        args.out, index=False)
    idx.close()
    print(f"DONE: {len(rows)} portfolio items over {len(days)} days "
          f"(avg pf-pool {pf_total/max(1,len(days)):.1f}/day, raw pool "
          f"{pool_total/max(1,len(days)):.0f}/day) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
