"""Build the STOCK dataset (prices + news) to run TRR on equities — closing the
"crypto vs stock" gap vs the assignment title.

Portfolio: 6 large-cap tech names with heavy news coverage + tight correlation
(relational structure) and a clean crash event (COVID, Mar 2020).
  - prices : yfinance daily close per ticker -> data/stockdata/prices/{T}.csv
  - news   : concat the 3 Kaggle headline files, filter to the portfolio + window,
             normalize to (date, title, assets, source) -> data/stockdata/stocknews.csv
"""
from __future__ import annotations

import os

import pandas as pd

TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
# Wide backtest window: dense news for these 6 tickers starts ~2016 (earlier
# years are <1 headline/day). Sharded into 6-month chunks downstream.
START = os.environ.get("STOCK_START", "2016-01-01")
END = os.environ.get("STOCK_END", "2020-06-15")
OUT = "data/stockdata"
NEWS_SRC = "data/stocknews"


def build_prices():
    import yfinance as yf
    os.makedirs(f"{OUT}/prices", exist_ok=True)
    for t in TICKERS:
        df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  [warn] no price data for {t}"); continue
        close = df["Close"]
        close = close[t] if hasattr(close, "columns") else close
        out = pd.DataFrame({"date": pd.to_datetime(df.index).strftime("%Y-%m-%d"),
                            "close": close.to_numpy().ravel()})
        out.to_csv(f"{OUT}/prices/{t}.csv", index=False)
        print(f"  {t}: {len(out)} days  {out['date'].iloc[0]}..{out['date'].iloc[-1]}")


def build_news():
    frames = []
    files = [("analyst_ratings_processed.csv", "title"),
             ("raw_analyst_ratings.csv", "headline"),
             ("raw_partner_headlines.csv", "headline")]
    tickset = set(TICKERS)
    for fname, tcol in files:
        p = os.path.join(NEWS_SRC, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, usecols=lambda c: c in (tcol, "date", "stock"))
        df = df.rename(columns={tcol: "title"})
        df = df[df["stock"].isin(tickset)]
        df["dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["dt", "title"])
        df = df[(df["dt"] >= START) & (df["dt"] <= END)]
        frames.append(df[["dt", "title", "stock"]])
    alln = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["dt", "title", "stock"])
    out = pd.DataFrame({
        "date": alln["dt"].dt.strftime("%Y-%m-%d"),
        "title": alln["title"].astype(str),
        "assets": alln["stock"].astype(str),
        "source": "kaggle-stocknews",
    }).sort_values("date")
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/stocknews.csv", index=False)
    print(f"  stock news: {len(out)} headlines, {out['date'].nunique()} days, "
          f"{out['date'].min()}..{out['date'].max()}, avg {len(out)/out['date'].nunique():.1f}/day")


if __name__ == "__main__":
    print("=== prices (yfinance) ===")
    build_prices()
    print("=== news ===")
    build_news()
