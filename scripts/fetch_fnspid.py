"""Stream-filter the 23GB FNSPID news CSV (read from stdin) down to just our 6
portfolio tickers — never storing the full file. Keeps title + date + symbol
(the fields TRR needs), matching our existing stocknews.csv schema.

Usage:
    curl -sL "<FNSPID nasdaq_exteral_data.csv URL>" | python -m scripts.fetch_fnspid
"""
from __future__ import annotations

import sys

import pandas as pd

TICKERS = set(os.environ.get(
    "FNSPID_TICKERS", "AAPL,AMZN,GOOGL,NVDA,TSLA,NFLX").split(","))
MIN_YEAR = int(os.environ.get("FNSPID_MIN_YEAR", "2016"))
OUT = os.environ.get("FNSPID_OUT", "data/fnspid/stocknews.csv")

import os
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

kept = 0
chunks_seen = 0
first_write = True
reader = pd.read_csv(
    sys.stdin,
    usecols=["Date", "Article_title", "Stock_symbol"],
    dtype=str,
    chunksize=100_000,
    on_bad_lines="skip",
    engine="c",
)
for ch in reader:
    chunks_seen += 1
    ch = ch[ch["Stock_symbol"].isin(TICKERS)]
    if not ch.empty:
        dt = pd.to_datetime(ch["Date"], errors="coerce", utc=True)
        ch = ch[dt.dt.year >= MIN_YEAR].copy()
        ch["date"] = pd.to_datetime(ch["Date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        out = pd.DataFrame({
            "date": ch["date"],
            "title": ch["Article_title"].astype(str),
            "assets": ch["Stock_symbol"].astype(str),
            "source": "fnspid",
        }).dropna(subset=["date"])
        out.to_csv(OUT, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        kept += len(out)
    if chunks_seen % 20 == 0:
        print(f"  [{chunks_seen*100}k rows scanned] kept={kept}", flush=True)

print(f"DONE: scanned ~{chunks_seen*100}k rows, kept {kept} headlines -> {OUT}", flush=True)
