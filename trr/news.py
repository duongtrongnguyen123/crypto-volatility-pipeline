"""News-data layer for the TRR crypto crash-detection pipeline.

The pipeline reasons over financial NEWS to detect upcoming crashes in the
PORTFOLIO. This module is the ingestion front-end: it loads an arbitrary
crypto-news file (``.jsonl`` or ``.csv``), normalizes every row into the shared
``trr.schema.NewsItem`` contract, and buckets items by calendar day for the
per-day Brainstorming/Reasoning phases.

Because we have NO real news dataset locally, ``trr/sample_news.jsonl`` ships a
deterministic SYNTHETIC corpus whose negative-headline clusters are aligned with
the real portfolio crash windows (LUNA/Terra, FTX, 3AC/Celsius, the Jan-2022
selloff) so the whole pipeline can be demonstrated offline end-to-end. The
synthetic items are clearly fictional/illustrative (``source: "synthetic"``).

Plugging in a REAL dataset
--------------------------
Download a Kaggle crypto-news dataset (e.g. a CryptoPanic / crypto-headlines
CSV) and simply point the loader at it::

    from trr.news import load_news, group_by_day
    news = load_news("/path/to/crypto_news.csv")
    by_day = group_by_day(news)

``load_news`` is schema-tolerant: the column-mapping below handles the common
header variants (timestamp/date/published_at, title/headline, body/content,
source/publisher, assets/tickers/currencies including CryptoPanic's list-of-
dicts ``currencies`` field), so most public datasets load with no extra work.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from typing import Any, Iterable

import pandas as pd

from trr.schema import PORTFOLIO, NewsItem

# --- Column-name variants we accept (first present wins) ---------------------
_TIMESTAMP_KEYS = ["timestamp", "date", "published_at", "time", "created_at"]
_TITLE_KEYS = ["title", "headline", "text", "content"]
_BODY_KEYS = ["body", "content", "text", "description", "summary"]
_SOURCE_KEYS = ["source", "source_title", "publisher", "domain"]
_ASSET_KEYS = ["assets", "tickers", "currencies", "symbols", "coins"]

_PORTFOLIO_SET = set(PORTFOLIO)

# Map common aliases / names to PORTFOLIO tickers. Anything that resolves to a
# portfolio ticker is kept; LUNA/UST etc. are kept verbatim (uppercased) because
# they carry crash signal even though they're not in the portfolio.
_ASSET_ALIASES = {
    "BITCOIN": "BTC",
    "XBT": "BTC",
    "BTC": "BTC",
    "ETHEREUM": "ETH",
    "ETHER": "ETH",
    "ETH": "ETH",
    "SOLANA": "SOL",
    "SOL": "SOL",
    "BINANCE": "BNB",
    "BNB": "BNB",
    "AVALANCHE": "AVAX",
    "AVAX": "AVAX",
    "DOGECOIN": "DOGE",
    "DOGE": "DOGE",
}


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the value of the first key in ``keys`` that has a usable value."""
    for key in keys:
        if key in row:
            val = row[key]
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
    return None


def normalize_asset(raw: str) -> str | None:
    """Normalize a single asset token to a ticker, or ``None`` if unusable."""
    if raw is None:
        return None
    token = str(raw).strip().upper()
    if not token:
        return None
    # Strip common quote-pair suffixes (BTCUSDT -> BTC, ETH-USD -> ETH).
    for sep in ("/", "-", "_"):
        if sep in token:
            token = token.split(sep)[0]
    for quote in ("USDT", "USD", "USDC", "BUSD"):
        if token.endswith(quote) and len(token) > len(quote):
            token = token[: -len(quote)]
    return _ASSET_ALIASES.get(token, token)


def _parse_assets(raw: Any) -> list[str]:
    """Parse the assets field from a list, delimited string, or list of dicts.

    Accepts CryptoPanic-style ``[{"code": "BTC", "title": "Bitcoin"}, ...]``,
    plain lists, and comma / ``|`` / ``;``-separated strings. Portfolio tickers
    are surfaced first (deduped, order-preserving), other resolved tokens kept.
    """
    if raw is None:
        return []
    tokens: list[str] = []

    if isinstance(raw, str):
        s = raw.strip()
        # A stringified JSON list (common in CSV exports).
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = json.loads(s)
            except (ValueError, TypeError):
                raw = None
                for sep in (",", "|", ";"):
                    s = s.replace(sep, ",")
                tokens = [t for t in s.strip("[]").split(",")]
        if isinstance(raw, str):
            for sep in ("|", ";"):
                raw = raw.replace(sep, ",")
            tokens = raw.split(",")

    if isinstance(raw, (list, tuple)):
        for elem in raw:
            if isinstance(elem, dict):
                tokens.append(str(elem.get("code") or elem.get("title") or ""))
            else:
                tokens.append(str(elem))

    out: list[str] = []
    portfolio_hits: list[str] = []
    for tok in tokens:
        norm = normalize_asset(tok)
        if not norm:
            continue
        if norm in _PORTFOLIO_SET:
            if norm not in portfolio_hits:
                portfolio_hits.append(norm)
        elif norm not in out:
            out.append(norm)
    return portfolio_hits + [t for t in out if t not in portfolio_hits]


def _parse_timestamp(raw: Any) -> dt.datetime | None:
    """Parse a timestamp to a UTC-naive ``datetime``, or ``None``."""
    if raw is None:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        ts = pd.to_datetime(raw, errors="coerce")
        if ts is pd.NaT or pd.isna(ts):
            return None
        return ts.to_pydatetime().replace(tzinfo=None)
    return ts.tz_convert(None).to_pydatetime()


def _row_to_item(row: dict[str, Any], index: int) -> NewsItem | None:
    """Normalize one raw record into a NewsItem (or ``None`` if unusable)."""
    timestamp = _parse_timestamp(_first_present(row, _TIMESTAMP_KEYS))
    title = _first_present(row, _TITLE_KEYS)
    if timestamp is None or title is None:
        return None
    title = str(title).strip()
    if not title:
        return None

    body_raw = _first_present(row, _BODY_KEYS)
    body = str(body_raw).strip() if body_raw is not None else ""
    # Don't duplicate the title into the body.
    if body == title:
        body = ""

    source_raw = _first_present(row, _SOURCE_KEYS)
    source = str(source_raw).strip() if source_raw is not None else ""

    assets = _parse_assets(_first_present(row, _ASSET_KEYS))

    item_id = row.get("id")
    if item_id is None or (isinstance(item_id, str) and not item_id.strip()):
        item_id = str(index)
    else:
        item_id = str(item_id)

    return NewsItem(
        id=item_id,
        timestamp=timestamp,
        title=title,
        body=body,
        source=source,
        assets=assets,
    )


def _read_records(path: str) -> list[dict[str, Any]]:
    """Read raw records from a ``.jsonl`` or ``.csv`` file."""
    lower = path.lower()
    records: list[dict[str, Any]] = []
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        records = data if isinstance(data, list) else data.get("results", [data])
    elif lower.endswith(".csv") or lower.endswith(".tsv"):
        delimiter = "\t" if lower.endswith(".tsv") else ","
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            records = [dict(r) for r in reader]
    else:
        raise ValueError(f"Unsupported news file type: {path!r} (use .jsonl/.csv)")
    return records


def load_news(path: str) -> list[NewsItem]:
    """Load a ``.jsonl`` or ``.csv`` news file into NewsItems, sorted by time.

    Robust to column-name variants and asset-field formats (see module docstring).
    Rows without a usable title or timestamp are skipped; ids are generated from
    the row index when absent.
    """
    records = _read_records(path)
    items: list[NewsItem] = []
    for index, row in enumerate(records):
        item = _row_to_item(row, index)
        if item is not None:
            items.append(item)
    items.sort(key=lambda it: it.timestamp)
    return items


def group_by_day(news: list[NewsItem]) -> dict[dt.date, list[NewsItem]]:
    """Bucket items by calendar day, chronological within each day."""
    by_day: dict[dt.date, list[NewsItem]] = {}
    for item in sorted(news, key=lambda it: it.timestamp):
        by_day.setdefault(item.timestamp.date(), []).append(item)
    return by_day


def load_sample_news() -> list[NewsItem]:
    """Load the bundled synthetic demo corpus (``trr/sample_news.jsonl``)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_news.jsonl")
    return load_news(path)


# --- Lightweight sentiment for the demo cross-check -------------------------
_NEG_WORDS = {
    "collapse", "insolvent", "insolvency", "contagion", "hack", "hacked",
    "exploit", "liquidation", "liquidated", "depeg", "depegs", "bankruptcy",
    "bankrupt", "plunge", "plunges", "selloff", "sell-off", "ban", "lawsuit",
    "sue", "sues", "fraud", "halt", "halts", "freeze", "freezes", "default",
    "crash", "crashes", "delist", "rout", "panic", "withdrawals", "probe",
}


def is_negative(item: NewsItem) -> bool:
    """Cheap keyword sentiment used only by the demo cross-check."""
    text = item.text().lower()
    return any(word in text for word in _NEG_WORDS)


def _main() -> None:
    news = load_sample_news()
    by_day = group_by_day(news)
    days = sorted(by_day)

    print(f"[news] loaded {len(news)} sample items")
    print(f"[news] date range: {days[0]} -> {days[-1]}")
    print(f"[news] distinct days with news: {len(days)}")

    print("[news] example items:")
    for item in news[:3]:
        print(f"    {item.timestamp.date()}  {item.assets}  "
              f"({item.source})  {item.title}")

    # Cross-check the negative-news clusters against the REAL crash labels.
    try:
        from trr.labels import crash_labels

        labels = crash_labels()
        crash_days = {ts.date() for ts in labels.index[labels["crash"] == 1]}
    except Exception as exc:  # pragma: no cover - price data may be absent
        print(f"[news] crash cross-check skipped (labels unavailable): {exc}")
        return

    neg_days = sorted(d for d, items in by_day.items() if any(is_negative(i) for i in items))
    print(f"[news] days with NEGATIVE news: {len(neg_days)}")

    def near_crash(day: dt.date, window: int = 1) -> bool:
        return any(
            (day + dt.timedelta(days=delta)) in crash_days
            for delta in range(-window, window + 1)
        )

    on_or_near = [d for d in neg_days if near_crash(d)]
    print(f"[news] negative-news days on/near a real crash day (+-1d): "
          f"{len(on_or_near)} / {len(neg_days)}")

    def covered(lo: str, hi: str) -> list[dt.date]:
        lo_d = dt.date.fromisoformat(lo)
        hi_d = dt.date.fromisoformat(hi)
        return [d for d in neg_days if lo_d <= d <= hi_d]

    luna = covered("2022-05-06", "2022-05-12")
    ftx = covered("2022-11-05", "2022-11-10")
    print(f"[news] LUNA/Terra window (2022-05-06..12): {len(luna)} neg-news days {luna}")
    print(f"[news] FTX window      (2022-11-05..10): {len(ftx)} neg-news days {ftx}")


if __name__ == "__main__":
    _main()
