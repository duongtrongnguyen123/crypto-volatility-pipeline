"""Tests for the live-news rule-based extractor + summarizer (webapp.live).

The live tab is descriptive (monitor + summarize), recency-weighted — no model
needed for the rule-based path, so these run headless.

Run: python tests/test_live_summary.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trr.schema import NewsItem
from webapp.live import extract_news_signals, summarize_live_news

_NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def _item(i, title, assets, age_min=0):
    return NewsItem(id=str(i), timestamp=_NOW - timedelta(minutes=age_min),
                    title=title, assets=assets)


def test_empty():
    sig = extract_news_signals([])
    assert sig["total"] == 0 and sig["stress"] == "Thấp"
    assert summarize_live_news([])["source"] == "none"


def test_recency_weighting_dominates():
    # 1 old positive vs 2 fresh negatives -> recent negativity should win.
    items = [_item(1, "AAPL upgraded, shares rally to record", ["AAPL"], age_min=20 * 60),
             _item(2, "Bank collapse triggers selloff and panic", ["TSLA"], age_min=2),
             _item(3, "NVDA plunges on recession crash fears", ["NVDA"], age_min=30)]
    sig = extract_news_signals(items)
    assert sig["stress"] in ("Cao", "Tăng"), sig
    assert sig["neg_ratio"] > 0.5, sig
    # the top salient-recent headline is a fresh negative one (not the stale positive)
    assert any(w in sig["top_recent"][0].title.lower()
               for w in ("plunge", "crash", "selloff", "panic", "recession")), sig["top_recent"][0].title
    # the stale positive AAPL carries far less weight than the fresh tickers
    tick = dict(sig["top_tickers"])
    assert tick.get("TSLA", 0) > tick.get("AAPL", 0)
    assert tick.get("NVDA", 0) > tick.get("AAPL", 0)


def test_calm_is_low():
    items = [_item(i, "Quiet session, stocks drift higher mildly", ["AAPL"], age_min=i * 10)
             for i in range(5)]
    sig = extract_news_signals(items)
    assert sig["stress"] == "Thấp", sig


def test_summary_rule_based_string():
    items = [_item(1, "Markets tumble as crash fears mount", ["NVDA"], age_min=1)]
    r = summarize_live_news(items, use_local_7b=False)
    assert r["source"] == "rule-based"
    assert "tin" in r["summary"] and "tiêu cực" in r["summary"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("[live-summary] all tests passed")
