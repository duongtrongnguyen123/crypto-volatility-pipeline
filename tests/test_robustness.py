"""Robustness / edge-case tests for the TRR LLM pipeline.

Covers the failure modes a real deployment hits: malformed/hallucinated LLM
output, empty or tiny news days, and division-by-zero-style degenerate input.
The pipeline must DEGRADE GRACEFULLY (no crash, valid probabilities) rather than
throw. No GPU / no network.

Run: python tests/test_robustness.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trr.llm import MockLLM, ReasoningLLM, extract_json
from trr.pipeline import TRRPipeline
from trr.schema import PORTFOLIO, NewsItem


class FaultyLLM(ReasoningLLM):
    """Simulates a hallucinating LLM: returns garbage / broken JSON every call.
    Exercises the real generate()->extract_json parsing path (not MockLLM's
    heuristic overrides), so it tests graceful fallback."""
    def __init__(self, mode="garbage"):
        self.mode = mode

    def generate(self, prompt, max_new_tokens=512, temperature=0.0):
        return {
            "garbage": "I think the market will... (model rambles, no JSON)",
            "broken": '[{"subject": "NEWS", "object": "BTC", "polarity": -1, ',  # truncated
            "wrongschema": '{"foo": 1, "bar": [2,3]}',
            "empty": "",
        }[self.mode]


def test_extract_json_never_crashes_on_garbage():
    for bad in ["", "no json here", "{unbalanced", "[1,2,", "```json\n{bad```",
                "null", "{}", "[]", "  ", "}{][", '{"a": }']:
        extract_json(bad)  # must not raise
    # valid still works
    assert extract_json('prefix {"crash_prob":0.7} x')["crash_prob"] == 0.7
    assert extract_json("[{\"a\":1},{\"b\":2}]") == [{"a": 1}, {"b": 2}]


def test_faulty_llm_extract_impacts_returns_empty_not_crash():
    n = NewsItem(id="1", timestamp=datetime(2022, 1, 1), title="x", assets=["BTC"])
    for mode in ("garbage", "broken", "wrongschema", "empty"):
        edges = FaultyLLM(mode).extract_impacts(n, PORTFOLIO)
        assert edges == [], f"{mode} should yield no edges, got {edges}"


def test_faulty_llm_predict_crash_defaults_safely():
    for mode in ("garbage", "broken", "wrongschema", "empty"):
        prob, _ = FaultyLLM(mode).predict_crash([(datetime(2022, 1, 1), "a", -1, "BTC")])
        assert 0.0 <= prob <= 1.0


def test_pipeline_survives_hallucinating_llm():
    from trr.news import group_by_day
    news = [NewsItem(id=str(i), timestamp=datetime(2022, 1, 1 + i % 5),
                     title="exchange hack contagion", assets=["BTC"]) for i in range(20)]
    df = TRRPipeline(llm=FaultyLLM("garbage"), batch=True, cross_batch=True).run(group_by_day(news))
    assert len(df) > 0
    assert df["crash_prob"].between(0, 1).all()
    assert (df["n_edges"] == 0).all()  # garbage -> no edges, but no crash


def test_pipeline_handles_empty_and_tiny_news_days():
    from datetime import date
    # A stream with empty days and a 1-item day.
    nb = {
        date(2022, 1, 1): [],
        date(2022, 1, 2): [NewsItem(id="a", timestamp=datetime(2022, 1, 2),
                                    title="btc steady", assets=["BTC"])],
        date(2022, 1, 3): [],
    }
    df = TRRPipeline(llm=MockLLM(), batch=True, cross_batch=True).run(nb)
    assert len(df) == 3
    assert df["crash_prob"].between(0, 1).all()
    # empty day produces a finite, defined probability (no NaN)
    assert df["crash_prob"].notna().all()


def test_predict_crash_empty_tuples():
    prob, _ = MockLLM().predict_crash([])
    assert prob == 0.0


def test_kafka_producer_parse_robust_to_malformed():
    """The Binance aggTrade parser must drop garbage (return None) rather than
    crash the WebSocket callback, and parse valid messages correctly."""
    import json as _json

    from ingestion.producer_price import parse_agg_trade
    good = _json.dumps({"s": "BTCUSDT", "p": "100.5", "q": "2.0", "T": 1700000000000, "m": False})
    rec = parse_agg_trade(good)
    assert rec["symbol"] == "BTCUSDT" and rec["price"] == 100.5 and rec["is_buyer_maker"] is False
    for bad in ["", "not json", "{}", '{"s":"BTC"}',                 # missing fields
                '{"s":"BTC","p":null,"q":"1","T":1,"m":true}',        # null price
                '{"s":"BTC","p":"x","q":"1","T":1,"m":true}',         # non-numeric
                '{"p":"1","q":"1","T":1,"m":true}']:                  # missing symbol
        assert parse_agg_trade(bad) is None, f"should drop: {bad}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} robustness tests passed.")


if __name__ == "__main__":
    _run_all()
