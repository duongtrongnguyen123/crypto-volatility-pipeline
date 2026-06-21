"""Smoke + correctness tests for the webapp/live demo code (pure, no network/GPU).

Protects the demo-critical paths added late: interactive figures, the 'Try it'
arbitrary-headline graph, the live advisory composer, and the live-RAG bank query.
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

from webapp import lib

_PRED = "kaggle/out_s9/crash/trr_predictions.csv"


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(_PRED):
        pytest.skip("no prediction CSV present")
    return lib.load_predictions(_PRED)


def test_animated_timeline(df):
    fig = lib.build_animated_timeline_figure(df, "t")
    assert len(fig.frames) > 0
    assert len(fig.data) >= 1


def test_try_it_graph_from_arbitrary_headline():
    from trr.schema import NewsItem
    items = [NewsItem(id="u0", timestamp=datetime(2026, 1, 1),
                      title="Major exchange hacked; contagion; cascading liquidations",
                      assets=["BTC", "ETH"])]
    d = lib.build_impact_graph_data(news_items=items)
    assert 0.0 <= d["crash_prob"] <= 1.0
    assert "nodes" in d and "edges" in d
    fig = lib.build_impact_graph_figure(d)
    assert fig is not None


def test_interactive_figure_builders():
    fd = lib.load_fig_data()
    if not fd:
        pytest.skip("fig_data.json not generated (run train.figures)")
    for fn in (lib.build_campaign_figure, lib.build_reliability_figure,
               lib.build_backtest_figure):
        assert len(fn(fd).data) >= 1


def test_advisory_composer_risk_levels():
    from webapp.live import compose_advisory
    hi = compose_advisory({"crash_prob": 0.8, "edges": [
        {"subject": "MACRO", "object": "NVDA", "polarity": -1, "weight": 0.9}],
        "rationale": "x", "backend": "test", "asof": "now"})
    lo = compose_advisory({"crash_prob": 0.05, "edges": [], "rationale": "",
                           "backend": "test", "asof": "now"})
    assert hi["risk_level"] == "HIGH" and lo["risk_level"] == "LOW"
    assert hi["at_risk_assets"] and hi["cautions"]
    assert "not financial advice" in hi["disclaimer"].lower()


def test_live_rag_query():
    """CausalRAG can answer a NEW (live) query against a labeled bank."""
    from datetime import date
    from trr.rag import CausalRAG
    texts = ["exchange hack panic market crash", "calm market mild gains tech",
             "exchange crackdown panic selloff market", "calm market quiet tech gains"]
    dates = [date(2022, 1, d) for d in (1, 2, 3, 4)]
    labels = [1, 0, 1, 0]
    rag = CausalRAG(k=2, min_sim=0.0).fit(texts, dates)
    block = rag.fewshot_for_query("exchange hack panic crash", labels)
    assert "ANALOGUES" in block and "2022-01-01" in block
