"""Smoke tests for the TRR (Temporal Relational Reasoning) pipeline.

No GPU / no transformers / no network — uses the deterministic MockLLM. Validates
the four-phase pipeline plumbing and the crash-label ground truth.

Run:
    python tests/test_trr.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trr.attention import pagerank_prune
from trr.llm import MockLLM, extract_json
from trr.memory import DecayMemory
from trr.news import group_by_day, load_sample_news
from trr.pipeline import TRRPipeline
from trr.schema import PORTFOLIO, ImpactEdge, NewsItem


def _news(nid, day, title, assets, body=""):
    return NewsItem(id=nid, timestamp=datetime(2022, 1, day), title=title,
                    body=body, source="t", assets=assets)


def test_extract_json_robustness():
    assert extract_json('prefix {"crash_prob": 0.7} suffix') == {"crash_prob": 0.7}
    assert extract_json("```json\n[1, 2, 3]\n```") == [1, 2, 3]
    assert extract_json("no json here") is None


def test_mockllm_negative_news_raises_crash():
    llm = MockLLM()
    bad = _news("a", 1, "Exchange hacked, BTC plunges in liquidation cascade", ["BTC"])
    good = _news("b", 2, "Spot ETF approval sparks record inflows and rally", ["BTC"])
    pb, _ = llm.predict_crash([e.as_tuple() for e in llm.extract_impacts(bad, PORTFOLIO)])
    pg, _ = llm.predict_crash([e.as_tuple() for e in llm.extract_impacts(good, PORTFOLIO)])
    assert pb > pg, f"negative news should score higher: {pb} !> {pg}"


def test_decay_memory_decays():
    mem = DecayMemory()
    e = ImpactEdge("NEWS:1", "BTC", -1, 1.0, datetime(2022, 1, 1), "1")
    mem.update([e], step=0)
    r_recent = mem.retrieve(current_step=0, lam=0.3)
    r_old = mem.retrieve(current_step=5, lam=0.3)
    rec = r_recent[0][1] if r_recent else 0.0
    old = r_old[0][1] if r_old else 0.0
    assert rec > old, f"relevance should decay over time: {rec} !> {old}"


def test_pagerank_prune_caps_and_prioritizes():
    edges = [
        ImpactEdge("NEWS:1", "BTC", -1, 0.9, datetime(2022, 1, 1), "1"),
        ImpactEdge("NEWS:2", "ETH", -1, 0.8, datetime(2022, 1, 1), "2"),
        ImpactEdge("NEWS:3", "RANDOMCO", 1, 0.5, datetime(2022, 1, 1), "3"),
    ]
    kept = pagerank_prune(edges, portfolio=PORTFOLIO, top_k=2)
    assert len(kept) <= 2
    objs = {e.object for e in kept}
    assert objs & set(PORTFOLIO), "should keep portfolio-adjacent edges"


def test_pipeline_elevates_on_crash_windows():
    import pandas as pd

    pred = TRRPipeline().run(group_by_day(load_sample_news()))
    assert len(pred) > 50
    idx = pd.to_datetime(pd.Index(pred.index))
    luna = pred.loc[((idx >= "2022-05-06") & (idx <= "2022-05-12")).tolist(), "crash_prob"]
    calm = pred.loc[((idx >= "2022-08-01") & (idx <= "2022-08-20")).tolist(), "crash_prob"]
    assert luna.mean() > calm.mean(), "LUNA window should outscore calm August"


def test_crash_labels_capture_known_events():
    from trr.labels import crash_labels

    df = crash_labels()
    # The single worst forward drawdown should be the FTX collapse (Nov 2022).
    worst_day = df["fwd_ret"].idxmin()
    assert worst_day.year == 2022 and worst_day.month == 11
    assert 0.02 < df["crash"].mean() < 0.25  # rare but present


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} TRR smoke tests passed.")


if __name__ == "__main__":
    _run_all()
