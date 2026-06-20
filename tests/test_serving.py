"""Smoke tests for the LOCAL live-serving tier (serving/).

No GPU, no Kafka, no network — uses the heuristic (MockLLM) backend and static
data via FastAPI's TestClient. Validates:
    * GET /health
    * POST /crash-risk (3 sample headlines) -> crash_prob in [0, 1]
    * bearish headlines score HIGHER than bullish ones
    * /signal/latest and /volatility degrade gracefully when artifacts absent
    * paper_trader.simulate: finite metrics + de-risk cuts max-drawdown on a crash

Run:
    python tests/test_serving.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from serving.api import create_app
from serving.model_backend import APIReasoningLLM, get_backend
from serving.paper_trader import simulate

BEARISH = [
    {"timestamp": "2026-06-20T09:00:00Z",
     "headline": "Major exchange hacked, BTC plunges in liquidation cascade",
     "assets": ["BTC"]},
    {"timestamp": "2026-06-20T10:00:00Z",
     "headline": "ETH dumps as contagion fear spreads, insolvency contagion",
     "assets": ["ETH"]},
    {"timestamp": "2026-06-20T11:00:00Z",
     "headline": "SOL collapse: validators halt after exploit; selloff deepens",
     "assets": ["SOL"]},
]

BULLISH = [
    {"timestamp": "2026-06-20T09:00:00Z",
     "headline": "BTC ETF inflows hit a record as adoption surges",
     "assets": ["BTC"]},
    {"timestamp": "2026-06-20T10:00:00Z",
     "headline": "ETH network upgrade rallies developer interest, bullish breakout",
     "assets": ["ETH"]},
    {"timestamp": "2026-06-20T11:00:00Z",
     "headline": "SOL partnership and adoption news drive a strong rally",
     "assets": ["SOL"]},
]


def _client() -> TestClient:
    return TestClient(create_app(backend="heuristic"))


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["status"] == "ok", body
    assert body["backend"] == "heuristic", body
    assert body["model_loaded"] is False, body
    print("[serving] OK /health:", body)


def test_crash_risk_in_range():
    r = _client().post("/crash-risk", json=BEARISH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["crash_prob"] <= 1.0, body
    assert body["n_edges"] >= 1, body
    assert isinstance(body["rationale"], str) and body["rationale"], body
    assert body["backend"] == "heuristic", body
    print(f"[serving] OK /crash-risk crash_prob={body['crash_prob']:.3f} "
          f"n_edges={body['n_edges']}")


def test_bearish_exceeds_bullish():
    client = _client()
    bear = client.post("/crash-risk", json=BEARISH).json()
    bull = client.post("/crash-risk", json=BULLISH).json()
    assert bear["crash_prob"] > bull["crash_prob"], (bear, bull)
    print(f"[serving] OK bearish {bear['crash_prob']:.3f} > "
          f"bullish {bull['crash_prob']:.3f}")


def test_crash_risk_empty():
    r = _client().post("/crash-risk", json=[])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crash_prob"] == 0.0 and body["n_edges"] == 0, body
    print("[serving] OK /crash-risk empty -> 0.0")


def test_signal_latest_graceful():
    # In the test env the crash-signal store should not exist -> graceful no-data.
    r = _client().get("/signal/latest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "available" in body, body
    if not body["available"]:
        assert "reason" in body, body
    print("[serving] OK /signal/latest graceful:", body.get("available"))


def test_volatility_graceful():
    r = _client().get("/volatility")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "available" in body, body
    if not body["available"]:
        assert "reason" in body and body["reason"], body
    print("[serving] OK /volatility graceful:", body.get("available"))


def test_api_backend_degrades_offline():
    # No LLM_API_URL/KEY set -> APIReasoningLLM must degrade to the heuristic.
    be = get_backend("api")
    assert isinstance(be, APIReasoningLLM)
    assert be.configured is False
    out = be.generate("test prompt")
    assert isinstance(out, str)
    print("[serving] OK api backend degrades offline")


def test_unknown_backend_raises():
    try:
        get_backend("nope")
    except ValueError:
        print("[serving] OK unknown backend raises")
        return
    raise AssertionError("unknown backend did not raise")


def test_paper_trader_finite_metrics():
    rng = np.random.default_rng(0)
    n = 50
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    signal = pd.Series(rng.uniform(0, 1, n), index=idx)
    fwd = pd.Series(0.001 + 0.01 * rng.standard_normal(n), index=idx)
    res = simulate(signal, fwd, threshold=0.5, cost_bps=10)
    for key in ("total_return", "sharpe", "max_drawdown"):
        assert math.isfinite(res[key]), (key, res[key])
    assert res["max_drawdown"] <= 0.0, res["max_drawdown"]
    assert len(res["equity_curve"]) == n, len(res["equity_curve"])
    print(f"[serving] OK paper_trader metrics finite: ret={res['total_return']:.3f} "
          f"sharpe={res['sharpe']:.2f} mdd={res['max_drawdown']:.3f}")


def test_derisk_reduces_drawdown_on_crash():
    # Craft a series: calm, then a crash window the signal flags in advance.
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    crash_prob = np.full(20, 0.1)
    fwd = np.full(20, 0.005)  # mild positive drift in calm periods
    # Days 8..12: signal high AND returns sharply negative (the crash).
    crash_prob[8:13] = 0.9
    fwd[8:13] = -0.10
    signal = pd.Series(crash_prob, index=idx)
    fwd_returns = pd.Series(fwd, index=idx)

    res = simulate(signal, fwd_returns, threshold=0.5, cost_bps=10)
    bh_mdd = res["buy_hold"]["max_drawdown"]
    strat_mdd = res["max_drawdown"]
    # De-risk should have a SHALLOWER (closer to 0) drawdown than buy&hold.
    assert strat_mdd > bh_mdd, (strat_mdd, bh_mdd)
    assert res["total_return"] > res["buy_hold"]["total_return"], res
    print(f"[serving] OK de-risk mdd {strat_mdd:.3f} > buy&hold mdd {bh_mdd:.3f} "
          f"(shallower); de-risk return {res['total_return']:.3f} > "
          f"buy&hold {res['buy_hold']['total_return']:.3f}")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n[serving] ALL {len(fns)} serving tests passed.")


if __name__ == "__main__":
    _run_all()
