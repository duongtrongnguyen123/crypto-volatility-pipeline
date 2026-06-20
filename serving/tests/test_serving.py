"""Smoke + integration tests for the LOCAL live-serving tier (serving/).

No GPU, no Kafka, no network — uses the heuristic (MockLLM) backend and static
data via FastAPI's TestClient. Validates:
    * GET  /health
    * POST /crash-risk (3 sample headlines) -> crash_prob in [0, 1]
    * bearish headlines score HIGHER than bullish ones
    * POST /predict (single-day + multi-day) -> crash_prob, pruned_edges, ...
    * GET  /backtest -> aggregated campaign AUROC results
    * /signal/latest and /volatility degrade gracefully when artifacts absent
    * paper_trader.simulate: finite metrics + de-risk cuts max-drawdown on a crash

Run:
    .venv/bin/python -m pytest serving/tests/ -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

# Make the repo root importable (serving/, trr/, config) regardless of cwd.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi.testclient import TestClient

from serving.api import create_app, read_campaign_results
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

# /predict uses {title, assets} headlines (no required timestamp).
PREDICT_BEARISH = [{"title": h["headline"], "assets": h["assets"]} for h in BEARISH]
PREDICT_BULLISH = [{"title": h["headline"], "assets": h["assets"]} for h in BULLISH]


def _client() -> TestClient:
    return TestClient(create_app(backend="heuristic"))


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["status"] == "ok", body
    assert body["backend"] == "heuristic", body
    assert body["model_loaded"] is False, body


def test_crash_risk_in_range():
    r = _client().post("/crash-risk", json=BEARISH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["crash_prob"] <= 1.0, body
    assert body["n_edges"] >= 1, body
    assert isinstance(body["rationale"], str) and body["rationale"], body
    assert body["backend"] == "heuristic", body


def test_bearish_exceeds_bullish():
    client = _client()
    bear = client.post("/crash-risk", json=BEARISH).json()
    bull = client.post("/crash-risk", json=BULLISH).json()
    assert bear["crash_prob"] > bull["crash_prob"], (bear, bull)


def test_crash_risk_empty():
    r = _client().post("/crash-risk", json=[])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crash_prob"] == 0.0 and body["n_edges"] == 0, body


# --- /predict ---------------------------------------------------------------
def test_predict_single_day():
    r = _client().post("/predict", json={"headlines": PREDICT_BEARISH})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["crash_prob"] <= 1.0, body
    assert body["n_edges"] >= 1, body
    assert isinstance(body["pruned_edges"], list) and body["pruned_edges"], body
    edge = body["pruned_edges"][0]
    assert set(edge) == {"subject", "object", "polarity", "weight"}, edge
    assert edge["polarity"] in (-1, 0, 1), edge
    assert 0.0 <= edge["weight"] <= 1.0, edge
    # n_edges reported must match the pruned-edge list length.
    assert body["n_edges"] == len(body["pruned_edges"]), body
    assert isinstance(body["rationale"], str) and body["rationale"], body
    assert body["backend"] == "heuristic", body


def test_predict_multi_day_schema():
    payload = {
        "days": [
            {"date": "2026-05-01",
             "headlines": [{"title": "Calm markets, BTC steady", "assets": ["BTC"]}]},
            {"date": "2026-05-02", "headlines": PREDICT_BEARISH},
        ]
    }
    r = _client().post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["crash_prob"] <= 1.0, body
    assert body["n_edges"] >= 1, body
    assert body["pruned_edges"], body


def test_predict_bearish_exceeds_bullish():
    client = _client()
    bear = client.post("/predict", json={"headlines": PREDICT_BEARISH}).json()
    bull = client.post("/predict", json={"headlines": PREDICT_BULLISH}).json()
    assert bear["crash_prob"] > bull["crash_prob"], (bear, bull)


def test_predict_empty():
    r = _client().post("/predict", json={"headlines": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crash_prob"] == 0.0, body
    assert body["n_edges"] == 0 and body["pruned_edges"] == [], body


# --- /backtest --------------------------------------------------------------
def test_backtest_endpoint():
    r = _client().get("/backtest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "windows" in body and isinstance(body["windows"], list), body
    assert body["n_windows"] == len(body["windows"]), body
    assert body["n_windows"] >= 1, body
    for w in body["windows"]:
        assert "window" in w and isinstance(w["window"], str) and w["window"], w
        for key in ("trr_auroc", "news_volume_auroc"):
            assert key in w, w
            if w[key] is not None:
                assert 0.0 <= w[key] <= 1.0, w
    if body["mean_trr_auroc"] is not None:
        assert 0.0 <= body["mean_trr_auroc"] <= 1.0, body
    assert isinstance(body["source"], str) and body["source"], body


def test_read_campaign_results_direct():
    rows = read_campaign_results()
    assert isinstance(rows, list) and rows, "expected campaign results"
    for r in rows:
        assert set(("window", "trr_auroc", "news_volume_auroc", "source")) <= set(r), r
    # At least one window must carry a real TRR AUROC in [0, 1].
    trr = [r["trr_auroc"] for r in rows if r["trr_auroc"] is not None]
    assert trr, "no TRR AUROC found in campaign results"
    assert all(0.0 <= v <= 1.0 for v in trr), trr


# --- graceful degradation + paper trader ------------------------------------
def test_signal_latest_graceful():
    r = _client().get("/signal/latest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "available" in body, body
    if not body["available"]:
        assert "reason" in body, body


def test_volatility_graceful():
    r = _client().get("/volatility")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "available" in body, body
    if not body["available"]:
        assert "reason" in body and body["reason"], body


def test_api_backend_degrades_offline():
    be = get_backend("api")
    assert isinstance(be, APIReasoningLLM)
    assert be.configured is False
    out = be.generate("test prompt")
    assert isinstance(out, str)


def test_unknown_backend_raises():
    try:
        get_backend("nope")
    except ValueError:
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


def test_derisk_reduces_drawdown_on_crash():
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    crash_prob = np.full(20, 0.1)
    fwd = np.full(20, 0.005)
    crash_prob[8:13] = 0.9
    fwd[8:13] = -0.10
    signal = pd.Series(crash_prob, index=idx)
    fwd_returns = pd.Series(fwd, index=idx)
    res = simulate(signal, fwd_returns, threshold=0.5, cost_bps=10)
    bh_mdd = res["buy_hold"]["max_drawdown"]
    strat_mdd = res["max_drawdown"]
    assert strat_mdd > bh_mdd, (strat_mdd, bh_mdd)
    assert res["total_return"] > res["buy_hold"]["total_return"], res
