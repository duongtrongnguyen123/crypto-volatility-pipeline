"""Calibrated meta-ensemble scorer for serving.

Loads the trained meta-learner (models/trr_meta.pkl, from `train/export.py`) and
turns a live (LLM crash_prob + news counts) plus the latest portfolio price
technicals into a calibrated ensemble crash probability — the training->serving
handoff. Degrades gracefully (returns None) if the model artifact is absent.

Live requests carry only news, so the price technicals are taken from the most
recent available bar (or `asof` if within range) as a proxy — documented in the
response `note`.
"""
from __future__ import annotations

import math
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_REPO, "models", "trr_meta.pkl")
_PRICE_DIR = os.path.join(_REPO, "data", "fnspid", "prices")

_CACHE: dict = {}


def _load_model():
    if "model" not in _CACHE:
        try:
            import joblib
            _CACHE["model"] = joblib.load(_MODEL_PATH)
        except Exception:  # noqa: BLE001 — missing artifact / no joblib -> disabled
            _CACHE["model"] = None
    return _CACHE["model"]


def _latest_technicals(asof: str | None = None):
    """Return (feature_dict, asof_date_str) from the most recent price bar."""
    if "tech" not in _CACHE:
        from train.features import TICKERS, _technical
        t = _technical(_PRICE_DIR).dropna()
        _CACHE["tech"] = t
    t = _CACHE["tech"]
    if asof is not None:
        import pandas as pd
        a = pd.to_datetime(asof).date()
        sub = t[[d <= a for d in t.index]]
        row = sub.iloc[-1] if len(sub) else t.iloc[-1]
    else:
        row = t.iloc[-1]
    return row.to_dict(), str(row.name)


def is_available() -> bool:
    return _load_model() is not None


def score_ensemble(crash_prob: float, n_news: int, n_edges: int,
                   asof: str | None = None) -> dict | None:
    """Combine the live LLM signal with price technicals via the meta-learner."""
    art = _load_model()
    if art is None:
        return None
    tech, tech_asof = _latest_technicals(asof)
    feat = {"crash_prob": float(crash_prob),
            "log_news": math.log1p(max(0, n_news)),
            "log_edges": math.log1p(max(0, n_edges)), **tech}
    vec = [[feat[name] for name in art["features"]]]
    prob = float(art["model"].predict_proba(vec)[:, 1][0])
    return {
        "ensemble_crash_prob": prob,
        "llm_crash_prob": float(crash_prob),
        "technicals_asof": tech_asof,
        "trained_on": art.get("trained_on"),
        "note": ("ensemble = trained meta-learner over LLM crash_prob + news "
                 "counts + portfolio price technicals (latest available bar "
                 f"{tech_asof} used as proxy for live technicals)"),
    }


if __name__ == "__main__":
    print("model available:", is_available())
    print(score_ensemble(0.6, 20, 12))
