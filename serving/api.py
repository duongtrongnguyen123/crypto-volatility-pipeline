"""FastAPI live-serving app for the crypto crash-prediction system.

This is the LOCAL (tier 3) serving surface. It runs the TRR temporal-relational
reasoning pipeline with a pluggable backend (default: the CPU heuristic, always
available) and exposes:

    GET  /health          -> {status, backend, model_loaded}
    POST /crash-risk      -> run TRR over recent headlines -> crash probability
    GET  /volatility      -> latest LSTM volatility prediction (if model+features)
    GET  /signal/latest   -> newest live crash-signal row from the Parquet store

Use `create_app(backend="heuristic")` to build an app with a chosen backend; the
module-level `app` (created at import) uses the env var `SERVING_BACKEND` (default
"heuristic") so `uvicorn serving.api:app` works for live deployment.
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from serving.model_backend import backend_label, get_backend, is_real_model
from trr.news import group_by_day
from trr.pipeline import TRRPipeline
from trr.schema import NewsItem

logger = logging.getLogger("serving.api")

CRASH_SIGNAL_DIR = os.path.join(config.FEATURES_DIR, "crash_signal")


# --- Request / response models ----------------------------------------------
class Headline(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp of the headline")
    headline: str = Field(..., description="The news headline text")
    assets: list[str] | None = Field(
        default=None, description="Optional explicit tickers, e.g. ['BTC','ETH']"
    )


class CrashRiskResponse(BaseModel):
    crash_prob: float
    n_edges: int
    rationale: str
    asof: str
    backend: str


def _headlines_to_news(headlines: list[Headline]) -> list[NewsItem]:
    """Convert request headlines into TRR NewsItems (tolerant timestamp parse)."""
    items: list[NewsItem] = []
    for i, h in enumerate(headlines):
        ts = pd.to_datetime(h.timestamp, utc=True, errors="coerce")
        if ts is pd.NaT or pd.isna(ts):
            ts = datetime.now(timezone.utc)
        else:
            ts = ts.tz_convert(None).to_pydatetime()
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        # Strip tz so it matches the rest of the pipeline (UTC-naive).
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        items.append(NewsItem(
            id=str(i),
            timestamp=ts,
            title=h.headline,
            assets=[a.upper() for a in (h.assets or [])],
        ))
    return items


def read_latest_signal(signal_dir: str = CRASH_SIGNAL_DIR) -> dict | None:
    """Read the most recent crash-signal row from the Parquet store.

    Returns the newest row (by window_start) as a dict, or None if the store is
    absent/empty. Schema (from processing/consumer_trr.py): window_start,
    window_end, n_edges, n_neg, assets_hit, crash_risk.
    """
    if not os.path.isdir(signal_dir):
        return None
    files = glob.glob(os.path.join(signal_dir, "**", "*.parquet"), recursive=True)
    if not files:
        return None
    try:
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    except Exception as exc:  # pragma: no cover - corrupt/partial parquet
        logger.warning("could not read crash-signal store: %s", exc)
        return None
    if df.empty:
        return None
    sort_col = "window_start" if "window_start" in df.columns else df.columns[0]
    df = df.sort_values(sort_col)
    row = df.iloc[-1].to_dict()
    # JSON-friendly coercion of timestamps.
    for k, v in list(row.items()):
        if isinstance(v, (pd.Timestamp, datetime)):
            row[k] = pd.Timestamp(v).isoformat()
    return row


def create_app(backend: str = "heuristic") -> FastAPI:
    """Build the FastAPI app with the chosen reasoning backend injected."""
    app = FastAPI(
        title="Crypto Crash-Prediction Serving",
        description="Local live-serving tier (TRR pipeline + LSTM volatility).",
        version="1.0.0",
    )
    llm = get_backend(backend)
    # batch + cross_batch: one brainstorm pass and one reason pass — the live path.
    pipeline = TRRPipeline(llm=llm, batch=True, cross_batch=True)
    app.state.backend_name = backend
    app.state.llm = llm
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": backend_label(llm),
            "model_loaded": is_real_model(llm),
        }

    @app.post("/crash-risk", response_model=CrashRiskResponse)
    def crash_risk(headlines: list[Headline]) -> CrashRiskResponse:
        now = datetime.now(timezone.utc).isoformat()
        if not headlines:
            return CrashRiskResponse(
                crash_prob=0.0, n_edges=0,
                rationale="no headlines provided", asof=now,
                backend=backend_label(llm),
            )
        items = _headlines_to_news(headlines)
        by_day = group_by_day(items)
        df = pipeline.run(by_day)
        # Most recent day's prediction is the live crash risk.
        last = df.iloc[-1]
        return CrashRiskResponse(
            crash_prob=float(last["crash_prob"]),
            n_edges=int(last["n_edges"]),
            rationale=str(last["rationale"]),
            asof=now,
            backend=backend_label(llm),
        )

    @app.get("/volatility")
    def volatility() -> dict:
        """Latest LSTM next-window volatility prediction, if available."""
        if not os.path.exists(config.MODEL_PATH):
            return {"available": False,
                    "reason": f"LSTM model not found at {config.MODEL_PATH}"}
        try:
            from ml.infer import predict_latest

            pred = predict_latest(source="parquet")
            return {"available": True, "volatility": float(pred),
                    "asof": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            # Missing feature store / too-few windows / torch issue: degrade.
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    @app.get("/signal/latest")
    def signal_latest() -> dict:
        """Most recent live crash-signal row from the Parquet store."""
        row = read_latest_signal()
        if row is None:
            return {"available": False,
                    "reason": "no live data yet — crash-signal store is empty; "
                              "start processing/consumer_trr.py to populate it"}
        return {"available": True, "signal": row}

    return app


# Module-level app for `uvicorn serving.api:app`.
app = create_app(os.getenv("SERVING_BACKEND", "heuristic"))
