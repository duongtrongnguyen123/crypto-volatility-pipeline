"""FastAPI live-serving app for the crypto crash-prediction system.

This is the LOCAL (tier 3) serving surface. It runs the TRR temporal-relational
reasoning pipeline with a pluggable backend (default: the CPU heuristic, always
available) and exposes:

    GET  /health          -> {status, backend, model_loaded}
    POST /crash-risk      -> run TRR over recent headlines -> crash probability
    POST /predict         -> run the ACTUAL TRRPipeline over one or more days of
                             headlines -> {crash_prob, n_edges, pruned_edges,
                             rationale} for the most recent day
    GET  /volatility      -> latest LSTM volatility prediction (if model+features)
    GET  /signal/latest   -> newest live crash-signal row from the Parquet store
    GET  /backtest        -> aggregated offline campaign AUROC results as JSON

Use `create_app(backend="heuristic")` to build an app with a chosen backend; the
module-level `app` (created at import) uses the env var `SERVING_BACKEND` (default
"heuristic") so `uvicorn serving.api:app` works for live deployment. If the env
var `SMALL_MODEL` is set, the default backend reasons with the real
`trr.llm.HFReasoningLLM` model; otherwise it uses the deterministic MockLLM.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from datetime import date, datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from serving.model_backend import backend_label, get_backend, is_real_model
from trr.attention import pagerank_prune
from trr.brainstorm import build_impact_graph
from trr.news import group_by_day
from trr.pipeline import TRRPipeline
from trr.schema import NewsItem

logger = logging.getLogger("serving.api")

CRASH_SIGNAL_DIR = os.path.join(config.FEATURES_DIR, "crash_signal")

# Repo root = parent of this serving/ package, used to locate the offline
# campaign reports (independent of the process working directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "reports", "stock_runs", "campaign")
RESULTS_TRR_MD = os.path.join(_REPO_ROOT, "reports", "RESULTS_TRR.md")


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


# --- /predict request / response models -------------------------------------
class PredictHeadline(BaseModel):
    """A single headline in a /predict request (timestamp is optional)."""
    title: str = Field(..., description="The news headline text")
    timestamp: str | None = Field(
        default=None, description="Optional ISO-8601 timestamp of the headline"
    )
    assets: list[str] | None = Field(
        default=None, description="Optional explicit tickers, e.g. ['BTC','ETH']"
    )


class PredictDay(BaseModel):
    """One day of headlines for /predict."""
    date: str = Field(..., description="Day in YYYY-MM-DD form")
    headlines: list[PredictHeadline] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Body for POST /predict.

    Either supply a multi-day stream via `days`, or a single day's headlines via
    `headlines` (which are treated as 'today'). At least one must be non-empty.
    """
    days: list[PredictDay] | None = None
    headlines: list[PredictHeadline] | None = None


class PrunedEdge(BaseModel):
    subject: str
    object: str
    polarity: int
    weight: float


class PredictResponse(BaseModel):
    crash_prob: float
    n_edges: int
    pruned_edges: list[PrunedEdge]
    rationale: str
    asof: str
    backend: str


# --- /backtest response models ----------------------------------------------
class BacktestWindow(BaseModel):
    window: str
    trr_auroc: float | None = None
    news_volume_auroc: float | None = None


class BacktestResponse(BaseModel):
    n_windows: int
    mean_trr_auroc: float | None = None
    mean_news_volume_auroc: float | None = None
    windows: list[BacktestWindow]
    source: str


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


def _predict_request_to_days(req: PredictRequest) -> dict[date, list[NewsItem]]:
    """Convert a /predict body into the pipeline's {date: [NewsItem]} stream.

    Supports both the multi-day `days` form and the single-day `headlines`
    shorthand (treated as today's date). News items get a UTC-naive timestamp
    anchored to their day so memory/decay ordering across days is correct.
    """
    days: list[PredictDay] = []
    if req.days:
        days = list(req.days)
    if req.headlines:
        today = datetime.now(timezone.utc).date().isoformat()
        days.append(PredictDay(date=today, headlines=req.headlines))

    stream: dict[date, list[NewsItem]] = {}
    nid = 0
    for day in days:
        d = date.fromisoformat(str(day.date))
        items: list[NewsItem] = []
        for h in day.headlines:
            ts = pd.to_datetime(h.timestamp, utc=True, errors="coerce") \
                if h.timestamp else pd.NaT
            if ts is pd.NaT or pd.isna(ts):
                dt = datetime(d.year, d.month, d.day)
            else:
                dt = ts.tz_convert(None).to_pydatetime()
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.replace(tzinfo=None)
            items.append(NewsItem(
                id=str(nid),
                timestamp=dt,
                title=h.title,
                assets=[a.upper() for a in (h.assets or [])],
            ))
            nid += 1
        # Days with no headlines are still part of the stream (memory-only day).
        stream.setdefault(d, []).extend(items)
    return stream


def _last_day_pruned_edges(pipeline: TRRPipeline, stream: dict, last_day: date,
                           step: int) -> list[PrunedEdge]:
    """Recompute the pruned impact sub-graph for the most recent day.

    The pipeline DataFrame only surfaces n_edges/rationale, so to expose the
    actual pruned edges we re-run the brainstorm + attention phases for the last
    day (the same deterministic phases the pipeline ran) against the memory it
    has just accumulated, then PageRank-prune to top_k.
    """
    day_news = stream.get(last_day, [])
    graph = build_impact_graph(
        day_news, pipeline.llm, pipeline.portfolio,
        batch=pipeline.batch, max_items=pipeline.max_items_per_day,
    )
    today_edges = list(graph.edges)
    decayed = pipeline.memory.retrieve(step, pipeline.lam)
    salient = [e for e, r in decayed if r >= pipeline.mem_min_relevance]
    combined = today_edges + [e for e in salient if e not in today_edges]
    pruned = pagerank_prune(combined, pipeline.portfolio, top_k=pipeline.top_k)
    return [
        PrunedEdge(
            subject=str(e.subject),
            object=str(e.object),
            polarity=int(e.polarity),
            weight=round(float(e.weight), 6),
        )
        for e in pruned
    ]


def _window_label(summary: dict) -> str:
    """Human window label from a campaign summary block."""
    start = summary.get("date_start")
    end = summary.get("date_end")
    if start and end:
        return f"{start} -> {end}"
    return start or end or "unknown"


def _iter_campaign_blocks(obj: dict):
    """Yield (summary, metrics) pairs from a campaign eval-results JSON.

    Two shapes exist in reports/stock_runs/campaign/:
      * flat:   {"summary": {...}, "metrics": {...}}
      * nested: {"<target>": {"summary": {...}, "metrics": {...}}, ...}
    """
    if "summary" in obj and "metrics" in obj:
        yield obj["summary"], obj.get("metrics", {})
        return
    for value in obj.values():
        if isinstance(value, dict) and "summary" in value:
            yield value["summary"], value.get("metrics", {})


def read_campaign_results(campaign_dir: str = CAMPAIGN_DIR) -> list[dict]:
    """Read + aggregate the offline campaign eval-results JSONs.

    Returns a sorted list of {window, trr_auroc, news_volume_auroc, source}.
    Single-class windows (no positive days -> no metrics) are skipped. Falls
    back to parsing reports/RESULTS_TRR.md headline table if no JSONs exist.
    """
    rows: list[dict] = []
    files = sorted(glob.glob(os.path.join(campaign_dir, "*.json")))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            logger.warning("could not read campaign file %s: %s", path, exc)
            continue
        for summary, metrics in _iter_campaign_blocks(obj):
            if not metrics:
                # Single-class / degenerate window: AUROC undefined -> skip.
                continue
            trr = (metrics.get("TRR") or {}).get("auroc")
            nv = (metrics.get("news_volume") or {}).get("auroc")
            if trr is None and nv is None:
                continue
            rows.append({
                "window": _window_label(summary),
                "trr_auroc": None if trr is None else round(float(trr), 4),
                "news_volume_auroc": None if nv is None else round(float(nv), 4),
                "source": os.path.basename(path),
            })
    rows.sort(key=lambda r: r["window"])
    if rows:
        return rows
    return _read_results_md(RESULTS_TRR_MD)


def _read_results_md(md_path: str = RESULTS_TRR_MD) -> list[dict]:
    """Fallback: parse the headline AUROC table from reports/RESULTS_TRR.md."""
    if not os.path.isfile(md_path):
        return []
    rows: list[dict] = []
    with open(md_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|") or "AUROC" in line or "---" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4:
                continue
            setup, model, window, auroc = cells[0], cells[1], cells[2], cells[3]
            # Pull the leading float out of the AUROC cell (drops emoji/notes).
            val = None
            for tok in auroc.replace("**", "").split():
                try:
                    val = float(tok)
                    break
                except ValueError:
                    continue
            if val is None:
                continue
            label = f"{setup} | {model} | {window}".replace("**", "").strip()
            rows.append({
                "window": label,
                "trr_auroc": round(val, 4),
                "news_volume_auroc": None,
                "source": os.path.basename(md_path),
            })
    return rows


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


def _resolve_llm(backend: str):
    """Pick the reasoning LLM for the app.

    If the env var ``SMALL_MODEL`` is set AND the caller did not request a more
    specific backend ("finetuned"/"api"), reason with the real
    ``trr.llm.HFReasoningLLM`` loaded from that model id/path; otherwise use the
    requested backend (default "heuristic" -> MockLLM). Loading the real model
    can fail (no GPU / transformers missing / bad id) — in that case we log and
    degrade to the deterministic MockLLM so the service still comes up.
    """
    small = os.getenv("SMALL_MODEL", "").strip()
    if small and (backend or "heuristic").lower() == "heuristic":
        try:
            from trr.llm import HFReasoningLLM

            device = os.getenv("TRR_MODEL_DEVICE", "cuda")
            logger.info("SMALL_MODEL=%r set; loading HFReasoningLLM.", small)
            return HFReasoningLLM(small, device=device)
        except Exception as exc:  # noqa: BLE001 - any load failure -> degrade
            logger.warning(
                "Could not load HFReasoningLLM from SMALL_MODEL=%r (%s); "
                "falling back to the heuristic MockLLM.", small, exc,
            )
    return get_backend(backend)


def create_app(backend: str = "heuristic") -> FastAPI:
    """Build the FastAPI app with the chosen reasoning backend injected."""
    app = FastAPI(
        title="Crypto Crash-Prediction Serving",
        description="Local live-serving tier (TRR pipeline + LSTM volatility).",
        version="1.0.0",
    )
    llm = _resolve_llm(backend)
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

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Run the ACTUAL TRRPipeline end-to-end over one or more days.

        Accepts either a multi-day stream ({"days":[{"date","headlines"}]}) or a
        single day's headlines ({"headlines":[...]}). Returns the most recent
        day's crash probability, edge count, the pruned impact sub-graph, and the
        reasoning rationale.
        """
        now = datetime.now(timezone.utc).isoformat()
        stream = _predict_request_to_days(req)
        if not stream:
            return PredictResponse(
                crash_prob=0.0, n_edges=0, pruned_edges=[],
                rationale="no headlines provided", asof=now,
                backend=backend_label(llm),
            )
        dates = sorted(stream.keys())
        # Fresh pipeline per request so accumulated memory is request-scoped.
        req_pipeline = TRRPipeline(llm=llm, batch=True, cross_batch=True)
        df = req_pipeline.run(stream, dates)
        last = df.iloc[-1]
        last_day = dates[-1]
        pruned = _last_day_pruned_edges(
            req_pipeline, stream, last_day, step=len(dates) - 1,
        )
        return PredictResponse(
            crash_prob=float(last["crash_prob"]),
            n_edges=int(last["n_edges"]),
            pruned_edges=pruned,
            rationale=str(last["rationale"]),
            asof=now,
            backend=backend_label(llm),
        )

    @app.post("/predict-ensemble")
    def predict_ensemble(req: PredictRequest) -> dict:
        """Run /predict, then fold the LLM signal + price technicals through the
        trained meta-learner (models/trr_meta.pkl) for a calibrated ensemble
        probability. Falls back to the raw LLM prob if the model is unavailable.
        """
        base = predict(req)
        stream = _predict_request_to_days(req)
        last_day = max(stream) if stream else None
        n_news = len(stream.get(last_day, [])) if last_day is not None else 0
        out = {"llm_crash_prob": base.crash_prob, "n_edges": base.n_edges,
               "n_news": n_news, "rationale": base.rationale, "asof": base.asof,
               "backend": base.backend}
        try:
            from serving.ensemble import score_ensemble
            ens = score_ensemble(base.crash_prob, n_news, base.n_edges, asof=None)
        except Exception:  # noqa: BLE001
            ens = None
        if ens is None:
            out["ensemble_available"] = False
            out["crash_prob"] = base.crash_prob
        else:
            out["ensemble_available"] = True
            out["crash_prob"] = ens["ensemble_crash_prob"]
            out.update({k: ens[k] for k in ("ensemble_crash_prob",
                        "technicals_asof", "trained_on", "note")})
        return out

    @app.get("/backtest", response_model=BacktestResponse)
    def backtest() -> BacktestResponse:
        """Aggregated offline campaign AUROC results (TRR vs news-volume)."""
        rows = read_campaign_results()
        windows = [BacktestWindow(**{k: r[k] for k in
                                     ("window", "trr_auroc", "news_volume_auroc")})
                   for r in rows]
        trr_vals = [r["trr_auroc"] for r in rows if r["trr_auroc"] is not None]
        nv_vals = [r["news_volume_auroc"] for r in rows
                   if r["news_volume_auroc"] is not None]
        source = rows[0]["source"] if rows else "none"
        # Collapse the source to a directory/file hint rather than one filename.
        if rows and all(r["source"].endswith(".json") for r in rows):
            source = "reports/stock_runs/campaign/*.json"
        elif rows:
            source = f"reports/{rows[0]['source']}"
        return BacktestResponse(
            n_windows=len(windows),
            mean_trr_auroc=(round(sum(trr_vals) / len(trr_vals), 4)
                            if trr_vals else None),
            mean_news_volume_auroc=(round(sum(nv_vals) / len(nv_vals), 4)
                                    if nv_vals else None),
            windows=windows,
            source=source,
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
