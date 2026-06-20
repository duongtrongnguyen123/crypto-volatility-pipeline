# Serving tier — local live crash-prediction service

This directory is **tier 3** of the crypto crash-prediction system: the
**local, online, always-available serving layer**. It runs the TRR
(Temporal Relational Reasoning) pipeline over recent news to produce a live
crash probability, serves the LSTM volatility forecast, and exposes the live
crash-signal stream — all from a small box with internet and an optional small
GPU.

## The three-tier architecture

```
  TIER 1: Kaggle offline lab            TIER 2: adapter        TIER 3: local serving
  (RTX 6000 Pro, NO internet,           artifact               (this dir; internet +
   batch-only kernels)                  (the hand-off)          optional small GPU)
  ──────────────────────────           ─────────────          ──────────────────────
  • run the big LLM zero-shot     ──►   LoRA / merged    ──►   • FastAPI + TRR pipeline
  • fine-tune a small LoRA              weights exported        • pluggable backend
    adapter on news→crash               & downloaded here       • runs continuously
  • cannot serve (no network)           = TRR_MODEL_DIR         • live CryptoPanic news
```

The **only** thing that crosses the air-gap is the trained adapter directory.
Locally you point `TRR_MODEL_DIR` at it and flip `SERVING_BACKEND=finetuned`.
Everything else — pipeline, prompts, API — is identical across backends.

This tier is **separate** from:
- the Kaggle batch lab (`kaggle/`, no internet), and
- the Kafka/Spark streaming stack (`docker-compose.yml` + `processing/`), which
  ingests live news and writes the crash-signal Parquet store this tier reads.

## Pluggable backends (`model_backend.get_backend`)

| `SERVING_BACKEND` | Backend | Notes |
|---|---|---|
| `heuristic` (default) | `trr.llm.MockLLM` | CPU, deterministic, no model/network. Always works — the safe live default and the test backend. |
| `finetuned` | `trr.llm.HFReasoningLLM` on `TRR_MODEL_DIR` | The Kaggle-trained adapter downloaded locally. Degrades to heuristic if `TRR_MODEL_DIR`/transformers/model are missing. |
| `api` | `APIReasoningLLM` (hosted LLM over HTTP) | Uses `LLM_API_URL` + `LLM_API_KEY` (+ `LLM_MODEL`). The local box has internet, so reasoning can be offloaded. Degrades to heuristic if unconfigured. |

Every backend degrades to the heuristic `MockLLM` rather than failing, so the
service always comes up.

## Endpoints (`serving/api.py`)

- `GET /health` → `{status, backend, model_loaded}`
- `POST /crash-risk` → body: JSON list of `{timestamp, headline, assets?}`;
  returns `{crash_prob, n_edges, rationale, asof, backend}`.
- `POST /predict` → run the **actual** `TRRPipeline` end-to-end over one or more
  days. Body is either a multi-day stream
  `{"days":[{"date":"YYYY-MM-DD","headlines":[{"title","assets?","timestamp?"}]}]}`
  or a single day's `{"headlines":[...]}`. Returns
  `{crash_prob, n_edges, pruned_edges:[{subject,object,polarity,weight}], rationale, asof, backend}`
  for the most recent day. Uses the MockLLM backend by default; if the env var
  `SMALL_MODEL` is set it reasons with the real `trr.llm.HFReasoningLLM`
  (degrading to MockLLM if the model can't be loaded).
- `GET /backtest` → aggregated offline campaign AUROC results read from
  `reports/stock_runs/campaign/*.json` (falling back to `reports/RESULTS_TRR.md`).
  Returns `{n_windows, mean_trr_auroc, mean_news_volume_auroc, windows:[{window, trr_auroc, news_volume_auroc}], source}`.
- `GET /volatility` → latest LSTM next-window volatility, or
  `{available: false, reason}` if the model / feature store is absent.
- `GET /signal/latest` → newest row from the crash-signal Parquet store, or a
  clear "no live data yet" message.

## Live news source

The local box has internet, so live headlines come from **CryptoPanic**
(`config.CRYPTOPANIC_URL`, token `CRYPTOPANIC_TOKEN`). The Kafka/Spark stack
ingests them into the `crypto-news` topic; `processing/consumer_trr.py` scores
them into the `crash-signal` Parquet store. `POST /crash-risk` also accepts
headlines directly, so the API can be driven ad-hoc without the streaming stack.

## Run locally

```bash
# Heuristic backend (no GPU, no secrets) — works out of the box:
uvicorn serving.api:app --reload --port 8000

# Fine-tuned adapter from Kaggle:
export TRR_MODEL_DIR=/path/to/downloaded/adapter
export SERVING_BACKEND=finetuned
uvicorn serving.api:app --port 8000

# Hosted-LLM backend:
export SERVING_BACKEND=api LLM_API_URL=... LLM_API_KEY=...
uvicorn serving.api:app --port 8000

# Dashboard:
streamlit run serving/dashboard.py

# Both via Docker (local tier only):
docker compose -f serving/docker-compose.serving.yml up
```

Example:

```bash
curl -s localhost:8000/crash-risk -H 'content-type: application/json' -d '[
  {"timestamp":"2026-06-20T09:00:00Z","headline":"Major exchange hacked, BTC plunges in liquidation cascade","assets":["BTC"]},
  {"timestamp":"2026-06-20T10:00:00Z","headline":"ETH dumps as contagion fear spreads","assets":["ETH"]}
]'

# End-to-end TRRPipeline prediction (single day):
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "headlines":[
    {"title":"Major exchange hacked, BTC plunges in liquidation cascade","assets":["BTC"]},
    {"title":"ETH dumps as contagion fear spreads","assets":["ETH"]}
  ]
}'

# Offline backtest (campaign AUROC summary):
curl -s localhost:8000/backtest
```

## Tests

```bash
cd /home/nduong/dev/bigdata
.venv/bin/pip install pytest fastapi httpx uvicorn
.venv/bin/python -m pytest serving/tests/ -q
```

## Paper trading (`serving/paper_trader.py`)

`simulate(signal, fwd_returns, threshold, cost_bps)` is a pure function that
de-risks when `crash_prob >= threshold` and returns the equity curve, total
return, Sharpe, max drawdown, and the buy&hold comparison. `run_live()` is the
transport-free streaming driver that polls new crash-signal rows and updates a
paper position. Both are used by the dashboard's equity panel.
