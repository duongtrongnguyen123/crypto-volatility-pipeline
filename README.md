# Temporal Relational Reasoning of Large Language Models for Stock Price Prediction

**Big Data course project** — adapted from [arXiv:2410.17266](https://arxiv.org/abs/2410.17266).

A **zero-shot LLM** reads financial **news** and predicts the probability that an
equal-weight portfolio of large-cap stocks **crashes** (≥ 6% drop) over the next
~3 trading days. The model is **never trained** — it *reasons*, in four phases:

> **Brainstorm** (news → impact graph) → **Memory** (time-decay) → **Attention** (PageRank prune) → **Reason** (LLM → crash probability)

The narrowing from "price prediction" (the paper's title) to **crash / tail-risk
prediction** is deliberate: direction and raw price are ≈ chance under weak-form
market efficiency, while crash risk carries a real signal from news.

📄 **Báo cáo & slide tiếng Việt:** [`docs/BAO_CAO_VI.md`](docs/BAO_CAO_VI.md) · [`docs/SLIDE_VI.md`](docs/SLIDE_VI.md)
📊 **Kết quả chi tiết:** [`reports/RESULTS_TRR.md`](reports/RESULTS_TRR.md)

---

## Why this is a Big Data project

| V | What we do | Numbers |
|---|---|---|
| **Volume** | stream-filter a 23 GB / 15.7M-article source → 2016–2023 corpus | **12 GB / 4.5M articles** |
| **Velocity** | live news daemon + Spark Structured Streaming | ~500 news/day, 60 s poll |
| **Variety** | company / macro / crypto / world news + OHLCV prices, multi-source | FNSPID + RSS + yfinance |
| **Value** | deployed crash-advisory web app + FastAPI | Streamlit + REST |

**Storage — "store enormous, serve tiny":** 12 GB corpus → date-indexed **SQLite**
(1.9 GB, ~44 ms/day lookup) → RAG-selected slice (~2 MB) served to the LLM.

**Distributed processing:**
- **Apache Spark** batch ETL: 12 GB corpus → **Parquet partitioned by year** (data-lake layout); 12 GB→718 MB in 101 s; queries read 4.5M rows in 2.4 s (~40×, partition-pruned). Same code runs on a cluster via `SPARK_MASTER=spark://…`.
- **20 Kaggle GPUs**: the 32B LLM backtest fans out to **40 shards** (20 base + 20 RAG) across 20 accounts × 2 notebooks — one ~20-minute wave instead of ~5 h.

---

## Method (TRR, `trr/`)

1. **Brainstorm** (`brainstorm.py`) — LLM turns each day's news into a directed **impact graph** (signed, weighted edges: news → entities → portfolio assets).
2. **Memory** (`memory.py`) — decaying store `R = exp(−t·λ)` carries the **temporal** signal across days (bad news fades over time).
3. **Attention** (`attention.py`) — PageRank-style prune to the top-*k* portfolio-relevant **relational** sub-graph.
4. **Reason** (`reason.py`) — LLM predicts crash probability from the pruned tuples.

**RAG** (`rag.py`, `corpus.py`, `select.py`) — two roles, both bound LLM cost to `O(days·k)`:
- *Retrieval-selection*: pick the *k* most portfolio-relevant headlines per day from the corpus.
- *Case-based few-shot* (causal lookback bank): retrieve similar **past labeled days** + their realized outcomes into the reasoning prompt.

**Models:** Qwen2.5-**32B** on Kaggle RTX 6000 Pro (offline batch) · Qwen2.5-**7B-AWQ** local on RTX 2060 SUPER (live). The pipeline code is identical for both and for the deterministic `MockLLM` used in tests.

**Portfolio:** AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX. **Labels** (`prices`/`targets`): a day is a *crash* if the equal-weight portfolio's forward 3-day low breaches −6%.

---

## Headline results (crash AUROC)

| Setup | AUROC |
|---|---|
| Stock COVID crash window | **0.785** (+RAG **0.847**) |
| Stock broad 2016–2020 | **0.710** |
| RAG lift (large-N) | **+0.074 (p = 0.009)** |
| news-volume baseline | ≈ 0.50 (signal comes from reasoning, not headline counts) |
| Full corpus 2016–2023 (portfolio-filtered) | *(running — see report)* |

**Honest notes:** small-N is the real ceiling (14–82 crash days, ~4% base rate);
direction/raw price ≈ chance (weak-form EMH); naively scaling to an *all-ticker*
corpus with crash-query selection **hurt** (relevance ≠ portfolio-relevance) and
was fixed by portfolio filtering. Full study + negatives in [`reports/RESULTS_TRR.md`](reports/RESULTS_TRR.md).

---

## Repo map

```
trr/         TRR pipeline (brainstorm, memory, attention, reason, rag, corpus, select, prices, targets)
kaggle/      self-contained 32B kernels + distributed deploy/poll/eval scripts (gen_corpus_shards, launch_corpus, eval_corpus)
processing/  Apache Spark — corpus ETL (spark_corpus_etl) + Structured Streaming consumers
train/       meta-learner, ablations, backtest, figures, significance
serving/     FastAPI (/predict, /predict-ensemble, /backtest)
webapp/      Streamlit live crash-advisory dashboard
scripts/     live_daemon, build_corpus_news, fetch_fnspid, daily cron
docs/        BAO_CAO_VI.md (report), SLIDE_VI.md (slides), ARCHITECTURE, REPORT, SLIDES
reports/     RESULTS_TRR.md (master results)
```

---

## How to run

```bash
# venv: /home/nduong/dev/bigdata/.venv/bin/python  (bare `python` not on PATH)
.venv/bin/python -m pytest tests/ serving/tests/ -q        # tests (75 passing)
bash scripts/run_all.sh                                    # reproduce analysis (no GPU)
.venv/bin/streamlit run webapp/app.py                      # web app -> http://localhost:8501

# Big-data pipeline
.venv/bin/python -m trr.corpus build                       # 12 GB corpus -> date-indexed SQLite
.venv/bin/python -m processing.spark_corpus_etl            # corpus -> partitioned Parquet lake
.venv/bin/python -m scripts.build_corpus_news              # RAG-select portfolio news per day
python kaggle/gen_corpus_shards.py 20                      # generate 40 distributed shards
```

Derived data (12 GB corpus, 1.9 GB index, Parquet lake, RAG slices) is **gitignored** — only the code that regenerates it is tracked.
