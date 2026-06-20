# TRR Speed-Layer Streaming Demo

`scripts/demo_streaming.py` is a **runnable, zero-infrastructure** simulation of
the TRR speed layer. It reproduces the end-to-end Kafka producer → Spark
Structured Streaming → live crash-signal dataflow entirely in one Python
process, so it can be run for a demo or graded **without a Kafka broker or a
Spark cluster**.

## Run

```bash
cd /home/nduong/dev/bigdata
.venv/bin/python scripts/demo_streaming.py                 # ~50 messages, then exits
.venv/bin/python scripts/demo_streaming.py --messages 120  # stream more
.venv/bin/python scripts/demo_streaming.py --threshold 0.4 --rate 200
DEMO_MESSAGES=30 .venv/bin/python scripts/demo_streaming.py
```

It is **bounded**: it streams `--messages` headlines (default 50, or the
`DEMO_MESSAGES` env var), prints a live updating view, then terminates cleanly.

### Arguments

| flag | default | meaning |
|------|---------|---------|
| `--messages` | 50 (`DEMO_MESSAGES`) | headlines to stream, then exit |
| `--rate` | 60 (`DEMO_RATE`) | producer messages per second |
| `--threshold` | 0.55 | rolling-signal level that raises an ALERT |
| `--every` | 1 | refresh the live view every N messages |

## What it simulates

The TRR system is a **lambda architecture**:

- **BATCH layer** — the heavy zero-shot LLM temporal-relational reasoning runs
  offline on GPU (`kaggle/trr_standalone.py`, a 32B model).
- **SPEED layer** — a low-latency crash-risk signal is produced from the live
  news stream by Kafka producers (`ingestion/`) feeding a Spark Structured
  Streaming job (`processing/consumer_trr.py`) that scores each headline with the
  deterministic, no-GPU TRR heuristic backend (`trr.llm.MockLLM`).

This demo is the speed layer, in-process.

## How it maps to the real Kafka / Spark components

| Demo component (`scripts/demo_streaming.py`) | Real speed-layer component |
|----------------------------------------------|----------------------------|
| `queue.Queue` "topic" | Kafka topic `crypto-news` |
| `_producer()` thread replaying a CSV/corpus | `ingestion/producer_news_replay.py` → `KafkaProducer` |
| consumer `while` loop pulling from the queue | Spark `readStream.format("kafka")` in `processing/consumer_trr.py` |
| `score_impacts(headline)` | `processing/consumer_trr.py::_score_impacts` registered as a Spark UDF (same function — imported directly when PySpark is available) |
| `CrashSignal` rolling + decaying score | Spark windowed aggregation `crash_risk = 0.1 + 0.7·(n_neg/n_edges) + 0.04·assets_hit` |
| exponential `decay` of the signal | the batch pipeline's `trr.memory.DecayMemory` carry-over across days |
| console table + `<<< ALERT` | Kafka sink `crash-signal` + Parquet store (`data/features/crash_signal`) consumed by dashboards / the batch layer |

The scorer is **the exact production function**. The demo first tries to import
`processing.consumer_trr._score_impacts`; that module imports `pyspark` at the
top level, so on a box without Spark the demo transparently falls back to a local
copy that calls the identical TRR backend
(`trr.llm.MockLLM().extract_impacts(...)`) — same numbers, no Spark. The startup
banner prints which path is active.

## Headline source

1. `data/fnspid/stocknews.csv` if present (reads a few hundred rows).
2. Otherwise the bundled synthetic crypto corpus
   (`trr.news.load_sample_news()`), whose negative-headline clusters are aligned
   with real crypto crash windows (LUNA/Terra, FTX, 3AC/Celsius) — that corpus
   pushes the rolling signal past the ALERT threshold so the alerting path is
   visibly exercised.

## Live view

Each line shows the message offset, the instantaneous per-headline crash
probability, the rolling/decaying signal, an ASCII risk bar, and the (truncated)
headline. When the rolling signal crosses `--threshold`, the line is flagged
`<<< ALERT`. A summary footer reports the processed count, the peak signal, the
number of alert messages, and the first alert.
