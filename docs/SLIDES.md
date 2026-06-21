# Presentation — TRR of LLMs for Stock Price Prediction

Slide-by-slide outline. All numbers are from
[`../reports/RESULTS_TRR.md`](../reports/RESULTS_TRR.md). See
[REPORT.md](REPORT.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for full detail.

---

## Slide 1 — Title

- **Temporal Relational Reasoning of LLMs for Stock Price Prediction**
- Big Data course project — zero-shot LLM crash detection over financial news
- Adapts Koa et al., NUS (arXiv:2410.17266) to crypto + equities
- Stack: Qwen2.5-32B on Kaggle RTX 6000 Pro + Kafka/Spark + FastAPI/Streamlit

## Slide 2 — The Problem

- Title says "price prediction" — but is the daily price/direction predictable?
- Target we actually solve: P(portfolio **crashes** in next 3 days), binary,
  imbalanced (~10-13% positive), scored by **AUROC**
- Universes: stocks (AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX), crypto (BTC, ETH,
  SOL, BNB, AVAX, DOGE)
- LLM reasons over **news**; price used only for labels + baselines

## Slide 3 — Why an LLM / why TRR?

- News is unstructured, relational (X affects Y), and time-dependent
- Crashes are contagion cascades — need reasoning over *relations* and over *time*
- Zero-shot: the LLM is never trained, it **reasons** — robust to regime shift
- Method = brainstorm -> memory -> attention -> reason (the paper's loop)

## Slide 4 — Architecture (lambda)

- **Batch layer**: Qwen2.5-32B zero-shot on Kaggle RTX 6000 Pro, NO internet —
  authoritative crash series (kaggle/trr_standalone.py)
- **Speed layer**: Kafka + Spark Structured Streaming — live news -> crash signal
  with a small model (ingestion/, processing/consumer_trr.py)
- **Serving tier**: FastAPI /crash-risk + /volatility + Streamlit dashboard +
  paper trader (serving/) — backends degrade to a safe heuristic
- Only the trained adapter crosses the air-gap

## Slide 5 — The 4 phases

- **1. Brainstorm** (brainstorm.py): news -> directed impact graph G=(Z,A);
  signed (+1/-1), weighted [0,1] "X impacts Y" edges toward the portfolio
- **2. Memory** (memory.py): decay store `R = exp(-t*lambda)` — carries signal
  across days, fades old news
- **3. Attention** (attention.py): portfolio-biased PageRank prunes to top_k
  most relevant edges
- **4. Reason** (reason.py): (time, subject, polarity, object) tuples -> crash_prob
  with base-rate calibration + 3 few-shot exemplars
- **+ RAG** (rag.py): causal TF-IDF retrieval of similar PAST days + their real
  outcomes as dynamic few-shot

## Slide 6 — Data scale & big-data character (honest)

- **Volume**: FNSPID **23.2 GB** raw, **stream-filtered** to 6 tickers → 3.6 MB
  (process-don't-store; verify via HTTP `Content-Length`). Stored corpora are
  **medium data** (~640 MB total) — we don't claim terabytes.
- **Variety** (multi-modal): news 40.8k + 31k headlines, Reddit ~940k posts,
  5-min OHLCV ~441k windows, macro/world feeds — structured + unstructured.
- **Velocity**: Kafka producers + Spark Structured Streaming + live yfinance/RSS + daily cron.
- **Distributed compute**: 32B campaign parallelised across **~18 Kaggle GPU accounts**, 6-month shards.
- Big-data *character* = Velocity + Variety + distributed processing + a 23 GB
  streaming filter — not raw stored volume.

## Slide 7 — Results: the campaign (stocks primary)

| Window | N crashes | TRR AUROC | +RAG (Δ, significance) |
|---|---|---|---|
| Stock — COVID 2019-20 | 14 | **0.785** | 0.847 (+0.063, n.s. — few events) |
| Stock — broad 2016-2020 (pooled) | 31 | 0.710 | **+0.074, p=0.009** |
| FNSPID — bear 2021-23 (pooled) | 41 | 0.550 | **+0.065, p=0.004** |
| Crypto — 2022-23 (cross-asset) | 76 | 0.530 | +0.012 (n.s.) |

- **N = crash events** (not days) — AUROC rests on **14–76 positives**, so quote
  *pooled, CI-backed* results; treat any single small-window cell as indicative only.
- **RAG is the robust win** — statistically significant (paired bootstrap) on the
  two larger-N windows; helps the local 7B +0.058 (≈ 32B baseline).
- Strongest on a concentrated panic; modest across broad regimes.

## Slide 8 — Results: economic value (the robust win)

- Leak-free walk-forward de-risk overlay (cash on highest-crash-prob days)
- **Stock 2018-2023: de-risk +205% / -45.0% maxDD / 0.97 Sharpe**
  vs buy&hold +161% / -50.2% / 0.80 — beats on **return AND drawdown**
- Precision@10 = 0.20 (3.2x base rate): the very top alerts are enriched
- Even a modest-AUROC detector is economically useful as a risk overlay
  (you only need to be right on the worst days)

## Slide 9 — Feasibility insight (the key takeaway)

- Price level: R2=0.999 — autocorrelation illusion, useless
- Raw return autocorr(1) = -0.07 (~0); **direction from news AUROC ~0.5** (chance)
- BUT |return| autocorr(1) = +0.20 (volatility **clusters**); returns left-skewed,
  19% of movement in the 5% biggest days
- **Predictability lives in the size + the tails, not the sign of the center**
- Weak-form EMH: direction infeasible, **tail-risk / crashes feasible**

## Slide 10 — Demo (live serving)

- 32B stays the offline batch predictor (65 GB, no internet on Kaggle)
- Live serving runs locally on a small GPU: real Qwen2.5-1.5B ran the full
  brainstorm->reason pipeline; Qwen-7B-AWQ (~5.5 GB) fits an 8 GB card
- FastAPI: `POST /crash-risk` (headlines -> crash_prob), `GET /volatility`
- Streamlit dashboard: live risk + equity curve; paper trader de-risks on signal

## Slide 11 — Honest limitations

- **Small N is the real ceiling**: every headline AUROC rests on **14–76 crash
  events**, so per-asset / single-window cells are *indicative only* — only the
  pooled, bootstrap-CI'd results are claimed. Don't over-read single cells.
- AUROC is modest and, on crypto, **not statistically separable from
  price-momentum** (only 76 crash events)
- Sentiment ensemble was leaky: 0.653 -> 0.577 leak-free
- Probabilities uncalibrated (use for ranking; isotonic Brier 0.199 -> 0.048)
- Learned heads (stacking 0.40-0.43, GNN 0.444) and test-time compute do NOT help
  under non-stationarity + few events — simplicity wins
- Data ends 2023-24; live coverage needs a news API

## Slide 12 — Conclusion

- Faithful zero-shot implementation of Koa et al., ported to stocks + crypto
- Real but weak-to-moderate crash detector: 0.53-0.85, best on concentrated panic
- Direction ~chance — honest with weak-form EMH; the signal is **tail-risk**
- Durable wins: zero-shot reasoning, fixed sentiment blend, calibration, and a
  **de-risk strategy with real economic value** in bull and bear regimes
