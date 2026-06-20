# Temporal Relational Reasoning of Large Language Models for Stock Price Prediction

**Big Data course — final report.** All quantitative results in this report are
taken verbatim from [`../reports/RESULTS_TRR.md`](../reports/RESULTS_TRR.md); no
numbers are invented. The system architecture is documented in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Abstract

We study whether a Large Language Model can predict equity and crypto market
**crashes** by reasoning over the **temporal** and **relational** structure of
financial **news**, following the framework of Koa et al.
([arXiv:2410.17266](https://arxiv.org/abs/2410.17266)). Our system is a
**zero-shot** four-phase pipeline — Brainstorm (news to directed impact graph),
Memory (time decay `R = exp(-t*lambda)`), Attention (portfolio-biased PageRank
prune), and Reason (LLM to crash probability) — run with **Qwen2.5-32B** on a
Kaggle RTX 6000 Pro GPU with no internet. We add a **RAG** case-based few-shot
extension that retrieves similar past days and their realized outcomes, a
Kafka/Spark streaming speed layer, and a FastAPI/Streamlit serving tier. Across a
multi-window campaign we find a real but weak-to-moderate signal: the **stock
COVID crash** window reaches **AUROC 0.785** (RAG **0.847**); the broad **stock
2016-2020** window **0.710** (where a news-volume baseline wins at **0.747**);
**crypto 2022-23** **0.530** (RAG **0.542**); and the **FNSPID 2021-23 bear
market** **0.550**, beating news-volume's **0.491**. Daily **direction**
prediction is near chance (**0.46** stocks, **~0.50** crypto). The honest
conclusion is that **predictability lives in the tails and in volatility, not in
the sign of daily returns**, consistent with the weak form of the Efficient
Market Hypothesis — the LLM is a crash / large-move detector, not a price oracle.

---

## 1. Introduction & Problem

The assignment title is *"Temporal Relational Reasoning of Large Language Models
for Stock Price Prediction."* Taken literally, "price prediction" invites a
regression on tomorrow's price or a classifier on tomorrow's direction. We show
why both literal targets are scientifically infeasible from news alone, and
reframe the task around the target that *is* feasible: **crash / large-drawdown
detection**.

**Task.** For each day, predict the probability that an equal-weight portfolio
**crashes** over the next 3 days. For the stock universe (AAPL, AMZN, GOOGL,
NVDA, TSLA, NFLX) a crash is a forward 3-day draw-down past -6%; for the crypto
universe (BTC, ETH, SOL, BNB, AVAX, DOGE) it is past -8%. The problem is binary
and **imbalanced** (~10-13% positive), so we score by **AUROC** rather than
accuracy. The LLM reasons **zero-shot / few-shot over news**; price is used only
to build labels and as an optional ensemble baseline.

**Why this is a Big Data problem.** The news corpora are large and heterogeneous
(the FNSPID source is 23 GB, stream-filtered to the six tickers), the LLM run is
GPU-bound and parallelised across ~18 Kaggle accounts in 6-month shards, and the
live system is a Kafka/Spark streaming stack. The deliverable spans data
engineering, distributed compute, and LLM reasoning.

---

## 2. Related Work

The method is a direct adaptation of **Koa, Li, Cheng, Sun and Chua (NUS),
"Temporal Relational Reasoning of Large Language Models for Detecting Stock
Portfolio Crashes" ([arXiv:2410.17266](https://arxiv.org/abs/2410.17266))**. That
work argues that crash detection requires reasoning over *relations* between
entities and over *time*, and proposes the brainstorm to memory to attention to
reason loop that this project re-implements. Our contributions relative to the
paper are: (1) a **crypto** adaptation and an **equities** port that closes the
literal "stock" gap; (2) a **causal RAG** case-based few-shot retriever; (3) a
**lambda architecture** (offline 32B batch layer + Kafka/Spark speed layer +
local serving tier); and (4) a **rigorous, honest evaluation** (bootstrap CIs,
leak-free ensembling, calibration, precision@K, and an economic backtest) that
corrects two of our own over-optimistic intermediate claims.

We benchmark the LLM against three quantitative baselines: **base rate**,
**news-volume / news-negativity** (counting negative headlines), and
**price-momentum** (price-only, no news).

---

## 3. Method

The pipeline (`trr/pipeline.py`) runs four phases per day in chronological order;
memory persists across days, which is the *temporal* mechanism, and the PageRank
prune is the *relational* mechanism. Full code-level detail and diagrams are in
[ARCHITECTURE.md](ARCHITECTURE.md).

1. **Brainstorm** (`trr/brainstorm.py`). The LLM turns each news item into a
   directed **impact graph** `G=(Z,A)`: news to intermediary entities to
   portfolio assets, as chains of signed (+1/-1), weighted (`[0,1]`) impact
   edges. On the large corpus this is **batched** — one LLM call per day over a
   capped set of headlines — because a per-article call count is infeasible on
   the GPU quota.
2. **Memory** (`trr/memory.py`). Each edge is stored with its day index; its
   relevance decays as **`R = exp(-(current_step - entry_step) * lambda)`** and
   edges below a salience cutoff are dropped. This carries a crash signal forward
   and lets it fade as the news ages.
3. **Attention** (`trr/attention.py`). A power-iteration PageRank over the edge
   node-graph, with the teleport vector biased toward the portfolio tickers,
   ranks nodes by relational closeness to the portfolio; edges are scored by
   endpoint importance * `|weight|` and pruned to the top_k.
4. **Reason** (`trr/reason.py`). The pruned edges become
   `(time, subject, polarity, object)` tuples; the LLM is prompted with an
   explicit **base-rate calibration** block and three worked **few-shot
   exemplars** (no-crash / contained-stress / contagion) and returns
   `{"crash_prob": 0..1, "rationale"}`.

**RAG extension** (`trr/rag.py`). A TF-IDF + cosine retriever finds the most
similar **past** days and injects their realized crash/no-crash outcomes as
dynamic case-based few-shot examples. It is **causal**: only days older than an
embargo >= the 3-day label horizon are eligible, so a day's own outcome can never
leak into its prediction.

**Modes.** The same pipeline supports `target_mode="crash"` (default),
`target_mode="direction"` (next-day `up_prob`), and a per-asset crash mode.

---

## 4. Data

| Source | Coverage | Volume | Role |
|---|---|---|---|
| `oliviervha/crypto-news` | 2021-10 to 2023-12 | 30.5k headlines (~43/day) | main crypto news corpus |
| `filipemunizz/bitcoin-news` | to 2024-10 | 5.8k headlines | 2024 regime test |
| Stock analyst-ratings news | 2019-06 to 2020-06 (ends 2020-06) | 5,517 headlines / 343 news-days | equities port (COVID window) |
| FNSPID (financial news) | 2021 to 2023 | 23 GB raw, stream-filtered to 6 tickers | bear-market equities window |
| `leukipp/reddit-crypto-data` | 2022 | 940k posts, 50 subreddits | social-post reasoning (negative result) |
| Fear & Greed index (alternative.me) | 2018 to 2026 | daily | sentiment ensemble baseline |
| yfinance daily closes | per window | 6 stocks | equity crash / direction labels |
| eth-alpha 5-min OHLCV | 2022-01 to 2026-03 | 6 assets | crypto crash labels |

The **23 GB FNSPID** corpus is the canonical Big Data input: it is too large to
load whole, so it is **stream-filtered** down to the six target tickers before
the pipeline sees it. Recent years are a hard data limit — stock analyst-ratings
news ends 2020-06, crypto news ends 2023-12 — so live coverage needs a news API
(Finnhub/GDELT), not a static corpus.

**Evaluation sizes.** Crypto 2022-23 = 712 days / 76 crashes; crypto 2024 = 284
days / 19 crashes; crypto 2022-only = 363 days / 63 crashes; stock COVID window =
261 trading days; stock 2016-2020 pooled = 9 shards / 31 crashes; FNSPID 2021-23
pooled = 41 crashes. The small positive counts are the dominant source of
evaluation variance.

---

## 5. Experiments

### 5.1 Multi-window campaign (Qwen2.5-32B, RTX 6000 Pro)

The headline campaign, parallelised across ~18 Kaggle accounts (6-month shards
concatenated into a pooled AUROC):

| Window | TRR crash AUROC | news-volume | RAG |
|---|---|---|---|
| Stock — COVID (2019-20) | **0.785** | 0.71 | **+0.06 -> 0.847** |
| Stock — 2016-2020 pooled (9 shards, 31 crashes) | 0.710 | **0.747** | — |
| Crypto — 2022-23 | 0.530 | 0.458 | +0.01 -> 0.542 |
| FNSPID — 2021-23 bear market pooled (41 crashes) | 0.550 | 0.491 | — |

Baselines on crypto 2022-23: price-momentum 0.550, base rate 0.107.

### 5.2 Crypto ablations (context)

| Setup | Model | Window | AUROC |
|---|---|---|---:|
| No few-shot (baseline) | Qwen2.5-14B | 2022-23 | 0.505 |
| News reasoning + few-shot | Qwen2.5-14B | 2022-23 | 0.560 |
| News reasoning + few-shot | Qwen2.5-32B | 2022-23 | 0.566 |
| News reasoning + few-shot | Qwen2.5-32B | 2024 (new regime) | 0.580 |
| News reasoning + few-shot | Qwen2.5-14B | 2024 | 0.376 |
| + Fear & Greed ensemble | 32B | 2022-23 | 0.653 (leaky; 0.577 leak-free) |
| Social-post reasoning (Reddit) | 32B | 2022 | 0.475-0.489 |

Key levers: **few-shot prompting** broke the zero-shot flatline (0.505 -> 0.566) —
a bigger gain than model size or hyperparameters; **model scale buys robustness**
(32B holds 0.566->0.580 out-of-regime, 14B collapses to 0.376); **slow decay +
wide focus wins** (`lam=0.6, top_k=30` beat `lam=0.9, top_k=15`).

### 5.3 Memory-horizon sweep (stocks, lambda)

| lambda | horizon | crash AUROC |
|---|---|---|
| 1.0 | ~3 d | 0.761 |
| **0.6** | **~5 d** | **0.785** |
| 0.3 | ~10 d | 0.744 |
| 0.2 | ~15 d | 0.746 |

Crash skill **peaks at a ~5-day (one-week) window** and declines with both
shorter and longer memory: stale impact edges dilute the signal.

### 5.4 Direction target (the literal "price prediction")

Next-day up/down from news: **AUROC 0.46 (stocks, 32B) / ~0.50 (crypto)** —
near-random, exactly as efficient-market theory predicts.

### 5.5 Rigorous and economic evaluation (crypto 2022-23)

Bootstrap (2000 resamples): TRR 0.566 [0.501, 0.630]; Fear & Greed 0.646
[0.580, 0.707]; price-momentum 0.550 [0.480, 0.619]. TRR clears chance only
barely and is **not statistically separable from price-momentum** (paired diff
+0.016, p=0.35). The 0.653 sentiment ensemble was an in-sample-fit artifact:
leak-free it collapses to **0.577 = TRR-only**. Probabilities are **uncalibrated**
(Brier 0.191 vs 0.095 base-rate) — use scores for ranking, not as literal
probabilities. Precision@K: **P@10 = 0.30** (~3x base rate), so the
highest-confidence alerts are meaningfully enriched.

The **strongest, most robust result is economic**: a strategy that de-risks to
cash on the top-20% highest-crash-probability days (decision at day *t*, return
realized *t+1*, no lookahead) beats buy-and-hold on **return and drawdown in both
regimes**:

| period | strategy | return | Sharpe | max drawdown |
|---|---|---:|---:|---:|
| 2022-23 (bear) | buy & hold | -39.3% | -0.01 | -75.4% |
| 2022-23 (bear) | **TRR de-risk** | **+4.2%** | **0.27** | **-61.5%** |
| 2024 (bull) | buy & hold | +22.1% | 0.72 | -40.7% |
| 2024 (bull) | **TRR de-risk** | **+31.5%** | **0.92** | **-32.5%** |

The drawdown reduction survives realistic 10 bps costs in both regimes.

### 5.6 Advanced techniques — what did NOT work

Learned downstream heads **underperform the raw zero-shot LLM signal** under
regime non-stationarity and few crash events: a stacked meta-learner fell to
0.404-0.426 (below chance; best single signal F&G 0.705); a Graph Attention
Network on the asset graph reached 0.444 vs 0.534 for the raw signal; and a
clean same-model self-consistency sweep was flat (K=1 0.524 -> K=3 0.531 -> K=5
0.508). The durable wins are the LLM's zero-shot reasoning, a *fixed* sentiment
blend, **isotonic calibration** (held-out Brier 0.199 -> 0.048), and the de-risk
strategy — not added capacity or test-time compute.

---

## 6. Feasibility Analysis

We measured the feasibility of each candidate target over 2012 days.

- **Price level** ("tomorrow = today"): R2 = 0.999 — an autocorrelation illusion,
  useless for decisions.
- **Raw return**: autocorr(1) = -0.07 (~0) — the sign of the daily return is not
  predictable from its own history.
- **Direction from news**: AUROC ~0.5 — near chance, confirmed by the live
  direction runs (0.46 / ~0.50).

These three are infeasible because daily returns are essentially a martingale
difference — the **weak form of the Efficient Market Hypothesis**: past prices
and public news are already in the price, so the *sign of the center* of the
return distribution carries no exploitable edge.

What **is** feasible lives in the second moment and the tails:

- **Volatility clusters**: `|return|` (absolute return) has autocorr(1) = +0.20 —
  large moves follow large moves, so the *size* of moves is predictable even when
  the sign is not.
- **Left-skewed, fat-tailed returns**: 19% of all movement is concentrated in the
  5% biggest days. Crashes are a structured, recurring, news-driven phenomenon.

So predictability lives in the **size** of moves and in the **tails**, not in the
sign of the center. This is precisely *why* crash detection works (0.71-0.85 on
concentrated panics) while direction fails (~0.5). Choosing crash / tail-risk as
the target is the scientifically honest reading of "stock price prediction."

---

## 7. Findings & Limitations

**Findings.**

1. **The method ports to equities** — the literal "stock" domain — with no code
   changes beyond a daily price loader (`trr/prices.py`), and cleanly surfaces
   the COVID crash (Mar 9-10 2020) as the two highest-risk days.
2. **Tail-risk is feasible, direction is not** — crash AUROC 0.71-0.85 on
   concentrated panics vs ~0.5 for direction, consistent with weak-form EMH.
3. **RAG helps where analogues exist** (stock/COVID +0.06 -> 0.847) but is
   regime-dependent and marginal on one-off shocks (crypto +0.01).
4. **News reasoning is largely orthogonal to price momentum** — within
   price-calm and price-alarmed strata TRR scores ~0.555 either way, so it
   carries complementary information rather than a price proxy.
5. **Economic value is the most robust finding** — the de-risk strategy improves
   return and drawdown in both regimes and survives costs.

**Honest negatives and limitations.**

1. **News-volume is competitive — and wins on calm, broad windows** (stock
   2016-2020: 0.747 vs TRR 0.710). On a single news-saturated panic, even a
   sentiment lexicon scores ~0.81, so simple baselines earn their keep. TRR's
   advantage shows where the "count headlines" trick fails — slow grind-downs
   with no panic spike (crypto, FNSPID bear market).
2. **Modest absolute AUROC and weak significance.** On crypto 2022-23 the
   news-reasoning AUROC is **not statistically separable from price-momentum**
   (small N: only 76 crash events). Only the large gaps (few-shot +0.06, COVID
   RAG +0.06) are clearly meaningful.
3. **Probabilities are uncalibrated** — use for ranking, not as literal
   probabilities (calibrate with isotonic regression).
4. **Sentiment lift is regime-dependent and leaky** — the 0.653 ensemble dropped
   to 0.577 under a leak-free protocol.
5. **Learned downstream models hurt** under non-stationarity and few events.
6. **A bug worth recording**: the reason prompts originally hard-coded the crypto
   portfolio, so the first 32B stock run scored ~chance (0.48, constant 0.10);
   threading the real portfolio universe into the prompts fixed it. A rigorous
   baseline (the 0.81 lexicon) exposed the bug.
7. **Data limits** — no dated news corpus past 2024 (crypto) / 2020-06 (stock
   analyst news); live coverage needs an API.

---

## 8. Conclusion

A zero-shot temporal-relational LLM pipeline over financial news is a faithful
implementation of Koa et al. and a **real but weak-to-moderate** crash detector:
strongest on a single concentrated panic (stock COVID **0.785**, RAG **0.847**),
modest across broad regimes (stock 2016-2020 **0.710**, crypto 2022-23 **0.530**,
FNSPID **0.550**), and at chance for daily direction (**0.46 / ~0.50**). The
scientifically honest reading of "stock price prediction" is **tail-risk
detection**: predictability lives in volatility clustering and the left tail, not
in the sign of daily returns (weak-form EMH). The durable, defensible wins are
the LLM's zero-shot reasoning, a fixed sentiment blend, calibration, and a
de-risking strategy whose **economic value** — better return and drawdown in
both bull and bear regimes — is the most robust result of the study. Added model
capacity (stacking, GNNs) and test-time compute do not help at this scale;
simplicity and calibration win.
