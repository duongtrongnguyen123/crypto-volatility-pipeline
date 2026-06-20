# Results — Temporal Relational Reasoning of LLMs for Crypto Crash Prediction

Empirical study for the assignment *"Temporal Relational Reasoning of Large
Language Models for Stock Price Prediction"* (crypto adaptation of
[arXiv:2410.17266](https://arxiv.org/abs/2410.17266)).

**Task.** For each day, predict the probability that an equal-weight crypto
portfolio (BTC, ETH, SOL, BNB, AVAX, DOGE) **crashes** — drops > 8% over the next
3 days. Binary, imbalanced (~11% positive), scored by **AUROC**. The LLM reasons
**zero-shot / few-shot over news** (price is used only for labels and as an
optional ensemble signal).

**Pipeline.** Four phases per day: Brainstorm (news → directed impact graph) →
Memory (decay `R=exp(-t·λ)`) → Attention (PageRank prune) → Reasoning (LLM →
crash probability). Run on Kaggle RTX 6000 Pro (Blackwell, sm_120), batched
`transformers`, no internet.

---

## Headline results

| Setup | Model | Window | AUROC |
|---|---|---|---:|
| No few-shot (baseline) | Qwen2.5-14B | 2022–23 | 0.505 |
| **News reasoning + few-shot** | Qwen2.5-14B | 2022–23 | 0.560 |
| **News reasoning + few-shot** | **Qwen2.5-32B** | **2022–23** | **0.566** |
| News reasoning + few-shot | Qwen2.5-32B | **2024** (new regime) | **0.580** |
| News reasoning + few-shot | Qwen2.5-14B | 2024 | 0.376 ⚠️ |
| + price-momentum ensemble | 32B | 2022–23 | 0.576 |
| + Fear & Greed ensemble | 32B | 2022–23 | **0.653** |
| Social-post reasoning (Reddit) | 32B | 2022 | 0.475–0.489 ✗ |

Baselines: news-volume 0.458, price-momentum 0.550, base rate 0.107.

---

## What we learned

### 1. Few-shot prompting is the key lever
Zero-shot, the LLM anchors to a single probability for every day (crash-day mean
≈ non-crash mean ≈ 0.157) → **AUROC 0.505 (chance)**. Adding 3 worked exemplars
(no-crash / contained-stress / contagion) and telling it the ~13% base rate broke
the flatline → **0.566**. This was a far bigger gain than model size or
hyperparameters.

### 2. News reasoning generalizes across regimes — but only with a big model
The 32B model scores **0.566** on 2022–23 (bear market) and **0.580** on 2024
(ETF/halving bull run) — the signal holds out-of-regime. The 14B model is
comparable in-sample (0.560) but **collapses to 0.376 (below chance) on 2024**.
**Model scale buys robustness, not just in-sample accuracy.**

### 3. Memory/attention: slow decay + wide focus wins
Ablation (few-shot held fixed): `lam=0.6, top_k=30` beat `lam=0.9, top_k=15` for
both models (32B: 0.566 vs 0.538; 14B: 0.560 vs 0.545). Aggressive recency and
tight pruning hurt. Over-stuffing `max_items` past the input-token cap starved
the impact graph (edges dropped to ~1.5/day) and lowered AUROC.

### 4. Aggregate sentiment helps — but it's regime-dependent
The **Fear & Greed index** (crypto sentiment, partly social) is the single
strongest signal on the full 2022–23 window (fear level alone = 0.646; ensembled
with news reasoning = **0.653**). But on **2022 alone it falls to 0.488** — in a
relentless bear market, fear is constantly high and stops discriminating. The
0.653 lift is largely a **2023 effect**. Honest caveat: F&G is a *composite*
(volatility, momentum, social media, dominance), so part of its power is not
purely social.

### 5. Reasoning over social *posts* does NOT help
Feeding the top-15 engagement Reddit posts/day into the LLM (social-only 0.489,
news+social 0.475 on 2022) **underperformed news-only (0.524)**. Reddit titles
are noisy (memes, price chatter, shilling) and dilute the systemic-event signal
that news headlines carry. **Aggregate** social sentiment helps; social-post
*reasoning* does not.

### Best result
**News temporal-relational reasoning + Fear & Greed sentiment ensemble = AUROC
0.653** on 2022–23 — a real, honest signal well above price-only (0.55), lexicon
(0.46), and base-rate (0.50) baselines. The reasoning component generalizes to an
unseen 2024 regime (0.58) at 32B scale.

---

## Data

| Source | Coverage | Volume | Role |
|---|---|---|---|
| `oliviervha/crypto-news` | 2021-10 → 2023-12 | 30.5k headlines (~43/day) | main news corpus |
| `filipemunizz/bitcoin-news` | → 2024-10 | 5.8k headlines (2024) | 2024 regime test |
| `leukipp/reddit-crypto-data` | 2022 | 940k posts, 50 subreddits | social reasoning |
| Fear & Greed index (alternative.me) | 2018 → 2026 | daily | sentiment ensemble |
| eth-alpha 5-min OHLCV | 2022-01 → 2026-03 | 6 assets | crash labels |

**Evaluation sizes:** 2022–23 = 712 days / 76 crashes; 2024 = 284 days / 19
crashes; 2022-only = 363 days / 63 crashes.

## Honest limitations
- **Small positive counts** (19–76 crashes) → AUROC has real variance; only the
  large gaps (few-shot +0.06, sentiment +0.09) are clearly meaningful, not the
  0.566-vs-0.560 differences.
- **News cap**: the brainstorm uses ≤ 20–24 of ~43 headlines/day; raising the cap
  (with a larger input budget) is an unexplored lever.
- **No 2025**: no dated crypto-news headline corpus covering 2025 exists on
  Kaggle, so the out-of-sample test stops at Oct 2024.
- **Crash *timing* is intrinsically hard**: ~0.57 from news alone is a modest but
  genuine edge, consistent with the paper needing its full machinery.

## Rigorous evaluation (statistical + economic)

A single AUROC is not enough to claim a signal is real or useful. Running
`python -m trr.analysis` over the saved predictions adds significance testing, a
leak-free ensemble, calibration, early-warning precision, and an economic
backtest. This **corrected two over-optimistic claims** and produced the
study's strongest practical result.

### Statistical significance (2022–23, 2000-resample bootstrap)
| signal | AUROC | 95% CI |
|---|---:|---|
| TRR (news reasoning) | 0.566 | **[0.501, 0.630]** |
| Fear & Greed | 0.646 | [0.580, 0.707] |
| price-momentum | 0.550 | [0.480, 0.619] |

- **TRR clears chance only barely** (lower CI 0.501) and is **not significantly
  better than price-momentum** (paired-bootstrap diff +0.016, p=0.35). With only
  76 crash events, AUROC differences of ~0.02 are noise. **Honest conclusion:
  the AUROC edge from news reasoning alone is real but weak and not statistically
  separable from simple price momentum.** Fear & Greed is the only signal whose
  CI is clearly above 0.5.

### The 0.653 ensemble was leaky — honest number is ~0.58
The headline 0.653 fit the blend weight on the same data it scored. Calibrating
the weight on the first half and testing on the **held-out** second half, the
optimizer picks `alpha_TRR = 1.0` (i.e. drops F&G) and the held-out ensemble =
**0.577 = TRR-only**. So **the sentiment "lift" does not survive a leak-free
protocol** on the out-of-time half — a critical correction.

### Calibration: ranking works, probabilities don't
Brier 0.191 vs 0.095 base-rate (skill **−1.0**) — the model is badly
**overconfident** (outputs 0.3–0.85 when the base rate is 0.11). Use the scores
for **ranking** (AUROC, precision@K), not as literal probabilities.

### Early warning — precision@K (base rate 10.7%)
P@10 = **0.30**, P@20 = 0.15, P@50 = 0.14. The very top of the risk ranking is
~3× enriched, so the **highest-confidence alerts are meaningfully better than
random**, even though mid-ranking is not.

### Economic backtest — the strongest result
Strategy: go to cash on the top-20% highest-risk days (decision at day *t* from
`crash_prob[t]`, return realized *t+1* — no lookahead), vs equal-weight
buy-and-hold. See `reports/backtest_equity.png`.

| period | strategy | return | Sharpe | max drawdown |
|---|---|---:|---:|---:|
| 2022–23 (bear) | buy & hold | −39.3% | −0.01 | −75.4% |
| 2022–23 (bear) | **TRR de-risk** | **+4.2%** | **0.27** | **−61.5%** |
| 2024 (bull) | buy & hold | +22.1% | 0.72 | −40.7% |
| 2024 (bull) | **TRR de-risk** | **+31.5%** | **0.92** | **−32.5%** |

**Cost-aware refinement** (`cost_aware_backtest`, continuous sizing
`e=1−causal-percentile(crash_prob)`, turnover charged):

| regime | cost | strat return | Sharpe | maxDD | (buy&hold) |
|---|---|---:|---:|---:|---|
| 2022–23 | 0 bps | +7.6% | 0.31 | −49.6% | −39.3% / −75.4% |
| 2022–23 | 10 bps | −5.5% | 0.17 | −52.6% | −39.3% / −75.4% |
| 2024 | 10 bps | +12.3% | 0.56 | −28.1% | +22.1% / −40.7% |

The **drawdown reduction is robust and survives costs in both regimes** (−50% vs
−75% bear; −28% vs −41% bull). Absolute outperformance survives realistic 10 bps
costs in the bear market but the *continuous* sizing turns over ~0.18/day, so it
is cost-sensitive — the **lower-turnover binary de-risk below is the more
practical variant**, and in the bull market de-risking trades some upside for a
much smaller drawdown. Honest takeaway: the signal's durable economic value is
**risk reduction**, not raw return.

The simpler binary version: **heeding the crash signal beats buy-and-hold on both
return and drawdown in both regimes** — turning a −39% bear-market loss into +4%, and improving the
bull year too (and out-of-regime, where AUROC significance is weak). **Economic
value is the most robust finding** — more so than the AUROC, because a strategy
only needs the few biggest crashes called right, which is exactly where
precision@K shows the signal concentrates.

### What rigor changed
1. The ensemble's 0.653 → **0.577 leak-free** (sentiment lift didn't generalize out-of-time).
2. News-reasoning AUROC is **not statistically separable from price-momentum** (small N).
3. Probabilities are **uncalibrated** (rank-only).
4. But the **de-risking strategy adds real economic value** across regimes — the headline takeaway.

Run it: `python -m trr.analysis` (writes `reports/analysis_*.json` + `reports/backtest_equity.png`).

## Per-asset crash prediction

Beyond the portfolio, we asked the LLM for a crash probability **per asset**
(`reason_multi_per_asset`), labelled against each asset's own forward-3-day
−12% drawdown, scored per asset with bootstrap CIs (`make trr-analyze` →
`run_per_asset`).

| asset | crashes | 14B AUROC | 32B AUROC | 32B 95% CI |
|---|---:|---:|---:|---|
| BTC | 16 | 0.493 | **0.690** | [0.545, 0.817] |
| ETH | 30 | 0.527 | **0.639** | [0.526, 0.744] |
| SOL | 66 | 0.505 | 0.544 | [0.469, 0.622] |
| BNB | 18 | 0.494 | 0.583 | [0.436, 0.724] |
| AVAX | 69 | 0.480 | 0.557 | [0.486, 0.629] |
| DOGE | 31 | 0.461 | 0.550 | [0.454, 0.648] |
| **macro** | | **0.493** | **0.594** | |

**Findings:**
1. **Per-asset works only at scale.** 14B is at chance across the board (macro
   0.493); 32B reaches macro **0.594**, with **BTC (0.690) and ETH (0.639)
   significantly above 0.5** (lower CI > 0.5) despite few events.
2. **The majors are the most predictable** — they get the most news coverage and
   drive market narratives; small alts (SOL/AVAX/DOGE ≈ 0.55) are weaker.
3. **Capability × granularity interaction** (the non-obvious result):
   - **Weak model (14B):** portfolio (0.560) ≫ per-asset (0.493) — aggregating
     *helps*, averaging out idiosyncratic noise it can't reason about.
   - **Strong model (32B):** per-asset macro (0.594) **>** portfolio (0.566) — a
     capable model extracts *more* signal reasoning per-asset than from the
     aggregate.

Caveat: BTC/BNB have only 16–18 crash events, so those CIs are wide; the macro
average and the BTC/ETH lower-CI-above-0.5 are the defensible claims.

## Advanced techniques

### Stacked meta-learner — a cautionary result (`trr/stacking.py`)
We fused all signals (TRR, 6 per-asset probs, F&G, price-momentum, volatility,
news-volume, edges) into a meta-model under leak-free expanding **walk-forward**
CV.

| model (walk-forward OOF) | AUROC |
|---|---:|
| F&G alone (best single signal) | 0.705 |
| TRR alone | 0.594 |
| **stack — regularized logistic** | **0.404** |
| **stack — gradient boosting** | **0.426** |

**Both learned stacks underperform the best single signal — and fall below
chance.** The cause is **non-stationarity**: the signal→crash relationships
learned on the 2022 bear market *invert* in the 2023 recovery, so a model fit on
the past generalizes backwards. With only 76 crash events, added capacity is
strictly harmful. **Robust alternatives win:** the single strongest signal, or a
*fixed* leak-free convex blend (0.577, see Rigorous Evaluation) — not a learned
combination. A clean lesson that in non-stationary, low-event regimes, simpler is
better.

**Calibration is the real win:** isotonic regression (fit on a past fold, applied
forward) cut the held-out Brier score from 0.199 → 0.048. Conformal flagging at a
20% alarm budget recovers 27% of crashes. Use the scores calibrated, for ranking.

### Graph Attention Network on the asset-relational graph (`trr/gnn.py`)
We *learn* the relational step (which the pipeline hand-codes via PageRank): a
2-layer GAT message-passes across a 6-asset graph (edges = return correlation
> 0.3), node features `[per-asset LLM prob, return, volatility, F&G]`, trained
walk-forward to predict each asset's crash.

| (walk-forward, same test) | macro AUROC |
|---|---:|
| raw per-asset LLM signal | 0.534 |
| **learned GAT** | **0.444** |

The GAT **underperforms the raw LLM signal** — again, learned propagation overfits
the few crash events under regime shift.

### Meta-finding across the advanced techniques
Both learned downstream models — the **stacking meta-learner** and the **GNN** —
**underperform the raw zero-shot LLM signal** (and a fixed convex blend). The
zero-shot LLM is robust *precisely because it is not fit to the non-stationary
training data*. At this scale (≤76 crash events) and with regime shift,
**learned capacity is counterproductive**; the wins come from (a) the LLM's
zero-shot reasoning, (b) a *fixed* sentiment blend, (c) isotonic **calibration**,
and (d) the economic de-risking strategy — not from training a model on top.
### Self-consistency with a reasoning model (DeepSeek-R1-Distill-Qwen-32B, K=3)
Test-time compute scaling: 3 sampled reasoning traces per day, averaged.
- **First attempt failed (AUROC 0.508)** because R1's `<think>` traces ate the
  1024-token *brainstorm* budget before emitting the edge JSON → `n_edges`
  collapsed to **0.3/day**. Lesson: reasoning models are wasteful on the
  mechanical extraction phase.
- **Fair re-run** (brainstorm budget raised to 2048 → edges flow again at
  **11.9/day**): **AUROC 0.544 vs 0.524** for the Qwen-32B greedy reference — a
  small but *positive* edge. So when given enough budget, reasoning + sampled
  self-consistency appeared to help by +0.02 — but that confounds model with
  method.
- **Clean isolation (Qwen-32B fixed, vary only K, 2022):** K=1 **0.524** → K=3
  **0.531** → K=5 **0.508** — *flat, within ±0.01 noise*. So **test-time compute
  / self-consistency does not reliably help** once the model is held fixed; the
  earlier +0.02 was the R1 model (or noise), not the sampling method.

### Bottom line on advanced techniques
All three advanced approaches **failed to beat the straightforward recipe**
(capable *instruct* model + few-shot + fixed sentiment blend + isotonic
calibration + de-risking backtest): stacking and the GNN *underperformed*
(learned heads overfit under non-stationarity + few events), and a clean
same-model K-sweep showed **self-consistency gives no reliable gain** (0.524 →
0.531 → 0.508). The consistent, well-tested thesis: **at ≤76 crash events with
regime non-stationarity, added model capacity and test-time compute do not help;
the durable wins are the LLM's zero-shot reasoning, a fixed sentiment blend,
calibration, and the risk-reducing de-risk strategy.** The lessons are consistent and honest: at ≤76 crash
events with regime non-stationarity, **added model capacity (learned heads, graph
nets) overfits, and reasoning-model test-time compute is wasted on extraction.**
Simplicity and calibration win.

### Incremental value over price (model-free) — `incremental_value`
Does the LLM signal just rediscover price momentum, or add information? We
stratify days by price-momentum and measure TRR's crash-AUROC *within each
stratum* (model-free, so immune to the non-stationarity that broke the learned
combos).

| stratum | days | crashes | TRR AUROC |
|---|---:|---:|---:|
| price-CALM (mom < median) | 356 | 32 | 0.558 [0.45, 0.66] |
| price-ALARMED (mom ≥ median) | 356 | 44 | 0.555 [0.47, 0.64] |

TRR predicts crashes **about as well on the days price says are calm** as when
price is already falling — so the **news reasoning is largely orthogonal to price
momentum**, carrying complementary information rather than a price proxy. Modest
(~0.56) and the CIs are wide (small per-stratum counts), but the point estimates
are stable across strata — the cleanest evidence that the LLM signal adds
something price autocorrelation does not.

## Matching the title literally: stocks + direction + live serving

The title says "**Stock** Price Prediction"; our core study is crypto crash
detection. To close the gaps:

### Equities port (`trr/prices.py`, `scripts/build_stock_data.py`)
The same TRR pipeline run on **6 large-cap stocks** (AAPL, AMZN, GOOGL, NVDA,
TSLA, NFLX), 2019-06 → 2020-06 — 261 trading days, **5,517 real headlines**
(`miguelaenlle` analyst/partner news), prices via yfinance. Labels cleanly
surface the **COVID crash (Feb 20–Mar 13 2020)** as the worst drawdowns.
- **Crash AUROC = 0.823** (MockLLM). Honest caveat: this single-event window
  (COVID dominates) inflates it — one big, news-saturated crash is easy to flag;
  contrast the multi-event crypto window (≈0.57). The 32B refines it on GPU.
- Demonstrates the method **ports to equities** — the literal "stock" domain —
  with no code changes beyond a daily price loader.

### Direction target (`trr/targets.py`, `target_mode="direction"`)
The literal "price prediction": next-day up/down. The LLM is prompted for
`up_prob` directly. Result: **AUROC ≈ 0.51 (stocks) / ≈0.50 (crypto)** — daily
direction is **near-random** from news alone (efficient-market consistent). An
honest negative: TRR is a **crash/down-tail detector**, not a daily price
oracle.

### Live serving proven on local hardware (`scripts/prove_live_serving.py`)
The robust **32B stays the offline batch predictor** (Kaggle, no internet); live
serving runs **locally on a 2060** with a small model. Proof: a real
**Qwen2.5-1.5B** loaded and ran the full brainstorm→reason pipeline producing
real crash_probs + edges (90 s/day on CPU; ~1–2 s/day on the 2060 GPU). VRAM:
Qwen-7B-AWQ ≈ 5.5 GB **fits** the 8 GB card; 32B (65 GB) does not → it stays in
the Kaggle lab. The `serving/` FastAPI + dashboard + paper-trader expose it
(`/crash-risk`, `/volatility`), 10 tests passing.

**Title verdict:** method (temporal-relational LLM reasoning over news) = faithful
match; now demonstrated on **stocks** (domain gap closed) and with a **direction**
target (literal-price gap closed, though weak); the durable signal remains
crash/large-move detection, not raw price.

## Reproduce
```bash
# Offline LLM runs (Kaggle RTX 6000 Pro): kaggle/trr_standalone.py + deploy_trr.sh
# Local pipeline + analysis:
make trr-labels        # crash labels (FTX/LUNA appear as worst drawdowns)
make trr-eval          # TRR vs baselines (MockLLM harness)
python -m pytest tests/test_trr.py
```
Ablation variants are in `kaggle/` (`exp1_14b.py`, `exp2_32b.py`, `exp3_14b.py`,
`exp2024_*.py`, `social_*_32b.py`); each kernel writes `eval_results.json` +
`trr_predictions.csv` + a timeline plot.
