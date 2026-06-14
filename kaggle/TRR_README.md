# TRR Crypto Crash Detection on Kaggle (NVIDIA Nemotron, RTX 6000 Pro)

Temporal Relational Reasoning (TRR) for crypto portfolio crash detection — the
crypto adaptation of **"Temporal Relational Reasoning of Large Language Models
for Detecting Stock Portfolio Crashes"** ([arXiv:2410.17266](https://arxiv.org/abs/2410.17266)).

The pipeline reasons over **real** financial **news** (the
[`oliviervha/crypto-news`](https://www.kaggle.com/datasets/oliviervha/crypto-news)
dataset — `cryptonews.csv`, 31,037 headlines, Oct-2021..Dec-2023) to detect
upcoming crashes in the portfolio `["BTC","ETH","SOL","BNB","AVAX","DOGE"]`,
scored against **real** price-derived crash labels (LUNA May-2022, FTX Nov-2022).

The LLM backend on Kaggle is **NVIDIA Nemotron**
(`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`), run zero-shot on
the RTX 6000 Pro (Blackwell, sm_120).

## The four phases

For each day, in chronological order (`trr/pipeline.py`):

1. **Brainstorm** — the LLM extracts directed *impact relations* (`X --polarity-->
   portfolio asset`) from the day's news, building that day's impact graph. On
   the real corpus this is **batched**: all of a day's headlines go into **one**
   LLM call (`extract_impacts_batch`) that returns every impact edge for the day,
   instead of one call per article (see "Batched daily brainstorming" below).
2. **Memory** — new edges are added to a decaying memory so accumulated negative
   impacts keep elevating the crash signal even after the news ages.
3. **Retrieve + Attention** — decayed edges are unioned with today's, then
   PageRank-pruned to the top-k portfolio-relevant sub-graph.
4. **Reasoning** — the LLM reasons over the temporal accumulation and relational
   spread of negative impacts and emits `crash_prob ∈ [0,1]` + a rationale.

The LLM backend is swappable (`trr/llm.py`): `MockLLM` (deterministic, offline)
for local tests, `HFReasoningLLM` (a local HuggingFace causal LM, e.g. Nemotron)
for the zero-shot Kaggle GPU run. The four phases are identical for both.

## Evaluation

`trr/evaluate.py` aligns the daily `crash_prob` to the real labels
(`trr.labels.crash_labels`, ~9.7% positive) on overlapping days and scores TRR
against:

- **base_rate** — constant majority predictor (AUROC 0.5 reference).
- **news_negativity** — a NON-LLM heuristic: per-day fraction of negative
  headlines by keyword lexicon. Tests whether relational/temporal reasoning
  beats naive headline counting.
- **price_momentum** — a PRICE-ONLY signal (trailing portfolio drawdown, no
  news). Tests whether news reasoning beats a pure price signal.

Primary metric is **AUROC** (crashes are rare); PR-AUC and thresholded
accuracy/precision/recall/F1 are also reported. It writes `trr/eval_results.json`
plus an ROC curve and a crash-probability timeline.

Run locally with the mock backend:

```bash
python -m trr.evaluate
```

> With `MockLLM` (itself lexicon-based) TRR and `news_negativity` score
> similarly — the local run only validates that the harness is correct. The real
> lift comes from Nemotron's relational + temporal reasoning on Kaggle.

## Deploy to Kaggle

The Kaggle account is `nguyenduongtrong`. Kernels run **without internet**, so
the code, price data, news, and the Nemotron model are all pre-staged.

1. **Kaggle API token** — put it at `~/.kaggle/kaggle.json`
   (kaggle.com/settings → API → Create New Token), `chmod 600`.

2. **Set the Nemotron model** (see next section) in
   `kaggle/trr-kernel-metadata.json`.

3. **Stage + push:**

   ```bash
   bash kaggle/deploy_trr.sh            # create dataset, else version it
   bash kaggle/deploy_trr.sh --update   # force a new dataset version
   ```

   This builds `kaggle/build_trr/` with `code/` (config.py + ml/ + the whole
   `trr/` package incl. the synthetic `sample_news.jsonl` SMOKE fallback) and
   `data/` (the six `*USDT_5min_long.csv` price files **only** — the real news is
   **not** bundled), creates/versions the private `crypto-trr-bundle` dataset,
   then pushes the `crypto-trr-nemotron` kernel. The real news and the Nemotron
   model are mounted from `dataset_sources` / `model_sources` (already set in
   `kaggle/trr-kernel-metadata.json`).

4. **Collect output:**

   ```bash
   kaggle kernels output nguyenduongtrong/crypto-trr-nemotron -p kaggle/out
   ```

   Artifacts: `trr_predictions.csv`, `eval_results.json`, `trr_timeline.png`.

## The Nemotron model (`model_sources`)

The competition model is **already set** in `kaggle/trr-kernel-metadata.json`:

```json
"model_sources": ["metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"]
```

This is the official NVIDIA Nemotron checkpoint in HuggingFace `transformers`
format. The kernel is **model-path-agnostic**: at runtime it auto-detects the
mounted HuggingFace model dir under `/kaggle/input` (a directory containing
`config.json` + a tokenizer file), so the slug above is all that needs to change
if you swap checkpoints. Kaggle only **mounts** a model listed in
`model_sources`; if none is mounted the kernel aborts with a clear message.

## The news dataset (`dataset_sources`)

The real news is the attached **`oliviervha/crypto-news`** dataset, mounted at
`/kaggle/input/crypto-news/cryptonews.csv` (31,037 headlines). It is listed in
`dataset_sources` alongside the code/price bundle:

```json
"dataset_sources": [
  "nguyenduongtrong/crypto-trr-bundle",
  "oliviervha/crypto-news"
]
```

The kernel's `_find_news_file` globs `/kaggle/input` for `*cryptonews*.csv`
first, so the real news is picked up automatically; `trr.news.load_news` parses
its `date,sentiment,source,subject,text,title,url` schema directly.

## Date window (`TRR_START` / `TRR_END`)

Running the LLM over all ~700 news days is more than the RTX 6000 Pro quota
needs to absorb for a first pass, so the kernel processes a **bounded date
window**. It defaults to the **FTX-collapse validation window**:

```
TRR_START = 2022-10-01
TRR_END   = 2022-12-15
```

(~76 days). Set the `TRR_START` / `TRR_END` kernel environment variables to
widen it — e.g. the **full run** `TRR_START=2022-01-01`, `TRR_END=2023-07-01`
once you have confirmed the bounded run is healthy. In SMOKE mode the window
defaults to a tiny `2022-11-05..2022-11-12` (also overridable via the same vars).

## Batched daily brainstorming (~1400 LLM calls, not 31k)

The naive Brainstorming phase calls the LLM **once per article**. On the 31,037-
article real corpus that is 31k+ generations — infeasible on the GPU quota. The
kernel instead runs the **batched** pipeline, `TRRPipeline(llm, batch=True,
max_items_per_day=40)`:

- **Brainstorm** aggregates each day's (up to 40 most-recent) headlines into
  **one** numbered prompt and asks for a single JSON array of impact edges, each
  tagged with the `news_idx` of its source headline (`extract_impacts_batch`).
- So the cost is **~1 brainstorm + 1 reason call per day** ≈ **2 calls/day**.
  Over the default ~76-day FTX window that is ~150 calls; over the full
  ~550-day window ≈ **~1100–1400 calls** — well within budget.

`batch=False` (the default for the local tests / `trr.evaluate`) preserves the
original per-article behaviour, so nothing else changes.

## The three-field GPU gate + sm_120 verification

Kaggle silently falls back to a **Tesla P100 (sm_60)** unless the kernel
metadata carries **all three** of the following — `kaggle/trr-kernel-metadata.json`
already does:

```json
"machine_shape": "NvidiaRtxPro6000",
"enable_gpu": true,
"competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]
```

`enable_gpu` must be the boolean `true` (not the string `"true"`). Missing any
one → silent P100 downgrade, and no modern torch kernel image exists for sm_60.

The kernel prints the allocated compute capability and **aborts on sm_60 only**;
it never aborts on CPU. After a run, verify the RTX 6000 Pro (Blackwell, sm_120)
was actually allocated:

```bash
grep -E "sm_1|compute_capability|device" kaggle/out/*.log
```

You want to see `sm_120` / the RTX 6000 Pro device name. `sm_60` means the gate
failed or the 30 hrs/week RTX 6000 Pro quota was exhausted. The kernel selects
**bf16** on sm_80+ and **fp16** on sm_70/75.

## No internet

The kernel sets `enable_internet: false`. Everything it needs is pre-staged:
project code + price CSVs in the `crypto-trr-bundle` dataset, the real news via
the `oliviervha/crypto-news` dataset, and the Nemotron weights via
`model_sources`. `transformers`/`torch` are already on the Kaggle image and are
lazy-imported by `HFReasoningLLM`.

## The synthetic SMOKE fallback

The real run always uses `oliviervha/crypto-news`. The bundled
`sample_news.jsonl` is **synthetic** (clearly fictional, `source: "synthetic"`),
with negative-headline clusters aligned to the real LUNA/FTX crash windows; it
ships inside `code/trr/` purely as the **SMOKE fallback** for when the real CSV
is not on disk. `trr.news.load_news` is schema-tolerant (handles `timestamp`/
`date`/`published_at`, `title`/`headline`, `body`/`content`, `source`/`publisher`,
and `assets`/`tickers`/`currencies` including CryptoPanic's list-of-dicts form),
so other public datasets load too if you change the `dataset_sources` entry.

## Local smoke test

Validate the full orchestration off-Kaggle on CPU with `MockLLM` and the
**batched** pipeline, writing `trr_predictions.csv` + `eval_results.json` +
`trr_timeline.png` under `/tmp/trr_smoke_out`:

```bash
SMOKE=1 python kaggle/trr_kernel.py
```

If the real `data/news_raw/oliviervha/cryptonews.csv` is present locally the
smoke run uses it (bounded to `2022-11-05..2022-11-12`); otherwise it falls back
to the synthetic `sample_news.jsonl`. CUDA is unavailable on CPU, so the GPU gate
prints a note and does **not** abort.
