# TRR Crypto Crash Detection on Kaggle (NVIDIA Nemotron, RTX 6000 Pro)

Temporal Relational Reasoning (TRR) for crypto portfolio crash detection — the
crypto adaptation of **"Temporal Relational Reasoning of Large Language Models
for Detecting Stock Portfolio Crashes"** ([arXiv:2410.17266](https://arxiv.org/abs/2410.17266)).

The pipeline reasons over financial **news** to detect upcoming crashes in the
portfolio `["BTC","ETH","SOL","BNB","AVAX","DOGE"]`, scored against **real**
price-derived crash labels (LUNA May-2022, FTX Nov-2022) and three baselines.

## The four phases

For each day, in chronological order (`trr/pipeline.py`):

1. **Brainstorm** — the LLM extracts directed *impact relations* (`X --polarity-->
   portfolio asset`) from the day's news, building that day's impact graph.
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
   `trr/` package incl. `sample_news.jsonl`) and `data/` (the six
   `*USDT_5min_long.csv` price files + `sample_news.jsonl` as the default news
   file), creates/versions the private `crypto-trr-bundle` dataset, then pushes
   the `crypto-trr-nemotron` kernel.

4. **Collect output:**

   ```bash
   kaggle kernels output nguyenduongtrong/crypto-trr-nemotron -p kaggle/out
   ```

   Artifacts: `trr_predictions.csv`, `eval_results.json`, `trr_roc.png`,
   `trr_timeline.png`.

## Setting the Nemotron model (`model_sources`)

The kernel is **model-path-agnostic**: at runtime it auto-detects the staged
HuggingFace model dir under `/kaggle/input` (a directory containing
`config.json` + a tokenizer file). But Kaggle only **mounts** a model if it is
listed in `model_sources`, so you must set it.

1. Find the Nemotron model on **Kaggle Models** (kaggle.com/models) for the
   nvidia-nemotron-model-reasoning-challenge competition. A model source slug has
   the form `owner/model/framework/variation`, for example:

   ```json
   "model_sources": ["nvidia/nemotron-nano/transformers/9b-v2"]
   ```

   **Confirm the exact slug on Kaggle Models** — the owner, model name,
   framework, and variation vary by which Nemotron checkpoint you choose. Copy
   it from the model page's "Add Input" / model-source string; do not assume the
   example above is current.

2. Put it in `kaggle/trr-kernel-metadata.json`:

   ```json
   "model_sources": ["<owner>/<model>/<framework>/<variation>"]
   ```

3. Re-run `bash kaggle/deploy_trr.sh`. If no model is mounted, the kernel aborts
   with a clear message pointing back here.

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
project code + price CSVs + news in the `crypto-trr-bundle` dataset, and the
Nemotron weights via `model_sources`. `transformers`/`torch` are already on the
Kaggle image and are lazy-imported by `HFReasoningLLM`.

## Swapping in a real news dataset

The bundled `sample_news.jsonl` is **synthetic** (clearly fictional,
`source: "synthetic"`), with negative-headline clusters aligned to the real
LUNA/FTX crash windows so the pipeline demonstrates end-to-end offline.

To use real headlines, stage a real crypto-news `.jsonl`/`.csv` (e.g. a
CryptoPanic / crypto-headlines export) into the bundle's `data/` dir in place of
(or alongside) `sample_news.jsonl`, or attach it as another `dataset_sources`
entry. The kernel's `_find_news_file` picks up the first news file it finds, and
`trr.news.load_news` is schema-tolerant (handles `timestamp`/`date`/
`published_at`, `title`/`headline`, `body`/`content`, `source`/`publisher`, and
`assets`/`tickers`/`currencies` including CryptoPanic's list-of-dicts form), so
most public datasets load with no extra work.

## Local smoke test

Validate the full orchestration off-Kaggle on CPU (MockLLM + sample news),
writing `trr_predictions.csv` + `eval_results.json` under `/tmp/trr_smoke_out`:

```bash
SMOKE=1 python kaggle/trr_kernel.py
```

CUDA is unavailable on CPU, so the GPU gate prints a note and does **not** abort.
