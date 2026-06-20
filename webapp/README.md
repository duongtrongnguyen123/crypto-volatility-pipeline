# TRR Stock Crash Radar — Web Platform

A Streamlit web platform for the **TRR (Temporal Relational Reasoning)**
stock-crash-prediction project. TRR is a zero-shot LLM pipeline (in the repo's
`trr/` package) that reads financial **news** and predicts portfolio **crash**
probability through four phases:

1. **Brainstorm** — turn each news item into a directed signed impact graph.
2. **Memory** — carry impacts forward with exponential time-decay.
3. **Attention** — PageRank-prune to the most portfolio-relevant subgraph.
4. **Reason** — the LLM reads the pruned graph and emits a crash probability.

This app visualizes the pipeline's outputs and runs a live, GPU-free demo of the
graph-building phases.

## What it shows

- **Crash-risk gauge** — a big-number gauge + metrics for the selected run's
  latest day, peak risk, actual-crash-day count, and a from-scratch rank AUROC.
- **Live impact graph** — runs the Brainstorm → Attention phases on the fly with
  the heuristic `MockLLM` (no GPU) over the bundled synthetic news corpus, laid
  out with networkx and drawn as a Plotly network (node colour = kind, size =
  PageRank, edge colour = impact polarity). Pick any demo day; tune the
  attention top-k.
- **Historical timeline** — predicted crash probability per day with actual
  crash days shaded red, plus an optional local stock-price overlay
  (AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX).
- **Per-day reasoning** — browse the LLM's natural-language rationale for any
  day, alongside the full predictions table.
- **Campaign & backtest results** — every Markdown table from
  `reports/RESULTS_TRR.md`, parsed and rendered interactively.

## Architecture

- `webapp/lib.py` — **pure** functions (no Streamlit). Loads prediction CSVs,
  builds the Plotly timeline and impact-graph figures, runs `TRRPipeline`'s
  Brainstorm/Attention/Reason over `load_sample_news()` with `MockLLM`, and
  parses the results tables. Importable and runnable headless. Its `__main__`
  block is a self-contained smoke test.
- `webapp/app.py` — the Streamlit UI, a thin shell that calls `lib.py` and adds
  caching, layout, and styling.

Data sources are read live from the repo:
- Predictions: `kaggle/out_*/crash/trr_predictions.csv`
- Prices: `data/fnspid/prices/{TICKER}.csv`
- Results: `reports/RESULTS_TRR.md`
- Live demo graph: `trr.news.load_sample_news()` + `trr.pipeline` + `trr.llm.MockLLM`

## Install

The repo's virtualenv at `/home/nduong/dev/bigdata/.venv` already has the
`trr/` package and its deps. Add the webapp's extra libraries:

```bash
cd /home/nduong/dev/bigdata
.venv/bin/pip install -r webapp/requirements.txt
```

## Run

```bash
cd /home/nduong/dev/bigdata
.venv/bin/streamlit run webapp/app.py
```

Then open http://localhost:8501 in a browser.

## Verify headless (no browser)

Run the pure-function smoke test — it exercises every `lib.py` function against
the repo's real data and prints a summary:

```bash
cd /home/nduong/dev/bigdata
.venv/bin/python -m webapp.lib
```

Expected tail: `== ALL OK ==`.

## Notes

- The live impact graph uses `MockLLM`, a deterministic lexicon heuristic — it
  needs no GPU and no network, so the graph demo always works. The campaign
  results and prediction CSVs were produced by the real Qwen2.5 runs on Kaggle.
- The bundled `trr/sample_news.jsonl` corpus is synthetic crypto news (BTC, ETH,
  SOL, …), so the live graph reasons over the crypto portfolio; the prediction
  CSVs and price overlays are the equities runs. Both are genuine TRR outputs.
- Crash predictions are research artefacts, not investment advice.
