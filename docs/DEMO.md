# Live demo guide (≈8–10 minutes)

A self-contained demo that runs **locally, no internet, no GPU** (the heavy
Qwen2.5-32B is the *offline batch* predictor — shown via its committed Kaggle
outputs; the live pieces use the deterministic MockLLM backend so nothing can
fail on stage).

## 0. One-time prep (run BEFORE the demo)
```bash
cd /home/nduong/dev/bigdata
bash scripts/demo.sh          # generates figures, exports the model, runs 64 tests
```
Open **two terminals** + a browser. Pre-open `reports/figures/*.png` and
`docs/REPORT.md` (the "Results at a glance" table) as backup slides.

---

## Part 1 — The problem & method (1 min, slides)
Use `docs/SLIDES.md` / `docs/ARCHITECTURE.md`. One sentence:
> "We predict market **crashes** by having an LLM reason over the **temporal**
> and **relational** structure of financial **news** — the TRR framework — and
> we evaluate it honestly."
Show the ARCHITECTURE.md Mermaid diagram (4 phases + lambda architecture).

## Part 2 — Live web platform (3 min) ⭐ the centerpiece
```bash
.venv/bin/streamlit run webapp/app.py      # -> http://localhost:8501
```
Walk through, top to bottom:
1. **Crash-risk gauge** — pick a day; the probability + the model's rationale.
2. **Live impact graph** — Brainstorm → Attention rendered as a network (this is
   the "relational" reasoning, built live via the pipeline).
3. **Historical timeline** — crash_prob vs the actual COVID crash days (shaded).
4. **Campaign & research figures** — the reliability curve, backtest equity
   curve, and campaign AUROC bars.

## Part 3 — Live serving API (2 min)
Terminal 2:
```bash
.venv/bin/uvicorn serving.api:app --port 8000
```
Then (terminal 1) feed it fresh "news" and get a live prediction:
```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "headlines":[{"title":"Major exchange halts withdrawals; contagion fears spread; liquidations cascade","assets":["BTC","ETH"]}]}' | python -m json.tool
curl -s localhost:8000/predict-ensemble -H 'Content-Type: application/json' -d '{
  "headlines":[{"title":"Markets calm; modest tech gains","assets":["AAPL"]}]}' | python -m json.tool
curl -s localhost:8000/backtest | python -m json.tool      # the offline campaign
```
Point out: the bearish headline returns a high `crash_prob` + the extracted
impact edges + a rationale; `/predict-ensemble` folds in the trained meta-model.

## Part 4 — Big-Data streaming speed layer (1.5 min)
Terminal 1:
```bash
.venv/bin/python scripts/demo_streaming.py --messages 40 --rate 250
```
A live console: headlines stream through an in-memory "Kafka topic" → consumer
scoring → a rolling, decaying crash signal with **ALERT** flags. Explain it
mirrors `processing/consumer_trr.py` (real Spark Structured Streaming) without
needing a broker on stage.

## Part 5 — Results & honesty (1 min, slides/REPORT)
Show `docs/REPORT.md` "Results at a glance":
- Crash AUROC 0.785 (COVID) → 0.847 with RAG; broad 2016–2020 0.710.
- **Economic backtest**: de-risk overlay **+205% / −45% drawdown / 0.97 Sharpe**
  vs buy-and-hold — the practical payoff.
- Honest negatives: direction ~chance (EMH); volume competitive on calm windows.

## Part 6 (optional) — Reproducibility & the real 32B
```bash
bash scripts/run_all.sh        # reproduces every result + figure + 64 tests
ls kaggle/out_*/eval_results.json   # the actual Qwen2.5-32B RTX 6000 Pro outputs
```
"Every number is reproducible from one command; the 32B runs are the committed
Kaggle outputs."

---

## Fallbacks / gotchas
- **Everything is local + deterministic (MockLLM)** — no network call on stage.
- If Streamlit's port is busy: `--server.port 8502`.
- If `models/trr_meta.pkl` is missing, `/predict-ensemble` degrades gracefully
  (returns the raw LLM prob) — re-create with `.venv/bin/python -m train.export`.
- Keep `reports/figures/*.png` open as static backup in case a service hiccups.

## Honesty note — live vs batch (say this if asked)
- **Scientific claims = batch evaluation** (curated, *labeled* historical news +
  the 32B): AUROC 0.785→0.847, economic backtest, calibration. This is where
  "the method works" is *proven*.
- **Live monitor = deployment proof only**: same pipeline runs real-time on local
  hardware (yfinance → TRR → signal on the 2060). Live news is sparse/noisy/
  unlabeled, so it demonstrates *runnability*, NOT that the 0.78 AUROC holds live.
- One line: "We prove the method works on labeled historical data; the live
  monitor proves it deploys."

## Optional — continuous live daemon (real-time, local)
Run a background loop that fetches yfinance prices+news every minute, keeps a
rolling 7-day window (TRR's measured memory horizon), and runs TRR each time new
headlines arrive; the webapp auto-displays its latest signal:
```bash
.venv/bin/python -m scripts.live_daemon --poll 60 --backend mock      # instant
.venv/bin/python -m scripts.live_daemon --poll 60 --backend 7b        # local Qwen-7B on the 2060
```
Retention is in MINUTES (`--retain-min`, default 10080 = 7 days). Do NOT set it to
a few minutes — TRR's temporal memory needs ~5 trading days; short retention
discards the multi-day signal.

## Daily advisory (the feasible cadence)
Run once a day (cron / scheduler) — produces a structured advisory the web shows:
```bash
.venv/bin/python -m scripts.daily_report --backend 7b    # local Qwen-7B + live-RAG
# or on Kaggle (32B, best): run the stock kernel daily, download eval -> same JSON shape
```
Output: data/live/daily_report.json (risk level, most-exposed assets, key drivers,
cautions, rationale). Daily matches the 3-day horizon; minute-level is not feasible.
