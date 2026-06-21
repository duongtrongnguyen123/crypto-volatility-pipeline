#!/usr/bin/env bash
# Daily TRR advisory — run by cron ~05:00 ICT (just after the US market close).
# Prefers the 32B on Kaggle (validated quality model); falls back to the local
# Qwen-7B (+live-RAG), then to the instant MockLLM, so it never silently fails.
# Writes data/live/daily_report.json (the webapp reads it) + appends a log.
cd /home/nduong/dev/bigdata || exit 1
LOG=data/live/daily_cron.log
mkdir -p data/live
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') — daily advisory =====" >> "$LOG"
if .venv/bin/python -m scripts.daily_kaggle >> "$LOG" 2>&1; then
  echo "[ok] 32B (Kaggle) advisory written" >> "$LOG"
elif .venv/bin/python -m scripts.daily_report --backend 7b >> "$LOG" 2>&1; then
  echo "[ok] local 7B advisory written (Kaggle fell back)" >> "$LOG"
else
  .venv/bin/python -m scripts.daily_report --backend mock >> "$LOG" 2>&1 \
    && echo "[ok] mock advisory written (7B + Kaggle fell back)" >> "$LOG"
fi
