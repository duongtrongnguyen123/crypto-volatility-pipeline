#!/usr/bin/env bash
# Daily TRR advisory — run by cron ~05:00 ICT (just after the US market close, so
# the day's prices + news are in). Tries the local Qwen-7B (real LLM + live-RAG);
# falls back to the instant MockLLM if the 7B run fails (e.g. GPU busy).
# Writes data/live/daily_report.json (the webapp reads it) + appends a log.
cd /home/nduong/dev/bigdata || exit 1
LOG=data/live/daily_cron.log
mkdir -p data/live
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') — daily advisory =====" >> "$LOG"
if .venv/bin/python -m scripts.daily_report --backend 7b >> "$LOG" 2>&1; then
  echo "[ok] 7B advisory written" >> "$LOG"
else
  echo "[warn] 7B failed; falling back to MockLLM" >> "$LOG"
  .venv/bin/python -m scripts.daily_report --backend mock >> "$LOG" 2>&1 \
    && echo "[ok] mock advisory written" >> "$LOG"
fi
