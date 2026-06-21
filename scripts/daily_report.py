"""Generate today's DAILY ADVISORY report (run once/day — cron or scheduler).

Fetches current news + prices, runs TRR (local 7B or MockLLM, with live-RAG
against the labeled historical bank), and writes a structured advisory
(data/live/daily_report.json) the web reads: risk level, most-exposed assets,
key driving events, cautions, rationale. This is the feasible cadence — daily,
matching the 3-day horizon — not minute-level.

Usage:
  .venv/bin/python -m scripts.daily_report                 # MockLLM (instant)
  .venv/bin/python -m scripts.daily_report --backend 7b    # local Qwen-7B + RAG
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "7b"], default="mock")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()
    from webapp.live import daily_report
    adv = daily_report(use_local_7b=(args.backend == "7b"), use_rag=not args.no_rag)
    print(f"[daily] {adv['asof']}  risk={adv['risk_level']} "
          f"crash_prob={adv['crash_prob']:.0%}  backend={adv['backend']}")
    if adv["at_risk_assets"]:
        print("  most exposed:", ", ".join(
            f"{a['ticker']}({a['exposure']})" for a in adv["at_risk_assets"]))
    for c in adv["cautions"]:
        print("  •", c)
    print("  ->", "data/live/daily_report.json")


if __name__ == "__main__":
    main()
