"""Continuous LOCAL live TRR daemon.

Loop:
  1. every --poll seconds, fetch current prices + headlines (yfinance),
  2. append NEW headlines to a rolling on-disk store (dedup),
  3. PRUNE entries older than --retain-min (rolling retention),
  4. run TRR over the retained window (local Qwen or MockLLM) -> data/live/signal.json,
     re-running the model only when new headlines arrived (or --force-every ticks),
  5. print a live status line.

Retention note: TRR's temporal memory peaks at a ~5 TRADING-DAY window (measured
λ sweep), so the default retention is 7 DAYS, not minutes — shorter retention
throws away the multi-day signal that the "Temporal" in TRR depends on.

Usage:
  .venv/bin/python -m scripts.live_daemon --poll 60 --retain-min 10080 --backend mock
  .venv/bin/python -m scripts.live_daemon --backend 7b --minutes 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

STORE_DIR = "data/live"
NEWS_PATH = f"{STORE_DIR}/news.jsonl"
SIGNAL_PATH = f"{STORE_DIR}/signal.json"
PRICES_PATH = f"{STORE_DIR}/prices.json"


def _now() -> float:
    return time.time()


def _load_store() -> list[dict]:
    if not os.path.exists(NEWS_PATH):
        return []
    out = []
    for line in open(NEWS_PATH):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _save_store(rows: list[dict]) -> None:
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(NEWS_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _to_newsitems(rows: list[dict]):
    from trr.schema import NewsItem
    items = []
    for r in rows:
        items.append(NewsItem(id=r["id"],
                              timestamp=datetime.fromtimestamp(r["ts"], timezone.utc).replace(tzinfo=None),
                              title=r["title"], source="yfinance", assets=[r["ticker"]]))
    return items


def tick(store: list[dict], retain_min: int, backend: str, force: bool, rag: bool = False):
    """One poll: fetch, merge, prune, (maybe) run model. Returns (store, ran)."""
    from webapp.live import (FEED_TICKERS, fetch_live_headlines, fetch_live_prices,
                             run_live, run_live_window)
    now = _now()
    # 1-2. fetch + merge new (dedup by ticker+title); keep the ARTICLE's pub time.
    # Fetch the FULL display feed (~50 large-caps + macro + crypto + world ≈ 500/day)
    # so the store/feed is rich; the prediction is filtered back to company+macro below.
    seen = {(r["ticker"], r["title"]) for r in store}
    fresh = 0
    for h in fetch_live_headlines(FEED_TICKERS, max_per=12, include_macro=True,
                                  include_crypto=True, include_world=True):
        key = (h.assets[0], h.title)
        if key not in seen:
            seen.add(key)
            art_ts = h.timestamp.replace(tzinfo=timezone.utc).timestamp()
            store.append({"id": h.id, "ts": art_ts, "ticker": h.assets[0], "title": h.title})
            fresh += 1
    # 3. prune by retention
    cutoff = now - retain_min * 60
    before = len(store)
    store = [r for r in store if r["ts"] >= cutoff]
    pruned = before - len(store)
    # prices snapshot
    prices, port_move = fetch_live_prices()
    with open(PRICES_PATH, "w") as f:
        json.dump({"prices": prices, "portfolio_move": port_move,
                   "asof": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
    # 4. summarize live news only when something changed (or forced). The 7B
    #    stays loaded in this long-running process; the web just reads the cached
    #    summary.json — so a web restart is instant and the model never reloads.
    ran = False
    if fresh or force:
        from webapp.live import write_live_summary
        items = _to_newsitems(store)              # full feed (with timestamps)
        out = write_live_summary(use_llm=(backend == "7b"), items=items)
        out["portfolio_move"] = port_move
        out["retained_news"] = len(store)
        with open(SIGNAL_PATH, "w") as f:         # keep signal.json for back-compat
            json.dump(out, f)
        ran = True
    return store, fresh, pruned, ran


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=60, help="seconds between fetches")
    ap.add_argument("--retain-min", type=int, default=7 * 24 * 60, help="retention (minutes; default 7 days)")
    ap.add_argument("--backend", choices=["mock", "7b"], default="mock")
    ap.add_argument("--force-every", type=int, default=15, help="run model every N polls even with no new news")
    ap.add_argument("--rag", action="store_true", help="inject analogues from the labeled historical bank")
    ap.add_argument("--minutes", type=float, default=0, help="run for N minutes (0 = forever)")
    args = ap.parse_args()

    os.makedirs(STORE_DIR, exist_ok=True)
    store = _load_store()
    print(f"[daemon] start backend={args.backend} poll={args.poll}s "
          f"retain={args.retain_min}min ({args.retain_min/1440:.1f}d) "
          f"store={len(store)} items", flush=True)
    deadline = _now() + args.minutes * 60 if args.minutes else None
    i = 0
    while True:
        i += 1
        force = (i % max(1, args.force_every) == 0)
        try:
            store, fresh, pruned, ran = tick(store, args.retain_min, args.backend, force, rag=args.rag)
            _save_store(store)
            sig = json.load(open(SIGNAL_PATH)) if os.path.exists(SIGNAL_PATH) else {}
            print(f"[{datetime.now(timezone.utc):%H:%M:%S}] tick {i}: +{fresh} new "
                  f"-{pruned} pruned | store={len(store)} | "
                  f"{'RAN ' if ran else 'skip'} crash_prob="
                  f"{sig.get('crash_prob','?')} edges={sig.get('n_edges','?')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] tick {i} error: {exc}", flush=True)
        if deadline and _now() >= deadline:
            print("[daemon] reached --minutes limit, stopping.", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
