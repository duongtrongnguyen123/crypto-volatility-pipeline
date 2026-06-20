#!/usr/bin/env python3
"""In-process Kafka/Spark-style DEMO of the TRR speed layer (zero infra).

The TRR system uses a lambda architecture:

    BATCH layer  — the heavy zero-shot LLM temporal-relational reasoning runs
                   OFFLINE on GPU (kaggle/trr_standalone.py, 32B model).
    SPEED layer  — a low-latency crash-risk signal is produced from the live
                   news stream by Kafka producers (ingestion/) feeding a Spark
                   Structured Streaming job (processing/consumer_trr.py) that
                   scores each headline with the deterministic, no-GPU TRR
                   heuristic backend (trr.llm.MockLLM).

The real speed layer needs a running Kafka broker and a Spark cluster. This
script reproduces the SAME dataflow entirely in-process so it can be run for a
demo / grading with no infrastructure at all:

    PRODUCER  thread  -> queue.Queue        ==  Kafka producer -> `crypto-news`
    CONSUMER  loop    -> _score_impacts()   ==  Spark readStream + scoring UDF
    rolling/decaying crash signal           ==  Spark windowed crash_risk agg
    live console table + ALERTs             ==  Kafka `crash-signal` sink

The scoring path is the exact same code as production: it imports
``processing.consumer_trr._score_impacts`` when importable (which itself calls
``trr.llm.MockLLM().extract_impacts``), and falls back to a local copy of that
function if PySpark is not installed. The crash-risk formula mirrors the one in
the Spark windowed aggregation in consumer_trr.py (negative-impact concentration
plus breadth across assets), and an exponential time-decay carries the signal
forward exactly as the TRR pipeline's DecayMemory does across days.

Run:
    .venv/bin/python scripts/demo_streaming.py                 # ~50 messages
    .venv/bin/python scripts/demo_streaming.py --messages 120  # more
    DEMO_MESSAGES=30 .venv/bin/python scripts/demo_streaming.py

It replays headlines from data/fnspid/stocknews.csv if present, otherwise the
bundled synthetic crypto corpus (trr.news.load_sample_news()), processes N
messages, prints a periodically-updating live view, and exits cleanly.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

# Make the repo root importable when run as `scripts/demo_streaming.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------- #
# Scoring backend: reuse the production speed-layer scorer if importable.
# --------------------------------------------------------------------------- #
def _resolve_score_impacts():
    """Return (score_impacts, source_label).

    Prefer processing.consumer_trr._score_impacts (the exact function the Spark
    job applies as a UDF). That module imports pyspark at top level, so on a box
    without Spark we fall back to a local copy that calls the identical TRR
    backend (trr.llm.MockLLM.extract_impacts) — same numbers, no Spark.
    """
    try:
        from processing.consumer_trr import _score_impacts as spark_scorer
        return spark_scorer, "processing.consumer_trr._score_impacts (Spark UDF)"
    except Exception:
        from trr.llm import MockLLM
        from trr.schema import PORTFOLIO, NewsItem

        _mock = MockLLM()

        def local_scorer(headline: str):
            item = NewsItem(id="s", timestamp=datetime(1970, 1, 1),
                            title=headline or "", assets=[])
            edges = _mock.extract_impacts(item, PORTFOLIO)
            return [(e.object, int(e.polarity), float(e.weight)) for e in edges]

        return local_scorer, "trr.llm.MockLLM.extract_impacts (in-process fallback)"


# --------------------------------------------------------------------------- #
# Headline source.
# --------------------------------------------------------------------------- #
def _load_headlines(limit: int) -> tuple[list[dict], str]:
    """Load up to ``limit`` headlines as {timestamp, headline, source} dicts.

    Tries data/fnspid/stocknews.csv first (a few hundred rows), then falls back
    to the bundled synthetic crypto corpus. Returns (records, source_label).
    """
    csv_path = os.path.join(_REPO_ROOT, "data", "fnspid", "stocknews.csv")
    if os.path.exists(csv_path):
        import csv as _csv

        records: list[dict] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            title_col = next((c for c in ("title", "headline", "text")
                              if c in (reader.fieldnames or [])), None)
            date_col = next((c for c in ("date", "timestamp", "published_at")
                             if c in (reader.fieldnames or [])), None)
            src_col = next((c for c in ("source", "publisher")
                            if c in (reader.fieldnames or [])), None)
            for row in reader:
                title = (row.get(title_col) or "").strip() if title_col else ""
                if not title:
                    continue
                records.append({
                    "timestamp": (row.get(date_col) or "").strip() if date_col else "",
                    "headline": title,
                    "source": (row.get(src_col) or "fnspid").strip() if src_col else "fnspid",
                })
                # Read a bounded slice (a few hundred) — enough to sample from.
                if len(records) >= max(limit, 400):
                    break
        if records:
            return records, f"data/fnspid/stocknews.csv ({len(records)} rows read)"

    # Fallback: bundled synthetic crypto corpus.
    from trr.news import load_sample_news

    items = load_sample_news()
    records = [{
        "timestamp": it.timestamp.isoformat(),
        "headline": it.title,
        "source": it.source or "synthetic",
    } for it in items]
    return records, f"trr.news.load_sample_news() ({len(records)} items)"


# --------------------------------------------------------------------------- #
# Rolling / decaying crash-risk state (the speed-layer signal).
# --------------------------------------------------------------------------- #
@dataclass
class CrashSignal:
    """Exponentially-decaying crash-risk signal updated per incoming headline.

    Mirrors consumer_trr.py's windowed crash_risk (negative-impact concentration
    + asset breadth) but smoothed over a rolling event window with a per-step
    decay, the streaming analogue of the batch pipeline's DecayMemory.
    """
    decay: float = 0.85          # per-message multiplicative decay of the signal
    threshold: float = 0.55      # ALERT when the rolling signal crosses this
    window: int = 12             # rolling window length for instantaneous risk

    rolling_signal: float = 0.0
    _recent: list = field(default_factory=list)  # recent per-msg (n_edges, n_neg, assets)
    peak: float = 0.0
    n_alerts: int = 0

    def _instant_risk(self) -> float:
        """Crash risk over the rolling window — same shape as the Spark agg."""
        if not self._recent:
            return 0.0
        n_edges = sum(r[0] for r in self._recent)
        n_neg = sum(r[1] for r in self._recent)
        assets_hit = len({a for r in self._recent for a in r[2]})
        if n_edges == 0:
            return 0.0
        # consumer_trr.py: 0.1 + 0.7*(n_neg/n_edges) + 0.04*assets_hit, capped 1.
        return min(1.0, 0.1 + 0.7 * (n_neg / n_edges) + 0.04 * assets_hit)

    def update(self, impacts: list) -> tuple[float, float, bool]:
        """Fold one headline's impacts into the signal.

        Returns (instant_crash_prob, rolling_signal, is_alert).
        """
        n_edges = len(impacts)
        n_neg = sum(1 for (_a, pol, _w) in impacts if pol < 0)
        assets = {a for (a, _p, _w) in impacts}

        self._recent.append((n_edges, n_neg, assets))
        if len(self._recent) > self.window:
            self._recent.pop(0)

        instant = self._instant_risk()
        # Decay the carried signal, then pull it toward the new instantaneous
        # risk (EWMA) — keeps a crash cluster elevated, lets quiet news fade.
        self.rolling_signal = max(self.decay * self.rolling_signal, instant)
        self.rolling_signal = min(1.0, self.rolling_signal)

        self.peak = max(self.peak, self.rolling_signal)
        is_alert = self.rolling_signal >= self.threshold
        if is_alert:
            self.n_alerts += 1
        return instant, self.rolling_signal, is_alert


# --------------------------------------------------------------------------- #
# Producer thread — replays headlines onto an in-memory "Kafka topic".
# --------------------------------------------------------------------------- #
def _producer(topic: "queue.Queue", records: list[dict], n_messages: int,
              rate: float, stop: threading.Event) -> None:
    """Replay ``n_messages`` headlines onto ``topic`` (cycling if needed)."""
    delay = 1.0 / rate if rate > 0 else 0.0
    for i in range(n_messages):
        if stop.is_set():
            break
        rec = dict(records[i % len(records)])
        rec["_offset"] = i
        rec["_ingest_ts"] = datetime.now().isoformat(timespec="seconds")
        topic.put(rec)
        if delay:
            time.sleep(delay)
    topic.put(None)  # poison pill -> clean consumer shutdown


# --------------------------------------------------------------------------- #
# Console view.
# --------------------------------------------------------------------------- #
def _bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="In-process Kafka/Spark-style demo of the TRR speed layer.")
    ap.add_argument("--messages", type=int,
                    default=int(os.getenv("DEMO_MESSAGES", "50")),
                    help="number of headlines to stream then exit (default 50)")
    ap.add_argument("--rate", type=float,
                    default=float(os.getenv("DEMO_RATE", "60")),
                    help="producer messages per second (default 60)")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="rolling-signal ALERT threshold (default 0.55)")
    ap.add_argument("--every", type=int, default=1,
                    help="refresh the live view every N messages (default 1)")
    args = ap.parse_args(argv)

    score_impacts, scorer_label = _resolve_score_impacts()
    records, source_label = _load_headlines(args.messages)
    signal = CrashSignal(threshold=args.threshold)

    print("=" * 78)
    print("  TRR SPEED LAYER — in-process Kafka/Spark streaming demo (no infra)")
    print("=" * 78)
    print(f"  topic source : {source_label}")
    print(f"  scorer       : {scorer_label}")
    print(f"  messages     : {args.messages}   rate: {args.rate}/s   "
          f"alert>= {args.threshold}")
    print(f"  mapping      : Queue==Kafka 'crypto-news' | scorer==Spark UDF | "
          f"signal==windowed crash_risk")
    print("-" * 78)
    header = f"{'#':>4}  {'crash':>6}  {'signal':>6}  {'risk-bar':<24}  headline"
    print(header)
    print("-" * 78)

    topic: queue.Queue = queue.Queue(maxsize=256)
    stop = threading.Event()
    prod = threading.Thread(
        target=_producer, args=(topic, records, args.messages, args.rate, stop),
        daemon=True, name="producer")
    prod.start()

    processed = 0
    alerts: list[tuple[int, str, float]] = []
    try:
        while True:
            rec = topic.get()
            if rec is None:  # poison pill
                break
            headline = rec["headline"]
            impacts = score_impacts(headline)
            instant, rolling, is_alert = signal.update(impacts)
            processed += 1

            if is_alert:
                alerts.append((rec["_offset"], headline, rolling))

            if processed % args.every == 0 or is_alert:
                flag = "  <<< ALERT" if is_alert else ""
                line = (f"{rec['_offset']:>4}  {instant:6.2f}  {rolling:6.2f}  "
                        f"{_bar(rolling):<24}  {_truncate(headline, 40)}{flag}")
                print(line, flush=True)
    except KeyboardInterrupt:
        stop.set()
        print("\n[demo] interrupted")
    finally:
        stop.set()
        prod.join(timeout=2.0)

    print("-" * 78)
    print(f"[demo] processed {processed} messages | peak signal {signal.peak:.2f} "
          f"| {signal.n_alerts} alert-message(s)")
    if alerts:
        print(f"[demo] first ALERT at msg #{alerts[0][0]} "
              f"(signal {alerts[0][2]:.2f}): {_truncate(alerts[0][1], 56)}")
    else:
        print("[demo] no crash ALERT crossed the threshold in this run")
    print("[demo] In production this signal is published to Kafka 'crash-signal' "
          "and a Parquet store (see processing/consumer_trr.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
