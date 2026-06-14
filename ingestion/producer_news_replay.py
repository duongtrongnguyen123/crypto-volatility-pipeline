"""News replay producer — stream a historical news CSV into Kafka `crypto-news`.

Lets the whole real-time pipeline (consumer_trr / consumer_sentiment) be demoed
end-to-end WITHOUT a live CryptoPanic token, by replaying a downloaded news
dataset (e.g. data/news_raw/oliviervha/cryptonews.csv) at an accelerated rate.

Run:
    python -m ingestion.producer_news_replay --csv data/news_raw/oliviervha/cryptonews.csv --rate 50
"""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

import config


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a news CSV into Kafka")
    ap.add_argument("--csv", default="data/news_raw/oliviervha/cryptonews.csv")
    ap.add_argument("--rate", type=float, default=50.0,
                    help="messages per second (accelerated replay)")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Tolerate the common column-name variants.
    title_col = next((c for c in ["title", "headline", "text"] if c in df.columns), None)
    date_col = next((c for c in ["date", "timestamp", "published_at"] if c in df.columns), None)
    src_col = next((c for c in ["source", "publisher"] if c in df.columns), None)
    if title_col is None:
        raise SystemExit(f"no title/headline column in {args.csv}: {list(df.columns)}")
    df = df.sort_values(date_col) if date_col else df
    if args.limit:
        df = df.head(args.limit)

    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )
    print(f"[news_replay] replaying {len(df)} rows -> '{config.TOPIC_NEWS}' "
          f"at {args.rate}/s")

    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    for _, row in df.iterrows():
        record = {
            "timestamp": str(row[date_col]) if date_col else "",
            "headline": str(row[title_col]),
            "url": str(row.get("url", "")),
            "source": str(row[src_col]) if src_col else "replay",
        }
        producer.send(config.TOPIC_NEWS, record)
        sent += 1
        if sent % 500 == 0:
            producer.flush()
            print(f"[news_replay] sent {sent}/{len(df)}")
        if delay:
            time.sleep(delay)
    producer.flush()
    print(f"[news_replay] done — {sent} headlines streamed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[news_replay] stopped")
