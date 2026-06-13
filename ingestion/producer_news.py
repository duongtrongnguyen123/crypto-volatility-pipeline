"""Producer 2 — CryptoPanic news polling -> Kafka topic `crypto-news`.

Polls the CryptoPanic free API every 60 seconds for BTC headlines and pushes
new (unseen) headlines into Kafka. De-duplicates by post id within the process
lifetime.

Run:
    CRYPTOPANIC_TOKEN=xxxx python -m ingestion.producer_news
"""
import json
import sys
import time

import requests
from kafka import KafkaProducer

import config


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )


def main() -> None:
    if not config.CRYPTOPANIC_TOKEN:
        print("[producer_news] WARNING: CRYPTOPANIC_TOKEN is empty. "
              "Set it in your environment/.env to receive real headlines.",
              file=sys.stderr)

    producer = make_producer()
    print(f"[producer_news] -> kafka {config.KAFKA_BOOTSTRAP_SERVERS} "
          f"topic '{config.TOPIC_NEWS}'")

    seen: set = set()
    params = {
        "auth_token": config.CRYPTOPANIC_TOKEN,
        "currencies": "BTC",
        "public": "true",
    }

    while True:
        try:
            resp = requests.get(config.CRYPTOPANIC_URL, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            new_count = 0
            for post in results:
                pid = post.get("id")
                if pid in seen:
                    continue
                seen.add(pid)
                record = {
                    "timestamp": post.get("published_at"),
                    "headline": post.get("title", ""),
                    "url": post.get("url", ""),
                    "source": (post.get("source") or {}).get("title", ""),
                }
                producer.send(config.TOPIC_NEWS, record)
                new_count += 1
            producer.flush()
            print(f"[producer_news] polled {len(results)} posts, "
                  f"{new_count} new")
        except Exception as exc:  # noqa: BLE001 - keep polling no matter what
            print(f"[producer_news] poll error: {exc}", file=sys.stderr)

        time.sleep(config.NEWS_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[producer_news] stopped")
