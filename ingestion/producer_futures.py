"""Producer 3 — Binance USD-M futures open interest + funding -> `crypto-futures`.

Polls the Binance futures REST API every FUTURES_POLL_INTERVAL_SECONDS:
    GET /fapi/v1/openInterest   -> current open interest (contracts)
    GET /fapi/v1/premiumIndex   -> lastFundingRate, markPrice

Emits a sample {timestamp, open_interest, funding_rate, mark_price} that the
market consumer windows to the 5-min grid. These match the historical
sum_open_interest / funding_rate features.

Run:
    python -m ingestion.producer_futures
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


def poll_once(session: requests.Session) -> dict:
    base = config.BINANCE_FAPI_BASE
    params = {"symbol": config.SYMBOL}

    oi = session.get(f"{base}/fapi/v1/openInterest", params=params, timeout=10).json()
    prem = session.get(f"{base}/fapi/v1/premiumIndex", params=params, timeout=10).json()

    return {
        "timestamp": int(time.time() * 1000),
        "open_interest": float(oi.get("openInterest", 0.0)),
        "funding_rate": float(prem.get("lastFundingRate", 0.0)),
        "mark_price": float(prem.get("markPrice", 0.0)),
    }


def main() -> None:
    producer = make_producer()
    session = requests.Session()
    print(f"[producer_futures] -> kafka {config.KAFKA_BOOTSTRAP_SERVERS} "
          f"topic '{config.TOPIC_FUTURES}'")

    while True:
        try:
            record = poll_once(session)
            producer.send(config.TOPIC_FUTURES, record)
            producer.flush()
            print(f"[producer_futures] OI={record['open_interest']:.1f} "
                  f"funding={record['funding_rate']:.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[producer_futures] poll error: {exc}", file=sys.stderr)
        time.sleep(config.FUTURES_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[producer_futures] stopped")
