"""Producer 1 — Binance aggTrade WebSocket -> Kafka topic `crypto-price`.

Streams live BTC/USDT aggregated trades and pushes a normalized record per
trade. Reconnects automatically if the socket drops.

Run:
    python -m ingestion.producer_price
"""
import json
import sys
import time

import websocket
from kafka import KafkaProducer

import config


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=50,
        acks=1,
    )


def main() -> None:
    producer = make_producer()
    print(f"[producer_price] -> kafka {config.KAFKA_BOOTSTRAP_SERVERS} "
          f"topic '{config.TOPIC_PRICE}'")

    def on_message(_ws, message: str) -> None:
        data = json.loads(message)
        # aggTrade fields: s=symbol, p=price, q=quantity, T=trade time(ms), m=buyer-maker
        record = {
            "symbol": data["s"],
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "trade_time": int(data["T"]),
            "is_buyer_maker": bool(data["m"]),
        }
        producer.send(config.TOPIC_PRICE, record)

    def on_error(_ws, error) -> None:
        print(f"[producer_price] ws error: {error}", file=sys.stderr)

    def on_close(_ws, status_code, msg) -> None:
        print(f"[producer_price] ws closed ({status_code} {msg}); reconnecting...")

    def on_open(_ws) -> None:
        print("[producer_price] connected to Binance aggTrade stream")

    # Reconnect loop.
    while True:
        ws = websocket.WebSocketApp(
            config.BINANCE_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=180, ping_timeout=10)
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[producer_price] stopped")
