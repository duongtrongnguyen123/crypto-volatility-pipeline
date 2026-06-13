"""Producer 5 — Binance liquidation (forced order) stream -> `crypto-liquidations`.

Subscribes to the futures `<symbol>@forceOrder` stream and emits each liquidation
with its notional value (price * qty). The market consumer sums these per 5-min
window into `liq_notional`, matching the historical liquidation feature.

Run:
    python -m ingestion.producer_liquidations
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
        acks=1,
    )


def main() -> None:
    producer = make_producer()
    url = f"{config.BINANCE_FUTURES_WS}/{config.SYMBOL_LOWER}@forceOrder"
    print(f"[producer_liquidations] -> kafka topic '{config.TOPIC_LIQUIDATIONS}'")

    def on_message(_ws, message: str) -> None:
        data = json.loads(message)
        order = data.get("o", {})
        if not order:
            return
        price = float(order.get("p", 0.0))
        qty = float(order.get("q", 0.0))
        side = order.get("S", "")  # SELL = long liquidation, BUY = short liquidation
        record = {
            "timestamp": int(data.get("E", time.time() * 1000)),
            "side": side,
            "price": price,
            "quantity": qty,
            "liq_notional": price * qty,
        }
        producer.send(config.TOPIC_LIQUIDATIONS, record)

    def on_error(_ws, error) -> None:
        print(f"[producer_liquidations] ws error: {error}", file=sys.stderr)

    def on_open(_ws) -> None:
        print("[producer_liquidations] connected to forceOrder stream")

    while True:
        ws = websocket.WebSocketApp(
            url, on_open=on_open, on_message=on_message, on_error=on_error
        )
        ws.run_forever(ping_interval=180, ping_timeout=10)
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[producer_liquidations] stopped")
