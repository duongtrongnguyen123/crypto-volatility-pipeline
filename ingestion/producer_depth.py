"""Producer 4 — Binance partial order-book depth -> `crypto-depth`.

Subscribes to the futures partial book-depth stream `<symbol>@depth20@100ms` and
emits a near-touch depth snapshot. `book_depth` is the total notional resting
within ~1% of the mid price on both sides — matching the historical
depth_-1.0pct + depth_1.0pct feature.

Run:
    python -m ingestion.producer_depth
"""
import json
import sys
import time

import websocket
from kafka import KafkaProducer

import config

NEAR_TOUCH_PCT = 0.01  # +/- 1% of mid


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )


def near_touch_notional(levels, mid: float, side: str) -> float:
    """Sum price*qty for levels within NEAR_TOUCH_PCT of mid."""
    total = 0.0
    for price_str, qty_str in levels:
        price = float(price_str)
        qty = float(qty_str)
        if side == "bid" and price >= mid * (1 - NEAR_TOUCH_PCT):
            total += price * qty
        elif side == "ask" and price <= mid * (1 + NEAR_TOUCH_PCT):
            total += price * qty
    return total


def main() -> None:
    producer = make_producer()
    url = f"{config.BINANCE_FUTURES_WS}/{config.SYMBOL_LOWER}@depth20@100ms"
    print(f"[producer_depth] -> kafka topic '{config.TOPIC_DEPTH}'")

    def on_message(_ws, message: str) -> None:
        data = json.loads(message)
        bids = data.get("b") or data.get("bids") or []
        asks = data.get("a") or data.get("asks") or []
        if not bids or not asks:
            return
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        record = {
            "timestamp": int(time.time() * 1000),
            "mid": mid,
            "book_depth": near_touch_notional(bids, mid, "bid")
            + near_touch_notional(asks, mid, "ask"),
        }
        producer.send(config.TOPIC_DEPTH, record)

    def on_error(_ws, error) -> None:
        print(f"[producer_depth] ws error: {error}", file=sys.stderr)

    def on_open(_ws) -> None:
        print("[producer_depth] connected to depth stream")

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
        print("\n[producer_depth] stopped")
