"""Spark Structured Streaming unit test — windowed price-feature transform.

Pushes a small STATIC trades DataFrame through the actual production transform
(`processing.consumer_price.price_features`) and asserts the 5-minute window
aggregation produces the right schema and values (vwap, OHLC range volatility,
taker long/short ratio, open/close return via the min/max-struct trick).

Requires pyspark + a JVM. Skips cleanly where Spark/Java is unavailable (e.g.
this dev sandbox) — run on the cluster / a Spark-enabled machine:
    python tests/test_spark_processing.py      # or: pytest tests/test_spark_processing.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _skip(msg):
    print(f"SKIP test_spark_processing ({msg}) — run on a Spark-enabled machine.")
    sys.exit(0)


def main():
    try:
        from pyspark.sql import SparkSession
    except Exception as e:  # noqa: BLE001
        _skip(f"pyspark import failed: {e}")

    try:
        spark = (SparkSession.builder.master("local[1]")
                 .appName("test-price-features")
                 .config("spark.sql.shuffle.partitions", "1")
                 .getOrCreate())
    except Exception as e:  # noqa: BLE001 — typically "Java gateway"/no JVM
        _skip(f"no JVM: {e}")

    from processing.consumer_price import price_features

    base = 1_700_000_000_000  # epoch ms; all within one 5-min window
    rows = [
        # symbol, price, quantity, trade_time(ms), is_buyer_maker
        ("BTCUSDT", 100.0, 2.0, base + 1_000, False),  # taker BUY  (maker=False)
        ("BTCUSDT", 110.0, 1.0, base + 2_000, True),   # taker SELL
        ("BTCUSDT",  90.0, 1.0, base + 3_000, True),   # taker SELL (sets the low)
        ("BTCUSDT", 105.0, 1.0, base + 4_000, False),  # taker BUY  (last -> close)
    ]
    df = spark.createDataFrame(
        rows, "symbol string, price double, quantity double, trade_time long, is_buyer_maker boolean")
    out = price_features(df).collect()

    assert len(out) == 1, f"expected one 5-min window, got {len(out)}"
    r = out[0]
    cols = set(r.asDict())
    expected = {"window_start", "window_end", "vwap", "price_return",
                "volume", "trade_count", "volatility", "taker_ls_ratio"}
    assert expected <= cols, f"missing columns: {expected - cols}"

    # vwap = sum(p*q)/sum(q) = (200+110+90+105)/5 = 101.0
    assert abs(r["vwap"] - 101.0) < 1e-6, r["vwap"]
    assert r["trade_count"] == 4 and abs(r["volume"] - 5.0) < 1e-6
    # volatility = (high-low)/open = (110-90)/100 = 0.20 ; open=first=100, close=last=105
    assert abs(r["volatility"] - 0.20) < 1e-6, r["volatility"]
    assert abs(r["price_return"] - 0.05) < 1e-6, r["price_return"]
    # taker-buy volume = 2+1 = 3 of 5 -> ratio 0.6
    assert abs(r["taker_ls_ratio"] - 0.6) < 1e-6, r["taker_ls_ratio"]

    spark.stop()
    print("PASS  test_spark_price_features (vwap/volatility/return/taker ratio + schema)")
    print("\n1 spark test passed.")


if __name__ == "__main__":
    main()
