"""Consumer 1 — `crypto-price` -> 5-minute price features -> `features-price`.

Spark Structured Streaming job. For each 5-minute tumbling window it computes:
    - vwap           : sum(price*qty) / sum(qty)
    - price_return   : (close - open) / open
    - volume         : sum(qty)
    - trade_count    : number of trades
    - volatility     : (high - low) / open   -- normalized range (matches offline)
    - taker_ls_ratio : taker-buy volume / total volume

`open`/`close`/`high`/`low` are recovered with the min/max-over-struct trick:
min(struct(event_time, price)).price is the earliest price, max(...).price the
latest. high/low use plain min/max(price). A Binance aggTrade has is_buyer_maker
== True when the buyer is the maker, i.e. the trade was taker-SELL; so taker-buy
volume is the quantity of trades with is_buyer_maker == False.

Run:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      processing/consumer_price.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

import config

TRADE_SCHEMA = StructType([
    StructField("symbol", StringType()),
    StructField("price", DoubleType()),
    StructField("quantity", DoubleType()),
    StructField("trade_time", LongType()),
    StructField("is_buyer_maker", BooleanType()),
])


def build_stream(spark: SparkSession):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", config.TOPIC_PRICE)
        .option("startingOffsets", "latest")
        .load()
    )

    trades = (
        raw.select(F.from_json(F.col("value").cast("string"), TRADE_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_time", (F.col("trade_time") / 1000.0).cast("timestamp"))
        # taker-buy quantity: aggressor is the buyer when is_buyer_maker is False.
        .withColumn(
            "taker_buy_qty",
            F.when(~F.col("is_buyer_maker"), F.col("quantity")).otherwise(F.lit(0.0)),
        )
    )

    agg = (
        trades.withWatermark("event_time", config.WATERMARK_DELAY)
        .groupBy(F.window("event_time", config.WINDOW_DURATION))
        .agg(
            (F.sum(F.col("price") * F.col("quantity")) / F.sum("quantity")).alias("vwap"),
            F.sum("quantity").alias("volume"),
            F.count(F.lit(1)).alias("trade_count"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.sum("taker_buy_qty").alias("taker_buy_volume"),
            F.min(F.struct("event_time", "price")).alias("first_row"),
            F.max(F.struct("event_time", "price")).alias("last_row"),
        )
        .withColumn("open", F.col("first_row.price"))
        .withColumn("close", F.col("last_row.price"))
        .withColumn("price_return", (F.col("close") - F.col("open")) / F.col("open"))
        .withColumn("volatility", (F.col("high") - F.col("low")) / F.col("open"))
        .withColumn(
            "taker_ls_ratio",
            F.when(F.col("volume") > 0, F.col("taker_buy_volume") / F.col("volume"))
            .otherwise(F.lit(0.5)),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "vwap",
            "price_return",
            "volume",
            "trade_count",
            "volatility",
            "taker_ls_ratio",
        )
    )
    return agg


def main() -> None:
    spark = (
        SparkSession.builder.appName("price-features")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    features = build_stream(spark)
    out = features.select(
        F.to_json(F.struct("*")).alias("value"),
        F.date_format("window_start", "yyyyMMddHHmm").alias("key"),
    )

    query = (
        out.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", config.TOPIC_FEATURES_PRICE)
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/features_price")
        .outputMode("update")
        .start()
    )
    print(f"[consumer_price] writing 5-min features -> '{config.TOPIC_FEATURES_PRICE}'")
    query.awaitTermination()


if __name__ == "__main__":
    main()
