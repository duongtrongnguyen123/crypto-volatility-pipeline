"""Feature joining — all live streams -> 11-feature Parquet rows.

Reads the windowed price features plus the news, futures, depth and liquidation
streams, aligns everything on the 5-minute window grid, and writes the merged
rows (config.FEATURE_COLUMNS) to ./data/features/ as Parquet.

Topology (all keyed on `window_start`, price is the driving/left side):

    features-price ──┐
    features-sentiment ─(window mean)─┐
    crypto-futures ────(window last)──┤  left-outer joins
    crypto-depth ──────(window mean)──┤
    crypto-liquidations (window sum)──┘
                                       └─► ./data/features/*.parquet

Missing values are coalesced to neutral defaults so a price window always
produces a row even when a given market stream is briefly absent.

Run:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      processing/feature_join.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

import config

PRICE_SCHEMA = StructType([
    StructField("window_start", TimestampType()),
    StructField("window_end", TimestampType()),
    StructField("vwap", DoubleType()),
    StructField("price_return", DoubleType()),
    StructField("volume", DoubleType()),
    StructField("trade_count", LongType()),
    StructField("volatility", DoubleType()),
    StructField("taker_ls_ratio", DoubleType()),
])

SENTIMENT_SCHEMA = StructType([
    StructField("timestamp", StringType()),
    StructField("headline", StringType()),
    StructField("sentiment_score", DoubleType()),
])

FUTURES_SCHEMA = StructType([
    StructField("timestamp", LongType()),
    StructField("open_interest", DoubleType()),
    StructField("funding_rate", DoubleType()),
    StructField("mark_price", DoubleType()),
])

DEPTH_SCHEMA = StructType([
    StructField("timestamp", LongType()),
    StructField("mid", DoubleType()),
    StructField("book_depth", DoubleType()),
])

LIQ_SCHEMA = StructType([
    StructField("timestamp", LongType()),
    StructField("side", StringType()),
    StructField("price", DoubleType()),
    StructField("quantity", DoubleType()),
    StructField("liq_notional", DoubleType()),
])

WIN = config.WINDOW_DURATION
WM = config.WATERMARK_DELAY


def read_topic(spark, topic, schema):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )
    return raw.select(
        F.from_json(F.col("value").cast("string"), schema).alias("d")
    ).select("d.*")


def windowed(df, time_col, aggs, out_col="w_start"):
    """Window a raw stream to the 5-min grid with the given aggregations."""
    return (
        df.withWatermark(time_col, WM)
        .groupBy(F.window(time_col, WIN))
        .agg(*aggs)
        .select(F.col("window.start").alias(out_col), "*")
        .drop("window")
        .withWatermark(out_col, WM)
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("feature-join")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # --- price: already 5-min windowed upstream -----------------------------
    price = (
        read_topic(spark, config.TOPIC_FEATURES_PRICE, PRICE_SCHEMA)
        .withWatermark("window_start", WM)
    )

    # --- sentiment: per-headline -> window mean -----------------------------
    sentiment = windowed(
        read_topic(spark, config.TOPIC_FEATURES_SENTIMENT, SENTIMENT_SCHEMA)
        .withColumn("event_time", F.to_timestamp("timestamp")),
        "event_time",
        [F.avg("sentiment_score").alias("sentiment_score")],
        out_col="s_start",
    )

    # --- futures: samples -> window last (OI/funding are state) -------------
    futures = windowed(
        read_topic(spark, config.TOPIC_FUTURES, FUTURES_SCHEMA)
        .withColumn("event_time", (F.col("timestamp") / 1000.0).cast("timestamp")),
        "event_time",
        [
            F.last("open_interest", ignorenulls=True).alias("open_interest"),
            F.last("funding_rate", ignorenulls=True).alias("funding_rate"),
        ],
        out_col="f_start",
    )

    # --- depth: snapshots -> window mean ------------------------------------
    depth = windowed(
        read_topic(spark, config.TOPIC_DEPTH, DEPTH_SCHEMA)
        .withColumn("event_time", (F.col("timestamp") / 1000.0).cast("timestamp")),
        "event_time",
        [F.avg("book_depth").alias("book_depth")],
        out_col="d_start",
    )

    # --- liquidations: events -> window sum of notional ---------------------
    liq = windowed(
        read_topic(spark, config.TOPIC_LIQUIDATIONS, LIQ_SCHEMA)
        .withColumn("event_time", (F.col("timestamp") / 1000.0).cast("timestamp")),
        "event_time",
        [F.sum("liq_notional").alias("liq_notional")],
        out_col="l_start",
    )

    def eq(col):
        # Equi-join on the 5-min grid with a watermark-friendly range guard.
        return (
            (F.col("window_start") == F.col(col))
            & (F.col(col) >= F.col("window_start") - F.expr(f"interval {WIN}"))
            & (F.col(col) <= F.col("window_start") + F.expr(f"interval {WIN}"))
        )

    merged = (
        price
        .join(sentiment, eq("s_start"), "leftOuter")
        .join(futures, eq("f_start"), "leftOuter")
        .join(depth, eq("d_start"), "leftOuter")
        .join(liq, eq("l_start"), "leftOuter")
        .select(
            "window_start",
            "window_end",
            "vwap",
            "price_return",
            "volume",
            "trade_count",
            "volatility",
            F.coalesce("sentiment_score", F.lit(0.0)).alias("sentiment_score"),
            F.coalesce("open_interest", F.lit(0.0)).alias("open_interest"),
            F.coalesce("funding_rate", F.lit(0.0)).alias("funding_rate"),
            F.coalesce("taker_ls_ratio", F.lit(0.5)).alias("taker_ls_ratio"),
            F.coalesce("book_depth", F.lit(0.0)).alias("book_depth"),
            F.coalesce("liq_notional", F.lit(0.0)).alias("liq_notional"),
        )
    )

    query = (
        merged.writeStream.format("parquet")
        .option("path", config.FEATURES_DIR)
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/feature_join")
        .outputMode("append")
        .start()
    )
    print(f"[feature_join] writing merged 11-feature rows -> {config.FEATURES_DIR}")
    query.awaitTermination()


if __name__ == "__main__":
    main()
