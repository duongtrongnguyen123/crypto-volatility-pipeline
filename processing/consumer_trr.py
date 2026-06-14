"""TRR speed layer — `crypto-news` -> live crash-risk signal -> `crash-signal`.

This is the real-time half of the lambda architecture for the TRR crash-detection
system. The heavy LLM temporal-relational reasoning runs OFFLINE on GPU
(kaggle/trr_standalone.py); this Spark Structured Streaming job produces a
low-latency crash-risk signal from the live news stream using a deterministic,
lightweight impact scorer (no GPU, no model load) so it can run continuously.

Pipeline:
    1. Read raw headlines from Kafka `crypto-news`.
    2. Score each headline into directed portfolio impacts with the TRR
       heuristic backend (trr.llm.MockLLM — a fast lexicon/asset scorer).
    3. Window the impacts into tumbling time buckets (Spark windowed agg).
    4. Emit a per-window crash-risk score = negative-impact concentration,
       to Kafka `crash-signal` and a Parquet store for the batch layer / serving.

Run:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      processing/consumer_trr.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

import config

NEWS_SCHEMA = StructType([
    StructField("timestamp", StringType()),
    StructField("headline", StringType()),
    StructField("url", StringType()),
    StructField("source", StringType()),
])

# Output of the per-headline impact scorer: list of (asset, polarity, weight).
IMPACT_SCHEMA = ArrayType(StructType([
    StructField("asset", StringType()),
    StructField("polarity", IntegerType()),
    StructField("weight", DoubleType()),
]))


def _score_impacts(headline: str):
    """Lightweight TRR Brainstorming for streaming: map a headline to directed
    portfolio impacts using the deterministic heuristic backend (no GPU).
    """
    from datetime import datetime

    from trr.llm import MockLLM
    from trr.schema import PORTFOLIO, NewsItem

    item = NewsItem(id="s", timestamp=datetime(1970, 1, 1),
                    title=headline or "", assets=[])
    edges = MockLLM().extract_impacts(item, PORTFOLIO)
    return [(e.object, int(e.polarity), float(e.weight)) for e in edges]


score_udf = F.udf(_score_impacts, IMPACT_SCHEMA)


def main() -> None:
    spark = (
        SparkSession.builder.appName("trr-crash-signal")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", config.TOPIC_NEWS)
        .option("startingOffsets", "latest")
        .load()
    )

    news = (
        raw.select(F.from_json(F.col("value").cast("string"), NEWS_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_time", F.coalesce(
            F.to_timestamp("timestamp"), F.current_timestamp()))
        .withColumn("impacts", score_udf(F.col("headline")))
    )

    # Explode impacts to one row per (headline, asset) directed edge.
    edges = news.select(
        "event_time",
        F.explode("impacts").alias("e"),
    ).select(
        "event_time",
        F.col("e.asset").alias("asset"),
        F.col("e.polarity").alias("polarity"),
        F.col("e.weight").alias("weight"),
    )

    # Window the impact graph into tumbling buckets and score crash risk =
    # weighted concentration of NEGATIVE impacts toward the portfolio.
    windowed = (
        edges.withWatermark("event_time", config.WATERMARK_DELAY)
        .groupBy(F.window("event_time", config.WINDOW_DURATION))
        .agg(
            F.count(F.lit(1)).alias("n_edges"),
            F.sum(F.when(F.col("polarity") < 0, 1).otherwise(0)).alias("n_neg"),
            F.sum(F.when(F.col("polarity") < 0, F.col("weight"))
                  .otherwise(0.0)).alias("neg_weight"),
            F.countDistinct("asset").alias("assets_hit"),
        )
        .withColumn(
            "crash_risk",
            F.least(F.lit(1.0),
                    0.1 + 0.7 * (F.col("n_neg") / F.greatest(F.col("n_edges"), F.lit(1)))
                    + 0.04 * F.col("assets_hit")),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "n_edges", "n_neg", "assets_hit", "crash_risk",
        )
    )

    # Sink 1: Kafka crash-signal topic (for live consumers / dashboards).
    kafka_q = (
        windowed.select(
            F.to_json(F.struct("*")).alias("value"),
            F.date_format("window_start", "yyyyMMddHHmm").alias("key"),
        )
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", config.TOPIC_CRASH_SIGNAL)
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/crash_signal_kafka")
        .outputMode("update")
        .start()
    )

    # Sink 2: Parquet store (batch layer / historical serving).
    parquet_q = (
        windowed.writeStream.format("parquet")
        .option("path", f"{config.FEATURES_DIR}/crash_signal")
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/crash_signal_parquet")
        .outputMode("append")
        .start()
    )

    print(f"[consumer_trr] live crash-risk -> '{config.TOPIC_CRASH_SIGNAL}' "
          f"+ {config.FEATURES_DIR}/crash_signal")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
