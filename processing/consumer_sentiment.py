"""Consumer 2 — `crypto-news` -> per-headline sentiment -> `features-sentiment`.

For each incoming headline it calls `score_sentiment(text)` (FinBERT) and emits
{timestamp, headline, sentiment_score}.

The scoring function is wrapped in a Spark UDF. FinBERT is loaded lazily inside
the worker process (module-level cache), so it materializes at most once per
executor. Run in local mode so the driver Python env has torch/transformers:

    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      processing/consumer_sentiment.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

import config

NEWS_SCHEMA = StructType([
    StructField("timestamp", StringType()),
    StructField("headline", StringType()),
    StructField("url", StringType()),
    StructField("source", StringType()),
])


def _score(text):
    # Imported inside the UDF so the closure is shipped to workers cleanly.
    from sentiment.finbert import score_sentiment
    return float(score_sentiment(text))


score_udf = F.udf(_score, DoubleType())


def main() -> None:
    spark = (
        SparkSession.builder.appName("sentiment-features")
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
        .withColumn("sentiment_score", score_udf(F.col("headline")))
        .select(
            F.coalesce(F.col("timestamp"), F.current_timestamp().cast("string")).alias("timestamp"),
            "headline",
            "sentiment_score",
        )
    )

    out = news.select(F.to_json(F.struct("*")).alias("value"))

    query = (
        out.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", config.TOPIC_FEATURES_SENTIMENT)
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/features_sentiment")
        .outputMode("append")
        .start()
    )
    print(f"[consumer_sentiment] scoring headlines -> '{config.TOPIC_FEATURES_SENTIMENT}'")
    query.awaitTermination()


if __name__ == "__main__":
    main()
