"""Spark corpus-ETL test — partitioned-Parquet data-lake transform.

Runs the year-partitioning ETL + a distributed GROUP BY on a tiny in-memory
DataFrame (no dependency on the 12 GB corpus). Skips cleanly where Spark/Java is
unavailable, matching tests/test_spark_processing.py.

Run: python tests/test_spark_processing.py  (or pytest)
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _skip(msg):
    print(f"SKIP test_spark_corpus ({msg}) — run on a Spark-enabled machine.")
    sys.exit(0)


def main():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except Exception as e:  # noqa: BLE001
        _skip(f"pyspark import failed: {e}")
    try:
        spark = (SparkSession.builder.master("local[1]")
                 .appName("test-corpus-etl")
                 .config("spark.sql.shuffle.partitions", "1")
                 .getOrCreate())
    except Exception as e:  # noqa: BLE001
        _skip(f"no JVM: {e}")

    try:
        rows = [("2016-03-01", "calm", "s", "AAPL", "fnspid"),
                ("2020-03-12", "crash plunge", "p", "NVDA", "fnspid"),
                ("2020-03-13", "panic selloff", "q", "AAPL", "fnspid"),
                ("2023-07-01", "rally", "r", "TSLA", "fnspid")]
        df = spark.createDataFrame(rows, ["date", "title", "summary", "assets", "source"])
        df = df.withColumn("year", F.substring("date", 1, 4))

        lake = os.path.join(tempfile.mkdtemp(), "news_parquet")
        df.write.mode("overwrite").partitionBy("year").parquet(lake)

        # partitions exist per distinct year
        parts = sorted(d for d in os.listdir(lake) if d.startswith("year="))
        assert parts == ["year=2016", "year=2020", "year=2023"], parts

        back = spark.read.parquet(lake)
        assert back.count() == 4
        # distributed GROUP BY (Spark infers the partition column's type on
        # read-back, so normalise the year key to str)
        by_year = {str(r["year"]): r["n"] for r in
                   back.groupBy("year").count().withColumnRenamed("count", "n").collect()}
        assert by_year == {"2016": 1, "2020": 2, "2023": 1}, by_year
        # partition pruning returns only the 2020 slice
        assert back.filter(F.col("year") == 2020).count() == 2
        print("[spark-corpus] OK: partitioned lake + distributed GROUP BY + pruning")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
