"""Spark batch ETL: 12 GB news corpus -> partitioned Parquet data lake.

This is the DISTRIBUTED-PROCESSING half of the storage story. Instead of the
single-process chunked filter, the same transform is expressed as a Spark job:

  read CSV (12 GB) --> clean/derive `year` --> write Parquet PARTITIONED BY year

Parquet-partitioned-by-year is the canonical big-data "data lake" / HDFS-style
layout: columnar + compressed + partition-pruned. A query that touches only 2020
reads only the year=2020 partition, not the whole corpus. The job runs on
`local[*]` (every core an executor) here, and the SAME code runs on a real
cluster by setting SPARK_MASTER=spark://host:7077 — nothing else changes.

Then Spark SQL runs distributed aggregations over the lake (per-year counts,
top tickers) to demonstrate partition pruning + parallel scan.

Run:
    SPARK_MASTER=local[*] python -m processing.spark_corpus_etl            # full
    python -m processing.spark_corpus_etl --sample 100000                  # quick
    python -m processing.spark_corpus_etl --query-only                     # skip ETL
"""
from __future__ import annotations

import argparse
import os
import time

CORPUS = os.environ.get("CORPUS_CSV", "data/fnspid_corpus/news.csv")
LAKE = os.environ.get("CORPUS_LAKE", "data/lake/news_parquet")


def build_spark():
    from pyspark.sql import SparkSession
    master = os.environ.get("SPARK_MASTER", "local[*]")
    spark = (SparkSession.builder.master(master)
             .appName("corpus-etl")
             .config("spark.sql.shuffle.partitions", os.environ.get("SHUFFLE", "64"))
             .config("spark.driver.memory", os.environ.get("DRIVER_MEM", "3g"))
             .config("spark.sql.files.maxPartitionBytes", "128m")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print(f"[spark] master={spark.sparkContext.master} "
          f"defaultParallelism={spark.sparkContext.defaultParallelism}", flush=True)
    return spark


def etl(spark, sample: int | None):
    from pyspark.sql import functions as F
    t0 = time.time()
    reader = (spark.read.option("header", True)
              .option("multiLine", True).option("quote", '"').option("escape", '"'))
    df = reader.csv(CORPUS)
    if sample:
        df = df.limit(sample)
    df = (df.filter(F.col("date").isNotNull() & (F.length("date") >= 10))
            .withColumn("year", F.substring("date", 1, 4))
            .select("date", "title", "summary", "assets", "source", "year"))
    (df.write.mode("overwrite").partitionBy("year")
       .parquet(LAKE if not sample else LAKE + "_sample"))
    print(f"[spark] ETL wrote partitioned Parquet in {time.time()-t0:.0f}s "
          f"-> {LAKE if not sample else LAKE + '_sample'}", flush=True)


def query(spark, sample: bool):
    from pyspark.sql import functions as F
    path = LAKE + ("_sample" if sample else "")
    lake = spark.read.parquet(path)
    lake.createOrReplaceTempView("news")
    print(f"[spark] lake rows = {lake.count():,}  partitions(year) = "
          f"{lake.select('year').distinct().count()}", flush=True)
    print("[spark] per-year article counts (distributed GROUP BY):")
    spark.sql("SELECT year, count(*) AS n FROM news GROUP BY year ORDER BY year").show(30, False)
    print("[spark] partition pruning demo — scan ONLY year=2020:")
    t0 = time.time()
    n2020 = spark.sql("SELECT count(*) FROM news WHERE year='2020'").collect()[0][0]
    print(f"   year=2020 rows={n2020:,} in {time.time()-t0:.1f}s (read 1 partition, not all)")
    print("[spark] top tickers (distributed):")
    (lake.withColumn("ticker", F.explode(F.split("assets", ",")))
         .filter(F.col("ticker") != "")
         .groupBy("ticker").count().orderBy(F.desc("count")).show(10, False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--query-only", action="store_true")
    args = ap.parse_args()
    spark = build_spark()
    try:
        if not args.query_only:
            etl(spark, args.sample)
        query(spark, sample=bool(args.sample))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
