# Convenience targets for the crypto analysis pipeline.
# Spark jobs run in local mode against localhost:9092 (the host listener).

KAFKA_PKG := org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1
SPARK_SUBMIT := spark-submit --packages $(KAFKA_PKG)

.PHONY: help up down topics \
        producer-price producer-news producer-futures producer-depth producer-liq \
        consumer-price consumer-sentiment feature-join \
        train train-quick infer sample clean

help:
	@echo "Infrastructure:"
	@echo "  make up                 start kafka + spark (docker compose)"
	@echo "  make down               stop infrastructure"
	@echo "  make topics             (re)create kafka topics"
	@echo "Ingestion (host):"
	@echo "  make producer-price     Binance aggTrades   -> crypto-price"
	@echo "  make producer-news      CryptoPanic news    -> crypto-news"
	@echo "  make producer-futures   OI + funding (REST) -> crypto-futures"
	@echo "  make producer-depth     order-book depth    -> crypto-depth"
	@echo "  make producer-liq       liquidations        -> crypto-liquidations"
	@echo "Processing (host, local spark):"
	@echo "  make consumer-price     crypto-price -> features-price (5-min)"
	@echo "  make consumer-sentiment crypto-news  -> features-sentiment (FinBERT)"
	@echo "  make feature-join       merge all -> ./data/features parquet"
	@echo "ML:"
	@echo "  make train              train LSTM on historical 5-min dataset"
	@echo "  make train-quick        fast smoke train (2 epochs, recent slice)"
	@echo "  make infer              predict next-window volatility"
	@echo "  make sample             synthetic live feature store (for testing)"

# --- infrastructure ---
up:
	docker compose up -d zookeeper kafka kafka-setup spark-master spark-worker

down:
	docker compose down

topics:
	bash scripts/create_topics.sh

# --- ingestion ---
producer-price:
	python -m ingestion.producer_price

producer-news:
	python -m ingestion.producer_news

producer-futures:
	python -m ingestion.producer_futures

producer-depth:
	python -m ingestion.producer_depth

producer-liq:
	python -m ingestion.producer_liquidations

# --- processing ---
consumer-price:
	$(SPARK_SUBMIT) processing/consumer_price.py

consumer-sentiment:
	$(SPARK_SUBMIT) processing/consumer_sentiment.py

feature-join:
	$(SPARK_SUBMIT) processing/feature_join.py

# --- ml ---
train:
	python -m ml.train --source historical --epochs 50

train-quick:
	python -m ml.train --source historical --epochs 2 --max-rows 20000

infer:
	python -m ml.infer

sample:
	python -m scripts.generate_sample_features --rows 3000

clean:
	rm -rf data/checkpoints/* data/features/*.parquet data/features/_spark_metadata
