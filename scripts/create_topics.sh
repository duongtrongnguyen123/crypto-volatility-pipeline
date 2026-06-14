#!/usr/bin/env bash
# Create the application Kafka topics. Safe to re-run.
# Usage: ./scripts/create_topics.sh
set -euo pipefail

BROKER="${KAFKA_CONTAINER_BROKER:-kafka:29092}"

for topic in crypto-price crypto-news crypto-futures crypto-depth crypto-liquidations features-price features-sentiment features-market crash-signal; do
  docker compose exec kafka kafka-topics \
    --bootstrap-server "$BROKER" \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions 1 --replication-factor 1
done

echo "Current topics:"
docker compose exec kafka kafka-topics --bootstrap-server "$BROKER" --list
