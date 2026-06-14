"""Central configuration for the crypto analysis pipeline.

All values can be overridden via environment variables so the same code runs
on the host (talking to localhost:9092) or inside the Spark containers
(talking to kafka:29092).

Design: train OFFLINE on the historical 5-minute eth-alpha dataset, then SERVE
LIVE — the streaming Kafka/Spark pipeline produces the same feature schema for
real-time inference.
"""
import os

# --- Kafka ------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_PRICE = "crypto-price"            # raw aggTrades
TOPIC_NEWS = "crypto-news"              # raw news headlines
TOPIC_FUTURES = "crypto-futures"        # raw open-interest + funding samples
TOPIC_DEPTH = "crypto-depth"            # raw order-book depth snapshots
TOPIC_LIQUIDATIONS = "crypto-liquidations"  # raw forced-order (liquidation) events

TOPIC_FEATURES_PRICE = "features-price"
TOPIC_FEATURES_SENTIMENT = "features-sentiment"
TOPIC_FEATURES_MARKET = "features-market"   # windowed futures + depth + liq
TOPIC_CRASH_SIGNAL = "crash-signal"         # live TRR crash-risk per news window

# --- Data sources -----------------------------------------------------------
SYMBOL = "BTCUSDT"
SYMBOL_LOWER = SYMBOL.lower()

BINANCE_WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL_LOWER}@aggTrade"
# Futures (USD-M) endpoints for OI / funding / liquidations / depth.
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
CRYPTOPANIC_TOKEN = os.getenv("CRYPTOPANIC_TOKEN", "")
NEWS_POLL_INTERVAL_SECONDS = 60
FUTURES_POLL_INTERVAL_SECONDS = 30   # OI + funding REST poll cadence

# --- Stream processing ------------------------------------------------------
# 5-minute windows to match the historical training resolution.
WINDOW_DURATION = "5 minutes"
WATERMARK_DELAY = "10 minutes"

# --- Storage paths ----------------------------------------------------------
FEATURES_DIR = os.getenv("FEATURES_DIR", "./data/features")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./data/checkpoints")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/lstm_volatility.pt")

# Historical 5-minute dataset (offline training source).
HISTORICAL_DIR = os.getenv("HISTORICAL_DIR", "/home/nduong/eth-alpha/data")

# --- ML ---------------------------------------------------------------------
# Order matters: this is the model input vector layout. The SAME 11 features are
# produced offline (from CSVs) and online (from the live pipeline).
FEATURE_COLUMNS = [
    "vwap",            # volume-weighted average price
    "price_return",    # (close - open) / open
    "volume",          # base-asset volume
    "trade_count",     # number of trades
    "volatility",      # (high - low) / open  -- normalized range, computable both offline & online
    "sentiment_score", # mean FinBERT score of news in window (0 historically; live signal)
    "open_interest",   # futures open interest
    "funding_rate",    # perpetual funding rate (forward-filled to 5-min grid)
    "taker_ls_ratio",  # taker buy volume / total volume
    "book_depth",      # near-touch order-book depth (notional within ~1%)
    "liq_notional",    # liquidation notional (buy + sell) in the window
]
TARGET_COLUMN = "volatility"  # we predict the NEXT window's volatility
SEQUENCE_LENGTH = 24          # past windows fed to the LSTM (24 x 5min = 2h)

# Heavy-tailed, non-negative features are log1p-compressed before standardizing.
# This is applied identically at training and inference time (stored values stay
# raw/interpretable; the transform happens at model-input time).
LOG_FEATURES = ["volume", "trade_count", "open_interest", "book_depth", "liq_notional"]
