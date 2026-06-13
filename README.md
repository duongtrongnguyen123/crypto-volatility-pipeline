# Real-Time Crypto Analysis & Volatility Prediction

End-to-end Big Data pipeline that predicts short-term BTC/USDT **volatility**. It
trains **offline** on 4+ years of historical 5-minute market data, and serves
**live**: streaming Binance trades + futures + order book + liquidations + news
through Kafka and Spark Structured Streaming, scoring news sentiment with
FinBERT, and feeding the latest feature window to a PyTorch LSTM.

Everything runs locally — no cloud. GPU (RTX 2060 Super) is used automatically
for FinBERT scoring and LSTM training when available.

## Train offline, serve live

The model is trained on a rich historical dataset and then deployed against the
live stream, which reproduces the **same 11-feature schema** in real time:

```
OFFLINE (training)
  historical 5-min CSVs ─► ml/historical.py ─► 11-feature matrix + target
                                              ─► ml/train.py ─► models/lstm_volatility.pt

LIVE (serving)
  Binance aggTrade  WS ─► crypto-price ────► consumer_price ──► features-price ─┐
  CryptoPanic news     ─► crypto-news  ────► consumer_sentiment ► features-sentiment ┤
  Binance OI/funding REST► crypto-futures ───────────────────────────────────────┤  feature_join
  Binance depth     WS ─► crypto-depth ──────────────────────────────────────────┤  (5-min join)
  Binance forceOrder WS─► crypto-liquidations ────────────────────────────────────┘     │
                                                                                        ▼
                                                       ./data/features/*.parquet ──► ml/infer.py
                                                                                  ► predicted volatility
```

## The 11 features (per 5-minute window)

| Feature | Definition | Live source | Historical source |
|---|---|---|---|
| `vwap` | Σ(price·qty)/Σ(qty) | aggTrade | `quote_volume/volume` |
| `price_return` | (close − open)/open | aggTrade | `*_5min_long.csv` |
| `volume` | Σ qty | aggTrade | `*_5min_long.csv` |
| `trade_count` | # trades | aggTrade | `n_trades` |
| `volatility` | (high − low)/open | aggTrade | OHLC range |
| `sentiment_score` | mean FinBERT score, ∈[−1,1] | CryptoPanic + FinBERT | 0 (no historical news) |
| `open_interest` | futures open interest | `/fapi/v1/openInterest` | `*_metrics_full.csv` |
| `funding_rate` | perp funding rate | `/fapi/v1/premiumIndex` | `*_funding.csv` (ffill) |
| `taker_ls_ratio` | taker-buy vol / total vol | aggTrade maker flag | `taker_buy_volume/volume` |
| `book_depth` | notional within ±1% | `@depth20@100ms` | `*_bookdepth_5min.csv` |
| `liq_notional` | Σ liquidation notional | `@forceOrder` | `*_liquidations_5min.csv` |

**Target:** the **next** window's `volatility` (regression, not raw price).

Notes on the historical data:
- BTC has no liquidations file, so ETH liquidations are used as a market-wide
  liquidation-stress proxy (configurable in `ml/historical.py`).
- News/sentiment has no historical record, so `sentiment_score` is 0 in training;
  it becomes a live signal at serving time. (Retrain on the accumulated live
  Parquet store — `make train` with `--source parquet` — to let the model learn
  sentiment once enough live data exists.)
- Heavy-tailed features (`volume`, `trade_count`, `open_interest`, `book_depth`,
  `liq_notional`) are `log1p`-compressed before standardizing — applied
  identically offline and online (`ml/dataset.feature_matrix`).

## Project structure

```
bigdata/
├── docker-compose.yml          # zookeeper, kafka (localhost:9092), spark master+worker
├── config.py                   # central config: topics, 11-feature schema, paths
├── requirements.txt · Makefile
├── ingestion/
│   ├── producer_price.py        # Binance aggTrade WS   -> crypto-price
│   ├── producer_news.py         # CryptoPanic API       -> crypto-news
│   ├── producer_futures.py      # OI + funding (REST)   -> crypto-futures
│   ├── producer_depth.py        # order-book depth WS   -> crypto-depth
│   └── producer_liquidations.py # forceOrder WS         -> crypto-liquidations
├── processing/
│   ├── consumer_price.py        # crypto-price -> features-price (5-min windows)
│   ├── consumer_sentiment.py    # crypto-news  -> features-sentiment (FinBERT UDF)
│   └── feature_join.py          # join all -> ./data/features parquet (11 features)
├── sentiment/
│   └── finbert.py               # score_sentiment(text) -> float in [-1, 1]
├── ml/
│   ├── historical.py            # merge historical CSVs -> feature matrix + target
│   ├── dataset.py               # frames -> standardized sequences (+ log1p, splits)
│   ├── model.py                 # LSTMVolatility: attention-pooled LSTM (PyTorch)
│   ├── train.py                 # train -> models/lstm_volatility.pt
│   ├── infer.py                 # load model + predict latest window
│   ├── baselines.py             # naive predictors (persistence, rolling, EWMA)
│   └── evaluate.py              # held-out test metrics + baseline comparison
├── kaggle/                     # RTX 6000 Pro GPU training deployment
│   ├── train_kernel.py          # Kaggle entrypoint (no-internet, GPU gate check)
│   ├── kernel-metadata.json     # the three-field RTX 6000 Pro gate
│   ├── dataset-metadata.json    # code + data bundle
│   ├── stage_and_deploy.sh      # upload dataset + push kernel
│   └── README.md                # deploy guide + GPU verification
├── scripts/
│   ├── create_topics.sh
│   └── generate_sample_features.py   # synthetic live feature store
├── tests/
│   └── test_smoke.py            # fast invariant checks (splits, shapes, model)
├── data/features/              # live parquet feature store (output)
├── reports/                    # evaluation plots (pred-vs-actual, scatter)
└── models/                     # trained checkpoints + eval_metrics.json (output)
```

Run the smoke tests with `make test` (no GPU/Kafka/network needed).

## Setup

```bash
pip install -r requirements.txt        # Python 3.10+, virtualenv/conda recommended
cp .env.example .env                    # add your CryptoPanic token

# Point HISTORICAL_DIR at the 5-min dataset (default already set):
#   HISTORICAL_DIR=/home/nduong/eth-alpha/data
```

## Quick start — train & predict (no infra needed)

The model trains on the historical dataset directly:

```bash
make train-quick     # 2-epoch smoke train on a recent slice (~seconds on CPU)
make infer           # predict next-window volatility
make evaluate        # test-set metrics + baseline comparison table

# Full training:
make train           # 50 epochs on the full 4-year, 441k-window dataset (local GPU)
make kaggle-deploy   # or train on Kaggle's RTX 6000 Pro (see below)
```

## Full live pipeline

```bash
make up               # start Kafka + Spark, auto-create topics
make topics           # (idempotent) ensure topics exist

# Ingestion — each in its own terminal:
make producer-price   make producer-news    make producer-futures
make producer-depth   make producer-liq

# Stream processing — each in its own terminal:
make consumer-price       # crypto-price -> features-price
make consumer-sentiment   # crypto-news  -> features-sentiment (downloads FinBERT once)
make feature-join         # merge all    -> ./data/features/*.parquet

# Inference against the live store:
make infer                # serves from parquet; falls back to historical if empty
```

Spark master UI: http://localhost:8080 — Kafka host listener: `localhost:9092`.

The Spark jobs run in **local mode** against `localhost:9092` (simplest for one
machine; also keeps `torch`/`transformers` available to the FinBERT UDF). The
`spark-master`/`spark-worker` containers satisfy the cluster requirement and host
the UI; add `--master spark://localhost:7077` to submit to the cluster instead.

## Model

`LSTMVolatility` (`ml/model.py`) — a 2-layer LSTM consuming sequences of
`SEQUENCE_LENGTH` (default 24 = 2h of 5-min windows) of the 11 features. Instead
of using only the final hidden state, it applies **additive (Bahdanau) attention
pooling** over all time steps, then a small MLP head regresses the next window's
volatility.

Training (`ml/train.py`) uses a **3-way chronological split** (train / val /
test), AdamW + weight decay, `ReduceLROnPlateau`, gradient clipping, and
**early stopping** on validation loss. On GPU it auto-selects mixed precision by
compute capability (**bf16** for sm_80+, **fp16** for sm_70/75, fp32 on CPU). The
checkpoint bundles weights, hyperparameters, the feature layout, the
standardizer, the split fractions, the training history, and validation metrics —
so inference and evaluation are fully self-contained.

## Evaluation & baselines

`make evaluate` (`ml/evaluate.py`) scores the trained model on the **held-out
test set** (the most recent 15%, never seen in training) and compares it against
naive baselines (`ml/baselines.py`): **persistence**, **rolling mean** (k=6, 12),
and **EWMA**. It reports RMSE, MAE, R², sMAPE/MAPE, and **directional accuracy**
(did it predict volatility up vs down?), writes `models/eval_metrics.json`, and
saves predicted-vs-actual and scatter plots to `reports/`.

Volatility is highly autocorrelated, so the EWMA/rolling baselines are
**strong** (R² ≈ 0.3) — beating them is the real bar, and the comparison table
makes the LSTM's added value explicit and honest. Full training to beat them runs
on the Kaggle GPU (below); a quick CPU `make train-quick` will not.

## Train on Kaggle RTX 6000 Pro GPU

`kaggle/` deploys offline training to Kaggle's NVIDIA RTX 6000 Pro (Blackwell,
sm_120). The historical CSVs + project code are staged as one **private** Kaggle
dataset (the kernel runs with **no internet**), and the kernel trains on GPU and
emits the checkpoint + metrics as downloadable outputs.

```bash
# One-time: place your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)
make kaggle-deploy     # stage data+code, create/version dataset, push kernel
make kaggle-output     # download outputs and verify the log shows sm_120
```

The RTX 6000 Pro is gated behind **three** kernel-metadata fields
(`machine_shape: NvidiaRtxPro6000`, `enable_gpu: true`,
`competition_sources: [...]`) — missing any one silently falls back to a Tesla
P100 (sm_60), which is broken on modern PyTorch. The kernel prints an explicit
`sm_<cap>` line and aborts on the P100 fallback. See `kaggle/README.md`.

## Configuration knobs (`config.py`)

- `WINDOW_DURATION` / `SEQUENCE_LENGTH` — time resolution and LSTM context length
- `FEATURE_COLUMNS` / `LOG_FEATURES` — model input layout and which features are log-scaled
- `SYMBOL`, `HISTORICAL_DIR`, Kafka topics, storage paths
```
