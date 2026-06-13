"""Train the LSTM volatility model.

By default trains OFFLINE on the historical 5-min eth-alpha dataset; pass
`--source parquet` to train on the live feature store instead.

Saves a single checkpoint to ./models/lstm_volatility.pt containing the model
weights, hyperparameters, the feature standardizer, and the feature layout.

Run:
    python -m ml.train --epochs 50                 # historical (default)
    python -m ml.train --source parquet --epochs 50
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from ml.dataset import Standardizer, load_feature_frame, make_sequences
from ml.model import LSTMVolatility


def train(epochs: int, batch_size: int, lr: float, val_frac: float,
          source: str, max_rows: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  source={source}")

    df = load_feature_frame(source)
    if max_rows and len(df) > max_rows:
        # Most-recent slice — handy for quick smoke runs on CPU.
        df = df.iloc[-max_rows:]
        print(f"[train] limited to last {len(df)} rows")
    X, y = make_sequences(df)
    print(f"[train] built {len(X)} sequences of shape {X.shape[1:]} "
          f"from {len(df)} windows")

    # Chronological split (no shuffling across the time boundary).
    n_val = max(1, int(len(X) * val_frac))
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    scaler = Standardizer.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMVolatility(input_size=len(config.FEATURE_COLUMNS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        print(f"[train] epoch {epoch:3d}  train_mse={train_loss:.6e}  "
              f"val_mse={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hparams": model.hparams,
                    "scaler": scaler.to_dict(),
                    "feature_columns": config.FEATURE_COLUMNS,
                    "sequence_length": config.SEQUENCE_LENGTH,
                    "target_column": config.TARGET_COLUMN,
                },
                config.MODEL_PATH,
            )

    print(f"[train] best val_mse={best_val:.6e} -> saved {config.MODEL_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LSTM volatility model")
    ap.add_argument("--source", choices=["historical", "parquet"], default="historical")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="train on only the most recent N rows (0 = all)")
    args = ap.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.val_frac, args.source,
          args.max_rows)


if __name__ == "__main__":
    main()
