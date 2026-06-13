"""Train the LSTM volatility model.

By default trains OFFLINE on the historical 5-min eth-alpha dataset; pass
`--source parquet` to train on the live feature store instead.

Training uses a 3-way CHRONOLOGICAL split (train / val / test). The model is
fit on train, the best epoch is selected on val (early stopping on val loss),
and the test split is left UNTOUCHED — ml/evaluate.py owns the held-out test
set using the identical partition from ml.dataset.chronological_split.

Saves a single checkpoint to ./models/lstm_volatility.pt containing the model
weights, hyperparameters, the feature standardizer, the feature layout, the
split fractions, the per-epoch history, and the best val metrics.

Run:
    python -m ml.train --epochs 50                 # historical (default)
    python -m ml.train --source parquet --epochs 50
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from ml.dataset import (
    Standardizer,
    chronological_split,
    load_feature_frame,
    make_sequences,
)
from ml.model import LSTMVolatility


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic where it does not break CPU/GPU execution.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_precision(device: str):
    """Pick (autocast_dtype, use_scaler) for the active device.

    bf16 autocast on sm_80+ (no scaler needed), fp16 autocast + GradScaler on
    sm_70/75, plain fp32 on older GPUs and on CPU.
    """
    if device != "cuda":
        return None, False
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16, False
    if major >= 7:
        return torch.float16, True
    return None, False


@torch.no_grad()
def _val_metrics(model, val_dl, loss_fn, device, autocast_dtype) -> tuple[float, float, float]:
    """Return (val_loss, rmse, mae) over the validation loader."""
    model.eval()
    total_loss, sq_err, abs_err, n = 0.0, 0.0, 0.0, 0
    for xb, yb in val_dl:
        xb, yb = xb.to(device), yb.to(device)
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                pred = model(xb)
                loss = loss_fn(pred, yb)
        else:
            pred = model(xb)
            loss = loss_fn(pred, yb)
        pred = pred.float()
        total_loss += loss.item() * len(xb)
        sq_err += torch.sum((pred - yb) ** 2).item()
        abs_err += torch.sum(torch.abs(pred - yb)).item()
        n += len(xb)
    return total_loss / n, (sq_err / n) ** 0.5, abs_err / n


def train(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    source: str = "historical",
    max_rows: int = 0,
    seed: int = 42,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> dict:
    """Train the LSTM and save the best-val checkpoint. Returns the val metrics."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype, use_scaler = _select_precision(device)
    dtype_name = (
        str(autocast_dtype).replace("torch.", "") if autocast_dtype is not None else "float32"
    )
    print(f"[train] device={device}  dtype={dtype_name}  source={source}  seed={seed}")

    df = load_feature_frame(source)
    if max_rows and len(df) > max_rows:
        # Most-recent slice — handy for quick smoke runs on CPU.
        df = df.iloc[-max_rows:]
        print(f"[train] limited to last {len(df)} rows")
    X, y = make_sequences(df)
    print(f"[train] built {len(X)} sequences of shape {X.shape[1:]} "
          f"from {len(df)} windows")

    # 3-way chronological split (no shuffling across time boundaries). The test
    # split is held out here and consumed only by ml/evaluate.py.
    train_idx, val_idx, test_idx = chronological_split(len(X), val_frac, test_frac)
    X_train, y_train = X[train_idx.start:train_idx.stop], y[train_idx.start:train_idx.stop]
    X_val, y_val = X[val_idx.start:val_idx.stop], y[val_idx.start:val_idx.stop]
    print(f"[train] split  train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)} (held out)")

    # Fit the scaler on TRAIN ONLY to avoid leakage from val/test.
    scaler = Standardizer.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    loader_kw = dict(num_workers=2, pin_memory=True) if device == "cuda" else {}
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, **loader_kw)

    model = LSTMVolatility(input_size=len(config.FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_scaler)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_metrics = {"rmse": float("inf"), "mae": float("inf")}
    epochs_since_improve = 0
    history: list[dict] = []
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = loss_fn(model(xb), yb)
            else:
                loss = loss_fn(model(xb), yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        val_loss, val_rmse, val_mae = _val_metrics(
            model, val_dl, loss_fn, device, autocast_dtype
        )
        scheduler.step(val_loss)

        lr_now = opt.param_groups[0]["lr"]
        print(f"[train] epoch {epoch:3d}  train_mse={train_loss:.6e}  "
              f"val_mse={val_loss:.6e}  val_rmse={val_rmse:.6e}  "
              f"val_mae={val_mae:.6e}  lr={lr_now:.2e}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "lr": lr_now,
        })

        if val_loss < best_val:
            best_val = val_loss
            best_metrics = {"rmse": val_rmse, "mae": val_mae}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs)")
                break

    # Restore the best-val weights before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": model.hparams,
            "scaler": scaler.to_dict(),
            "feature_columns": config.FEATURE_COLUMNS,
            "sequence_length": config.SEQUENCE_LENGTH,
            "target_column": config.TARGET_COLUMN,
            "splits": {"val_frac": float(val_frac), "test_frac": float(test_frac)},
            "history": history,
            "val_metrics": best_metrics,
        },
        config.MODEL_PATH,
    )

    print(f"[train] best val_mse={best_val:.6e}  "
          f"rmse={best_metrics['rmse']:.6e}  mae={best_metrics['mae']:.6e} "
          f"-> saved {config.MODEL_PATH}")
    return best_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LSTM volatility model")
    ap.add_argument("--source", choices=["historical", "parquet"], default="historical")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="train on only the most recent N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stopping patience in epochs")
    args = ap.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        source=args.source,
        max_rows=args.max_rows,
        seed=args.seed,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
