"""Held-out TEST evaluation of the LSTM volatility model.

Training (ml/train.py) reports only validation metrics; the test split is left
untouched. This module owns that held-out set: it loads the trained checkpoint,
rebuilds the model and scaler, and scores the model on the SAME most-recent
test_frac the trainer carved out (via the identical chronological_split).

It then scores naive baselines (ml/baselines.py) on the *exact same* test
sequences and prints a side-by-side comparison, so the LSTM's value is
demonstrated rather than assumed. Metrics: RMSE, MAE, R^2, sMAPE, and
directional accuracy (did the model call the up/down move of next-window
volatility relative to the last observed window?).

Artifacts:
    models/eval_metrics.json   full per-method metric dict + test size + split
    reports/pred_vs_actual.png predicted/actual volatility over the test timeline
    reports/scatter.png        predicted-vs-actual scatter with a y=x reference

Run from the project root:
    python -m ml.evaluate --source historical --max-rows 20000
    python -m ml.evaluate --model-path /tmp/agent2_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

import config
from ml.baselines import baseline_predictions, last_observed_volatility
from ml.dataset import (
    Standardizer,
    chronological_split,
    feature_matrix,  # noqa: F401  (re-exported convenience; used indirectly)
    load_feature_frame,
    make_sequences,
)
from ml.model import LSTMVolatility

# Volatility floor for percentage error. Below this the relative error explodes;
# we report sMAPE (symmetric, bounded) instead of raw MAPE and additionally
# expose a masked MAPE computed only over targets >= EPS for transparency.
EPS = 1e-6

METRIC_KEYS = ("rmse", "mae", "r2", "smape", "mape_masked", "directional_acc")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, last_obs: np.ndarray
) -> Dict[str, float]:
    """Standard regression metrics + directional accuracy for one predictor.

    Args:
        y_true:   actual next-window volatility [N].
        y_pred:   predicted next-window volatility [N].
        last_obs: most-recent observed volatility per sample [N] — the reference
                  the up/down direction is measured against.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    last_obs = np.asarray(last_obs, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # sMAPE: symmetric, bounded in [0, 200%], well-defined even near-zero vol.
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.where(denom > EPS, 2.0 * np.abs(y_pred - y_true) / denom, 0.0)
    smape = float(100.0 * smape_terms.mean())

    # Masked MAPE: classic MAPE but only over targets above the volatility floor,
    # so near-zero denominators can't blow it up. Reported alongside sMAPE.
    mask = np.abs(y_true) >= EPS
    if mask.any():
        mape_masked = float(
            100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))
        )
    else:
        mape_masked = float("nan")

    # Directional accuracy: sign of the predicted change vs the actual change,
    # both measured against the last observed volatility. A flat call (sign 0)
    # only counts as correct when the actual move is also flat.
    pred_dir = np.sign(y_pred - last_obs)
    true_dir = np.sign(y_true - last_obs)
    directional_acc = float(np.mean(pred_dir == true_dir))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "mape_masked": mape_masked,
        "directional_acc": directional_acc,
    }


# --------------------------------------------------------------------------- #
# Model prediction
# --------------------------------------------------------------------------- #
def _load_checkpoint(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"checkpoint not found: {model_path!r}. Train a model first "
            "(python -m ml.train)."
        )
    return torch.load(model_path, map_location="cpu", weights_only=False)


@torch.no_grad()
def _model_predict(
    model: LSTMVolatility, X: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Batched forward pass over scaled sequences -> predictions [N]."""
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).float()
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.empty(0, np.float32)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save_plots(
    y_true: np.ndarray, y_pred: np.ndarray, out_dir: str, max_points: int = 500
) -> list[str]:
    """Save the timeline and scatter plots; return the written paths.

    Degrades gracefully: if matplotlib is unavailable, returns [] and the caller
    proceeds with table + JSON only.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display required
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[evaluate] matplotlib unavailable ({exc}); skipping plots")
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    # (1) Predicted vs actual over the test timeline (first max_points for clarity).
    n = min(max_points, len(y_true))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(n), y_true[:n], label="actual", linewidth=1.1, color="#1f77b4")
    ax.plot(
        np.arange(n), y_pred[:n], label="LSTM predicted",
        linewidth=1.1, color="#d62728", alpha=0.85,
    )
    ax.set_title(f"Next-window volatility: predicted vs actual (first {n} test points)")
    ax.set_xlabel("test sequence index (chronological)")
    ax.set_ylabel("volatility")
    ax.legend(loc="upper right")
    fig.tight_layout()
    timeline_path = os.path.join(out_dir, "pred_vs_actual.png")
    fig.savefig(timeline_path, dpi=120)
    plt.close(fig)
    written.append(timeline_path)

    # (2) Scatter of predicted vs actual with a y=x reference line.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color="#2c7fb8", edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y = x")
    ax.set_title("LSTM predicted vs actual volatility")
    ax.set_xlabel("actual volatility")
    ax.set_ylabel("predicted volatility")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left")
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, "scatter.png")
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    written.append(scatter_path)

    return written


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def _format_table(metrics: Dict[str, Dict[str, float]], model_name: str) -> str:
    """Render a fixed-width comparison table; the model row is marked with '*'."""
    cols = METRIC_KEYS
    headers = {
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R^2",
        "smape": "sMAPE%",
        "mape_masked": "MAPE%",
        "directional_acc": "DirAcc",
    }
    name_w = max(len("method"), *(len(k) for k in metrics)) + 2
    col_w = 12

    def fmt(v: float) -> str:
        if v != v:  # NaN
            return "nan"
        return f"{v:.6f}" if abs(v) < 1e4 else f"{v:.4e}"

    lines = []
    head = "method".ljust(name_w) + "".join(headers[c].rjust(col_w) for c in cols)
    lines.append(head)
    lines.append("-" * len(head))
    # Model first (marked), then baselines in a stable order.
    order = [model_name] + [m for m in metrics if m != model_name]
    for name in order:
        mark = " *" if name == model_name else "  "
        row = (name + mark).ljust(name_w)
        row += "".join(fmt(metrics[name][c]).rjust(col_w) for c in cols)
        lines.append(row)
    lines.append("")
    lines.append("* = trained LSTM model.  DirAcc = directional accuracy "
                 "(up/down vs last observed).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(
    source: str = "historical",
    model_path: str | None = None,
    out_dir: str = "reports",
    max_rows: int = 0,
) -> dict:
    """Evaluate the trained model + baselines on the held-out test set.

    Returns the full metrics dict (also written to models/eval_metrics.json).
    Importable so a Kaggle kernel can call `evaluate(...)` directly.
    """
    model_path = model_path or config.MODEL_PATH
    ckpt = _load_checkpoint(model_path)

    seq_len = int(ckpt.get("sequence_length", config.SEQUENCE_LENGTH))
    splits = ckpt.get("splits", {"val_frac": 0.15, "test_frac": 0.15})
    val_frac = float(splits.get("val_frac", 0.15))
    test_frac = float(splits.get("test_frac", 0.15))

    # Rebuild model + scaler from the checkpoint.
    model = LSTMVolatility(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = Standardizer.from_dict(ckpt["scaler"])

    # Load features. CRITICAL: mirror ml/train.py's max_rows handling — it slices
    # df.iloc[-max_rows:] BEFORE building sequences, so to land on the identical
    # test split we must slice the frame the same way here.
    df = load_feature_frame(source)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
        print(f"[evaluate] limited to last {len(df)} rows")

    X, y = make_sequences(df, seq_len)
    _, _, test_idx = chronological_split(len(X), val_frac, test_frac)
    test_idx_arr = np.asarray(list(test_idx), dtype=np.int64)

    X_test = X[test_idx_arr]
    y_test = y[test_idx_arr]
    X_test_scaled = scaler.transform(X_test)

    print(f"[evaluate] source={source}  model={model_path}")
    print(f"[evaluate] sequences={len(X)}  test={len(y_test)}  seq_len={seq_len}  "
          f"val_frac={val_frac}  test_frac={test_frac}")

    # Reference point for directional accuracy = last observed volatility of each
    # test sequence (== the persistence baseline by construction).
    last_obs = last_observed_volatility(df, test_idx, seq_len)

    # Model predictions on the held-out test set.
    y_model = _model_predict(model, X_test_scaled)

    # Baseline predictions on the *same* test sequences.
    base_preds = baseline_predictions(df, test_idx, seq_len)

    model_name = "LSTM"
    all_preds: Dict[str, np.ndarray] = {model_name: y_model, **base_preds}

    metrics = {
        name: regression_metrics(y_test, pred, last_obs)
        for name, pred in all_preds.items()
    }

    # Comparison table.
    table = _format_table(metrics, model_name)
    print()
    print(table)
    print()

    # Plots (model predictions vs actual). Skipped gracefully if no matplotlib.
    plot_paths = _save_plots(y_test, y_model, out_dir)
    if plot_paths:
        print("[evaluate] wrote plots: " + ", ".join(plot_paths))

    # Persist the full result. Deterministic: no timestamps, sorted keys.
    result = {
        "source": source,
        "model_path": os.path.abspath(model_path),
        "test_size": int(len(y_test)),
        "num_sequences": int(len(X)),
        "sequence_length": seq_len,
        "split": {"val_frac": val_frac, "test_frac": test_frac},
        "model_name": model_name,
        "metrics": metrics,
        "plots": [os.path.abspath(p) for p in plot_paths],
    }
    metrics_dir = os.path.dirname(config.MODEL_PATH) or "."
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[evaluate] wrote metrics -> {metrics_path}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out test evaluation of the LSTM")
    ap.add_argument("--source", choices=["historical", "parquet"], default="historical")
    ap.add_argument("--model-path", default=None,
                    help=f"checkpoint to evaluate (default {config.MODEL_PATH})")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="evaluate on only the most recent N rows (mirror training)")
    ap.add_argument("--out-dir", default="reports", help="directory for the plots")
    args = ap.parse_args()
    evaluate(
        source=args.source,
        model_path=args.model_path,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
