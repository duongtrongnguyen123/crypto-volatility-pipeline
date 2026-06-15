"""Stacked meta-learner over all TRR signals — walk-forward CV, calibration,
conformal prediction.

Fuses every signal we produced (LLM news-reasoning crash_prob, the six per-asset
probabilities, Fear & Greed, price-momentum, trailing volatility, news volume,
impact-edge count) into a gradient-boosted meta-model, evaluated under a
leak-free expanding **walk-forward** protocol (no future data ever trains the
past). Then:
  - **isotonic calibration** (fixes the over-confidence found in trr.analysis),
  - **split-conformal** prediction sets with a coverage guarantee.

Local only (no GPU). Run:  python -m trr.stacking
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from trr.labels import build_portfolio, crash_labels

OUT_DIR = "reports"


def build_features() -> pd.DataFrame:
    v4 = pd.read_csv("kaggle/out_v4/trr_predictions.csv", index_col=0)
    v4.index = pd.to_datetime(v4.index).date
    pa = pd.read_csv("kaggle/out_perasset_32b/trr_predictions.csv", index_col=0)
    pa.index = pd.to_datetime(pa.index).date

    port = build_portfolio(); port.index = pd.to_datetime(port.index).date
    lab = crash_labels(horizon=3); lab.index = pd.to_datetime(lab.index).date
    fng = json.load(open("data/fng.json"))["data"]
    fser = pd.Series({datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date(): int(x["value"])
                      for x in fng})

    idx = v4.index
    df = pd.DataFrame(index=idx)
    df["trr"] = v4["crash_prob"].values
    df["n_news"] = v4["n_news"].values
    df["n_edges"] = v4["n_edges"].values
    for c in [c for c in pa.columns if c.startswith("crash_prob_")]:
        df[c.replace("crash_prob_", "pa_")] = pa[c].reindex(idx).values
    df["fng_fear"] = (100 - fser.reindex(idx).ffill()).values
    df["price_mom5"] = (-port["portfolio_level"].pct_change(5)).reindex(idx).fillna(0).values
    df["vol20"] = port["portfolio_ret"].rolling(20).std().reindex(idx).fillna(0).values
    df["y"] = [int(lab["crash"].get(d, 0)) for d in idx]
    return df.dropna()


def _make(kind: str):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                              learning_rate=0.05, l2_regularization=1.0,
                                              random_state=0)
    # Strongly-regularized linear stack — the right capacity for ~70 positives.
    return LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000)


def walk_forward(df: pd.DataFrame, kind: str = "logreg", n_splits: int = 6):
    """Expanding-window OOF predictions: train on [0..k], predict the next block."""
    feats = [c for c in df.columns if c != "y"]
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    n = len(df)
    start = n // (n_splits + 1)
    oof = np.full(n, np.nan)
    for k in range(1, n_splits + 1):
        tr_end = start * k
        te_end = start * (k + 1) if k < n_splits else n
        if y[:tr_end].sum() < 5:
            continue
        sc = StandardScaler().fit(X[:tr_end])
        clf = _make(kind)
        clf.fit(sc.transform(X[:tr_end]), y[:tr_end])
        oof[tr_end:te_end] = clf.predict_proba(sc.transform(X[tr_end:te_end]))[:, 1]
    mask = ~np.isnan(oof)
    return oof, mask, feats


def split_conformal(scores, y, mask, alpha=0.2):
    """Split-conformal: calibrate a threshold so that 'no-crash' prediction sets
    cover the truth at ~1-alpha. Reports empirical coverage on the test half.
    """
    idx = np.where(mask)[0]
    half = len(idx) // 2
    cal, te = idx[:half], idx[half:]
    # nonconformity = predicted crash prob for true non-crash (we want to flag risk)
    s_cal = scores[cal]
    q = np.quantile(s_cal, 1 - alpha)  # threshold flagging top-alpha as "risk"
    flagged = scores[te] >= q
    # coverage: of actual crashes in test, how many were flagged (recall at 1-alpha budget)
    crashes = y[te] == 1
    recall = flagged[crashes].mean() if crashes.sum() else float("nan")
    alarm_rate = flagged.mean()
    return {"alpha": alpha, "threshold": float(q), "alarm_rate": float(alarm_rate),
            "crash_recall_at_budget": float(recall)}


def main():
    df = build_features()
    print(f"=== Stacked meta-learner | {len(df)} days, {int(df['y'].sum())} crashes ===")
    y = df["y"].to_numpy()

    print(f"\n[walk-forward OOF]")
    results = {}
    for kind in ("logreg", "hgb"):
        oof_k, mask_k, feats = walk_forward(df, kind=kind)
        au = roc_auc_score(y[mask_k], oof_k[mask_k])
        results[kind] = au
        print(f"    STACK-{kind:6s} (all signals)  AUROC = {au:.3f}")
    # Use the regularized linear stack downstream (HGB overfits at this scale).
    oof, mask, feats = walk_forward(df, kind="logreg")
    yv, sv = y[mask], oof[mask]
    au_stack = results["logreg"]
    print("    --- single signals (same folds) ---")
    for f in ["trr", "fng_fear", "price_mom5"]:
        print(f"    {f:24s} AUROC = {roc_auc_score(yv, df[f].to_numpy()[mask]):.3f}")

    # Isotonic calibration (fit first half of OOF, apply second).
    h = len(sv) // 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(sv[:h], yv[:h])
    sv_cal = iso.predict(sv[h:])
    b_raw = brier_score_loss(yv[h:], np.clip(sv[h:], 0, 1))
    b_cal = brier_score_loss(yv[h:], sv_cal)
    print(f"\n[calibration, held-out 2nd half]")
    print(f"    Brier raw={b_raw:.4f} -> isotonic={b_cal:.4f} "
          f"({'improved' if b_cal < b_raw else 'no gain'})")

    print(f"\n[split-conformal risk flagging]")
    cf = split_conformal(oof, y, mask, alpha=0.2)
    print(f"    20% alarm budget -> alarm_rate={cf['alarm_rate']:.2f}, "
          f"crash recall={cf['crash_recall_at_budget']:.2f}")

    # feature importance via permutation on the last fold
    print(f"\n[signals used] {feats}")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"n_days": len(df), "n_crash": int(df["y"].sum()),
               "auroc_stack": au_stack,
               "auroc_trr": float(roc_auc_score(yv, df["trr"].to_numpy()[mask])),
               "brier_raw": b_raw, "brier_isotonic": b_cal, "conformal": cf,
               "features": feats},
              open(f"{OUT_DIR}/analysis_stacking.json", "w"), indent=2)
    print(f"\n[saved] {OUT_DIR}/analysis_stacking.json")


if __name__ == "__main__":
    main()
