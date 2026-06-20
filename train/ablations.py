"""Ablations + rigor for the TRR meta-learner.

1. WITHIN-SOURCE fairness test — the out-of-time split in run.py trains on
   analyst-ratings news and tests on FNSPID, so the LLM's crash_prob feature
   suffers a source shift. Here we time-split *within each source* so the LLM
   signal is evaluated on its own distribution — a fair test of whether it adds
   value over technicals.
2. CALIBRATION — reliability of the walk-forward OOF probabilities + Brier
   score, raw vs isotonic-recalibrated (honest: isotonic fit on an earlier
   time-half, evaluated on the later half).
3. PRECISION@K — of the K highest-risk days, how many were real crashes (the
   metric that matters for an alerting overlay).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from train.features import FEATURES_FULL, FEATURES_TECH, build_dataset
from train.run import _gbm


def _auroc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def within_source(df):
    out = {}
    for era in ["2016_2020", "2021_2023"]:
        d = df[df["era"] == era].sort_index()
        n = len(d); cut = int(n * 0.6)
        tr, te = d.iloc[:cut], d.iloc[cut:]
        yte = te["label_true"].to_numpy(); ytr = tr["label_true"].to_numpy()
        row = {"n_train": int(len(tr)), "n_test": int(len(te)),
               "test_crashes": int(yte.sum())}
        row["LLM_zeroshot"] = _auroc(yte, te["crash_prob"].to_numpy())
        if ytr.sum() > 0 and 0 < yte.sum() < len(yte):
            mt = _gbm().fit(tr[FEATURES_TECH], ytr)
            row["GBM_technical"] = _auroc(yte, mt.predict_proba(te[FEATURES_TECH])[:, 1])
            mf = _gbm().fit(tr[FEATURES_FULL], ytr)
            row["GBM_full"] = _auroc(yte, mf.predict_proba(te[FEATURES_FULL])[:, 1])
        out[era] = row
    return out


def _oof(df, n_splits=6):
    d = df.sort_index()
    X, y = d[FEATURES_FULL].to_numpy(), d["label_true"].to_numpy()
    oof = np.full(len(d), np.nan)
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y[tr_i].sum() == 0:
            continue
        oof[te_i] = _gbm().fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
    d = d.assign(oof=oof).dropna(subset=["oof"])
    return d


def calibration(d):
    y = d["label_true"].to_numpy(); p = d["oof"].to_numpy()
    # reliability bins
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rel = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            rel.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int(m.sum()),
                        "pred_mean": float(p[m].mean()), "obs_freq": float(y[m].mean())})
    # honest isotonic: fit on earlier time-half, eval Brier on later half
    half = len(d) // 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[:half], y[:half])
    brier_raw = brier_score_loss(y[half:], p[half:])
    brier_iso = brier_score_loss(y[half:], iso.predict(p[half:]))
    return {"reliability": rel, "brier_raw": float(brier_raw),
            "brier_isotonic": float(brier_iso)}


def precision_at_k(d, ks=(10, 20, 30)):
    s = d.sort_values("oof", ascending=False)
    y = s["label_true"].to_numpy()
    base = float(d["label_true"].mean())
    return {f"P@{k}": float(y[:k].mean()) for k in ks} | {"base_rate": base}


def main():
    df = build_dataset()
    ws = within_source(df)
    d = _oof(df)
    cal = calibration(d)
    pk = precision_at_k(d)

    L = ["# TRR meta-learner — ablations & rigor\n",
         "## 1. Within-source fairness (time-split inside each news source)\n",
         "| Era | n_test | crashes | LLM zero-shot | GBM technical | GBM full |",
         "|---|---|---|---|---|---|"]
    for era, r in ws.items():
        L.append(f"| {era} | {r['n_test']} | {r['test_crashes']} | "
                 f"{r.get('LLM_zeroshot',float('nan')):.3f} | {r.get('GBM_technical',float('nan')):.3f} "
                 f"| {r.get('GBM_full',float('nan')):.3f} |")
    L += ["\n## 2. Calibration (walk-forward OOF)\n",
          f"Brier: raw={cal['brier_raw']:.4f}  isotonic={cal['brier_iso'] if 'brier_iso' in cal else cal['brier_isotonic']:.4f}\n",
          "| prob bin | n | predicted | observed |", "|---|---|---|---|"]
    for r in cal["reliability"]:
        L.append(f"| {r['bin']} | {r['n']} | {r['pred_mean']:.3f} | {r['obs_freq']:.3f} |")
    L += ["\n## 3. Precision@K (highest-risk days)\n",
          f"base rate = {pk['base_rate']:.3f}"]
    for k in (10, 20, 30):
        L.append(f"- P@{k} = {pk[f'P@{k}']:.2f}  ({pk[f'P@{k}']/pk['base_rate']:.1f}x base)")

    report = "\n".join(L)
    print(report)
    with open("reports/ablation_results.md", "w") as f:
        f.write(report + "\n")
    with open("reports/stock_runs/ablation_metrics.json", "w") as f:
        json.dump({"within_source": ws, "calibration": cal, "precision_at_k": pk},
                  f, indent=2, default=str)
    print("\n[ablations] wrote reports/ablation_results.md")


if __name__ == "__main__":
    main()
