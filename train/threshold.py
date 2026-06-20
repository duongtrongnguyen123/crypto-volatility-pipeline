"""Operating-threshold selection + confusion matrix for the crash detector.

AUROC is threshold-free; in deployment you must pick an alert threshold. This
sweeps the walk-forward OOF probabilities, reports the precision/recall/F1
trade-off, and prints the confusion matrix at two operating points:
  - max-F1 (balanced), and
  - a high-precision point (>=0.40 precision) for low-false-alarm alerting.
"""
from __future__ import annotations

import json

import numpy as np

from train.ablations import _oof
from train.features import build_dataset


def _confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f1": f1, "f2": f2}


def main():
    d = _oof(build_dataset())
    y, p = d["label_true"].to_numpy(), d["oof"].to_numpy()
    base = y.mean()

    def block(name, t, c, extra=""):
        return (f"### {name} (threshold = {t:.3f}){extra}\n"
                f"precision={c['precision']:.2f}  recall={c['recall']:.2f}  "
                f"F1={c['f1']:.2f}  F2={c['f2']:.2f}  lift={c['precision']/base:.1f}x\n\n"
                f"| | pred crash | pred calm |\n|---|---|---|\n"
                f"| **actual crash** | {c['tp']} (TP) | {c['fn']} (FN) |\n"
                f"| **actual calm**  | {c['fp']} (FP) | {c['tn']} (TN) |\n")

    L = ["# Operating-threshold selection (walk-forward OOF)\n",
         f"{len(d)} days, {int(y.sum())} crashes ({base:.1%} base rate). At this base "
         "rate with a moderate-AUROC detector, absolute precision is structurally "
         "limited; the decision-relevant view is the **alert rate** (how many days "
         "you flag) vs recall/precision lift.\n",
         "## Operating points by alert rate (top-k% riskiest days)\n"]
    json_pts = {}
    for rate in (0.05, 0.10, 0.15, 0.20):
        t = float(np.quantile(p, 1 - rate))
        c = _confusion(y, (p >= t).astype(int))
        L.append(block(f"Alert on riskiest {int(rate*100)}% of days", t, c))
        json_pts[f"alert_{int(rate*100)}pct"] = {"threshold": t, **c}
    # max-F1 for completeness
    grid = np.unique(np.round(np.quantile(p, np.linspace(0.5, 0.999, 80)), 5))
    best_f1 = max(((t, _confusion(y, (p >= t).astype(int))) for t in grid),
                  key=lambda r: r[1]["f1"])
    L.append(block("Max-F1 operating point", best_f1[0], best_f1[1]))
    json_pts["max_f1"] = {"threshold": float(best_f1[0]), **best_f1[1]}
    report = "\n".join(L)
    print(report)
    with open("reports/threshold_results.md", "w") as f:
        f.write(report + "\n")
    with open("reports/stock_runs/threshold_metrics.json", "w") as f:
        json.dump(json_pts, f, indent=2, default=str)
    print("\n[threshold] wrote reports/threshold_results.md")


if __name__ == "__main__":
    main()
