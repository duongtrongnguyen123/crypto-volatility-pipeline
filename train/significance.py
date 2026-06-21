"""Paired bootstrap significance of the RAG-vs-baseline crash-AUROC deltas.

A +0.07 AUROC gain on ~31 crash events could be noise. This pools the baseline
and RAG per-day predictions for each window (same days), and paired-bootstraps
the AUROC difference (resample days with replacement) to get a 95% CI on the
delta — does the improvement exclude zero?
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
B = 2000

# window -> (baseline tags, rag tags)
WINDOWS = {
    "stock_COVID_2019_20": (["stock_base"], ["stock_rag"]),
    "stock_broad_2018_20": ([f"s{i}" for i in range(5, 10)],
                            [f"rag_s{i}" for i in range(5, 10)]),
    "FNSPID_bear_2021_23": ([f"f{i}" for i in range(1, 6)],
                            [f"rag_f{i}" for i in range(1, 6)]),
}


def _pool(tags):
    fr = []
    for t in tags:
        p = f"kaggle/out_{t}/crash/trr_predictions.csv"
        if os.path.exists(p):
            fr.append(pd.read_csv(p, index_col=0)[["crash_prob", "label_true"]])
    if not fr:
        return None
    a = pd.concat(fr)
    return a[~a.index.duplicated(keep="first")]


def _boot_delta(y, pb, pr):
    n = len(y)
    out = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        out.append(roc_auc_score(ys, pr[idx]) - roc_auc_score(ys, pb[idx]))
    return np.array(out)


def main():
    rows = []
    for w, (btags, rtags) in WINDOWS.items():
        b, r = _pool(btags), _pool(rtags)
        if b is None or r is None:
            continue
        j = b.join(r, lsuffix="_b", rsuffix="_r", how="inner").dropna()
        y = j["label_true_b"].to_numpy()
        pb, pr = j["crash_prob_b"].to_numpy(), j["crash_prob_r"].to_numpy()
        if not (0 < y.sum() < len(y)):
            continue
        d = roc_auc_score(y, pr) - roc_auc_score(y, pb)
        bd = _boot_delta(y, pb, pr)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p_le0 = float((bd <= 0).mean())  # bootstrap p that RAG does not help
        rows.append({"window": w, "n_days": int(len(y)), "crashes": int(y.sum()),
                     "delta_auroc": float(d), "ci95_lo": float(lo),
                     "ci95_hi": float(hi), "boot_p_delta_le_0": p_le0,
                     "significant": bool(lo > 0)})

    L = ["# RAG vs baseline — paired bootstrap significance (2000 resamples)\n",
         "| Window | days | crashes | ΔAUROC | 95% CI | p(Δ≤0) | sig? |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['window']} | {r['n_days']} | {r['crashes']} | "
                 f"{r['delta_auroc']:+.3f} | [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] "
                 f"| {r['boot_p_delta_le_0']:.3f} | {'**yes**' if r['significant'] else 'no'} |")
    n_sig = sum(r["significant"] for r in rows)
    L.append(f"\n{n_sig}/{len(rows)} windows show a statistically significant RAG gain "
             f"(95% CI excludes 0). Few crash events widen the CIs — read accordingly.\n")
    report = "\n".join(L)
    print(report)
    with open("reports/significance_results.md", "w") as f:
        f.write(report + "\n")
    with open("reports/stock_runs/significance_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("[significance] wrote reports/significance_results.md")


if __name__ == "__main__":
    main()
