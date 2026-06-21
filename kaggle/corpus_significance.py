"""Bootstrap significance of the RAG lift on the merged corpus shard predictions.

Paired-by-day bootstrap (resample days with replacement) of AUROC(RAG) -
AUROC(base) over the merged 40-shard corpus series, for COVID / broad-2016-2020 /
full-2016-2023 windows. Prints the one-sided p-value that RAG > base, plus the
news-volume baseline AUROC. Makes the slide/report p-values reproducible.

Run: python kaggle/corpus_significance.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "kaggle/out_corpus")
WINDOWS = [("COVID", "2019-06-03", "2020-06-10"),
           ("Broad 2016-2020", "2016-01-01", "2020-12-31"),
           ("Full 2016-2023", "2016-01-01", "2023-12-31")]


def _merge(prefix: str) -> pd.DataFrame:
    fr = [pd.read_csv(f) for d in sorted(glob.glob(os.path.join(OUT, f"{prefix}*")))
          for f in [os.path.join(d, "crash", "trr_predictions.csv")] if os.path.exists(f)]
    if not fr:
        return pd.DataFrame()
    return pd.concat(fr).drop_duplicates("day").sort_values("day").set_index("day")


def _boot(base, rag, lo, hi, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    b = base[(base.index >= lo) & (base.index <= hi)]
    r = rag[(rag.index >= lo) & (rag.index <= hi)]
    idx = b.index.intersection(r.index)
    y = b.loc[idx, "label_true"].to_numpy()
    pb, pr = b.loc[idx, "crash_prob"].to_numpy(), r.loc[idx, "crash_prob"].to_numpy()
    nv = b.loc[idx, "n_news"].to_numpy()
    if y.sum() < 2 or y.sum() > len(y) - 2:
        return None
    n, diffs = len(y), []
    for _ in range(B):
        s = rng.integers(0, n, n)
        if y[s].sum() == 0 or y[s].sum() == len(s):
            continue
        diffs.append(roc_auc_score(y[s], pr[s]) - roc_auc_score(y[s], pb[s]))
    diffs = np.array(diffs)
    return {"n": n, "pos": int(y.sum()),
            "base": round(roc_auc_score(y, pb), 3), "rag": round(roc_auc_score(y, pr), 3),
            "news_vol": round(roc_auc_score(y, nv), 3),
            "delta": round(roc_auc_score(y, pr) - roc_auc_score(y, pb), 3),
            "p_rag_gt_base": round(float((diffs <= 0).mean()), 3)}


def main():
    base, rag = _merge("cb"), _merge("cr")
    if base.empty or rag.empty:
        print("no shard predictions in", OUT); return
    print(f"{'window':18} {'n':>5} {'pos':>4} {'base':>6} {'rag':>6} "
          f"{'Δ':>7} {'p':>6} {'news_vol':>9}")
    for nm, lo, hi in WINDOWS:
        s = _boot(base, rag, lo, hi)
        if s:
            print(f"{nm:18} {s['n']:>5} {s['pos']:>4} {s['base']:>6} {s['rag']:>6} "
                  f"{s['delta']:>+7} {s['p_rag_gt_base']:>6} {s['news_vol']:>9}")


if __name__ == "__main__":
    main()
