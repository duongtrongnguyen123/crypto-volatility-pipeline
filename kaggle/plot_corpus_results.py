"""Up-to-date results figure: AUROC grouped bars from the merged corpus shards.

Reads kaggle/out_corpus/* (40 shards), computes per-window AUROC for base / RAG /
news-volume, and renders reports/figures/fig_corpus_auroc.png — matching the
deck (slides 14/15). p-values from kaggle/corpus_significance.py.

Run: python kaggle/plot_corpus_results.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle/out_corpus")
FIG = os.path.join(ROOT, "reports/figures/fig_corpus_auroc.png")
WINDOWS = [("COVID\n(2019-06…2020-06)", "2019-06-03", "2020-06-10", "0.082"),
           ("Rộng\n2016–2020", "2016-01-01", "2020-12-31", "0.66"),
           ("Toàn bộ\n2016–2023", "2016-01-01", "2023-12-31", "0.047")]


def _merge(pre):
    fr = [pd.read_csv(f) for d in sorted(glob.glob(os.path.join(OUT, f"{pre}*")))
          for f in [os.path.join(d, "crash", "trr_predictions.csv")] if os.path.exists(f)]
    return pd.concat(fr).drop_duplicates("day").sort_values("day").set_index("day")


def _au(df, lo, hi, col):
    s = df[(df.index >= lo) & (df.index <= hi)]
    y = s["label_true"].to_numpy()
    if y.sum() < 2 or y.sum() > len(y) - 2:
        return None
    return roc_auc_score(y, s[col])


def main():
    base, rag = _merge("cb"), _merge("cr")
    names, bb, rr, nn, ps = [], [], [], [], []
    for nm, lo, hi, p in WINDOWS:
        names.append(nm); ps.append(p)
        bb.append(_au(base, lo, hi, "crash_prob"))
        rr.append(_au(rag, lo, hi, "crash_prob"))
        nn.append(_au(base, lo, hi, "n_news"))

    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w, bb, w, label="Base (TRR)", color="#94a3b8")
    b2 = ax.bar(x, rr, w, label="RAG", color="#16a34a")
    b3 = ax.bar(x + w, nn, w, label="News-volume (baseline)", color="#60a5fa")
    ax.axhline(0.5, ls="--", lw=1, color="#dc2626")
    ax.text(len(names) - 0.5, 0.505, "Ngẫu nhiên (0.50)", color="#dc2626",
            fontsize=8, ha="right", va="bottom")
    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            if h:
                ax.text(r.get_x() + r.get_width() / 2, h + 0.006, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)
    for i, p in enumerate(ps):
        mark = "✓" if float(p) < 0.05 else ""
        ax.text(x[i], (rr[i] or 0) + 0.045, f"p={p}{mark}", ha="center",
                fontsize=8, color=("#16a34a" if float(p) < 0.05 else "#64748b"))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.40, 0.90)
    ax.set_title("AUROC dự đoán crash — corpus FNSPID 2016–2023 (lọc theo danh mục)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.text(0, 0.41, "Tham chiếu best-case (analyst-news): COVID 0.785 / +RAG 0.847 · rộng 0.710",
            fontsize=7.5, color="#475569")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"wrote {FIG}")
    print("base:", [round(v, 3) if v else None for v in bb])
    print("rag :", [round(v, 3) if v else None for v in rr])
    print("nvol:", [round(v, 3) if v else None for v in nn])


if __name__ == "__main__":
    main()
