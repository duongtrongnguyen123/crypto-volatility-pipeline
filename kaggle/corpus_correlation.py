"""Honest continuous eval: does crash_prob track the DEPTH of the drawdown?

Binary AUROC uses only the rare crash days (~78). This instead correlates the
predicted crash_prob with the ACTUAL forward 3-day portfolio drawdown over EVERY
day (~2000) — a far larger effective sample. A model that's calibrated to
severity should show: higher crash_prob -> more negative forward return
(negative Spearman), and a monotone decile table.

Run: python kaggle/corpus_correlation.py
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle/out_corpus")
OHLCV = os.path.join(ROOT, "data/ohlcv")
TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
HORIZON = 3


def forward_drawdown() -> pd.Series:
    """Forward HORIZON-day worst drawdown of the equal-weight portfolio, per day."""
    closes = {}
    for t in TICKERS:
        d = pd.read_csv(os.path.join(OHLCV, f"{t}.csv"))
        d.columns = [c.lower() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        closes[t] = d.set_index("date")["close"]
    px = pd.DataFrame(closes).sort_index().ffill().dropna()
    rets = px.pct_change().fillna(0.0)
    lvl = (1.0 + rets.mean(axis=1)).cumprod()
    fwd_low = lvl.iloc[::-1].rolling(HORIZON, min_periods=1).min().iloc[::-1].shift(-1)
    return (fwd_low / lvl - 1.0).dropna()           # <= 0 means a drawdown ahead


def merge(pre) -> pd.DataFrame:
    fr = [pd.read_csv(f) for d in sorted(glob.glob(os.path.join(OUT, f"{pre}*")))
          for f in [os.path.join(d, "crash", "trr_predictions.csv")] if os.path.exists(f)]
    return pd.concat(fr).drop_duplicates("day").sort_values("day").set_index("day")


def report(name, df, fwd):
    j = df.join(fwd.rename("fwd_ret"), how="inner").dropna(subset=["crash_prob", "fwd_ret"])
    p, prob = j["crash_prob"].to_numpy(), j["fwd_ret"].to_numpy()
    rho, rp = spearmanr(p, prob)
    r, _ = pearsonr(p, prob)
    print(f"\n=== {name} (n={len(j)} days) ===")
    print(f"Spearman(crash_prob, fwd_ret) = {rho:+.3f}  (p={rp:.1e})   "
          f"[negative = higher prob -> deeper drop = GOOD]")
    print(f"Pearson  (crash_prob, fwd_ret) = {r:+.3f}")
    # decile table: mean forward return per crash_prob decile
    j = j.copy()
    j["decile"] = pd.qcut(j["crash_prob"].rank(method="first"), 10, labels=False) + 1
    tab = j.groupby("decile").agg(mean_crash_prob=("crash_prob", "mean"),
                                  mean_fwd_ret=("fwd_ret", "mean"),
                                  worst_fwd_ret=("fwd_ret", "min"),
                                  n=("fwd_ret", "size"))
    print("decile (1=lowest prob … 10=highest):")
    for d, row in tab.iterrows():
        print(f"  D{int(d):2d}  prob~{row.mean_crash_prob:.2f}  "
              f"mean fwd_ret {row.mean_fwd_ret:+.3%}  worst {row.worst_fwd_ret:+.2%}")
    return rho


def _decile_means(df, fwd):
    j = df.join(fwd.rename("fwd_ret"), how="inner").dropna(subset=["crash_prob", "fwd_ret"]).copy()
    j["decile"] = pd.qcut(j["crash_prob"].rank(method="first"), 10, labels=False) + 1
    return j.groupby("decile")["fwd_ret"].mean() * 100   # %


def save_figure(base, rag, fwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_path = os.path.join(ROOT, "reports/figures/fig_corpus_severity.png")
    bm, rm = _decile_means(base, fwd), _decile_means(rag, fwd)
    x = np.arange(1, 11); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, bm.reindex(x).values, w, label="Base", color="#94a3b8")
    ax.bar(x + w / 2, rm.reindex(x).values, w, label="RAG", color="#16a34a")
    ax.axhline(0, color="#475569", lw=0.8)
    ax.set_xlabel("crash_prob decile  (1 = lowest predicted risk → 10 = highest)")
    ax.set_ylabel("mean forward 3-day return (%)")
    ax.set_title("Severity check: higher predicted crash_prob → deeper actual drawdown\n"
                 "(corpus 2016–2023, all 1,860 days · Spearman base −0.087, RAG −0.104, p<1e-4)",
                 fontsize=10, weight="bold")
    ax.set_xticks(x); ax.legend()
    fig.tight_layout(); os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {fig_path}")


def main():
    fwd = forward_drawdown()
    base, rag = merge("cb"), merge("cr")
    report("BASE", base, fwd)
    report("RAG", rag, fwd)
    save_figure(base, rag, fwd)
    print("\n(Continuous severity eval over all days — complements the binary AUROC.)")


if __name__ == "__main__":
    main()
