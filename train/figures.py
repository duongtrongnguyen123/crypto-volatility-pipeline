"""Generate figures for the report / webapp: reliability curve, backtest equity
curve, and the campaign AUROC comparison. Saves PNGs to reports/figures/.
Reproducible from the committed prediction CSVs + prices.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train.ablations import _oof
from train.features import TICKERS, build_dataset
from trr.prices import build_portfolio_daily

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)


def fig_reliability(d):
    y, p = d["label_true"].to_numpy(), d["oof"].to_numpy()
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#c0392b", label="TRR meta-learner (OOF)")
    ax.set_xlabel("predicted crash probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability curve (walk-forward OOF)"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/reliability.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_backtest(d):
    port = build_portfolio_daily("data/fnspid/prices", TICKERS)
    port.index = pd.to_datetime(port.index).date
    fwd = port["portfolio_ret"].shift(-1)
    d = d.join(fwd.rename("fwd_ret")).dropna(subset=["fwd_ret"])
    thr = d["oof"].quantile(0.85)
    invested = (d["oof"] < thr).astype(float)
    bh = (1 + d["fwd_ret"]).cumprod()
    strat = (1 + invested * d["fwd_ret"]).cumprod()
    x = pd.to_datetime(d.index)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, bh, label="buy & hold", color="#7f8c8d")
    ax.plot(x, strat, label="TRR de-risk overlay", color="#27ae60", lw=2)
    ax.set_title("Economic backtest — crash signal as de-risking overlay")
    ax.set_ylabel("growth of $1"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/backtest_equity.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_campaign():
    rows = [("Stock\nCOVID", 0.785, 0.71), ("Stock\n2016-20", 0.710, 0.747),
            ("Crypto\n2022-23", 0.530, 0.458), ("FNSPID\n2021-23", 0.550, 0.491)]
    labels = [r[0] for r in rows]; trr = [r[1] for r in rows]; nv = [r[2] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, trr, w, label="TRR (32B)", color="#2980b9")
    ax.bar(x + w/2, nv, w, label="news-volume baseline", color="#bdc3c7")
    ax.axhline(0.5, color="k", ls=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("crash AUROC")
    ax.set_ylim(0.4, 0.9); ax.set_title("Campaign: TRR vs news-volume baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(f"{OUT}/campaign_auroc.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def dump_fig_data(d):
    """Dump the data behind each figure as JSON so the web can render INTERACTIVE
    (Plotly) versions instead of static PNGs."""
    import json
    y, p = d["label_true"].to_numpy(), d["oof"].to_numpy()
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rel = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            rel.append({"pred": float(p[m].mean()), "obs": float(y[m].mean())})
    port = build_portfolio_daily("data/fnspid/prices", TICKERS)
    port.index = pd.to_datetime(port.index).date
    fwd = port["portfolio_ret"].shift(-1)
    dd = d.join(fwd.rename("fwd_ret")).dropna(subset=["fwd_ret"])
    thr = dd["oof"].quantile(0.85)
    inv = (dd["oof"] < thr).astype(float)
    bh = (1 + dd["fwd_ret"]).cumprod()
    strat = (1 + inv * dd["fwd_ret"]).cumprod()
    campaign = [("Stock COVID", 0.785, 0.71), ("Stock 2016-20", 0.710, 0.747),
                ("Crypto 2022-23", 0.530, 0.458), ("FNSPID bear", 0.550, 0.491)]
    data = {
        "reliability": rel,
        "backtest": {"dates": [str(x) for x in dd.index],
                     "buy_hold": [float(v) for v in bh],
                     "trr_derisk": [float(v) for v in strat]},
        "campaign": [{"window": w, "trr": t, "news_vol": n} for w, t, n in campaign],
    }
    with open("reports/stock_runs/fig_data.json", "w") as f:
        json.dump(data, f)


def main():
    d = _oof(build_dataset())
    fig_reliability(d)
    fig_backtest(d)
    fig_campaign()
    dump_fig_data(d)
    print(f"[figures] wrote {OUT}/*.png + reports/stock_runs/fig_data.json")


if __name__ == "__main__":
    main()
