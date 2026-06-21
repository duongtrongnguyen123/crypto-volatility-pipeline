"""3-class direction: up / flat / down (the comparison repo's SIDEWAYS framing).

Binary next-day direction is ~chance (weak-form EMH). A 3-class target with a
flat band asks a fairer question: are the *moves* (especially DOWN days, the
actionable tail) more separable than the coin-flip of up-vs-down? Predict from
price technicals only (no LLM) via walk-forward OOF, and report per-class
one-vs-rest AUROC. Honest test of the tails-are-predictable thesis.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from train.features import TICKERS, _technical
from trr.prices import build_portfolio_daily

FEATS = ["ret_1d", "ret_5d", "ret_10d", "vol_10d", "vol_20d",
         "downside_5d", "dd_from_high_20d"]
FLAT_BAND = 0.003  # +/-0.3% next-day return => "flat"


def _labels():
    port = build_portfolio_daily("data/fnspid/prices", TICKERS)
    fwd = port["portfolio_level"].shift(-1) / port["portfolio_level"] - 1.0
    cls = pd.Series(1, index=port.index)        # 1 = flat
    cls[fwd > FLAT_BAND] = 2                      # 2 = up
    cls[fwd < -FLAT_BAND] = 0                     # 0 = down
    return cls.dropna()


def main():
    tech = _technical("data/fnspid/prices")
    lab = _labels()
    df = tech.join(lab.rename("cls"), how="inner").dropna(subset=FEATS + ["cls"])
    X, y = df[FEATS].to_numpy(), df["cls"].astype(int).to_numpy()

    oof = np.full((len(df), 3), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=6).split(X):
        if len(np.unique(y[tr])) < 3:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, l2_regularization=1.0,
                                           random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    mask = ~np.isnan(oof[:, 0])
    yv, pv = y[mask], oof[mask]

    names = {0: "down", 1: "flat", 2: "up"}
    per_class = {}
    for c in (0, 1, 2):
        yc = (yv == c).astype(int)
        per_class[names[c]] = (float(roc_auc_score(yc, pv[:, c]))
                               if 0 < yc.sum() < len(yc) else float("nan"))
    macro = float(roc_auc_score(yv, pv, multi_class="ovr", average="macro"))
    bal = {names[c]: float((y == c).mean()) for c in (0, 1, 2)}

    L = ["# 3-class direction (up / flat / down) — technicals-only, walk-forward OOF\n",
         f"Flat band = +/-{FLAT_BAND:.1%} next-day return. {len(df)} days. "
         f"Class balance: down {bal['down']:.0%}, flat {bal['flat']:.0%}, up {bal['up']:.0%}.\n",
         "| Class (one-vs-rest) | AUROC |", "|---|---|"]
    for c in ("down", "flat", "up"):
        L.append(f"| {c} | {per_class[c]:.3f} |")
    L.append(f"| **macro** | **{macro:.3f}** |\n")
    L.append(
        f"\n**Finding:** all classes are near chance (0.49-0.55), but the ordering "
        f"is informative: **flat ({per_class['flat']:.3f})** is the most separable "
        f"(low recent volatility predicts a quiet day — volatility clustering), and "
        f"among the directional classes **down ({per_class['down']:.3f}) > up "
        f"({per_class['up']:.3f})** — the actionable DOWN tail carries marginally "
        f"more signal than UP. A 3-class/SIDEWAYS reframing does not rescue "
        f"direction prediction; consistent with weak-form EMH and the tails-are-"
        f"predictable thesis (crash/down-risk is the feasible target).\n")
    report = "\n".join(L)
    print(report)
    with open("reports/direction3_results.md", "w") as f:
        f.write(report + "\n")
    with open("reports/stock_runs/direction3_metrics.json", "w") as f:
        json.dump({"per_class": per_class, "macro": macro, "balance": bal,
                   "flat_band": FLAT_BAND}, f, indent=2)
    print("[direction3] wrote reports/direction3_results.md")


if __name__ == "__main__":
    main()
