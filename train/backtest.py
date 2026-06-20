"""Economic backtest — does the crash signal actually protect capital?

Uses leak-free walk-forward out-of-fold (OOF) crash probabilities from the
meta-learner to drive a simple de-risking rule: on days the model flags
elevated crash risk, move the portfolio to cash; otherwise stay invested.
Compares the resulting equity curve to buy-and-hold on total return, max
drawdown, and Sharpe — the practical payoff of crash detection.

No lookahead: the position for day t+1 uses only the OOF probability from day t
(itself trained on strictly earlier folds), and earns the realised next-day
portfolio return.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from train.features import FEATURES_FULL, TICKERS, build_dataset
from train.run import _gbm
from trr.prices import build_portfolio_daily


def _oof_probs(df, n_splits=6):
    d = df.sort_index()
    X, y = d[FEATURES_FULL].to_numpy(), d["label_true"].to_numpy()
    oof = np.full(len(d), np.nan)
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y[tr_i].sum() == 0:
            continue
        oof[te_i] = _gbm().fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
    d = d.copy(); d["oof"] = oof
    return d.dropna(subset=["oof"])


def _metrics(daily_ret):
    eq = (1 + daily_ret).cumprod()
    total = eq.iloc[-1] - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = daily_ret.mean() / (daily_ret.std() + 1e-12) * np.sqrt(252)
    return {"total_return": float(total), "max_drawdown": float(dd),
            "sharpe": float(sharpe), "ann_vol": float(daily_ret.std() * np.sqrt(252))}


def main():
    df = _oof_probs(build_dataset())
    port = build_portfolio_daily("data/fnspid/prices", TICKERS)
    port.index = pd.to_datetime(port.index).date
    fwd = port["portfolio_ret"].shift(-1)  # next-day return earned by today's position
    df = df.join(fwd.rename("fwd_ret")).dropna(subset=["fwd_ret"])

    bh = df["fwd_ret"]
    # threshold = OOF prob that flags the riskiest ~15% of days -> go to cash
    thr = df["oof"].quantile(0.85)
    invested = (df["oof"] < thr).astype(float)
    strat = invested * df["fwd_ret"]

    out = {"buy_and_hold": _metrics(bh), "trr_derisk": _metrics(strat),
           "threshold": float(thr), "days": int(len(df)),
           "days_in_cash": int((invested == 0).sum()),
           "period": f"{df.index.min()}..{df.index.max()}"}

    lines = ["# Economic backtest — crash signal as a de-risking overlay\n",
             f"Leak-free walk-forward OOF probs; {out['days']} days {out['period']}; "
             f"de-risk to cash on the riskiest 15% days ({out['days_in_cash']} days in cash).\n",
             "| Strategy | Total return | Max drawdown | Sharpe | Ann vol |",
             "|---|---|---|---|---|"]
    for k in ["buy_and_hold", "trr_derisk"]:
        m = out[k]
        lines.append(f"| {k} | {m['total_return']:+.1%} | {m['max_drawdown']:.1%} "
                     f"| {m['sharpe']:.2f} | {m['ann_vol']:.1%} |")
    dd_cut = out["trr_derisk"]["max_drawdown"] - out["buy_and_hold"]["max_drawdown"]
    lines.append(f"\nDrawdown improved by {dd_cut:+.1%} (positive = shallower/better); "
                 f"return {out['trr_derisk']['total_return']-out['buy_and_hold']['total_return']:+.1%} "
                 f"vs buy-and-hold.\n")
    report = "\n".join(lines)
    print(report)
    with open("reports/backtest_results.md", "w") as f:
        f.write(report + "\n")
    with open("reports/stock_runs/backtest_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[backtest] wrote reports/backtest_results.md")


if __name__ == "__main__":
    main()
