"""Rigorous evaluation of the TRR crash predictions.

Goes beyond a single AUROC to address the questions that decide whether the
signal is *real* and *useful*:

  1. Statistical significance — bootstrap CIs on AUROC, and paired-bootstrap
     tests for differences vs baselines (is 0.57 distinguishable from 0.50?).
  2. Leak-free ensemble — the headline 0.653 fit the blend weight on the same
     data it scored; here we calibrate the weight on the first half and report
     the held-out second half.
  3. Calibration — Brier score + reliability, are the probabilities meaningful?
  4. Early-warning quality — precision@K on the highest-risk days.
  5. Economic value — a de-risking backtest: equity curve / Sharpe / max
     drawdown vs equal-weight buy-and-hold (no lookahead).

Inputs are the saved Kaggle predictions (kaggle/out_v4/...) + price labels +
the Fear & Greed index. Run:  python -m trr.analysis
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from trr.labels import build_portfolio, crash_labels

RNG = np.random.default_rng(0)
OUT_DIR = "reports"


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def load_aligned(pred_csv: str, horizon: int = 3) -> pd.DataFrame:
    """Align model predictions with crash labels, F&G, and forward returns."""
    pred = pd.read_csv(pred_csv, index_col=0)
    pred.index = pd.to_datetime(pred.index).date

    lab = crash_labels(horizon=horizon)
    lab.index = pd.to_datetime(lab.index).date

    port = build_portfolio()
    port.index = pd.to_datetime(port.index).date

    # Fear & Greed (fetched to data/fng.json).
    fng = None
    if os.path.exists("data/fng.json"):
        raw = json.load(open("data/fng.json"))["data"]
        fng = pd.Series({
            datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date(): int(x["value"])
            for x in raw
        })

    idx = pred.index
    df = pd.DataFrame(index=idx)
    df["crash_prob"] = pred["crash_prob"].values
    df["crash"] = [int(lab["crash"].get(d, 0)) for d in idx]
    df["port_ret"] = [float(port["portfolio_ret"].get(d, 0.0)) for d in idx]
    df["fng_fear"] = (100 - fng.reindex(idx).ffill()).values if fng is not None else np.nan
    df["price_mom"] = (-port["portfolio_level"].pct_change(5)).reindex(idx).fillna(0.0).values
    return df


# --------------------------------------------------------------------------- #
# 1. Statistical significance
# --------------------------------------------------------------------------- #
def bootstrap_auroc(y, s, n: int = 2000):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(y, s)
    stats = []
    for _ in range(n):
        bp = RNG.choice(pos, len(pos), replace=True)
        bn = RNG.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        stats.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return base, lo, hi


def paired_bootstrap_diff(y, s_a, s_b, n: int = 2000):
    """P(AUROC_a <= AUROC_b) under paired resampling — one-sided test that a>b."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(pos, len(pos), True), RNG.choice(neg, len(neg), True)])
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float((diffs <= 0).mean())  # mean diff, p-value


# --------------------------------------------------------------------------- #
# 2. Leak-free ensemble (calibrate weight on first half, test on second)
# --------------------------------------------------------------------------- #
def leakfree_ensemble(df: pd.DataFrame):
    n = len(df)
    cut = n // 2
    cal, test = df.iloc[:cut], df.iloc[cut:]

    def norm(s, ref):
        return (s - ref.min()) / (ref.max() - ref.min() + 1e-9)

    best_a, best_au = 1.0, -1
    for a in np.linspace(0, 1, 21):
        s = a * norm(cal["crash_prob"], cal["crash_prob"]) + (1 - a) * norm(cal["fng_fear"], cal["fng_fear"])
        if cal["crash"].sum() and cal["crash"].sum() < len(cal):
            au = roc_auc_score(cal["crash"], s)
            if au > best_au:
                best_au, best_a = au, a
    s_test = (best_a * norm(test["crash_prob"], cal["crash_prob"])
              + (1 - best_a) * norm(test["fng_fear"], cal["fng_fear"]))
    test_au = roc_auc_score(test["crash"], s_test) if test["crash"].nunique() > 1 else float("nan")
    naive_au = roc_auc_score(test["crash"], test["crash_prob"]) if test["crash"].nunique() > 1 else float("nan")
    return {"alpha_trr": best_a, "cal_auroc": best_au,
            "test_auroc_ensemble": test_au, "test_auroc_trr_only": naive_au,
            "cal_days": cut, "test_days": n - cut}


# --------------------------------------------------------------------------- #
# 3. Calibration, 4. precision@K
# --------------------------------------------------------------------------- #
def calibration(df: pd.DataFrame):
    p = df["crash_prob"].clip(0, 1)
    brier = brier_score_loss(df["crash"], p)
    base = brier_score_loss(df["crash"], np.full(len(df), df["crash"].mean()))
    return {"brier": float(brier), "brier_baserate": float(base),
            "skill_score": float(1 - brier / base)}


def precision_at_k(df: pd.DataFrame, ks=(10, 20, 50)):
    order = df.sort_values("crash_prob", ascending=False)
    base = df["crash"].mean()
    return {f"P@{k}": float(order["crash"].head(k).mean()) for k in ks} | {"base_rate": float(base)}


# --------------------------------------------------------------------------- #
# 5. Economic backtest (no lookahead)
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, threshold: float = None):
    """De-risk when crash_prob is high. Decision at day t uses crash_prob[t];
    return is realized on t+1 (port_ret shifted) -> no lookahead.
    """
    p = df["crash_prob"].values
    thr = threshold if threshold is not None else np.quantile(p, 0.80)
    exposure = np.where(p >= thr, 0.0, 1.0)          # flat when high risk
    fwd_ret = np.append(df["port_ret"].values[1:], 0.0)  # r[t+1]
    strat = exposure * fwd_ret
    bh = fwd_ret

    def stats(r):
        eq = np.cumprod(1 + r)
        total = eq[-1] - 1
        sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(365)
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / peak).min()
        return {"total_return": float(total), "sharpe": float(sharpe),
                "max_drawdown": float(mdd), "equity": eq}

    s, b = stats(strat), stats(bh)
    return {"threshold_pct": 80, "time_in_market": float(exposure.mean()),
            "strategy": {k: v for k, v in s.items() if k != "equity"},
            "buy_hold": {k: v for k, v in b.items() if k != "equity"},
            "_eq_strat": s["equity"], "_eq_bh": b["equity"],
            "_dates": list(df.index)}


# --------------------------------------------------------------------------- #
def incremental_value(df: pd.DataFrame):
    """Does the LLM signal catch crashes that PRICE momentum misses? Model-free
    (no learned combo -> immune to the non-stationarity that broke stacking).

    Stratify days by price-momentum (below/above median). Within each stratum,
    measure the TRR crash-AUROC. If TRR > 0.5 on the 'price says calm' days, the
    news reasoning adds value beyond price autocorrelation.
    """
    med = df["price_mom"].median()
    out = {}
    for name, sub in [("price-CALM (mom<median)", df[df["price_mom"] < med]),
                      ("price-ALARMED (mom>=median)", df[df["price_mom"] >= med])]:
        y = sub["crash"].to_numpy()
        if 0 < y.sum() < len(y):
            b, lo, hi = bootstrap_auroc(y, sub["crash_prob"].to_numpy(), n=2000)
            out[name] = {"days": int(len(sub)), "crashes": int(y.sum()),
                         "trr_auroc": b, "ci": [lo, hi]}
        else:
            out[name] = {"days": int(len(sub)), "crashes": int(y.sum()), "trr_auroc": None}
    return out


def cost_aware_backtest(df: pd.DataFrame, cost_bps: float = 10.0, lookback: int = 60):
    """Continuous-sizing, transaction-cost-aware backtest (fully causal).

    Exposure e[t] = 1 - (causal trailing-percentile rank of crash_prob[t] over
    the last `lookback` days) — scale out of the market as today's crash risk is
    high relative to its recent history. Return realized t+1; turnover charged at
    `cost_bps`. No lookahead, no fitted calibration.
    """
    p = df["crash_prob"].to_numpy()
    fwd = np.append(df["port_ret"].to_numpy()[1:], 0.0)  # r[t+1]
    n = len(p)
    expo = np.ones(n)
    for t in range(n):
        lo = max(0, t - lookback)
        window = p[lo : t + 1]
        rank = (window < p[t]).mean()           # causal percentile of today
        expo[t] = 1.0 - rank
    cost = cost_bps / 1e4
    turnover = np.abs(np.diff(np.concatenate([[1.0], expo])))
    strat = expo * fwd - cost * turnover
    bh = fwd

    def stats(r):
        eq = np.cumprod(1 + r)
        sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(365)
        mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
        return {"total_return": float(eq[-1] - 1), "sharpe": float(sharpe),
                "max_drawdown": float(mdd)}

    return {"cost_bps": cost_bps, "avg_exposure": float(expo.mean()),
            "turnover_per_day": float(turnover.mean()),
            "strategy_net": stats(strat), "buy_hold": stats(bh)}


def _plots(df, bt, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    x = pd.to_datetime(pd.Index(bt["_dates"]))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, bt["_eq_bh"], label="buy & hold", lw=1.4)
    ax.plot(x, bt["_eq_strat"], label="TRR de-risk strategy", lw=1.4)
    ax.set_title("Economic backtest — equity curve (TRR crash-signal de-risking)")
    ax.set_ylabel("equity (×)"); ax.legend()
    fig.savefig(f"{out_dir}/backtest_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(pred_csv: str = "kaggle/out_v4/trr_predictions.csv", label: str = "v4 (32B, 2022-23)"):
    df = load_aligned(pred_csv)
    print(f"=== {label} | {len(df)} days, {int(df['crash'].sum())} crashes "
          f"({df['crash'].mean():.1%}) ===")

    print("\n[1] AUROC with 95% bootstrap CI (2000 resamples):")
    signals = {"TRR (news reasoning)": df["crash_prob"], "Fear&Greed": df["fng_fear"],
               "price-momentum": df["price_mom"]}
    aurocs = {}
    for name, s in signals.items():
        b, lo, hi = bootstrap_auroc(df["crash"], s)
        aurocs[name] = (b, lo, hi)
        print(f"    {name:24s} {b:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n[2] Significance vs chance (AUROC=0.5) and vs price baseline:")
    mdiff, p_vs_price = paired_bootstrap_diff(df["crash"], df["crash_prob"], df["price_mom"])
    _, p_vs_half = paired_bootstrap_diff(df["crash"], df["crash_prob"],
                                         pd.Series(RNG.permutation(df["crash_prob"].values), index=df.index))
    print(f"    TRR vs price: mean diff {mdiff:+.3f}, p={p_vs_price:.3f}")
    print(f"    TRR > 0.5 lower CI? {'YES' if aurocs['TRR (news reasoning)'][1] > 0.5 else 'NO'}")

    print("\n[3] Leak-free ensemble (calibrate alpha on 1st half, test on 2nd):")
    lf = leakfree_ensemble(df)
    print(f"    alpha_TRR={lf['alpha_trr']:.2f} | held-out: TRR-only={lf['test_auroc_trr_only']:.3f} "
          f"-> ensemble={lf['test_auroc_ensemble']:.3f}")

    print("\n[4] Calibration:")
    cal = calibration(df)
    print(f"    Brier={cal['brier']:.4f} (base-rate {cal['brier_baserate']:.4f}, "
          f"skill {cal['skill_score']:+.3f})")

    print("\n[5] Early-warning precision@K:")
    pk = precision_at_k(df)
    print(f"    {pk}")

    print("\n[6] Economic backtest (de-risk top-20% risk days, no lookahead):")
    bt = backtest(df)
    s, b = bt["strategy"], bt["buy_hold"]
    print(f"    {'':14s} {'return':>9s} {'Sharpe':>7s} {'maxDD':>8s}")
    print(f"    buy & hold    {b['total_return']:>8.1%} {b['sharpe']:>7.2f} {b['max_drawdown']:>8.1%}")
    print(f"    TRR de-risk   {s['total_return']:>8.1%} {s['sharpe']:>7.2f} {s['max_drawdown']:>8.1%}"
          f"   (in-market {bt['time_in_market']:.0%})")

    _plots(df, bt, OUT_DIR)
    result = {"label": label, "n_days": len(df), "n_crash": int(df["crash"].sum()),
              "auroc_ci": {k: list(v) for k, v in aurocs.items()},
              "trr_vs_price_p": p_vs_price, "leakfree_ensemble": lf,
              "calibration": cal, "precision_at_k": pk,
              "backtest": {"strategy": s, "buy_hold": b, "time_in_market": bt["time_in_market"]}}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(result, open(f"{OUT_DIR}/analysis_{label.split()[0]}.json", "w"), indent=2)
    return result


def run_per_asset(pred_csv: str, threshold: float = 0.12):
    """Per-asset AUROC: each asset's crash_prob_{T} vs that asset's crash label."""
    from trr.labels import asset_crash_labels
    from trr.schema import PORTFOLIO

    pred = pd.read_csv(pred_csv, index_col=0)
    pred.index = pd.to_datetime(pred.index).date
    lab = asset_crash_labels(threshold=threshold)
    lab.index = pd.to_datetime(lab.index).date

    print(f"=== Per-asset crash prediction ({len(pred)} days, threshold {threshold:.0%}/3d) ===")
    print(f"    {'asset':6s} {'crashes':>8s} {'base':>6s} {'AUROC':>7s}  95% CI")
    rows, aurocs = [], []
    for a in PORTFOLIO:
        col = f"crash_prob_{a}"
        if col not in pred.columns:
            continue
        y = np.array([int(lab.get(f"{a}_crash", pd.Series()).get(d, 0)) for d in pred.index])
        if y.sum() < 3 or y.sum() == len(y):
            print(f"    {a:6s} {int(y.sum()):>8d}  (too few positives — skipped)")
            continue
        b, lo, hi = bootstrap_auroc(y, pred[col].values)
        aurocs.append(b)
        print(f"    {a:6s} {int(y.sum()):>8d} {y.mean():>6.1%} {b:>7.3f}  [{lo:.3f}, {hi:.3f}]")
        rows.append({"asset": a, "crashes": int(y.sum()), "base_rate": float(y.mean()),
                     "auroc": b, "ci": [lo, hi]})
    if aurocs:
        print(f"    {'MACRO':6s} {'':>8s} {'':>6s} {np.mean(aurocs):>7.3f}  (avg across assets)")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"threshold": threshold, "per_asset": rows,
               "macro_auroc": float(np.mean(aurocs)) if aurocs else None},
              open(f"{OUT_DIR}/analysis_per_asset.json", "w"), indent=2)
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "per-asset":
        run_per_asset(sys.argv[2] if len(sys.argv) > 2 else "kaggle/out_perasset/trr_predictions.csv")
        raise SystemExit
    run("kaggle/out_v4/trr_predictions.csv", "v4 (32B, 2022-23)")
    if os.path.exists("kaggle/out_2024_32b/trr_predictions.csv"):
        print("\n" + "=" * 60)
        run("kaggle/out_2024_32b/trr_predictions.csv", "2024-32B (out-of-regime)")
