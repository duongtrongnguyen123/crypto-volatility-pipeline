"""Does the crash signal generalize to other price targets?

The headline TRR result evaluates one signal (`crash_prob` from the LLM news
reasoning) against one target (the 3-day portfolio crash). This module reuses
the *same already-computed signals* — the saved `crash_prob` (kaggle/out_v4),
the Fear & Greed fear level, and price momentum — and scores them against the
NEW price targets from `trr.targets`:

    direction (up/down, h=1 and h=3) : binary  -> AUROC + bootstrap CI
    forward return  (h=1, h=3)       : regr.   -> Spearman rho + sign-accuracy
    severity (fwd drawdown, h=3)     : regr.   -> Spearman rho + sign-accuracy
    volatility regime (3 classes)    : multi   -> macro one-vs-rest AUROC + acc

The point is honesty about generalization: `crash_prob` was trained to fire on
DOWN moves, so for "price up" we score it as a DOWN predictor (AUROC < 0.5 means
"good at the opposite direction"); we report the direction explicitly and also
the flipped `1 - crash_prob` as an UP predictor.

Run:  python -m trr.eval_targets
Writes: reports/analysis_targets.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config  # noqa: F401  (ensures HISTORICAL_DIR env default is set)
from trr.analysis import bootstrap_auroc
from trr.labels import build_portfolio
from trr.targets import (
    direction_labels,
    return_target,
    severity_target,
    vol_regime_labels,
)

OUT_DIR = "reports"
PRED_CSV = "kaggle/out_v4/trr_predictions.csv"


# --------------------------------------------------------------------------- #
# Data assembly — the three reusable signals on the prediction window.
# --------------------------------------------------------------------------- #
def load_signals(pred_csv: str = PRED_CSV) -> pd.DataFrame:
    """Assemble the saved signals (crash_prob, F&G fear, price momentum) on the
    prediction window, indexed by `date`. Identical signal construction to
    `trr.analysis.load_aligned` so the comparison is apples-to-apples.
    """
    pred = pd.read_csv(pred_csv, index_col=0)
    pred.index = pd.to_datetime(pred.index).date

    port = build_portfolio()
    port.index = pd.to_datetime(port.index).date

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
    df["fng_fear"] = (100 - fng.reindex(idx).ffill()).values if fng is not None else np.nan
    # Negative 5-day price momentum: high when price has been FALLING (a
    # "risk-on/off" proxy aligned with crash_prob's direction).
    df["price_mom_down"] = (-port["portfolio_level"].pct_change(5)).reindex(idx).fillna(0.0).values
    return df


def _align(signals: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join signals with a target on the date index."""
    tgt = target.copy()
    tgt.index = pd.to_datetime(tgt.index).date
    common = [d for d in signals.index if d in set(tgt.index)]
    return signals.loc[common], tgt.reindex(common)


# --------------------------------------------------------------------------- #
# Metrics per target family.
# --------------------------------------------------------------------------- #
def _binary_auroc(y: np.ndarray, s: np.ndarray) -> dict:
    """AUROC + 95% bootstrap CI; NaN-safe (returns None on degenerate y)."""
    if len(np.unique(y)) < 2:
        return {"auroc": None, "ci": [None, None]}
    b, lo, hi = bootstrap_auroc(y, s, n=2000)
    return {"auroc": float(b), "ci": [float(lo), float(hi)]}


def eval_direction(signals: pd.DataFrame, horizon: int) -> dict:
    """Direction (price UP) target. crash_prob predicts DOWN, so we report it
    both as a DOWN predictor (AUROC vs `down=1-up`) and flipped as an UP
    predictor (`1 - crash_prob` vs `up`). Honest framing of the sign.
    """
    sig, up = _align(signals, direction_labels(horizon)["up"])
    up = up.to_numpy().astype(int)
    down = 1 - up
    out = {"horizon": horizon, "n": int(len(up)), "up_rate": float(up.mean()),
           "signals": {}}
    # All three signals are oriented toward DOWN moves -> score against `down`.
    for name in ("crash_prob", "fng_fear", "price_mom_down"):
        s = sig[name].to_numpy()
        as_down = _binary_auroc(down, s)
        as_up = _binary_auroc(up, -s)  # flip: high signal -> low up_prob
        out["signals"][name] = {"auroc_predict_DOWN": as_down["auroc"],
                                "ci_DOWN": as_down["ci"],
                                "auroc_predict_UP_flipped": as_up["auroc"]}
    return out


def eval_regression(signals: pd.DataFrame, target: pd.Series, name: str,
                    sign_signal: str = "crash_prob", expect_sign: int = -1) -> dict:
    """Spearman rho + sign-accuracy for a continuous target.

    `expect_sign` is the sign the *down-oriented* signals should correlate with
    the target: -1 for forward return (down signal high -> return low), +1 for
    severity/drawdown magnitude (down signal high -> deeper drawdown).
    Sign-accuracy: how often the chosen signal's predicted move direction
    (derived from `expect_sign`) matches the realized sign of the target.
    """
    sig, tgt = _align(signals, target)
    y = tgt.to_numpy(dtype=float)
    out = {"name": name, "n": int(len(y)), "expect_sign": expect_sign, "signals": {}}
    for col in ("crash_prob", "fng_fear", "price_mom_down"):
        s = sig[col].to_numpy(dtype=float)
        rho, p = spearmanr(s, y)
        out["signals"][col] = {"spearman": float(rho), "p_value": float(p)}
    # Sign-accuracy for the primary signal: predicted target sign = expect_sign
    # when the signal is above its median (elevated), else the opposite.
    s = sig[sign_signal].to_numpy(dtype=float)
    elevated = s >= np.median(s)
    pred_sign = np.where(elevated, expect_sign, -expect_sign)
    realized_sign = np.sign(y)
    mask = realized_sign != 0
    sign_acc = float((pred_sign[mask] == realized_sign[mask]).mean()) if mask.any() else None
    out["sign_accuracy"] = {"signal": sign_signal, "accuracy": sign_acc}
    return out


def eval_vol_regime(signals: pd.DataFrame, horizon: int = 1) -> dict:
    """Volatility regime (3-class) target. Macro one-vs-rest AUROC of the
    down-oriented signals (high fear/crash risk -> higher volatility regime) and
    the accuracy of a simple tercile argmax baseline from `crash_prob`.
    """
    sig, reg = _align(signals, vol_regime_labels(horizon)["regime"])
    y = reg.to_numpy().astype(int)
    out = {"horizon": horizon, "n": int(len(y)),
           "class_counts": {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
           "signals": {}}
    classes = sorted(np.unique(y))
    for col in ("crash_prob", "fng_fear", "price_mom_down"):
        s = sig[col].to_numpy(dtype=float)
        aucs = []
        for c in classes:
            yc = (y == c).astype(int)
            if len(np.unique(yc)) == 2:
                aucs.append(roc_auc_score(yc, s))
        out["signals"][col] = {"macro_ovr_auroc": float(np.mean(aucs)) if aucs else None}
    # Argmax baseline: bucket crash_prob into terciles, predict that regime.
    s = sig["crash_prob"].to_numpy(dtype=float)
    lo, hi = np.quantile(s, [1 / 3, 2 / 3])
    pred = np.where(s <= lo, 0, np.where(s <= hi, 1, 2))
    out["crash_prob_tercile_accuracy"] = float((pred == y).mean())
    out["majority_baseline_accuracy"] = float(max(np.bincount(y)) / len(y))
    return out


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def run(pred_csv: str = PRED_CSV) -> dict:
    signals = load_signals(pred_csv)
    n_days = len(signals)
    print(f"=== Generalization of the crash signal to NEW price targets "
          f"({n_days} days, {min(signals.index)} -> {max(signals.index)}) ===")
    print("Signals scored: crash_prob (LLM news reasoning), fng_fear "
          "(Fear&Greed), price_mom_down (-5d momentum). All oriented toward "
          "DOWN moves.\n")

    result: dict = {"pred_csv": pred_csv, "n_days": n_days}

    # 1. Direction (binary). -------------------------------------------------
    print("[1] DIRECTION (price UP/DOWN) — AUROC of each signal as a DOWN "
          "predictor (>0.5 = predicts down moves):")
    print(f"    {'horizon':>7s}  {'signal':16s} {'AUROC(DOWN)':>12s} "
          f"{'95% CI':>18s}  {'AUROC(UP,flip)':>15s}")
    result["direction"] = {}
    for h in (1, 3):
        dr = eval_direction(signals, h)
        result["direction"][f"h{h}"] = dr
        for name, m in dr["signals"].items():
            ci = m["ci_DOWN"]
            ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else "      n/a"
            print(f"    {h:>7d}  {name:16s} {_fmt(m['auroc_predict_DOWN']):>12s} "
                  f"{ci_s:>18s}  {_fmt(m['auroc_predict_UP_flipped']):>15s}")
    print("    (up-rate ~50%; AUROC>0.5 = the signal genuinely ranks DOWN days "
          "higher.)\n")

    # 2. Forward return (regression). ---------------------------------------
    print("[2] FORWARD RETURN (regression) — Spearman rho vs each signal "
          "(expect NEGATIVE: down-signal high -> return low):")
    print(f"    {'horizon':>7s}  {'signal':16s} {'spearman':>9s} {'p':>8s}   "
          f"sign-acc(crash_prob)")
    result["return"] = {}
    for h in (1, 3):
        rr = eval_regression(signals, return_target(h), f"return_h{h}", expect_sign=-1)
        result["return"][f"h{h}"] = rr
        for col, m in rr["signals"].items():
            extra = f"   {_fmt(rr['sign_accuracy']['accuracy'])}" if col == "crash_prob" else ""
            print(f"    {h:>7d}  {col:16s} {_fmt(m['spearman']):>9s} "
                  f"{_fmt(m['p_value']):>8s}{extra}")
    print()

    # 3. Severity (forward drawdown, regression). ---------------------------
    print("[3] SEVERITY (forward 3d max-drawdown magnitude) — Spearman rho "
          "(expect POSITIVE: down-signal high -> deeper drawdown):")
    sv = eval_regression(signals, severity_target(3), "severity_h3", expect_sign=+1)
    result["severity"] = sv
    print(f"    {'signal':16s} {'spearman':>9s} {'p':>8s}")
    for col, m in sv["signals"].items():
        print(f"    {col:16s} {_fmt(m['spearman']):>9s} {_fmt(m['p_value']):>8s}")
    print(f"    sign-accuracy (crash_prob): {_fmt(sv['sign_accuracy']['accuracy'])}\n")

    # 4. Volatility regime (multi-class). -----------------------------------
    print("[4] VOLATILITY REGIME (3-class low/med/high) — macro one-vs-rest "
          "AUROC (>0.5 = ranks high-vol days higher):")
    vr = eval_vol_regime(signals, horizon=1)
    result["vol_regime"] = vr
    print(f"    {'signal':16s} {'macro-OvR AUROC':>16s}")
    for col, m in vr["signals"].items():
        print(f"    {col:16s} {_fmt(m['macro_ovr_auroc']):>16s}")
    print(f"    crash_prob tercile-argmax accuracy: "
          f"{_fmt(vr['crash_prob_tercile_accuracy'])} "
          f"(majority baseline {_fmt(vr['majority_baseline_accuracy'])})\n")

    # Verdict. ---------------------------------------------------------------
    verdict = _verdict(result)
    result["verdict"] = verdict
    print("VERDICT (does the crash signal generalize?):")
    for line in verdict:
        print(f"  - {line}")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(result, open(f"{OUT_DIR}/analysis_targets.json", "w"), indent=2)
    print(f"\nWrote {OUT_DIR}/analysis_targets.json")
    return result


def _verdict(result: dict) -> list[str]:
    """Plain-language read of where crash_prob generalizes and where it doesn't."""
    out = []
    d1 = result["direction"]["h1"]["signals"]["crash_prob"]["auroc_predict_DOWN"]
    d3 = result["direction"]["h3"]["signals"]["crash_prob"]["auroc_predict_DOWN"]

    def verdict_word(auroc, lo=0.52, hi=0.55):
        if auroc is None:
            return "no signal"
        if auroc >= hi:
            return "predicts"
        if auroc >= lo:
            return "weakly predicts"
        if auroc <= 0.48:
            return "anti-correlated"
        return "does NOT predict (~chance)"

    out.append(f"direction h=1: crash_prob {verdict_word(d1)} down moves "
               f"(AUROC {_fmt(d1)})")
    out.append(f"direction h=3: crash_prob {verdict_word(d3)} down moves "
               f"(AUROC {_fmt(d3)})")
    sev_rho = result["severity"]["signals"]["crash_prob"]["spearman"]
    out.append(f"severity (drawdown): crash_prob Spearman {_fmt(sev_rho)} "
               f"({'positive as expected' if sev_rho and sev_rho > 0.05 else 'weak/absent'})")
    ret3 = result["return"]["h3"]["signals"]["crash_prob"]["spearman"]
    out.append(f"forward 3d return: crash_prob Spearman {_fmt(ret3)} "
               f"({'negative as expected' if ret3 and ret3 < -0.05 else 'weak/absent'})")
    vr = result["vol_regime"]["signals"]["crash_prob"]["macro_ovr_auroc"]
    out.append(f"volatility regime: crash_prob macro-OvR AUROC {_fmt(vr)} "
               f"({'ranks high-vol days higher' if vr and vr > 0.55 else 'weak'})")
    return out


if __name__ == "__main__":
    run()
