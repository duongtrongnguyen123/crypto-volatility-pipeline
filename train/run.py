"""Train + evaluate a TRR meta-learner and answer: does TRAINING help?

Compares, on a true OUT-OF-TIME / cross-source split (train 2016-2020 analyst
news, test 2021-2023 FNSPID news) AND a walk-forward CV:
  - LLM-only         : the zero-shot crash_prob (no training)
  - news-volume      : log news count (no training)
  - technical GBM    : trained on price technicals + news counts (NO LLM signal)
  - full ensemble GBM: trained on technicals + LLM crash_prob (the meta-learner)
  - logistic stack   : linear blend of all features

Honest question: does the trained ensemble beat the zero-shot LLM and the
volume baseline, and does the LLM signal add value on top of technicals?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train.features import FEATURES_FULL, FEATURES_TECH, build_dataset

RANDOM = 0


def _gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        class_weight="balanced", random_state=RANDOM)


def _logit():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000,
                                            C=0.5, random_state=RANDOM))


def _score(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "base_rate": float(np.mean(y))}


def out_of_time(df):
    tr = df[df["era"] == "2016_2020"]
    te = df[df["era"] == "2021_2023"]
    ytr, yte = tr["label_true"].to_numpy(), te["label_true"].to_numpy()
    res = {}
    # zero-shot / baselines (no training) — rank test by the raw feature
    res["LLM_zeroshot"] = _score(yte, te["crash_prob"].to_numpy())
    res["news_volume"] = _score(yte, te["log_news"].to_numpy())
    # trained: technical-only (no LLM)
    m = _gbm().fit(tr[FEATURES_TECH], ytr)
    res["GBM_technical_only"] = _score(yte, m.predict_proba(te[FEATURES_TECH])[:, 1])
    # trained: full ensemble (technicals + LLM)
    mf = _gbm().fit(tr[FEATURES_FULL], ytr)
    res["GBM_full_ensemble"] = _score(yte, mf.predict_proba(te[FEATURES_FULL])[:, 1])
    # trained: logistic stack
    ml = _logit().fit(tr[FEATURES_FULL], ytr)
    res["Logistic_stack"] = _score(yte, ml.predict_proba(te[FEATURES_FULL])[:, 1])
    # permutation importance of the full ensemble on the test set
    pi = permutation_importance(mf, te[FEATURES_FULL], yte, scoring="roc_auc",
                                n_repeats=20, random_state=RANDOM)
    imp = sorted(zip(FEATURES_FULL, pi.importances_mean), key=lambda x: -x[1])
    return res, imp, len(tr), len(te), int(ytr.sum()), int(yte.sum())


def walk_forward(df, n_splits=5):
    """Pooled out-of-fold AUROC: LLM-only vs full ensemble, time-ordered."""
    d = df.sort_index()
    X, y = d[FEATURES_FULL].to_numpy(), d["label_true"].to_numpy()
    llm = d["crash_prob"].to_numpy()
    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_ens = np.full(len(d), np.nan)
    for tr_i, te_i in tss.split(X):
        if y[tr_i].sum() == 0:
            continue
        m = _gbm().fit(X[tr_i], y[tr_i])
        oof_ens[te_i] = m.predict_proba(X[te_i])[:, 1]
    mask = ~np.isnan(oof_ens)
    return {
        "ensemble_walkforward": _score(y[mask], oof_ens[mask]),
        "LLM_zeroshot_samefolds": _score(y[mask], llm[mask]),
        "n_eval": int(mask.sum()),
    }


def main():
    df = build_dataset()
    oot, imp, ntr, nte, ctr, cte = out_of_time(df)
    wf = walk_forward(df)

    lines = ["# TRR meta-learner — does training help?\n",
             f"Dataset: {len(df)} days {df.index.min()}..{df.index.max()}, "
             f"{int(df['label_true'].sum())} crashes ({df['label_true'].mean():.1%}).\n",
             f"Out-of-time split: train 2016-2020 (n={ntr}, {ctr} crashes, analyst news) "
             f"-> test 2021-2023 (n={nte}, {cte} crashes, FNSPID news).\n",
             "\n## Out-of-time / cross-source test (2021-2023)\n",
             "| Model | AUROC | PR-AUC |", "|---|---|---|"]
    for k, v in sorted(oot.items(), key=lambda x: -x[1]["auroc"]):
        lines.append(f"| {k} | {v['auroc']:.3f} | {v['pr_auc']:.3f} |")
    lines.append(f"\n(test base rate {oot['LLM_zeroshot']['base_rate']:.3f})\n")
    lines.append("\n## Walk-forward CV (pooled out-of-fold, 5 splits)\n")
    lines.append("| Model | AUROC | PR-AUC |"); lines.append("|---|---|---|")
    for k in ["ensemble_walkforward", "LLM_zeroshot_samefolds"]:
        lines.append(f"| {k} | {wf[k]['auroc']:.3f} | {wf[k]['pr_auc']:.3f} |")
    lines.append(f"\n(n_eval={wf['n_eval']})\n")
    lines.append("\n## Full-ensemble permutation importance (test AUROC drop)\n")
    for f, v in imp:
        lines.append(f"- {f}: {v:+.4f}")

    report = "\n".join(lines)
    print(report)
    with open("reports/training_results.md", "w") as fh:
        fh.write(report + "\n")
    with open("reports/stock_runs/training_metrics.json", "w") as fh:
        json.dump({"out_of_time": oot, "walk_forward": wf,
                   "importance": imp}, fh, indent=2, default=str)
    print("\n[train] wrote reports/training_results.md + training_metrics.json")


if __name__ == "__main__":
    main()
