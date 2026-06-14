"""Rigorous evaluation of the TRR crash-detection pipeline.

Runs the four-phase TRR pipeline (arXiv:2410.17266, crypto adaptation) over a
news stream, aligns its daily crash probabilities to the REAL price-derived
crash labels (`trr.labels.crash_labels`), and scores it against three baselines
on the SAME aligned days:

    base_rate        constant majority predictor (AUROC 0.5 reference).
    news_negativity  a NON-LLM heuristic: per-day fraction of negative
                     headlines by a keyword lexicon. Tests whether the LLM's
                     relational/temporal reasoning beats naive headline counting.
    price_momentum   a PRICE-ONLY signal (no news): trailing portfolio
                     drawdown. Tests whether news reasoning beats pure price.

Primary metric is AUROC (crashes are rare/imbalanced); PR-AUC, plus
accuracy/precision/recall/F1 at the pipeline's label threshold, are also
reported. An ROC-curve plot and a crash-probability timeline (real crash days
marked) are written to the output dir. Everything is deterministic.

CLI:
    python -m trr.evaluate [--news-path P] [--out-dir reports] [--lam L] [--top-k K]

With the default MockLLM (which is itself lexicon-based), TRR and the
news_negativity baseline score similarly — that is expected and is noted in the
output. The harness is what matters: swapping in the Nemotron HFReasoningLLM on
Kaggle is where the relational/temporal reasoning earns its lift.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from trr.labels import build_portfolio, crash_labels
from trr.llm import MockLLM, ReasoningLLM, _NEG
from trr.news import group_by_day, load_news, load_sample_news
from trr.pipeline import TRRPipeline
from trr.schema import NewsItem

# Where the canonical machine-readable results land (deterministic).
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")


# --------------------------------------------------------------------------- #
# Baseline scores (computed on the same days the pipeline produced).
# --------------------------------------------------------------------------- #
def _news_negativity_scores(
    news_by_day: dict, days: list
) -> pd.Series:
    """Per-day fraction of that day's headlines that are negative by lexicon.

    Reuses `trr.llm._NEG` so it is exactly the lexicon the MockLLM keys off —
    making it a fair, purely count-based foil for the relational reasoning.
    A day with no news scores 0.
    """
    scores = {}
    for day in days:
        items = news_by_day.get(day, [])
        if not items:
            scores[day] = 0.0
            continue
        neg = 0
        for it in items:
            words = set(_tokenize(it.text()))
            if words & _NEG:
                neg += 1
        scores[day] = neg / len(items)
    return pd.Series(scores, name="news_negativity")


def _tokenize(text: str) -> list:
    import re

    return re.findall(r"[a-z]+", text.lower())


def _price_momentum_scores(days: list, lookback: int = 3) -> pd.Series:
    """Price-only crash score: trailing `lookback`-day portfolio drawdown.

    Higher score == more recent decline. Uses the equal-weight portfolio level
    from `build_portfolio`; a negative trailing return maps to a positive crash
    score (clamped to [0, 1]). No news whatsoever.
    """
    try:
        port = build_portfolio()
    except Exception as exc:  # pragma: no cover - price data may be absent
        warnings.warn(f"price_momentum unavailable ({exc}); using zeros")
        return pd.Series({d: 0.0 for d in days}, name="price_momentum")

    level = port["portfolio_level"]
    # Trailing return over the lookback window, by calendar day.
    trailing = level / level.shift(lookback) - 1.0
    # Map by date for alignment with the (date-indexed) prediction frame.
    by_date = {ts.date(): val for ts, val in trailing.items()}

    scores = {}
    for day in days:
        r = by_date.get(day, np.nan)
        if r is None or (isinstance(r, float) and np.isnan(r)):
            scores[day] = 0.0
        else:
            # Negative trailing return -> positive crash score; scale a -20%
            # drawdown to ~1.0 so the signal spans [0, 1] usefully.
            scores[day] = float(min(1.0, max(0.0, -r / 0.20)))
    return pd.Series(scores, name="price_momentum")


# --------------------------------------------------------------------------- #
# Metric helpers.
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """AUROC / PR-AUC + thresholded accuracy/precision/recall/F1.

    AUROC and PR-AUC need both classes present; if the aligned window is
    single-class they are reported as None with a warning.
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) == 2
    if both_classes:
        auroc = float(roc_auc_score(y_true, score))
        pr_auc = float(average_precision_score(y_true, score))
    else:
        warnings.warn("aligned window is single-class; AUROC/PR-AUC skipped")
        auroc = None
        pr_auc = None

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _fmt(x) -> str:
    return "  n/a " if x is None else f"{x:6.3f}"


# --------------------------------------------------------------------------- #
# Plots (Agg backend; degrade gracefully if matplotlib is missing).
# --------------------------------------------------------------------------- #
def _plot_roc(y_true, scores_by_method: dict, out_dir: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"matplotlib unavailable; ROC plot skipped ({exc})")
        return None

    if len(np.unique(y_true)) < 2:
        warnings.warn("single-class window; ROC plot skipped")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    for method, score in scores_by_method.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, score)
            auc = roc_auc_score(y_true, score)
        except ValueError:
            continue
        lw = 2.4 if method == "TRR" else 1.3
        ax.plot(fpr, tpr, lw=lw, label=f"{method} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("TRR crash detection — ROC vs baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out_dir, "trr_roc.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_timeline(
    days, trr_score, y_true, out_dir: str
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"matplotlib unavailable; timeline plot skipped ({exc})")
        return None

    days = list(days)
    y_true = np.asarray(y_true, dtype=int)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(days, trr_score, color="#1f77b4", lw=1.6, label="TRR crash_prob")
    # Shade real crash days.
    first = True
    for d, c in zip(days, y_true):
        if c == 1:
            ax.axvspan(
                d, d, color="#d62728", alpha=0.35, lw=3,
                label="real crash day" if first else None,
            )
            first = False
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("crash probability")
    ax.set_title("TRR daily crash probability vs real crash days")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(out_dir, "trr_timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main entry point.
# --------------------------------------------------------------------------- #
def evaluate(
    news_path: str = None,
    llm: ReasoningLLM = None,
    out_dir: str = "reports",
    lam: float = 0.3,
    top_k: int = 30,
) -> dict:
    """Evaluate TRR crash predictions against the real labels and baselines.

    Returns a dict with per-method metrics, the aligned window summary, and the
    saved artifact paths. Also writes `trr/eval_results.json` and two plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    llm = llm if llm is not None else MockLLM()

    # 1. Load news and run the pipeline.
    news = load_sample_news() if news_path is None else load_news(news_path)
    news_by_day = group_by_day(news)
    pipe = TRRPipeline(llm=llm, lam=lam, top_k=top_k)
    preds = pipe.run(news_by_day)
    label_threshold = pipe.label_threshold

    # 2. Align predictions to the real labels on overlapping days.
    labels = crash_labels()
    # Both frames keyed by python date for a clean inner join.
    pred_s = pd.Series(
        preds["crash_prob"].to_numpy(),
        index=[d for d in preds.index],
        name="crash_prob",
    )
    label_s = pd.Series(
        labels["crash"].to_numpy(dtype=int),
        index=[ts.date() for ts in labels.index],
        name="crash",
    )
    aligned = pd.concat([pred_s, label_s], axis=1, join="inner").dropna()
    aligned = aligned.sort_index()
    if aligned.empty:
        raise RuntimeError(
            "No overlap between prediction days and label days — check the news "
            "date range vs the price history."
        )

    days = list(aligned.index)
    y_true = aligned["crash"].to_numpy(dtype=int)
    trr_score = aligned["crash_prob"].to_numpy(dtype=float)
    base_rate = float(y_true.mean())

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        warnings.warn(
            "aligned window has a single class — AUROC/PR-AUC cannot be computed."
        )

    # 3. Baselines on the SAME aligned days.
    neg_s = _news_negativity_scores(news_by_day, days).reindex(days).fillna(0.0)
    mom_s = _price_momentum_scores(days).reindex(days).fillna(0.0)
    # base_rate: constant predictor at the base rate (AUROC is undefined for a
    # constant score -> 0.5 by definition; report it explicitly).
    base_score = np.full(len(days), base_rate, dtype=float)

    scores_by_method = {
        "TRR": trr_score,
        "news_negativity": neg_s.to_numpy(dtype=float),
        "price_momentum": mom_s.to_numpy(dtype=float),
    }

    # 4. Metrics per method.
    results: dict = {}
    results["TRR"] = _metrics(y_true, trr_score, label_threshold)
    results["news_negativity"] = _metrics(y_true, neg_s.to_numpy(), 0.5)
    results["price_momentum"] = _metrics(y_true, mom_s.to_numpy(), 0.5)
    # base_rate: a constant score has AUROC 0.5 by definition; the thresholded
    # predictor is the majority class.
    majority = int(base_rate >= 0.5)
    base_pred = np.full(len(days), majority, dtype=int)
    results["base_rate"] = {
        "auroc": 0.5 if n_classes == 2 else None,
        "pr_auc": base_rate if n_classes == 2 else None,
        "accuracy": float(accuracy_score(y_true, base_pred)),
        "precision": float(precision_score(y_true, base_pred, zero_division=0)),
        "recall": float(recall_score(y_true, base_pred, zero_division=0)),
        "f1": float(f1_score(y_true, base_pred, zero_division=0)),
    }

    # 5. Plots.
    roc_path = _plot_roc(y_true, scores_by_method, out_dir)
    timeline_path = _plot_timeline(days, trr_score, y_true, out_dir)

    # 6. Assemble + persist the full result dict (deterministic, JSON-safe).
    summary = {
        "n_days": len(days),
        "date_start": str(days[0]),
        "date_end": str(days[-1]),
        "n_crash_days": int(y_true.sum()),
        "base_rate": base_rate,
        "label_threshold": label_threshold,
        "lam": lam,
        "top_k": top_k,
        "llm": type(llm).__name__,
        "single_class_window": n_classes < 2,
    }
    out = {
        "summary": summary,
        "metrics": results,
        "artifacts": {
            "roc_plot": roc_path,
            "timeline_plot": timeline_path,
            "results_json": RESULTS_PATH,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    _print_table(out)
    return out


def _print_table(out: dict) -> None:
    """Pretty comparison table; the TRR row is marked."""
    s = out["summary"]
    print()
    print("=" * 72)
    print(
        f"[eval] TRR crash detection  |  llm={s['llm']}  "
        f"days={s['n_days']} ({s['date_start']}..{s['date_end']})"
    )
    print(
        f"[eval] crash days={s['n_crash_days']}  base_rate={s['base_rate']:.3f}  "
        f"label_threshold={s['label_threshold']}"
    )
    print("=" * 72)
    header = f"{'method':<18}{'AUROC':>8}{'PR-AUC':>8}{'F1':>8}{'prec':>8}{'recall':>8}"
    print(header)
    print("-" * len(header))
    order = ["TRR", "news_negativity", "price_momentum", "base_rate"]
    for method in order:
        m = out["metrics"][method]
        mark = " <- TRR" if method == "TRR" else ""
        print(
            f"{method:<18}"
            f"{_fmt(m['auroc'])}{_fmt(m['pr_auc'])}"
            f"{_fmt(m['f1'])}{_fmt(m['precision'])}{_fmt(m['recall'])}"
            f"{mark}"
        )
    print("-" * len(header))
    if s["llm"] == "MockLLM":
        print(
            "[eval] NOTE: MockLLM is itself lexicon-based, so TRR and the "
            "news_negativity\n"
            "       baseline score similarly here. The harness is correct; the "
            "real lift\n"
            "       comes from the Nemotron HFReasoningLLM on Kaggle, whose "
            "relational +\n"
            "       temporal reasoning is what news_negativity / price_momentum "
            "cannot do."
        )
    a = out["artifacts"]
    if a["roc_plot"]:
        print(f"[eval] ROC plot      -> {a['roc_plot']}")
    if a["timeline_plot"]:
        print(f"[eval] timeline plot -> {a['timeline_plot']}")
    print(f"[eval] results json  -> {a['results_json']}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the TRR crash-detection pipeline.")
    p.add_argument("--news-path", default=None,
                   help="news .jsonl/.csv (default: bundled sample_news.jsonl)")
    p.add_argument("--out-dir", default="reports", help="plot output dir")
    p.add_argument("--lam", type=float, default=0.3, help="memory decay rate")
    p.add_argument("--top-k", type=int, default=30, help="attention prune size")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    evaluate(
        news_path=args.news_path,
        out_dir=args.out_dir,
        lam=args.lam,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
