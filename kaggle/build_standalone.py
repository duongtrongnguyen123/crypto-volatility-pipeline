"""Assemble a SELF-CONTAINED Kaggle kernel from the trr/ package.

Inlines schema/llm/memory/attention/brainstorm/reason/pipeline/news/labels into
one script so the kernel needs NO code dataset (only the price-data, news, and
model datasets). This eliminates the bundle version-pin race entirely.

Output: kaggle/trr_standalone.py  (set as code_file for a fresh kernel).
"""
import os

MODULES = ["schema", "llm", "memory", "attention", "brainstorm",
           "reason", "pipeline", "news", "labels"]

HEADER = '''"""TRR crypto crash detection — SELF-CONTAINED Kaggle kernel (Qwen / Nemotron).

All TRR code is inlined below (no code dataset). Attaches only:
  - a price-data dataset (the *_5min_long.csv files)  -> crash labels
  - oliviervha/crypto-news (cryptonews.csv)           -> news
  - a HuggingFace model (Qwen2.5-14B-Instruct, etc.)  -> the reasoner
Runs zero-shot on the RTX 6000 Pro (sm_120), no internet.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

BUILD_TAG = "standalone-v2-calib"


class config:  # shim: only HISTORICAL_DIR is referenced by the inlined code
    HISTORICAL_DIR = os.environ.get("HISTORICAL_DIR", "/kaggle/input")
'''

MAIN = '''

# =========================================================================== #
# Kernel orchestration (self-contained — no code dataset needed).
# =========================================================================== #
KAGGLE_WORKING = "/kaggle/working"
SMOKE_OUT_DIR = "/tmp/trr_smoke_out"
LOCAL_NEWS_CSV = "data/news_raw/oliviervha/cryptonews.csv"
DEFAULT_START, DEFAULT_END = "2022-10-01", "2022-12-15"
SMOKE_START, SMOKE_END = "2022-11-05", "2022-11-12"
MAX_ITEMS_PER_DAY = 40


def _is_smoke():
    return os.environ.get("SMOKE", "0") == "1"


def _glob1(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None


def _gpu_gate():
    import torch
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available (CPU/SMOKE).", flush=True)
        return "float32"
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  sm_{major}{minor}  torch={torch.__version__}", flush=True)
    if (major, minor) == (6, 0):
        print("[gpu] FATAL: P100/sm_60 fallback — the three-field RTX 6000 Pro gate failed.", flush=True)
        sys.exit(1)
    return "bfloat16" if major >= 8 else "float16"


def _find_model_dir():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if any(os.path.exists(os.path.join(d, t)) for t in
               ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            return d
    return None


def _save_outputs(pred_df, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(out_dir, "trr_predictions.csv"))
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(pred_df)), pred_df["crash_prob"], label="crash_prob")
        crash = pred_df["label"].to_numpy()
        for i, c in enumerate(crash):
            if c == 1:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)
        ax.set_title("TRR crash probability vs actual crash days (shaded)")
        ax.legend()
        fig.savefig(os.path.join(out_dir, "trr_timeline.png"), dpi=120, bbox_inches="tight")
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)


def _evaluate(pred_df, out_dir):
    """AUROC / PR-AUC of TRR crash_prob vs price-derived labels + baselines."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = crash_labels()
    lab = labels.copy()
    lab.index = pd.to_datetime(lab.index).date
    df = pred_df.copy()
    df["crash"] = [int(lab["crash"].get(d, 0)) for d in df.index]

    y = df["crash"].to_numpy()
    metrics = {"summary": {"n_days": int(len(df)), "n_crash_days": int(y.sum()),
                           "base_rate": float(y.mean()),
                           "date_start": str(df.index.min()),
                           "date_end": str(df.index.max())},
               "metrics": {}}
    if 0 < y.sum() < len(y):
        metrics["metrics"]["TRR"] = {
            "auroc": float(roc_auc_score(y, df["crash_prob"])),
            "pr_auc": float(average_precision_score(y, df["crash_prob"])),
        }
        # news-volume baseline: more news ~ more attention ~ risk
        if "n_news" in df:
            metrics["metrics"]["news_volume"] = {
                "auroc": float(roc_auc_score(y, df["n_news"])),
                "pr_auc": float(average_precision_score(y, df["n_news"])),
            }
    else:
        metrics["summary"]["single_class_window"] = True
    _save_outputs(df, metrics, out_dir)
    return metrics


def main():
    smoke = _is_smoke()
    print(f"[kernel] BUILD={BUILD_TAG} mode={'SMOKE' if smoke else 'KAGGLE'}", flush=True)

    # price data dir -> HISTORICAL_DIR
    price = _glob1("/kaggle/input/**/BTCUSDT_5min_long.csv",
                   os.path.join(config.HISTORICAL_DIR, "BTCUSDT_5min_long.csv"))
    if smoke and not price:
        price = "/home/nduong/eth-alpha/data/BTCUSDT_5min_long.csv"
    config.HISTORICAL_DIR = os.path.dirname(price)
    print(f"[kernel] HISTORICAL_DIR={config.HISTORICAL_DIR}", flush=True)

    # news
    news_csv = _glob1("/kaggle/input/**/*cryptonews*.csv", "/kaggle/input/**/*crypto*news*.csv")
    if smoke and not news_csv and os.path.exists(LOCAL_NEWS_CSV):
        news_csv = LOCAL_NEWS_CSV
    print(f"[kernel] news={news_csv}", flush=True)
    news = load_news(news_csv)

    start = os.environ.get("TRR_START", SMOKE_START if smoke else DEFAULT_START)
    end = os.environ.get("TRR_END", SMOKE_END if smoke else DEFAULT_END)
    out_dir = SMOKE_OUT_DIR if smoke else (KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else "/tmp")

    dtype = _gpu_gate()
    if smoke:
        llm = MockLLM()
    else:
        model_dir = _find_model_dir()
        print(f"[kernel] model dir: {model_dir}", flush=True)
        llm = HFReasoningLLM(model_path=model_dir, dtype=dtype)

    print(f"[kernel] window {start}..{end}  news_items={len(news)}", flush=True)
    pipe = TRRPipeline(llm=llm, batch=True, max_items_per_day=MAX_ITEMS_PER_DAY, lam=0.6)
    pred = pipe.run(group_by_day(news), start=start, end=end)
    print(f"[kernel] predicted {len(pred)} days", flush=True)

    metrics = _evaluate(pred, out_dir)
    print(f"[kernel] metrics: {json.dumps(metrics.get('metrics', {}))}", flush=True)
    print(f"[kernel] wrote outputs -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def strip_module(src: str) -> str:
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("from trr.") or s == "import config" or s.startswith("from __future__"):
            continue
        # Drop each module's __main__ self-test block (and everything after it).
        if s.startswith("if __name__"):
            break
        out.append(line)
    return "\n".join(out)


def build() -> str:
    parts = [HEADER]
    for m in MODULES:
        src = open(os.path.join("trr", f"{m}.py")).read()
        parts.append(f"\n# ===================== trr/{m}.py =====================\n")
        parts.append(strip_module(src))
    parts.append(MAIN)
    return "\n".join(parts)


if __name__ == "__main__":
    code = build()
    out = os.path.join("kaggle", "trr_standalone.py")
    with open(out, "w") as f:
        f.write(code)
    print(f"wrote {out} ({len(code)} bytes, {code.count(chr(10))} lines)")
