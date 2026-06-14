"""Kaggle entrypoint for the TRR crash-detection run with NVIDIA Nemotron.

This runs ON Kaggle (kernel_type=script) against the RTX 6000 Pro (Blackwell,
sm_120) GPU, zero-shot — no fine-tuning, no internet. The Nemotron model, the
project code, and the data are all pre-staged (see kaggle/deploy_trr.sh):

    /kaggle/input/<bundle-slug>/code/   <- config.py + ml/ + trr/ (the package)
    /kaggle/input/<bundle-slug>/data/   <- the price CSVs + a news file
    /kaggle/input/<nemotron-model>/     <- a HuggingFace model dir (config.json
                                           + tokenizer); auto-detected at runtime.

Pipeline: load the staged news, run the four-phase TRRPipeline with an
HFReasoningLLM(Nemotron) backend to get daily crash probabilities, then call
trr.evaluate.evaluate(...) with that same backend to score it against the real
price-derived labels and the baselines, writing predictions + metrics + plots
into /kaggle/working.

Two modes:
    - Kaggle run (default) : detect the mounts, RTX 6000 Pro, Nemotron, full run.
    - SMOKE mode (SMOKE=1) : off-Kaggle on CPU with MockLLM + the bundled
                             sample news, tiny, to validate orchestration. Writes
                             a predictions CSV + eval json under /tmp.

HARDWARE GATE: Kaggle silently falls back to a Tesla P100 (sm_60) — which has no
torch kernel image for modern stacks — unless the kernel metadata carries all
three of machine_shape=NvidiaRtxPro6000, enable_gpu=true (boolean), and the
nvidia-nemotron-model-reasoning-challenge competition grant. This script prints
the allocated compute capability and aborts loudly on sm_60 only; it never
aborts on CPU (so the smoke run completes).
"""
from __future__ import annotations

import glob
import os
import sys

# Kaggle's writable output dir — anything written here becomes a downloadable
# kernel output artifact.
KAGGLE_WORKING = "/kaggle/working"

# Off-Kaggle smoke paths (CPU).
SMOKE_OUT_DIR = "/tmp/trr_smoke_out"

# Local copy of the real Olivier Vha crypto-news CSV (used by SMOKE if present).
LOCAL_NEWS_CSV = "data/news_raw/oliviervha/cryptonews.csv"

# Default REAL-run date window: the FTX collapse (Nov 2022). Bounding the first
# real run to ~75 days keeps the LLM call count (and RTX 6000 Pro quota) cheap;
# widen via TRR_START / TRR_END for the full 2022-01..2023-07 run (see README).
DEFAULT_START = "2022-10-01"
DEFAULT_END = "2022-12-15"
# A tiny window for the SMOKE run so it finishes fast on CPU.
SMOKE_START = "2022-11-05"
SMOKE_END = "2022-11-12"

# Cap on headlines fed to ONE daily brainstorm call.
MAX_ITEMS_PER_DAY = 40


def _is_smoke() -> bool:
    return os.environ.get("SMOKE", "0") == "1"


def _date_window(smoke: bool) -> tuple[str, str]:
    """Resolve the (start, end) date window from TRR_START / TRR_END env vars."""
    if smoke:
        start = os.environ.get("TRR_START", SMOKE_START)
        end = os.environ.get("TRR_END", SMOKE_END)
    else:
        start = os.environ.get("TRR_START", DEFAULT_START)
        end = os.environ.get("TRR_END", DEFAULT_END)
    return start, end


# --------------------------------------------------------------------------- #
# Mount discovery.
# --------------------------------------------------------------------------- #
def _diagnose_input() -> None:
    """Print what is actually mounted under /kaggle/input (for debugging mounts)."""
    print("[mount] listing /kaggle/input (depth<=3):", flush=True)
    root = "/kaggle/input"
    if not os.path.isdir(root):
        print("[mount]   /kaggle/input does not exist!", flush=True)
        return
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        print(f"[mount]   {dirpath}/  ({len(dirnames)} dirs, {len(filenames)} files)",
              flush=True)
        for f in filenames[:6]:
            print(f"[mount]       {f}", flush=True)


def _find_kaggle_mount() -> tuple[str, str]:
    """Locate the staged code + data dirs anywhere under /kaggle/input.

    Robust to mount nesting and to `--dir-mode zip` (extract code.zip/data.zip
    if the tree isn't already extracted). Strategy: recursively find the `trr`
    package (a dir containing pipeline.py) for the code dir, and the price CSVs
    for the data dir.
    """
    import zipfile

    # If the bundle came as zips, extract them first so the recursive search hits.
    for z in glob.glob("/kaggle/input/**/code.zip", recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall("/tmp/trr_bundle/code")
        except Exception:
            pass
    for z in glob.glob("/kaggle/input/**/data.zip", recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall("/tmp/trr_bundle/data")
        except Exception:
            pass

    search_roots = ["/kaggle/input", "/tmp/trr_bundle"]

    # code_dir = parent of the trr/ package (identified by trr/pipeline.py).
    code_dir = None
    for root in search_roots:
        hits = sorted(glob.glob(f"{root}/**/trr/pipeline.py", recursive=True))
        if hits:
            code_dir = os.path.dirname(os.path.dirname(hits[0]))
            break

    # data_dir = dir holding the price CSVs.
    data_dir = None
    for root in search_roots:
        hits = sorted(glob.glob(f"{root}/**/BTCUSDT_5min_long.csv", recursive=True))
        if hits:
            data_dir = os.path.dirname(hits[0])
            break

    if code_dir is None or data_dir is None:
        _diagnose_input()
        raise FileNotFoundError(
            f"Could not locate staged bundle (code_dir={code_dir}, "
            f"data_dir={data_dir}). Is nguyenduongtrong/crypto-trr-bundle "
            "attached to the kernel and fully processed?"
        )
    print(f"[mount] code_dir={code_dir}", flush=True)
    print(f"[mount] data_dir={data_dir}", flush=True)
    return code_dir, data_dir


def _find_news_file(data_dir: str) -> str:
    """Find the news file for the run.

    The REAL run uses the attached `oliviervha/crypto-news` dataset, which mounts
    at /kaggle/input/crypto-news/cryptonews.csv — so we glob /kaggle/input for any
    *cryptonews*.csv / *crypto*news*.csv first. Failing that (or off-Kaggle), we
    fall back to a *news* file in the staged bundle data dir, then the bundled
    sample_news.jsonl that deploy_trr.sh copies in.
    """
    input_patterns = (
        "/kaggle/input/**/*cryptonews*.csv",
        "/kaggle/input/**/*crypto*news*.csv",
        "/kaggle/input/**/*crypto-news*.csv",
    )
    for pat in input_patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]

    data_patterns = ("*cryptonews*.csv", "*news*.csv", "*news*.jsonl",
                     "sample_news.jsonl", "*.jsonl")
    for pat in data_patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        "No news file found. Attach the oliviervha/crypto-news dataset "
        "(cryptonews.csv) to the kernel, or stage a *news* file into the bundle "
        f"data dir ({data_dir})."
    )


def _find_model_dir() -> str | None:
    """Auto-detect a staged HuggingFace model dir under /kaggle/input.

    A valid model dir contains a config.json plus a tokenizer file. The exact
    Nemotron slug is set via model_sources in the kernel metadata (see
    TRR_README.md); this stays model-path-agnostic and just finds it on disk.
    Returns the dir, or None if none is staged (smoke / mis-config).
    """
    config_hits = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    tokenizer_names = (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "vocab.json",
    )
    for cfg in config_hits:
        d = os.path.dirname(cfg)
        # Skip the staged-code dir (it is not a model).
        if os.path.isdir(os.path.join(d, "trr")):
            continue
        if any(os.path.exists(os.path.join(d, t)) for t in tokenizer_names):
            return d
    return None


# --------------------------------------------------------------------------- #
# Env / import path.
# --------------------------------------------------------------------------- #
def _configure_env() -> tuple[str, str, str]:
    """Set HISTORICAL_DIR + the import path BEFORE importing the project code.

    Returns (code_dir, data_dir, out_dir).
    """
    if _is_smoke():
        # Off-Kaggle: import the repo we already sit in.
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.environ.get("HISTORICAL_DIR", "/home/nduong/eth-alpha/data")
        os.environ["HISTORICAL_DIR"] = data_dir
        out_dir = SMOKE_OUT_DIR
    else:
        code_dir, data_dir = _find_kaggle_mount()
        os.environ["HISTORICAL_DIR"] = data_dir
        out_dir = KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else "/tmp"

    os.makedirs(out_dir, exist_ok=True)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return code_dir, data_dir, out_dir


# --------------------------------------------------------------------------- #
# GPU diagnostics + hardware gate.
# --------------------------------------------------------------------------- #
def _gpu_diagnostics_and_gate() -> str:
    """Print GPU diagnostics, enforce the RTX 6000 Pro gate, return the dtype.

    Aborts (exit 1) ONLY on the P100/sm_60 fallback. On CPU (no CUDA) it prints
    a note and returns "float32" — the smoke run on this box has no GPU and must
    still complete. bf16 for sm_80+, fp16 for sm_70/75.
    """
    import torch

    available = torch.cuda.is_available()
    print(f"[gpu] torch={torch.__version__}  cuda_available={available}")

    if not available:
        print("[gpu] CUDA not available — running on CPU (expected in SMOKE mode).")
        return "float32"

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"  # explicit, greppable capability marker.
    print(f"[gpu] device={name}")
    print(f"[gpu] compute_capability={major}.{minor}  ({sm})")

    if (major, minor) == (6, 0):
        print("=" * 72)
        print("[gpu] FATAL: allocated a Tesla P100 (sm_60) — RTX 6000 Pro GATE FAILED.")
        print("[gpu] No modern torch kernel image for sm_60; the run cannot proceed.")
        print("[gpu] kernel-metadata.json MUST contain ALL THREE of:")
        print('[gpu]   "machine_shape": "NvidiaRtxPro6000"')
        print('[gpu]   "enable_gpu": true                 (boolean, not "true")')
        print('[gpu]   "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]')
        print("[gpu] Missing any one -> silent P100 downgrade. Also check the")
        print("[gpu] 30 hrs/week RTX 6000 Pro quota has not been exhausted.")
        print("=" * 72)
        sys.exit(1)

    if major >= 8:
        print(f"[gpu] OK: {sm} supports bf16 (sm_80+).")
        return "bfloat16"
    if major >= 7:
        print(f"[gpu] OK: {sm} supports fp16 (sm_70/75).")
        return "float16"
    print(f"[gpu] WARNING: {sm} older than sm_70 — falling back to fp32.")
    return "float32"


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> int:
    smoke = _is_smoke()
    print(f"[kernel] mode={'SMOKE' if smoke else 'KAGGLE'}")

    code_dir, data_dir, out_dir = _configure_env()
    print(f"[kernel] code_dir={code_dir}")
    print(f"[kernel] data_dir={data_dir}")
    print(f"[kernel] out_dir={out_dir}")
    print(f"[kernel] HISTORICAL_DIR={os.environ['HISTORICAL_DIR']}")

    # Import the project code only AFTER sys.path + env are set.
    from trr.llm import MockLLM
    from trr.news import group_by_day, load_news, load_sample_news
    from trr.pipeline import TRRPipeline

    dtype = _gpu_diagnostics_and_gate()

    start, end = _date_window(smoke)
    print(f"[kernel] date window: {start} .. {end}  (TRR_START/TRR_END)")

    # --- Resolve the news source ------------------------------------------ #
    # Real run: the attached oliviervha/crypto-news CSV. SMOKE: the same CSV
    # locally if present, else the bundled synthetic sample.
    if smoke:
        local_csv = os.path.join(code_dir, LOCAL_NEWS_CSV)
        if os.path.isfile(local_csv):
            news_path = local_csv
            news = load_news(news_path)
            print(f"[kernel] SMOKE: using local real news CSV -> {news_path}")
        else:
            news_path = None
            news = load_sample_news()
            print("[kernel] SMOKE: local news CSV absent — using bundled "
                  "sample_news.jsonl")
    else:
        news_path = _find_news_file(data_dir)
        news = load_news(news_path)
        print(f"[kernel] news file: {news_path}")

    # --- Build the LLM backend -------------------------------------------- #
    if smoke:
        print("[kernel] SMOKE: using deterministic MockLLM backend")
        llm = MockLLM()
    else:
        from trr.llm import HFReasoningLLM

        model_dir = _find_model_dir()
        if model_dir is None:
            print("=" * 72)
            print("[kernel] FATAL: no Nemotron model dir found under /kaggle/input.")
            print("[kernel] Set model_sources in the kernel metadata to the chosen")
            print("[kernel] Nemotron model (see kaggle/TRR_README.md) and re-push.")
            print("=" * 72)
            return 1
        print(f"[kernel] Nemotron model dir: {model_dir}")
        llm = HFReasoningLLM(model_path=model_dir, dtype=dtype)

    # --- Run the BATCHED pipeline -> daily predictions -------------------- #
    # Batched brainstorming: ONE LLM call per day (not per article). Over the
    # bounded window that is ~75 days x (1 brainstorm + 1 reason) generations —
    # feasible on the RTX 6000 Pro quota; per-article on 31k news would not be.
    by_day = group_by_day(news)
    print(f"[kernel] news items={len(news)}  days_with_news={len(by_day)}")

    days = [d for d in sorted(by_day.keys())
            if str(start) <= str(d) <= str(end)]
    print(f"[kernel] processing {len(days)} day(s) in window "
          f"(batched, max_items_per_day={MAX_ITEMS_PER_DAY})")

    pipe = TRRPipeline(llm=llm, batch=True, max_items_per_day=MAX_ITEMS_PER_DAY)

    # Step day-by-day so the Kaggle log shows liveness (day index / total).
    rows = []
    for i, day in enumerate(days, 1):
        pred = pipe._step(i - 1, by_day.get(day, []), day)
        rows.append((day, pred))
        print(f"[kernel]   day {i}/{len(days)}  {day}  "
              f"n_news={pred.n_news}  crash_prob={pred.crash_prob:.3f}",
              flush=True)

    import pandas as pd

    preds = pd.DataFrame(
        {
            "crash_prob": [p.crash_prob for _, p in rows],
            "label": [p.label for _, p in rows],
            "n_news": [p.n_news for _, p in rows],
            "n_edges": [p.n_edges for _, p in rows],
            "rationale": [p.rationale for _, p in rows],
        },
        index=pd.Index([d for d, _ in rows], name="day"),
    )

    preds_path = os.path.join(out_dir, "trr_predictions.csv")
    preds.to_csv(preds_path)
    print(f"[kernel] wrote daily predictions -> {preds_path} ({len(preds)} rows)")

    # --- Evaluate vs real labels (inline AUROC over the window) ----------- #
    results = _evaluate_window(preds, out_dir)

    import json

    eval_path = os.path.join(out_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[kernel] wrote eval results -> {eval_path}")

    print(f"[kernel] done. TRR AUROC={results['metrics']['TRR'].get('auroc')}  "
          f"window={results['summary']['date_start']}.."
          f"{results['summary']['date_end']}  "
          f"crash_days={results['summary']['n_crash_days']}")
    return 0


def _evaluate_window(preds, out_dir: str) -> dict:
    """Score the windowed predictions against the real crash labels.

    Aligns the daily `crash_prob` to `trr.labels.crash_labels` on the overlapping
    days and computes AUROC / PR-AUC (when both classes are present) plus the base
    rate. A timeline plot is written when matplotlib is available. Self-contained
    so the windowed real run does not re-run the whole pipeline via evaluate().
    """
    import numpy as np
    import pandas as pd

    from trr.labels import crash_labels

    labels = crash_labels()
    pred_s = pd.Series(
        preds["crash_prob"].to_numpy(dtype=float),
        index=[d for d in preds.index], name="crash_prob",
    )
    label_s = pd.Series(
        labels["crash"].to_numpy(dtype=int),
        index=[ts.date() for ts in labels.index], name="crash",
    )
    aligned = pd.concat([pred_s, label_s], axis=1, join="inner").dropna()
    aligned = aligned.sort_index()

    y_true = aligned["crash"].to_numpy(dtype=int)
    score = aligned["crash_prob"].to_numpy(dtype=float)
    n_classes = len(np.unique(y_true)) if len(y_true) else 0

    auroc = pr_auc = None
    if n_classes == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(y_true, score))
        pr_auc = float(average_precision_score(y_true, score))
    else:
        print("[eval] aligned window is single-class — AUROC/PR-AUC skipped.")

    days = list(aligned.index)
    summary = {
        "n_days": len(days),
        "date_start": str(days[0]) if days else None,
        "date_end": str(days[-1]) if days else None,
        "n_crash_days": int(y_true.sum()) if len(y_true) else 0,
        "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "single_class_window": n_classes < 2,
    }
    print(f"[eval] aligned {summary['n_days']} day(s)  "
          f"crash_days={summary['n_crash_days']}  "
          f"base_rate={summary['base_rate']:.3f}  AUROC={auroc}")

    # Timeline plot (best-effort).
    timeline_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(days, score, color="#1f77b4", lw=1.6, label="TRR crash_prob")
        first = True
        for d, c in zip(days, y_true):
            if c == 1:
                ax.axvspan(d, d, color="#d62728", alpha=0.35, lw=3,
                           label="real crash day" if first else None)
                first = False
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("crash probability")
        ax.set_title("TRR daily crash probability vs real crash days (window)")
        ax.legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        timeline_path = os.path.join(out_dir, "trr_timeline.png")
        fig.savefig(timeline_path, dpi=120)
        plt.close(fig)
        print(f"[eval] timeline plot -> {timeline_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[eval] timeline plot skipped ({exc})")

    return {
        "summary": summary,
        "metrics": {"TRR": {"auroc": auroc, "pr_auc": pr_auc}},
        "artifacts": {"timeline_plot": timeline_path},
    }


if __name__ == "__main__":
    sys.exit(main())
