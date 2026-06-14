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


def _is_smoke() -> bool:
    return os.environ.get("SMOKE", "0") == "1"


# --------------------------------------------------------------------------- #
# Mount discovery.
# --------------------------------------------------------------------------- #
def _find_kaggle_mount() -> tuple[str, str]:
    """Locate the staged code + data dirs on Kaggle.

    Returns (code_dir, data_dir). The dataset slug is unknown ahead of time, so
    glob for any */code under /kaggle/input that contains the `trr/` package.
    """
    code_candidates = sorted(glob.glob("/kaggle/input/*/code"))
    code_dir = None
    for cand in code_candidates:
        if os.path.isdir(os.path.join(cand, "trr")):
            code_dir = cand
            break
    if code_dir is None:
        raise FileNotFoundError(
            "Could not find a staged code dir containing the trr/ package under "
            "/kaggle/input/*/code. Did the crypto-trr-bundle dataset get "
            "attached to the kernel?"
        )
    mount = os.path.dirname(code_dir)
    data_dir = os.path.join(mount, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Found code at {code_dir} but no sibling data dir at {data_dir}."
        )
    return code_dir, data_dir


def _find_news_file(data_dir: str) -> str:
    """Find a news file (.jsonl/.csv) in the staged data dir.

    Prefers an explicit *news* file; falls back to the bundled sample_news.jsonl
    that deploy_trr.sh copies in. A real Kaggle crypto-news dataset can be staged
    here instead and will be picked up automatically.
    """
    patterns = ("*news*.jsonl", "*news*.csv", "sample_news.jsonl", "*.jsonl")
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No news file (.jsonl/.csv) found under {data_dir}. deploy_trr.sh copies "
        "trr/sample_news.jsonl into data/ as the default news file."
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
    from trr.evaluate import evaluate
    from trr.llm import MockLLM
    from trr.news import group_by_day, load_news, load_sample_news
    from trr.pipeline import TRRPipeline

    dtype = _gpu_diagnostics_and_gate()

    # --- Build the LLM backend -------------------------------------------- #
    if smoke:
        # Off-Kaggle CPU: deterministic mock backend over the bundled sample.
        print("[kernel] SMOKE: using MockLLM + bundled sample_news.jsonl")
        llm = MockLLM()
        news = load_sample_news()
        news_path = None  # evaluate() will reload the sample itself.
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

        news_path = _find_news_file(data_dir)
        print(f"[kernel] news file: {news_path}")
        news = load_news(news_path)

    # --- Run the pipeline -> daily predictions ---------------------------- #
    by_day = group_by_day(news)
    print(f"[kernel] news items={len(news)}  days_with_news={len(by_day)}")
    pipe = TRRPipeline(llm=llm)
    preds = pipe.run(by_day)

    preds_path = os.path.join(out_dir, "trr_predictions.csv")
    preds.to_csv(preds_path)
    print(f"[kernel] wrote daily predictions -> {preds_path} ({len(preds)} rows)")

    # --- Evaluate vs real labels + baselines (same Nemotron backend) ------ #
    # evaluate() re-runs the pipeline internally with this llm so its predictions
    # and metrics are self-consistent; it also writes trr/eval_results.json.
    results = evaluate(
        news_path=news_path,
        llm=llm,
        out_dir=out_dir,
    )

    # Surface the eval json into the output dir under a stable name too.
    import json

    eval_path = os.path.join(out_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"[kernel] wrote eval results -> {eval_path}")

    trr_auroc = results["metrics"]["TRR"].get("auroc")
    print(f"[kernel] done. TRR AUROC={trr_auroc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
