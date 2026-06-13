"""Kaggle training entrypoint for the crypto-volatility LSTM.

This script runs ON Kaggle (kernel_type=script) against the RTX 6000 Pro
(Blackwell, sm_120) GPU. Kaggle kernels run WITHOUT internet, so all code and
data are pre-staged as a private Kaggle dataset (see kaggle/stage_and_deploy.sh).

Layout of the staged dataset once mounted on Kaggle:
    /kaggle/input/<dataset-slug>/code/   <- config.py + ml/ (the project code)
    /kaggle/input/<dataset-slug>/data/   <- the 5 historical 5-min CSVs

Two modes:
    - Kaggle run  (default)  : detect the mounted dataset, full training run.
    - SMOKE mode  (SMOKE=1)  : run off-Kaggle on the local repo + local data,
                               tiny max_rows / 2 epochs, to validate the whole
                               orchestration on CPU before deploying.

HARDWARE GATE: Kaggle silently falls back to a Tesla P100 (sm_60) — which has
no torch 2.10+ kernel image — unless the kernel metadata carries all three of
machine_shape=NvidiaRtxPro6000, enable_gpu=true, and the Nemotron
competition_sources grant. This script GREPS the allocated compute capability
from the logs and aborts loudly on sm_60 so a silent downgrade can never pass
unnoticed.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

# Kaggle's writable output dir — anything copied here becomes a downloadable
# kernel output artifact.
KAGGLE_WORKING = "/kaggle/working"

# Local smoke-test paths (off-Kaggle, CPU).
SMOKE_HISTORICAL_DIR = "/home/nduong/eth-alpha/data"
SMOKE_MODEL_PATH = "/tmp/agent1_smoke.pt"


def _is_smoke() -> bool:
    return os.environ.get("SMOKE", "0") == "1"


def _find_kaggle_mount() -> tuple[str, str]:
    """Locate the staged dataset on Kaggle.

    Returns (code_dir, data_dir). The dataset slug is unknown ahead of time, so
    glob for any */code under /kaggle/input.
    """
    code_candidates = sorted(glob.glob("/kaggle/input/*/code"))
    if not code_candidates:
        raise FileNotFoundError(
            "Could not find a staged code dir under /kaggle/input/*/code. "
            "Did the crypto-volatility-bundle dataset get attached to the kernel?"
        )
    code_dir = code_candidates[0]
    mount = os.path.dirname(code_dir)
    data_dir = os.path.join(mount, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Found code at {code_dir} but no sibling data dir at {data_dir}."
        )
    return code_dir, data_dir


def _configure_env() -> str:
    """Set HISTORICAL_DIR + MODEL_PATH and the code import path, BEFORE config
    is imported. Returns the code dir that was added to sys.path.
    """
    if _is_smoke():
        # Off-Kaggle: import the repo we already sit in, read local data.
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["HISTORICAL_DIR"] = SMOKE_HISTORICAL_DIR
        os.environ["MODEL_PATH"] = SMOKE_MODEL_PATH
    else:
        # On Kaggle: import the staged code, read the staged data, write the
        # checkpoint into the writable working dir.
        code_dir, data_dir = _find_kaggle_mount()
        os.environ["HISTORICAL_DIR"] = data_dir
        os.environ["MODEL_PATH"] = os.path.join(KAGGLE_WORKING, "lstm_volatility.pt")

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return code_dir


def _gpu_diagnostics_and_gate() -> None:
    """Print GPU diagnostics and enforce the RTX 6000 Pro hardware gate.

    Aborts (exit 1) ONLY on the specific P100/sm_60 fallback. On CPU (no CUDA)
    it prints a note and returns — the smoke run on this box has no GPU and must
    still complete.
    """
    import torch

    available = torch.cuda.is_available()
    print(f"[gpu] torch={torch.__version__}  cuda_available={available}")

    if not available:
        print("[gpu] CUDA not available — running on CPU (expected in SMOKE mode).")
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    # Explicit, greppable capability marker for the post-run log check.
    sm = f"sm_{major}{minor}"
    print(f"[gpu] device={name}")
    print(f"[gpu] compute_capability={major}.{minor}  ({sm})")

    if (major, minor) == (6, 0):
        # P100 fallback — torch 2.10+ ships no kernel image for sm_60.
        print("=" * 72)
        print("[gpu] FATAL: allocated a Tesla P100 (sm_60) — the RTX 6000 Pro GATE FAILED.")
        print("[gpu] PyTorch 2.10+ has no kernel image for sm_60; training cannot run.")
        print("[gpu] kernel-metadata.json MUST contain ALL THREE of:")
        print('[gpu]   "machine_shape": "NvidiaRtxPro6000"')
        print('[gpu]   "enable_gpu": true                 (boolean, not "true")')
        print('[gpu]   "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]')
        print("[gpu] Missing any one -> silent P100 downgrade. Also check the")
        print("[gpu] 30 hrs/week RTX 6000 Pro quota has not been exhausted.")
        print("=" * 72)
        sys.exit(1)

    if major >= 8:
        print(f"[gpu] OK: {sm} supports bf16 autocast (sm_80+).")
    elif major >= 7:
        print(f"[gpu] OK: {sm} supports fp16 autocast + GradScaler (sm_70/75).")
    else:
        print(f"[gpu] WARNING: {sm} is older than sm_70 — training falls back to fp32.")


def _copy_outputs(model_path: str, metrics_path: str | None) -> None:
    """Copy the checkpoint (+ metrics JSON if any) into /kaggle/working so they
    are exposed as downloadable kernel outputs. No-op off-Kaggle.
    """
    if _is_smoke() or not os.path.isdir(KAGGLE_WORKING):
        return
    for src in (model_path, metrics_path):
        if src and os.path.exists(src):
            dst = os.path.join(KAGGLE_WORKING, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                print(f"[out] copied {src} -> {dst}")


def main() -> int:
    smoke = _is_smoke()
    print(f"[kernel] mode={'SMOKE' if smoke else 'KAGGLE'}")

    code_dir = _configure_env()
    print(f"[kernel] code_dir={code_dir}")
    print(f"[kernel] HISTORICAL_DIR={os.environ['HISTORICAL_DIR']}")
    print(f"[kernel] MODEL_PATH={os.environ['MODEL_PATH']}")

    # Import config/ml only AFTER the env vars + sys.path are set.
    import config
    from ml.train import train

    _gpu_diagnostics_and_gate()

    # Full run on Kaggle; tiny run in SMOKE so it finishes on CPU in seconds.
    if smoke:
        train_kwargs = dict(
            source="historical",
            epochs=2,
            batch_size=64,
            max_rows=4000,
        )
    else:
        train_kwargs = dict(
            source="historical",
            epochs=60,
            batch_size=256,
        )
    print(f"[kernel] train({train_kwargs})")
    metrics = train(**train_kwargs)
    print(f"[kernel] train done: {metrics}")

    model_path = config.MODEL_PATH
    metrics_path = None

    # Evaluation is owned by a teammate (ml/evaluate.py), built in parallel and
    # possibly not present yet. Degrade gracefully: never let a missing/broken
    # evaluator lose the trained model.
    try:
        from ml.evaluate import evaluate

        out_dir = KAGGLE_WORKING if (not smoke and os.path.isdir(KAGGLE_WORKING)) else "/tmp"
        print(f"[kernel] evaluate(model_path={model_path}, out_dir={out_dir})")
        eval_metrics = evaluate(
            source="historical",
            model_path=model_path,
            out_dir=out_dir,
        )
        print(f"[kernel] evaluate done: {eval_metrics}")
        # Pick up the metrics JSON the evaluator wrote, if present.
        jsons = sorted(glob.glob(os.path.join(out_dir, "*metric*.json")))
        if jsons:
            metrics_path = jsons[-1]
            print(f"[kernel] metrics json: {metrics_path}")
    except Exception as exc:  # noqa: BLE001 — evaluation is strictly optional.
        print(f"[kernel] evaluation skipped ({type(exc).__name__}: {exc})")

    _copy_outputs(model_path, metrics_path)
    print(f"[kernel] checkpoint at {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
