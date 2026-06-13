# Kaggle GPU deployment — crypto-volatility LSTM

Train the LSTM on Kaggle's **NVIDIA RTX 6000 Pro** (Blackwell, sm_120, 102 GB
VRAM). Kaggle kernels run **without internet**, so the project code and the
historical 5-min data are pre-staged as a private Kaggle dataset and the kernel
imports them from the mount at `/kaggle/input/<slug>/`.

Files in this directory:

| File | Purpose |
| --- | --- |
| `train_kernel.py` | Runs ON Kaggle: detects the mount, sets env, trains, evaluates, saves outputs. Also supports a local `SMOKE=1` dry-run on CPU. |
| `kernel-metadata.json` | Kernel config — carries the three-field RTX 6000 Pro gate. |
| `dataset-metadata.json` | Config for the `crypto-volatility-bundle` dataset (code + data). |
| `stage_and_deploy.sh` | Builds `build/`, uploads the dataset, pushes the kernel. |

## 1. Get a Kaggle API token

1. Go to <https://www.kaggle.com/settings> -> **API** -> **Create New Token**.
2. This downloads `kaggle.json`. Place it and lock it down:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

The account username is **nguyenduongtrong** (matches the dataset/kernel ids).

## 2. Deploy

```bash
bash kaggle/stage_and_deploy.sh
```

This stages `kaggle/build/code` (`config.py` + `ml/` + `requirements.txt`) and
`kaggle/build/data` (the 5 CSVs, ~150 MB), creates or versions the
`nguyenduongtrong/crypto-volatility-bundle` dataset, then pushes the
`nguyenduongtrong/crypto-volatility-lstm` kernel. Re-run with `--update` to
force a new dataset version after changing code or data.

The data source defaults to `/home/nduong/eth-alpha/data`; override with
`SRC=/path/to/data bash kaggle/stage_and_deploy.sh`.

## 3. The RTX 6000 Pro three-field gate (read this)

The RTX 6000 Pro pool is gated behind the Nemotron competition compute grant.
`kernel-metadata.json` MUST contain **all three** of these, or Kaggle silently
allocates a Tesla P100 (sm_60) — which has no kernel image in torch 2.10+ and
fails with `CUDA error: no kernel image is available for execution on the
device`:

```json
{
  "machine_shape": "NvidiaRtxPro6000",
  "enable_gpu": true,
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]
}
```

Notes:
- `enable_gpu` is a **boolean** `true`, not the string `"true"`.
- `competition_sources` is the non-obvious gate: same kernel + same
  `machine_shape` without the competition link still downgrades to P100.
- These kernels run with `enable_internet: false`. torch/pandas/numpy are
  already on the Kaggle image, so nothing is pip-installed at run time.

`train_kernel.py` enforces this in code: it prints the compute capability and
**aborts with `exit 1` on sm_60**, so a silent downgrade can never pass as a
successful run.

## 4. Quota and verifying sm_120 after a run

- **Quota: 30 hours/week** of RTX 6000 Pro per account, weekly reset. If
  exhausted, Kaggle falls back to P100 **even with correct config** — so always
  verify the allocated GPU from the log, never assume.

Pull the kernel output and grep the log:

```bash
kaggle kernels output nguyenduongtrong/crypto-volatility-lstm -p kaggle/out
grep -E "sm_1|GPU|device" kaggle/out/*.log
```

You want to see `sm_120` and the RTX 6000 Pro device name. `sm_60` means the
gate failed or the quota ran out. On sm_80+ the run uses bf16 autocast; on
sm_70/75 it uses fp16 + GradScaler.

## 5. No internet -> no live sentiment (expected)

The kernel has no internet, so FinBERT / news sentiment is **not** computed
during training. This is fine and by design: the historical feature builder
sets `sentiment_score = 0` for every window (there is no historical news), and
the live serving pipeline supplies the real sentiment signal at inference time.
The model trains on the same 11-feature layout either way.

## Local smoke test (no Kaggle needed)

Validate the whole orchestration on CPU against the local repo and data:

```bash
cd /home/nduong/dev/bigdata
SMOKE=1 python kaggle/train_kernel.py
```

It prints GPU diagnostics (CUDA unavailable on a CPU box — it does **not**
abort, only the specific sm_60 case aborts), trains a tiny 2-epoch model, and
saves `/tmp/agent1_smoke.pt`. Evaluation runs if `ml/evaluate.py` exists and is
skipped gracefully otherwise.
