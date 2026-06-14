#!/usr/bin/env bash
#
# Stage the TRR crash-detection code + data into a private Kaggle dataset, then
# push the kernel that runs the four-phase TRR pipeline with the NVIDIA Nemotron
# model on the RTX 6000 Pro GPU (zero-shot, no internet).
#
# PREREQUISITE: a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com/settings
# -> API -> Create New Token), chmod 600. Without it the kaggle CLI cannot
# authenticate and every command below fails.
#
# The Nemotron model is set in trr-kernel-metadata.json's model_sources
# (metric/nemotron-3-nano-30b-a3b-bf16/transformers/default) and the real news is
# attached via the oliviervha/crypto-news dataset_source — see kaggle/TRR_README.md.
# The kernel auto-detects the model dir at runtime; Kaggle mounts it from
# model_sources.
#
# Usage:
#   bash kaggle/deploy_trr.sh            # auto: create dataset, else version it
#   bash kaggle/deploy_trr.sh --update   # force a new dataset VERSION
#   SRC=/some/other/data bash kaggle/deploy_trr.sh
#
# The REAL news comes from the attached oliviervha/crypto-news dataset
# (cryptonews.csv), listed in trr-kernel-metadata.json's dataset_sources — NOT
# from this bundle. The bundle therefore carries only code + price CSVs. The
# synthetic trr/sample_news.jsonl still ships inside the code/ package as the
# kernel's SMOKE fallback.
set -euo pipefail

# --- Resolve paths (script lives in <repo>/kaggle) --------------------------
KAGGLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${KAGGLE_DIR}/.." && pwd)"
BUILD_DIR="${KAGGLE_DIR}/build_trr"
CODE_DIR="${BUILD_DIR}/code"
DATA_DIR="${BUILD_DIR}/data"

# Source of the historical 5-min CSVs (override with SRC=...).
SRC="${SRC:-/home/nduong/eth-alpha/data}"

# The kaggle CLI is assumed to be on PATH for the user.
KAGGLE="${KAGGLE:-kaggle}"

# The 5 daily-close price CSVs trr.labels.build_portfolio needs (the 6 PORTFOLIO
# tickers; ETH is a symlink to the same file, so 6 reads / 5 distinct files +
# whatever the loader resolves). We stage all six *_5min_long.csv explicitly.
CSVS=(
  "BTCUSDT_5min_long.csv"
  "ETHUSDT_5min_long.csv"
  "SOLUSDT_5min_long.csv"
  "BNBUSDT_5min_long.csv"
  "AVAXUSDT_5min_long.csv"
  "DOGEUSDT_5min_long.csv"
)

FORCE_UPDATE=0
[[ "${1:-}" == "--update" ]] && FORCE_UPDATE=1

# --- 1. Build the staging dir -----------------------------------------------
echo "[stage] cleaning ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${CODE_DIR}" "${DATA_DIR}"

echo "[stage] copying code -> ${CODE_DIR}"
cp "${REPO_DIR}/config.py" "${CODE_DIR}/"
cp -r "${REPO_DIR}/ml" "${CODE_DIR}/ml"
# The whole trr/ package INCLUDING sample_news.jsonl.
cp -r "${REPO_DIR}/trr" "${CODE_DIR}/trr"
if [[ -f "${REPO_DIR}/requirements.txt" ]]; then
  cp "${REPO_DIR}/requirements.txt" "${CODE_DIR}/requirements.txt"
fi
# Drop compiled caches so the upload stays lean.
find "${CODE_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "[stage] copying price data from ${SRC} -> ${DATA_DIR}"
for csv in "${CSVS[@]}"; do
  if [[ ! -f "${SRC}/${csv}" ]]; then
    echo "[stage] ERROR: missing ${SRC}/${csv}" >&2
    exit 1
  fi
  cp "${SRC}/${csv}" "${DATA_DIR}/"
  echo "[stage]   + ${csv}"
done

# NOTE: the news is NOT staged into the bundle — the real run reads the attached
# oliviervha/crypto-news dataset (cryptonews.csv), and the synthetic fallback
# travels inside code/trr/sample_news.jsonl for SMOKE.

# The dataset metadata must sit at the ROOT of the upload dir.
cat > "${BUILD_DIR}/dataset-metadata.json" <<'JSON'
{
  "title": "Crypto TRR Bundle (code + price data)",
  "id": "nguyenduongtrong/crypto-trr-bundle",
  "licenses": [
    {
      "name": "other"
    }
  ],
  "isPrivate": true
}
JSON

echo "[stage] staged contents:"
find "${BUILD_DIR}" -maxdepth 2 -type f | sort

# --- 2. Create or version the bundle dataset --------------------------------
# --dir-mode zip preserves the code/ and data/ subdirs so the kernel mounts them
# at /kaggle/input/<slug>/code and /data.
if [[ "${FORCE_UPDATE}" -eq 1 ]]; then
  echo "[deploy] pushing new dataset VERSION (--update)"
  "${KAGGLE}" datasets version -p "${BUILD_DIR}" -m "update" --dir-mode zip
else
  echo "[deploy] attempting dataset CREATE (falls back to version if it exists)"
  if "${KAGGLE}" datasets create -p "${BUILD_DIR}" --dir-mode zip; then
    echo "[deploy] dataset created"
  else
    echo "[deploy] create failed (dataset likely already exists) -> versioning"
    "${KAGGLE}" datasets version -p "${BUILD_DIR}" -m "update" --dir-mode zip
  fi
fi

# --- 3. Push the kernel -----------------------------------------------------
# `kaggle kernels push` expects the metadata file to be named exactly
# kernel-metadata.json, so stage it + the kernel script into a temp dir.
PUSH_DIR="$(mktemp -d)"
trap 'rm -rf "${PUSH_DIR}"' EXIT
cp "${KAGGLE_DIR}/trr_kernel.py" "${PUSH_DIR}/trr_kernel.py"
cp "${KAGGLE_DIR}/trr-kernel-metadata.json" "${PUSH_DIR}/kernel-metadata.json"

echo "[deploy] pushing kernel from ${PUSH_DIR}"
"${KAGGLE}" kernels push -p "${PUSH_DIR}"

# --- 4. Verification hint ---------------------------------------------------
cat <<'EOF'

[deploy] done.

model_sources is set to metric/nemotron-3-nano-30b-a3b-bf16/transformers/default
and the news is attached via the oliviervha/crypto-news dataset (see
kaggle/TRR_README.md). The run is bounded to the FTX window by default; widen it
with the TRR_START / TRR_END kernel env vars for the full run. After the kernel
runs on Kaggle, pull its output and confirm the RTX 6000 Pro (sm_120) was
allocated — NOT a silent P100 (sm_60) fallback:

  kaggle kernels output nguyenduongtrong/crypto-trr-nemotron -p kaggle/out
  grep -E "sm_1|compute_capability|device" kaggle/out/*.log

You want "sm_120" / the RTX 6000 Pro device name. "sm_60" means the three-field
hardware gate failed or the 30 hrs/week RTX 6000 Pro quota was exhausted.
The kernel writes trr_predictions.csv + eval_results.json + the ROC/timeline
plots into the kernel output.
EOF
