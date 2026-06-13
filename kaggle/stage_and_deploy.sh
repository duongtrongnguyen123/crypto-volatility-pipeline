#!/usr/bin/env bash
#
# Stage the crypto-volatility code + historical data into a private Kaggle
# dataset, then push the training kernel that runs on the RTX 6000 Pro GPU.
#
# PREREQUISITE: a Kaggle API token must exist at ~/.kaggle/kaggle.json
# (kaggle.com/settings -> API -> Create New Token), chmod 600. Without it the
# kaggle CLI cannot authenticate and every command below fails.
#
# Usage:
#   bash kaggle/stage_and_deploy.sh            # auto: create dataset, else version it
#   bash kaggle/stage_and_deploy.sh --update   # force a new dataset VERSION
#   SRC=/some/other/data bash kaggle/stage_and_deploy.sh
#
set -euo pipefail

# --- Resolve paths (script lives in <repo>/kaggle) --------------------------
KAGGLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${KAGGLE_DIR}/.." && pwd)"
BUILD_DIR="${KAGGLE_DIR}/build"
CODE_DIR="${BUILD_DIR}/code"
DATA_DIR="${BUILD_DIR}/data"

# Source of the historical 5-min CSVs (override with SRC=...).
SRC="${SRC:-/home/nduong/eth-alpha/data}"

# The kaggle CLI is assumed to be on PATH for the user.
KAGGLE="${KAGGLE:-kaggle}"

# The 5 CSVs the offline feature builder (ml/historical.py) needs (~150MB).
CSVS=(
  "BTCUSDT_5min_long.csv"
  "BTCUSDT_metrics_full.csv"
  "BTCUSDT_funding.csv"
  "BTCUSDT_bookdepth_5min.csv"
  "ETHUSDT_liquidations_5min.csv"
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
# A minimal requirements list documents the runtime deps (already present on
# the Kaggle image; not pip-installed there since the kernel has no internet).
cp "${REPO_DIR}/requirements.txt" "${CODE_DIR}/requirements.txt"
# Drop compiled caches so the upload stays lean.
find "${CODE_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "[stage] copying data from ${SRC} -> ${DATA_DIR}"
for csv in "${CSVS[@]}"; do
  if [[ ! -f "${SRC}/${csv}" ]]; then
    echo "[stage] ERROR: missing ${SRC}/${csv}" >&2
    exit 1
  fi
  cp "${SRC}/${csv}" "${DATA_DIR}/"
  echo "[stage]   + ${csv}"
done

# The dataset metadata must sit at the ROOT of the upload dir.
cp "${KAGGLE_DIR}/dataset-metadata.json" "${BUILD_DIR}/dataset-metadata.json"

echo "[stage] staged contents:"
find "${BUILD_DIR}" -maxdepth 2 -type f | sort

# --- 2. Create or version the bundle dataset --------------------------------
# --dir-mode zip preserves the code/ and data/ subdirs as a single zip the
# kernel mounts at /kaggle/input/<slug>/.
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
echo "[deploy] pushing kernel from ${KAGGLE_DIR}"
"${KAGGLE}" kernels push -p "${KAGGLE_DIR}"

# --- 4. Verification hint ---------------------------------------------------
cat <<'EOF'

[deploy] done.

After the kernel finishes running on Kaggle, pull its output and confirm the
RTX 6000 Pro (sm_120) was allocated — NOT a silent P100 (sm_60) fallback:

  kaggle kernels output nguyenduongtrong/crypto-volatility-lstm -p kaggle/out
  grep -E "sm_1|GPU|device" kaggle/out/*.log

You want to see "sm_120" / the RTX 6000 Pro device name. "sm_60" means the
three-field hardware gate failed or the 30 hrs/week quota was exhausted.
EOF
