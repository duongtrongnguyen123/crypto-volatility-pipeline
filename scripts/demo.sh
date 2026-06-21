#!/usr/bin/env bash
# Pre-demo prep: generate figures, export the meta-model, run the test suite,
# and print the exact commands for the live demo (see docs/DEMO.md).
# Local only — no internet/GPU needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python

echo "==> exporting trained meta-model (models/trr_meta.pkl)"; $PY -m train.export | tail -1
echo "==> generating figures (reports/figures/)";              $PY -m train.figures | tail -1
echo "==> regression guard (full test suite)";                 $PY -m pytest tests/ serving/tests/ -q | tail -1
cat <<'EOF'

================  READY TO DEMO  (see docs/DEMO.md)  ================
Terminal A — web platform:
  .venv/bin/streamlit run webapp/app.py            # http://localhost:8501
Terminal B — serving API:
  .venv/bin/uvicorn serving.api:app --port 8000
Terminal C — streaming speed layer:
  .venv/bin/python scripts/demo_streaming.py --messages 40 --rate 250

Live prediction (with API running):
  curl -s localhost:8000/predict -H 'Content-Type: application/json' \
    -d '{"headlines":[{"title":"Exchange halts withdrawals; contagion; liquidations","assets":["BTC","ETH"]}]}' | python -m json.tool
====================================================================
EOF
