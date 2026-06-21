#!/usr/bin/env bash
# SERIAL launcher for the full-corpus 2016-2023 stock backtest.
# 20 live accounts × 2 notebooks = 40 shards (cb{j} base + cr{j} rag per window j).
# Each account gets ONE shared dataset (flat: stocknews.csv + 6 price CSVs) and
# pushes its two kernels. MUST be the ONLY process touching ~/.kaggle while it
# runs (concurrent cred-swappers corrupt which account a push lands on).
#
# Usage:
#   bash kaggle/launch_corpus.sh           # launch ALL 20 accounts
#   bash kaggle/launch_corpus.sh TESTONE   # launch only the first account (de-risk)
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt
MODE="${1:-ALL}"
# live account lines (dead 5,6,8,16 excluded) -> windows 0..19 in order
LINES="1 2 3 4 7 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24"
DSID="stock-corpus-2016-2023"
MODEL="qwen-lm/qwen2.5/transformers/32b-instruct/1"
COMP="nvidia-nemotron-model-reasoning-challenge"

# stash acct1 to restore at the end
cp ~/.kaggle/access_token /tmp/ck_tok 2>/dev/null || true
cp ~/.kaggle/kaggle.json  /tmp/ck_json 2>/dev/null || true

setauth(){ local n=$1 u k
  u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json
  echo "$u"; }

j=0
for n in $LINES; do
  u=$(setauth "$n")
  who=$($KG config view 2>&1 | grep -i username | awk '{print $NF}')
  if [ "$who" != "$u" ]; then echo "[w$j/$u] AUTH FAIL (got '$who') — SKIP"; j=$((j+1)); continue; fi
  echo "=== window $j on $u ==="

  # dataset (idempotent): upload the flat corpus dataset if not already present
  dd="kaggle/dsc_$u"; rm -rf "$dd"; mkdir -p "$dd"; cp kaggle/sd_corpus/* "$dd"/
  printf '{"title":"stock corpus 2016-2023","id":"%s/%s","licenses":[{"name":"other"}],"isPrivate":true}\n' "$u" "$DSID" > "$dd/dataset-metadata.json"
  if $KG datasets files "$u/$DSID" 2>/dev/null | grep -q stocknews; then
    # content changed (portfolio-filtered) -> push a NEW VERSION; kernels pick up latest
    echo "  [ds/$u] new version: $($KG datasets version -p "$dd" -m "portfolio-filtered" --dir-mode zip 2>&1 | tail -1)"
    sleep 25
  else
    echo "  [ds/$u] $($KG datasets create -p "$dd" --dir-mode zip 2>&1 | tail -1)"
    for w in $(seq 1 25); do sleep 6
      $KG datasets files "$u/$DSID" 2>/dev/null | grep -q stocknews && { echo "  [ds/$u] ready"; break; }
    done
  fi

  for cfg in cb cr; do
    tag="$cfg$j"; kid="sc-$tag"          # kernel slug (>=5 chars)
    st=$($KG kernels status "$u/$kid" 2>&1 | tail -1)
    # re-run COMPLETE/ERROR kernels (a push = new version = re-run); only skip in-flight
    case "$st" in *RUNNING*|*QUEUED*) echo "  [$kid/$u] $st — SKIP (in flight)"; continue;; esac
    pd="kaggle/pk_$tag"; rm -rf "$pd"; mkdir -p "$pd"; cp "kaggle/cshards/$tag.py" "$pd/$tag.py"
    cat > "$pd/kernel-metadata.json" <<JSON
{"id":"$u/$kid","title":"$kid","code_file":"$tag.py","language":"python","kernel_type":"script","is_private":true,"enable_gpu":true,"enable_tpu":false,"enable_internet":false,"machine_shape":"NvidiaRtxPro6000","competition_sources":["$COMP"],"dataset_sources":["$u/$DSID"],"kernel_sources":[],"model_sources":["$MODEL"]}
JSON
    echo "  [$kid/$u] $($KG kernels push -p "$pd" 2>&1 | tail -1)"
  done

  j=$((j+1))
  if [ "$MODE" = "TESTONE" ]; then echo "TESTONE: stopping after first account"; break; fi
done

# restore acct1 auth
cp /tmp/ck_tok ~/.kaggle/access_token 2>/dev/null || true
cp /tmp/ck_json ~/.kaggle/kaggle.json 2>/dev/null || true
echo "DONE restored: $($KG config view 2>&1 | grep -i username | awk '{print $NF}')"
