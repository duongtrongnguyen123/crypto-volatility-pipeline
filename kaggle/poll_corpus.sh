#!/usr/bin/env bash
# Unified poller/collector for the 40 corpus shards. Cycles every live account,
# checks its two kernels, and downloads any that are COMPLETE into
# kaggle/out_corpus/<tag>/. ONE cred-swapper at a time (don't run with launcher).
#
# Usage: bash kaggle/poll_corpus.sh         # one pass
#        bash kaggle/poll_corpus.sh loop    # repeat until all 40 downloaded
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt
LINES="1 2 3 4 7 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24"
OUT=kaggle/out_corpus; mkdir -p "$OUT"

cp ~/.kaggle/access_token /tmp/ck_tok 2>/dev/null || true
cp ~/.kaggle/kaggle.json  /tmp/ck_json 2>/dev/null || true
setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; echo "$u"; }

one_pass(){
  local done=0 total=0 j=0
  for n in $LINES; do
    u=$(setauth "$n"); j_in=$j
    for cfg in cb cr; do
      tag="$cfg$j_in"; kid="sc-$tag"; total=$((total+1))
      if [ -f "$OUT/$tag/crash/trr_predictions.csv" ]; then done=$((done+1)); continue; fi
      st=$($KG kernels status "$u/$kid" 2>&1 | tail -1)
      case "$st" in
        *COMPLETE*)
          mkdir -p "$OUT/$tag"
          $KG kernels output "$u/$kid" -p "$OUT/$tag" >/dev/null 2>&1
          [ -f "$OUT/$tag/crash/trr_predictions.csv" ] && { echo "[$tag/$u] downloaded"; done=$((done+1)); } \
            || echo "[$tag/$u] COMPLETE but no predictions yet"
          ;;
        *RUNNING*|*QUEUED*) echo "[$tag/$u] $st" ;;
        *ERROR*|*CANCEL*)   echo "[$tag/$u] !! $st" ;;
        *) echo "[$tag/$u] $st" ;;
      esac
    done
    j=$((j+1))
  done
  cp /tmp/ck_tok ~/.kaggle/access_token 2>/dev/null || true
  cp /tmp/ck_json ~/.kaggle/kaggle.json 2>/dev/null || true
  echo "=== $done/$total downloaded ==="
  [ "$done" -ge "$total" ] && return 0 || return 1
}

if [ "${1:-once}" = "loop" ]; then
  while ! one_pass; do echo "--- sleeping 120s ---"; sleep 120; done
  echo "ALL SHARDS DOWNLOADED"
else
  one_pass
fi
