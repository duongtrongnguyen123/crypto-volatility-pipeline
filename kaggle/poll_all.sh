#!/usr/bin/env bash
# Unified poller for ALL active kernels: 4 RAG-vs-baseline + 9 backtest shards.
# Single process, swaps ~/.kaggle/access_token per check, restores acct1 at rest.
# MUST be the only process touching ~/.kaggle. Fetches every output when all done.
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt

# lineno : slug : tag
JOBS=(
  "1:nguyenduongtrong/crypto-trr-rag:crypto_rag"
  "2:zhongzhing/crypto-trr-base:crypto_base"
  "3:hduong/stock-trr-rag:stock_rag"
  "4:truongdinhduc06/stock-trr-base:stock_base"
  "14:leehoangquan006/stock-shard-s1:s1"
  "15:hiulnho/stock-shard-s2:s2"
  "7:namphmbuwu/stock-shard-s3:s3"
  "17:annabee12/stock-shard-s4:s4"
  "9:dfbjdsbds/stock-shard-s5:s5"
  "10:truongdv006/stock-shard-s6:s6"
  "11:tuetrandoanminh/stock-shard-s7:s7"
  "12:tuananhtran37/stock-shard-s8:s8"
  "13:phmthanhlm24022379/stock-shard-s9:s9"
)

setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1(){ cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }

for cycle in $(seq 1 120); do
  running=0
  for j in "${JOBS[@]}"; do
    IFS=: read -r n slug tag <<< "$j"
    setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1)
    short=$(echo "$st" | grep -oE "RUNNING|QUEUED|COMPLETE|ERROR|CANCEL|[A-Za-z]+$" | head -1)
    echo "[cyc $cycle][$tag] $short"
    case "$st" in *RUNNING*|*QUEUED*) running=$((running+1));; esac
  done
  echo "--- cycle $cycle: $running still running ---"
  if [ "$running" -eq 0 ]; then
    echo "=== ALL DONE — fetching outputs ==="
    for j in "${JOBS[@]}"; do
      IFS=: read -r n slug tag <<< "$j"
      setauth "$n"
      out="kaggle/out_${tag}"; rm -rf "$out"; mkdir -p "$out"
      $KG kernels output "$slug" -p "$out" >/dev/null 2>&1
      echo "[$tag] $(ls $out 2>/dev/null | tr '\n' ' ')"
    done
    restore1
    echo "=== FETCH COMPLETE ==="
    break
  fi
  restore1
  sleep 150
done
