#!/usr/bin/env bash
# Poll the 4 RAG-vs-baseline kernels (crypto + stock, RAG on/off) across 4
# accounts. Swaps ~/.kaggle/access_token per account; restores acct1 at rest.
# When all are terminal, pulls each output to kaggle/out_<tag>/ and exits.
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
)

setauth() { local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1() { cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }

for cycle in $(seq 1 90); do
  running=0
  for j in "${JOBS[@]}"; do
    IFS=: read -r n slug tag <<< "$j"
    setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1)
    echo "[cyc $cycle][$tag] $st"
    case "$st" in *RUNNING*|*QUEUED*) running=$((running+1));; esac
  done
  echo "--- cycle $cycle: $running still running ---"
  if [ "$running" -eq 0 ]; then
    echo "=== ALL DONE — fetching outputs ==="
    for j in "${JOBS[@]}"; do
      IFS=: read -r n slug tag <<< "$j"
      setauth "$n"
      out="kaggle/out_${tag}"; rm -rf "$out"; mkdir -p "$out"
      $KG kernels output "$slug" -p "$out" 2>&1 | tail -1
      echo "[$tag] $(ls $out 2>/dev/null | tr '\n' ' ')"
    done
    restore1
    echo "=== FETCH COMPLETE ==="
    break
  fi
  restore1
  sleep 150
done
