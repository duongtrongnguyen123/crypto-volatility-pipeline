#!/usr/bin/env bash
# Poller for the remaining 7 kernels: 2 crypto RAG/baseline + 5 FNSPID shards.
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt
JOBS=(
  "1:nguyenduongtrong/crypto-trr-rag:crypto_rag"
  "2:zhongzhing/crypto-trr-base:crypto_base"
  "18:annguyncng/fnspid-f1:f1"
  "19:dungnguyenhuy/fnspid-f2:f2"
  "20:chongnh/fnspid-f3:f3"
  "21:khunht/fnspid-f4:f4"
  "22:dnglethnh/fnspid-f5:f5"
)
setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token; printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1(){ cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }
for cycle in $(seq 1 120); do
  running=0
  for j in "${JOBS[@]}"; do
    IFS=: read -r n slug tag <<< "$j"; setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1)
    echo "[cyc $cycle][$tag] $(echo "$st"|grep -oE 'RUNNING|QUEUED|COMPLETE|ERROR|CANCEL'|head -1)"
    case "$st" in *RUNNING*|*QUEUED*) running=$((running+1));; esac
  done
  echo "--- cycle $cycle: $running still running ---"
  if [ "$running" -eq 0 ]; then
    echo "=== ALL DONE — fetching ==="
    for j in "${JOBS[@]}"; do
      IFS=: read -r n slug tag <<< "$j"; setauth "$n"
      out="kaggle/out_${tag}"; rm -rf "$out"; mkdir -p "$out"
      $KG kernels output "$slug" -p "$out" >/dev/null 2>&1
      echo "[$tag] $(ls $out 2>/dev/null | tr '\n' ' ')"
    done
    restore1; echo "=== FETCH COMPLETE ==="; break
  fi
  restore1; sleep 150
done
