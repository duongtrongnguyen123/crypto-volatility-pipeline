#!/usr/bin/env bash
set -u; cd /home/nduong/dev/bigdata; KG=.venv/bin/kaggle; ACC=accounts.txt
JOBS=("1:nguyenduongtrong/crypto-graphrag:crypto_graphrag" "2:zhongzhing/crypto-base-gr:crypto_base_gr")
setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token; printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1(){ cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }
for cyc in $(seq 1 90); do r=0
  for j in "${JOBS[@]}"; do IFS=: read -r n slug tag <<< "$j"; setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1); echo "[cyc $cyc][$tag] $(echo "$st"|grep -oE 'RUNNING|QUEUED|COMPLETE|ERROR'|head -1)"
    case "$st" in *RUNNING*|*QUEUED*) r=$((r+1));; esac; done
  echo "--- cyc $cyc: $r running ---"
  if [ "$r" -eq 0 ]; then echo "=== DONE — fetching ==="
    for j in "${JOBS[@]}"; do IFS=: read -r n slug tag <<< "$j"; setauth "$n"
      o="kaggle/out_${tag}"; rm -rf "$o"; mkdir -p "$o"; $KG kernels output "$slug" -p "$o" >/dev/null 2>&1
      echo "[$tag] $(ls $o 2>/dev/null|tr '\n' ' ')"; done
    restore1; echo "=== FETCH COMPLETE ==="; break; fi
  restore1; sleep 150
done
