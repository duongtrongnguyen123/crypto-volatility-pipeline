#!/usr/bin/env bash
set -u; cd /home/nduong/dev/bigdata; KG=.venv/bin/kaggle; ACC=accounts.txt
JOBS=(
 "1:nguyenduongtrong/crypto-graphrag-chains:chains_crypto"
 "9:dfbjdsbds/rag-s5:rag_s5" "10:truongdv006/rag-s6:rag_s6"
 "11:tuetrandoanminh/rag-s7:rag_s7" "12:tuananhtran37/rag-s8:rag_s8"
 "13:phmthanhlm24022379/rag-s9:rag_s9"
 "18:annguyncng/rag-f1:rag_f1" "19:dungnguyenhuy/rag-f2:rag_f2"
 "20:chongnh/rag-f3:rag_f3" "21:khunht/rag-f4:rag_f4" "22:dnglethnh/rag-f5:rag_f5"
)
setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token; printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1(){ cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }
for cyc in $(seq 1 110); do r=0
  for j in "${JOBS[@]}"; do IFS=: read -r n slug tag <<< "$j"; setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1)
    case "$st" in *RUNNING*|*QUEUED*) r=$((r+1));; esac; done
  echo "--- cyc $cyc: $r/${#JOBS[@]} running ---"
  if [ "$r" -eq 0 ]; then echo "=== ALL DONE — fetching ==="
    for j in "${JOBS[@]}"; do IFS=: read -r n slug tag <<< "$j"; setauth "$n"
      o="kaggle/out_${tag}"; rm -rf "$o"; mkdir -p "$o"; $KG kernels output "$slug" -p "$o" >/dev/null 2>&1
      echo "[$tag] $(ls $o 2>/dev/null|grep -c json) json"; done
    restore1; echo "=== FETCH COMPLETE ==="; break; fi
  restore1; sleep 150
done
