#!/usr/bin/env bash
set -u; cd /home/nduong/dev/bigdata; KG=.venv/bin/kaggle
SLUG=nguyenduongtrong/crypto-graphrag-chains
for cyc in $(seq 1 90); do
  st=$($KG kernels status "$SLUG" 2>&1 | tail -1)
  echo "[cyc $cyc] $(echo "$st"|grep -oE 'RUNNING|QUEUED|COMPLETE|ERROR'|head -1)"
  case "$st" in *RUNNING*|*QUEUED*) sleep 150;; *)
    echo "=== DONE — fetching ==="; rm -rf kaggle/out_crypto_graphrag_chains
    mkdir -p kaggle/out_crypto_graphrag_chains
    $KG kernels output "$SLUG" -p kaggle/out_crypto_graphrag_chains >/dev/null 2>&1
    echo "fetched: $(ls kaggle/out_crypto_graphrag_chains|tr '\n' ' ')"; break;; esac
done
