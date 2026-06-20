#!/usr/bin/env bash
# Idempotent SERIAL launcher for the 9 stock backtest shards (2016-2020).
# One account per shard; skips shards already RUNNING/QUEUED/COMPLETE.
# MUST be the only process touching ~/.kaggle while it runs (no concurrent poller).
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt

# tag:lineno  (s3 already launched on namphmbuwu = line 7)
MAP="s1:5 s2:6 s3:7 s4:8 s5:9 s6:10 s7:11 s8:12 s9:13"

setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; echo "$u"; }

for m in $MAP; do
  tag=${m%:*}; n=${m#*:}
  u=$(setauth $n)
  who=$($KG config view 2>&1 | grep -i username | awk '{print $NF}')
  if [ "$who" != "$u" ]; then echo "[$tag/$u] AUTH FAIL (got $who) — SKIP"; continue; fi

  st=$($KG kernels status "$u/stock-shard-$tag" 2>&1 | tail -1)
  case "$st" in
    *RUNNING*|*QUEUED*|*COMPLETE*) echo "[$tag/$u] already $st — SKIP"; continue;;
  esac

  # dataset (idempotent)
  dd="kaggle/wds_$tag"; rm -rf "$dd"; cp -r kaggle/build_stock_wide "$dd"
  printf '{"title":"stock wide %s","id":"%s/stock-trr-wide","licenses":[{"name":"other"}],"isPrivate":true}\n' "$tag" "$u" > "$dd/dataset-metadata.json"
  if ! $KG datasets files "$u/stock-trr-wide" 2>/dev/null | grep -q stocknews; then
    $KG datasets create -p "$dd" --dir-mode zip >/dev/null 2>&1
    for w in $(seq 1 16); do sleep 7; $KG datasets files "$u/stock-trr-wide" 2>/dev/null | grep -q stocknews && break; done
  fi

  pd="kaggle/ps_$tag"; rm -rf "$pd"; mkdir -p "$pd"; cp kaggle/shard_${tag}.py "$pd/shard_${tag}.py"
  cat > "$pd/kernel-metadata.json" <<JSON
{"id":"$u/stock-shard-$tag","title":"stock-shard-$tag","code_file":"shard_${tag}.py","language":"python","kernel_type":"script","is_private":true,"enable_gpu":true,"enable_tpu":false,"enable_internet":false,"machine_shape":"NvidiaRtxPro6000","competition_sources":["nvidia-nemotron-model-reasoning-challenge"],"dataset_sources":["$u/stock-trr-wide"],"kernel_sources":[],"model_sources":["qwen-lm/qwen2.5/transformers/32b-instruct/1"]}
JSON
  echo "[$tag/$u] $($KG kernels push -p "$pd" 2>&1 | tail -1)"
done

cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json
echo "DONE restored: $($KG config view 2>&1 | grep -i username | awk '{print $NF}')"
