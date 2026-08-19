#!/usr/bin/env bash
# 内容对拍相位:serve 栈 + decode-parity 引擎(kit 同款,prefix caching 关)
# ABA 三遍:sharegpt -> random -> sharegpt2(控漂移)
set -u
cd /workspace/examples/backends/vllm
mkdir -p /results/content
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "dynamo[.]frontend" 2>/dev/null; pkill -9 -f "fpm_[l]istener" 2>/dev/null
pkill -9 -f "etcd --data-dir /tmp/fpm-forward-[e]tcd" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
sleep 3
for _ in $(seq 1 45); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break; sleep 4
done
rm -rf /tmp/fpm-forward-etcd
etcd --data-dir /tmp/fpm-forward-etcd --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 --listen-peer-urls http://127.0.0.1:2380 \
  > /results/content/etcd.log 2>&1 &
nats-server > /results/content/nats.log 2>&1 &
sleep 3
curl -sf http://127.0.0.1:2379/health >/dev/null || { echo CONTENT-FAIL-etcd; exit 4; }
DYN_HTTP_PORT=8000 python3 -m dynamo.frontend > /results/content/frontend.log 2>&1 &
python3 /tmp/fpm-serve/fpm_listener.py /results/content/fpm_stream.jsonl > /results/content/listener.log 2>&1 &
export DYN_FORWARDPASS_METRIC_PORT=20380
bash /tmp/fpm-serve/serve_run_tep4_decode.sh > /results/content/engine.log 2>&1 &
READY=0
for i in $(seq 1 200); do
  curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && { READY=1; break; }
  sleep 5
done
[ "$READY" = 1 ] || { echo CONTENT-FAIL-engine-ready; tail -20 /results/content/engine.log; exit 5; }
echo CONTENT-ENGINE-READY
export L3_TOKENIZER=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
export L3_SHAREGPT_PATH=/workspace/model_cache/fpm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
for PASS in sharegpt random sharegpt2; do
  CONTENT=${PASS%2}
  echo "== pass $PASS (content=$CONTENT)"
  L3_CONTENT=$CONTENT L3_PASS=$PASS python3 /tmp/fpm-serve/content_decode_driver.py \
    /tmp/fpm-serve/content_plan.csv /results/content/fpm_stream.jsonl \
    /results/content/content_windows.tsv 2>&1 | tee -a /results/content/driver.log
done
echo CONTENT-PHASE-DONE
