#!/usr/bin/env bash
# FIX validation: decode engine with the capture list extended to 2048 (the
# prefill config already does this), so the graph boundary moves beyond any
# reachable per-rank batch. Rerun the cliff windows: POOL 2048 and 2052.
# Prediction if the DP-straddle mechanism is right: both drop from the eager
# level (~110ms) to the FULL-graph level (~62-66ms).
set -uo pipefail
DP=4
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
WORKDIR=/tmp/fpm-serve
cd /workspace/examples/backends/vllm

pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "dynamo[.]frontend" 2>/dev/null; pkill -9 -f "fpm_listener[.]py" 2>/dev/null
pkill -9 -f "etcd --data-dir" 2>/dev/null; pkill -9 -f "nats-server" 2>/dev/null
sleep 8
rm -rf /tmp/fpm-forward-etcd
etcd --data-dir /tmp/fpm-forward-etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  --listen-peer-urls http://127.0.0.1:2380 \
  >/results/etcd_fix.log 2>&1 &
nats-server >/results/nats_fix.log 2>&1 &
sleep 3
DYN_HTTP_PORT=8000 python3 -m dynamo.frontend >/results/frontend_fix.log 2>&1 &
STREAM=/results/fpm_stream_fix.jsonl
python3 "$WORKDIR/fpm_listener.py" "$STREAM" >/results/listener_fix.log 2>&1 &
export DYN_FORWARDPASS_METRIC_PORT=20380

source "$WORKDIR/fpm_env.sh"
export HF_HOME=$TOK
export FPM_RUN_ID=fpm-fix-capture
export DYN_FPM_WORKER_ID="${FPM_RUN_ID}-node0"
CAPTURE='{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024,1056,1088,1120,1152,1184,1216,1248,1280,1312,1344,1376,1408,1440,1472,1504,1536,1568,1600,1632,1664,1696,1728,1760,1792,1824,1856,1888,1920,1952,1984,2016,2048],"max_cudagraph_capture_size":2048}'
setsid python3 -m dynamo.vllm \
  --model "$TOK" --served-model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 --distributed-executor-backend mp \
  --distributed-timeout-seconds 1800 --no-enable-log-requests \
  --no-enable-prefix-caching --max-model-len -1 \
  --compilation-config "$CAPTURE" \
  --data-parallel-backend mp \
  --dump-config-to /results/resolved-config-fix.json \
  >/results/engine_fix.log 2>&1 &
ENGINE_PID=$!
ready=0
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && { ready=1; break; }
  kill -0 $ENGINE_PID 2>/dev/null || break
  sleep 5
done
[ $ready -eq 1 ] || { echo "FIX_ENGINE_FAIL" >> /results/probe_timing.log; tail -8 /results/engine_fix.log; exit 1; }
echo "FIX_READY=$(date +%s)" >> /results/probe_timing.log
for POOL in 2048 2052; do
  s0=$(wc -l < "$STREAM" 2>/dev/null || echo 0)
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
    --dataset-name random --random-input-len 1 --random-output-len 700 \
    --random-range-ratio 0 --num-prompts $POOL --max-concurrency $POOL \
    --request-rate inf --ignore-eos --seed $((RANDOM)) >/results/pw_fix_pool${POOL}.log 2>&1
  e0=$(wc -l < "$STREAM")
  printf 'FIX\t%s\t1\t700\t扩表\t%s\t%s\n' "$POOL" "$s0" "$e0" >> /results/probe_windows.tsv
  echo "FIX_POOL${POOL}_DONE=$(date +%s) ($s0..$e0)" >> /results/probe_timing.log
done
pkill -9 -f "dynamo[.]vllm" 2>/dev/null
echo "FIX_DONE=$(date +%s)" >> /results/probe_timing.log
