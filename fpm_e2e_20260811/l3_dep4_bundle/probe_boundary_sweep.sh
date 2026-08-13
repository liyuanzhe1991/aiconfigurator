#!/usr/bin/env bash
# Boundary kill-shot: one boot, four pool sizes bracketing the DP graph cliff.
# POOL=1984/2020 -> all ranks <=512 -> predict FULL-graph level (~60-62ms);
# POOL=2048 -> some rank >512 (router imbalance +-5) -> predict eager (~110);
# POOL=2044 -> coin-flip on the router draw; either level proves cliff width.
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
  >/results/etcd_b4.log 2>&1 &
nats-server >/results/nats_b4.log 2>&1 &
sleep 3
DYN_HTTP_PORT=8000 python3 -m dynamo.frontend >/results/frontend_b4.log 2>&1 &
STREAM=/results/fpm_stream_b4.jsonl
python3 "$WORKDIR/fpm_listener.py" "$STREAM" >/results/listener_b4.log 2>&1 &
export DYN_FORWARDPASS_METRIC_PORT=20380
bash "$WORKDIR/serve_run_dep4_decode.sh" >/results/engine_b4.log 2>&1 &
ENGINE_PID=$!
ready=0
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && { ready=1; break; }
  kill -0 $ENGINE_PID 2>/dev/null || break
  sleep 5
done
[ $ready -eq 1 ] || { echo "BOOT4_ENGINE_FAIL" >> /results/probe_timing.log; exit 1; }
echo "BOOT4_READY=$(date +%s)" >> /results/probe_timing.log
for POOL in 1984 2020 2044 2048; do
  s0=$(wc -l < "$STREAM" 2>/dev/null || echo 0)
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
    --dataset-name random --random-input-len 1 --random-output-len 700 \
    --random-range-ratio 0 --num-prompts $POOL --max-concurrency $POOL \
    --request-rate inf --ignore-eos --seed $((RANDOM)) >/results/pw_b4_pool${POOL}.log 2>&1
  e0=$(wc -l < "$STREAM")
  printf 'B4\t%s\t1\t700\t池扫\t%s\t%s\n' "$POOL" "$s0" "$e0" >> /results/probe_windows.tsv
  echo "BOOT4_POOL${POOL}_DONE=$(date +%s) ($s0..$e0)" >> /results/probe_timing.log
done
pkill -9 -f "dynamo[.]vllm" 2>/dev/null
echo "SWEEP_DONE=$(date +%s)" >> /results/probe_timing.log
