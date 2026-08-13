#!/usr/bin/env bash
# 512-graph lottery discriminator: 3 independent engine boots, each running
# only the C=512 and C=513 windows (decode config identical to the formal
# collection cell and to the L3 decode phase). Full stack (etcd/nats/frontend/
# listener/engine) is bounced per boot so every boot is an independent draw;
# each boot writes its own stream file.
set -uo pipefail
DP=4
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
WORKDIR=/tmp/fpm-serve
cd /workspace/examples/backends/vllm

stop_stack() {
  pkill -9 -f "dynamo[.]vllm" 2>/dev/null
  pkill -9 -f "VLLM[:]:" 2>/dev/null
  pkill -9 -f "dynamo[.]frontend" 2>/dev/null
  pkill -9 -f "fpm_listener[.]py" 2>/dev/null
  pkill -9 -f "etcd --data-dir" 2>/dev/null
  pkill -9 -f "nats-server" 2>/dev/null
  sleep 8
}

for boot in 1 2 3; do
  echo "BOOT${boot}_START=$(date +%s)" >> /results/probe_timing.log
  stop_stack
  rm -rf /tmp/fpm-forward-etcd
  etcd --data-dir /tmp/fpm-forward-etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://127.0.0.1:2379 \
    --listen-peer-urls http://127.0.0.1:2380 \
    >/results/etcd_b${boot}.log 2>&1 &
  nats-server >/results/nats_b${boot}.log 2>&1 &
  sleep 3
  curl -sf http://127.0.0.1:2379/health >/dev/null || { echo "BOOT${boot}_ETCD_FAIL" >> /results/probe_timing.log; continue; }
  DYN_HTTP_PORT=8000 python3 -m dynamo.frontend >/results/frontend_b${boot}.log 2>&1 &
  STREAM=/results/fpm_stream_b${boot}.jsonl
  python3 "$WORKDIR/fpm_listener.py" "$STREAM" >/results/listener_b${boot}.log 2>&1 &
  export DYN_FORWARDPASS_METRIC_PORT=20380
  bash "$WORKDIR/serve_run_dep4_decode.sh" >/results/engine_b${boot}.log 2>&1 &
  ENGINE_PID=$!
  ready=0
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && { ready=1; break; }
    kill -0 $ENGINE_PID 2>/dev/null || break
    sleep 5
  done
  [ $ready -eq 1 ] || { echo "BOOT${boot}_ENGINE_FAIL" >> /results/probe_timing.log; tail -5 /results/engine_b${boot}.log; continue; }
  echo "BOOT${boot}_READY=$(date +%s)" >> /results/probe_timing.log
  for C in 512 513; do
    POOL=$((C*DP))
    s0=$(wc -l < "$STREAM" 2>/dev/null || echo 0)
    vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
      --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
      --dataset-name random --random-input-len 1 --random-output-len 1055 \
      --random-range-ratio 0 --num-prompts $POOL --max-concurrency $POOL \
      --request-rate inf --ignore-eos --seed $((RANDOM)) >/results/pw_b${boot}_${C}.log 2>&1
    e0=$(wc -l < "$STREAM")
    printf 'B%s\t%s\t1\t1055\t探针\t%s\t%s\n' "$boot" "$C" "$s0" "$e0" >> /results/probe_windows.tsv
    echo "BOOT${boot}_C${C}_DONE=$(date +%s) ($s0..$e0)" >> /results/probe_timing.log
  done
done
stop_stack
echo "PROBE_DONE=$(date +%s)" >> /results/probe_timing.log
