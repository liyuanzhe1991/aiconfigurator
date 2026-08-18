#!/usr/bin/env bash
# Ground-truth serving stack (channel A): etcd + nats + dynamo.frontend +
# dynamo.vllm worker (same engine args as the formal collection, benchmark
# mode OFF, FPM stream ON) + ZMQ listener dumping per-step metrics.
set -uo pipefail
WORKDIR=/tmp/fpm-serve

rm -rf /tmp/fpm-forward-etcd
etcd --data-dir /tmp/fpm-forward-etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  --listen-peer-urls http://127.0.0.1:2380 \
  >/results/etcd.log 2>&1 &
nats-server >/results/nats.log 2>&1 &
sleep 3
curl -sf http://127.0.0.1:2379/health || { echo "etcd not healthy"; exit 1; }

DYN_HTTP_PORT=8000 python3 -m dynamo.frontend >/results/frontend.log 2>&1 &

python3 "$WORKDIR/fpm_listener.py" /results/fpm_stream.jsonl >/results/listener.log 2>&1 &

# FPM stream on (legacy explicit-port path activates InstrumentedScheduler),
# benchmark mode off — every real serving step gets published.
export DYN_FORWARDPASS_METRIC_PORT=20380
bash "$WORKDIR/serve_run_tp4_prefill.sh" >/results/engine.log 2>&1 &
ENGINE_PID=$!

echo "stack up: etcd+nats+frontend+listener+engine(pid=$ENGINE_PID); waiting for model ready"
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax; then
    echo "ENGINE-READY after ${i}0s"
    exit 0
  fi
  kill -0 $ENGINE_PID 2>/dev/null || { echo "ENGINE-DIED"; tail -20 /results/engine.log; exit 1; }
  sleep 10
done
echo "ENGINE-READY-TIMEOUT"; exit 1
