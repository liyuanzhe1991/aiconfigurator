#!/usr/bin/env bash
# Ground-truth serving stack (channel A): etcd + nats + dynamo.frontend +
# dynamo.vllm worker (same engine args as the formal collection, benchmark
# mode OFF, FPM stream ON) + ZMQ listener dumping per-step metrics.
set -uo pipefail
WORKDIR=/tmp/fpm-serve

# 引擎 boot 前排空守卫(方法论:显存 <1GB 才放行)
for _ in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break
  sleep 4
done

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
bash "$WORKDIR/serve_run_tp2_prefill.sh" >/results/engine.log 2>&1 &
ENGINE_PID=$!

echo "stack up: etcd+nats+frontend+listener+engine(pid=$ENGINE_PID); waiting for model ready"
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax; then
    # 模型入列表不等于路由可用(worker 实例注册有滞后窗口,期间 completions 404)
    # ——必须真打一条 completions 探针等 200 才算就绪
    for j in $(seq 1 60); do
      CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8000/v1/completions         -H "Content-Type: application/json"         -d "{\"model\":\"MiniMaxAI/MiniMax-M2.7\",\"prompt\":\"hi\",\"max_tokens\":1}")
      if [ "$CODE" = "200" ]; then
        echo "ENGINE-READY after ${i}0s (+completions probe ${j}0s)"
        exit 0
      fi
      sleep 10
    done
    echo "COMPLETIONS-PROBE-TIMEOUT (模型已入列表但路由 10 分钟未通)"; exit 1
  fi
  kill -0 $ENGINE_PID 2>/dev/null || { echo "ENGINE-DIED"; tail -20 /results/engine.log; exit 1; }
  sleep 10
done
echo "ENGINE-READY-TIMEOUT"; exit 1
