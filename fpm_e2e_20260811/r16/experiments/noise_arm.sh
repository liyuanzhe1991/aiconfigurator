#!/usr/bin/env bash
# 噪声因果臂:quiet=复刻节点探针;noisy=引擎前注入 48 自旋 + 4 dd 系统调用负载
# 目的:在"快节点"C 上人为制造主机 CPU 争抢,看是否复现"慢节点"签名
set -uo pipefail
ARM=${1:?quiet|noisy}
OUT=/results/$ARM; mkdir -p "$OUT"
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "etcd --data-dir /tmp/fpm-forward-[e]tcd" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
[ -f /tmp/noise.pids ] && { xargs -r kill -9 < /tmp/noise.pids 2>/dev/null; rm -f /tmp/noise.pids; }
sleep 3
for _ in $(seq 1 45); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break; sleep 4
done
rm -rf /tmp/fpm-forward-etcd
etcd --data-dir /tmp/fpm-forward-etcd --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 --listen-peer-urls http://127.0.0.1:2380 \
  >"$OUT/etcd.log" 2>&1 &
nats-server >"$OUT/nats.log" 2>&1 &
sleep 3
curl -sf http://127.0.0.1:2379/health >/dev/null || { echo "NOISE-FAIL-etcd $ARM"; exit 4; }
SNAP=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
export HF_HOME=$SNAP
export FLASHINFER_CUBIN_DIR=$SNAP/flashinfer-cubins
export NCCL_CUMEM_ENABLE=1
export DYN_BENCH_PREFILL_CONTENT=sharegpt
export DYN_FPM_GC_POLICY=freeze
export DYN_FPM_GC_FREEZE_INTERVAL_S=60
export DYN_FPM_WORKER_ID=noise-$ARM-node0
export DYN_BENCHMARK_POINTS_FILE=/tmp/regime/points_arm1.json
if [ "$ARM" = noisy ]; then
  for i in $(seq 1 48); do ( while :; do :; done ) & echo $! >> /tmp/noise.pids; done
  for i in $(seq 1 4); do ( dd if=/dev/zero of=/dev/null bs=64k ) 2>/dev/null & echo $! >> /tmp/noise.pids; done
  echo "noise injected: $(wc -l < /tmp/noise.pids) procs"
fi
cat /proc/loadavg > "$OUT/loadavg_start.txt"
ulimit -l unlimited 2>/dev/null; ulimit -n 1048576 2>/dev/null
date +%s > "$OUT/start.epoch"
python3 -m dynamo.vllm --model $SNAP --served-model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel \
  --kv-cache-dtype fp8 --distributed-executor-backend mp --distributed-timeout-seconds 1800 \
  --no-enable-log-requests --benchmark-timeout 10800 --benchmark-mode decode --benchmark-warmup-iterations 5 \
  --max-model-len -1 --dump-config-to "$OUT/resolved-config.json" \
  --benchmark-output-path "$OUT/benchmark.json" > "$OUT/engine.log" 2>&1
E=$?
cat /proc/loadavg > "$OUT/loadavg_end.txt"
[ -f /tmp/noise.pids ] && { xargs -r kill -9 < /tmp/noise.pids 2>/dev/null; rm -f /tmp/noise.pids; }
date +%s > "$OUT/end.epoch"
echo "NOISE-ARM-DONE $ARM exit=$E bench=$(ls -l $OUT/benchmark.json 2>/dev/null | awk '{print $5}')"
exit $E
