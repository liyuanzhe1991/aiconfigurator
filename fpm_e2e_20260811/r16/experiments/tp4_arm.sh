#!/usr/bin/env bash
# tp4 最小机制实验单臂:fake(kvwarm 被 moe_tp skip)| warm(sed 放行后真实链)
# 引擎命令逐字取自 R16 tp4 decode cell,仅两处受控偏离:
#   1) 去掉 --no-enable-prefix-caching(两臂同开,warm 谓词的 prefix 前提)
#   2) DYN_BENCHMARK_POINTS_FILE 换成 24 点鼓形探针
set -uo pipefail
ARM=${1:?fake|warm}
OUT=/results/$ARM; mkdir -p "$OUT"
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "etcd --data-dir /tmp/fpm-forward-[e]tcd" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
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
curl -sf http://127.0.0.1:2379/health >/dev/null || { echo "TP4EXP-FAIL-etcd $ARM"; exit 4; }
SNAP=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
export HF_HOME=$SNAP
export FLASHINFER_CUBIN_DIR=$SNAP/flashinfer-cubins
export NCCL_CUMEM_ENABLE=1
export DYN_BENCH_PREFILL_CONTENT=sharegpt
export DYN_FPM_GC_POLICY=freeze
export DYN_FPM_GC_FREEZE_INTERVAL_S=60
export DYN_FPM_WORKER_ID=tp4exp-$ARM-node0
export DYN_BENCHMARK_POINTS_FILE=/tmp/tp4exp/points.json
ulimit -l unlimited 2>/dev/null; ulimit -n 1048576 2>/dev/null
date +%s > "$OUT/start.epoch"
python3 -m dynamo.vllm --model $SNAP --served-model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --kv-cache-dtype fp8 --distributed-executor-backend mp --distributed-timeout-seconds 1800 \
  --no-enable-log-requests --benchmark-timeout 10800 --benchmark-mode decode --benchmark-warmup-iterations 5 \
  --max-model-len -1 --dump-config-to "$OUT/resolved-config.json" \
  --benchmark-output-path "$OUT/benchmark.json" > "$OUT/engine.log" 2>&1
E=$?
date +%s > "$OUT/end.epoch"
echo "TP4EXP-ARM-DONE $ARM exit=$E bench=$(ls -l $OUT/benchmark.json 2>/dev/null | awk '{print $5}')"
exit $E
