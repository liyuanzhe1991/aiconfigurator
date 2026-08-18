#!/usr/bin/env bash
# timing 补丁验证 boot 器(烘焙镜像 gc-realcontent-20260818 专用):
# 每 boot 先把调度器还原为烘焙态(kvwarm+content+argsdump 已内含),
# ARM=timing 时在其上叠加 timing_patch.py;不再重跑三补丁链。
# 用法: timingchk_run.sh <engine_script> <points_json> <outdir> <ARM: base|timing>
set -uo pipefail
ENG=${1:?}; PTS=${2:?}; OUT=${3:?}; ARM=${4:?}
W=/tmp/fpm-kvwarm
SP=/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py
mkdir -p "$OUT"
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "dynamo[.]frontend" 2>/dev/null
pkill -9 -f "etcd --data-dir /tmp/fpm-forward-[e]tcd" 2>/dev/null; pkill -9 -f "nats-serve[r]" 2>/dev/null
for _ in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break
  sleep 4
done
# 烘焙态快照(首 boot 建立;后续 boot 由此还原,保证 ARM 顺序无关)
[ -f "$W/baked_scheduler.py" ] || cp "$SP" "$W/baked_scheduler.py" || exit 5
cp "$W/baked_scheduler.py" "$SP" || exit 5
if [ "$ARM" = "timing" ]; then
  python3 "$W/timing_patch.py" || exit 3
fi
rm -rf /tmp/fpm-forward-etcd
etcd --data-dir /tmp/fpm-forward-etcd --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 --listen-peer-urls http://127.0.0.1:2380 \
  >"$OUT/etcd.log" 2>&1 &
nats-server >"$OUT/nats.log" 2>&1 &
sleep 3
curl -sf http://127.0.0.1:2379/health >/dev/null || { echo CHK-FAIL-etcd; exit 4; }
export DYN_BENCH_KV_WARMUP_CACHE_DIR=/workspace/model_cache/fpm_datasets
export DYN_BENCHMARK_POINTS_FILE="$W/$PTS"
export DYN_BENCH_KV_WARMUP=on
export DYN_BENCH_PREFILL_CONTENT=sharegpt
export DYN_BENCH_POOL_TAG=""
export BENCH_OUT="$OUT/benchmark_on.json"
export CFG_OUT="$OUT/resolved-config_on.json"
date +%s > "$OUT/chk_start.epoch"
nohup bash "$W/$ENG" >"$OUT/engine.log" 2>&1 &
echo "CHK-LAUNCHED arm=$ARM pts=$PTS out=$OUT"
