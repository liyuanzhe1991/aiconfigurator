#!/usr/bin/env bash
# fpm-verify 真值采收编排(固定方法论,r15 实战工装收编)。
# 用法: stage_and_run.sh <topo: tep4|dep4|tp4> <ctx> [ns]
#   1) 建 Guaranteed 采收 pod(/results 挂共享 PVC,抢占免疫)
#   2) exec-cat + 双端 sha256 布置 kit(禁 kubectl cp)
#   3) 串行相位:decode(锁步窗)→ prefill(burst + mixed)
#   4) 结果留在 PVC;取件用 fetch_results.sh(分块+逐块 sha)
# 依赖:tsh 会话有效;长采收(>4h)前建议重登(teleport 会话级流劣化,记过档)。
set -uo pipefail
TOPO=${1:?tep4|dep4|tp4}
CTX=${2:?kubectl context}
NS=${3:-yuanli-aic}
PHASES=${4:-all}   # all | prefill(保留既有 decode 产物,只重跑 burst+mixed)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$TOPO"
POD="fpm-l3-$TOPO-agg"
K() { kubectl --context "$CTX" "$@"; }

K apply -f "$DIR/k8s_deploy.yaml"
for _ in $(seq 1 60); do
  [ "$(K get pod $POD -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ] && break
  sleep 10
done
[ "$(K get pod $POD -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ] || { echo "HARVEST-FAIL pod"; exit 1; }

K exec -n $NS $POD -- mkdir -p /tmp/fpm-serve
stage() {
  local ref got
  ref=$(shasum -a 256 "$1" | cut -d" " -f1)
  for _ in 1 2 3 4 5; do
    K exec -i -n $NS $POD -- bash -c "cat > /tmp/fpm-serve/$(basename "$1")" < "$1" 2>/dev/null
    got=$(K exec -n $NS $POD -- sha256sum "/tmp/fpm-serve/$(basename "$1")" 2>/dev/null | cut -d" " -f1)
    [ "$ref" = "$got" ] && return 0
  done
  echo "STAGE-FAIL $(basename "$1")"; exit 1
}
for f in "$DIR"/*.sh "$DIR"/*.py "$DIR"/*.csv "$DIR"/*.json; do
  [ -f "$f" ] || continue
  [ "$(basename "$f")" = "k8s_deploy.yaml" ] && continue
  stage "$f"
done
if [ "$PHASES" = "prefill" ]; then
  K exec -n $NS $POD -- bash -c 'chmod +x /tmp/fpm-serve/*.sh; rm -f /results/burst_windows.tsv /results/mixed_windows.tsv /results/phase_all.log /results/phase_prefill_all.log /results/burst_driver.log /results/phase_mixed.log; touch /results/fpm_stream.jsonl'
else
  K exec -n $NS $POD -- bash -c 'chmod +x /tmp/fpm-serve/*.sh; rm -f /results/fpm_stream.jsonl /results/*_windows.tsv /results/phase_*.log /results/l3_timing.log; touch /results/fpm_stream.jsonl'
fi

# 相位串行(pod 内 nohup 自持续;laptop 侧轮询相位完成标记)
# 全链一个 nohup 会话:serve_stack 的守护进程(etcd/nats/frontend/listener)
# 必须活在链内,否则 exec 会话结束即死。
if [ "$PHASES" = "prefill" ]; then
  K exec -n $NS $POD -- bash -c 'cd /workspace/examples/backends/vllm && nohup bash -c "WORKDIR=/tmp/fpm-serve bash /tmp/fpm-serve/serve_stack.sh 2>&1 | tee /results/serve_stack.log; bash /tmp/fpm-serve/phase_prefill_entry.sh 2>&1 | tee /results/phase_prefill_all.log; echo HARVEST-ALL-DONE >> /results/phase_all.log" >/tmp/fpm-serve/nohup.log 2>&1 & echo LAUNCHED'
else
  K exec -n $NS $POD -- bash -c 'cd /workspace/examples/backends/vllm && nohup bash -c "WORKDIR=/tmp/fpm-serve bash /tmp/fpm-serve/serve_stack.sh 2>&1 | tee /results/serve_stack.log; bash /tmp/fpm-serve/phase_decode.sh 2>&1 | tee /results/phase_decode.log; bash /tmp/fpm-serve/phase_prefill_entry.sh 2>&1 | tee /results/phase_prefill_all.log; echo HARVEST-ALL-DONE >> /results/phase_all.log" >/tmp/fpm-serve/nohup.log 2>&1 & echo LAUNCHED'
fi
echo "HARVEST-LAUNCHED topo=$TOPO pod=$POD(结果落 PVC:_yuanli_l3_results/r16_$TOPO)"
