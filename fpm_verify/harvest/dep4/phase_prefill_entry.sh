#!/usr/bin/env bash
# prefill 相位入口(固定方法论):切 prefill-parity 引擎 → burst → mixed
set -uo pipefail
cd /workspace/examples/backends/vllm
echo "T_prefill_stack_start=$(date +%s)" >> /results/l3_timing.log
pkill -9 -f "dynamo[.]vllm"; pkill -9 -f "VLLM[:]:"; sleep 8
bash /tmp/fpm-serve/serve_run_dep4_prefill.sh >/results/engine_prefill.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && break; sleep 5; done
echo "T_prefill_ready=$(date +%s)" >> /results/l3_timing.log
export L3_DP_MODE=1
python3 /tmp/fpm-serve/burst_driver.py /tmp/fpm-serve/prefill_plan.csv /results/fpm_stream.jsonl /results/burst_windows.tsv 2>&1 | tee /results/burst_driver.log
echo "T_burst_done=$(date +%s)" >> /results/l3_timing.log
bash /tmp/fpm-serve/phase_mixed.sh 2>&1 | tee /results/phase_mixed.log
echo PREFILL-PHASE-DONE >> /results/phase_prefill.log
