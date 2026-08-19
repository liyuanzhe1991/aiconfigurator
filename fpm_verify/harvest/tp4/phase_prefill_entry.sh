#!/usr/bin/env bash
# prefill 相位入口(固定方法论):切 prefill-parity 引擎 → burst → mixed
set -uo pipefail
cd /workspace/examples/backends/vllm
echo "T_prefill_stack_start=$(date +%s)" >> /results/l3_timing.log
pkill -9 -f "dynamo[.]vllm"; pkill -9 -f "VLLM[:]:"
for _ in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break
  sleep 4
done
export DYN_FORWARDPASS_METRIC_PORT=20380
bash /tmp/fpm-serve/serve_run_tp4_prefill.sh >/results/engine_prefill.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && break; sleep 5; done
echo "T_prefill_ready=$(date +%s)" >> /results/l3_timing.log
export L3_TOKENIZER=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
export L3_SHAREGPT_PATH=/workspace/model_cache/fpm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json

python3 /tmp/fpm-serve/burst_driver.py /tmp/fpm-serve/prefill_plan.csv /results/fpm_stream.jsonl /results/burst_windows.tsv 2>&1 | tee /results/burst_driver.log
echo "T_burst_done=$(date +%s)" >> /results/l3_timing.log
bash /tmp/fpm-serve/phase_mixed.sh 2>&1 | tee /results/phase_mixed.log
echo PREFILL-PHASE-DONE >> /results/phase_prefill.log
