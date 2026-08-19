#!/usr/bin/env bash
# dep4 decode 相:ShareGPT 驱动默认(L3_DECODE_DRIVER=bench 回退);
# plan 的 C 为每 rank 目标,驱动内部 total=C*DP;容量护栏在驱动内
# (L3_CAPACITY_GUARD,默认 9M token,与旧 bench 路径同值)。
set -uo pipefail
DP=2
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
cd /workspace/examples/backends/vllm
echo "T_decode_stack_start=$(date +%s)" >> /results/l3_timing.log
pkill -9 -f "dynamo[.]vllm"; pkill -9 -f "VLLM[:]:"
for _ in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${USED:-99999}" -lt 1000 ] && break
  sleep 4
done
export DYN_FORWARDPASS_METRIC_PORT=20380
bash /tmp/fpm-serve/serve_run_dep2_decode.sh >/results/engine_decode.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && break; sleep 5; done
echo "T_decode_ready=$(date +%s)" >> /results/l3_timing.log
if [ "${L3_DECODE_DRIVER:-sharegpt}" = "bench" ]; then
  tail -n +2 /tmp/fpm-serve/decode_plan.csv | while IFS=, read grp C isl osl kind boots; do
    POOL=$((C*DP))
    [ $((POOL*isl+POOL*osl)) -gt 9000000 ] && { echo "skip C=$C isl=$isl (容量护栏)"; continue; }
    s0=$(wc -l < /results/fpm_stream.jsonl)
    vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
      --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
      --dataset-name random --random-input-len $isl --random-output-len $osl \
      --random-range-ratio 0 --num-prompts $POOL --max-concurrency $POOL \
      --request-rate inf --ignore-eos --seed $((RANDOM)) >/results/dw_${C}_${isl}.log 2>&1
    e0=$(wc -l < /results/fpm_stream.jsonl)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$grp" "$POOL" "$isl" "$osl" "$kind" "$s0" "$e0" >> /results/decode_windows.tsv
    echo "窗 per-rank目标C=$C isl=$isl done ($s0..$e0) $(date +%H:%M:%S)"
  done
else
  export L3_CAPACITY_GUARD=2000000
  export L3_TOKENIZER="$TOK"
  export L3_SHAREGPT_PATH=/workspace/model_cache/fpm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
  python3 /tmp/fpm-serve/decode_driver.py /tmp/fpm-serve/decode_plan.csv \
    /results/fpm_stream.jsonl /results/decode_windows.tsv $DP \
    2>&1 | tee /results/decode_driver.log
fi
echo "T_decode_done=$(date +%s)" >> /results/l3_timing.log
echo DECODE-PHASE-DONE >> /results/phase_decode.log
