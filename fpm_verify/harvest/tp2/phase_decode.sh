#!/usr/bin/env bash
# decode 相:切 decode-parity 引擎 → decode_driver.py(ShareGPT 奇数池,
# 默认;内容 ABA 实验实证 random 池偏快 -1.51%)。回退:
# L3_DECODE_DRIVER=bench 走旧 vllm bench serve random 路径。
# 注意:新驱动窗口写 v3 九列且 isl==1 窗带 lockstep 标;bench 路径七列。
set -uo pipefail
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
bash /tmp/fpm-serve/serve_run_tp2_decode.sh >/results/engine_decode.log 2>&1 &
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && break; sleep 5; done
echo "T_decode_ready=$(date +%s)" >> /results/l3_timing.log
if [ "${L3_DECODE_DRIVER:-sharegpt}" = "bench" ]; then
  tail -n +2 /tmp/fpm-serve/decode_plan.csv | while IFS=, read grp C isl osl kind boots; do
    s0=$(wc -l < /results/fpm_stream.jsonl)
    vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
      --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
      --dataset-name random --random-input-len $isl --random-output-len $osl \
      --random-range-ratio 0 --num-prompts $C --max-concurrency $C \
      --request-rate inf --ignore-eos --seed $((RANDOM)) >/results/dw_${C}_${isl}.log 2>&1
    e0=$(wc -l < /results/fpm_stream.jsonl)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$grp" "$C" "$isl" "$osl" "$kind" "$s0" "$e0" >> /results/decode_windows.tsv
    echo "窗 C=$C isl=$isl osl=$osl done ($s0..$e0) $(date +%H:%M:%S)"
  done
else
  export L3_CAPACITY_GUARD=600000
  export L3_TOKENIZER="$TOK"
  export L3_SHAREGPT_PATH=/workspace/model_cache/fpm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
  python3 /tmp/fpm-serve/decode_driver.py /tmp/fpm-serve/decode_plan.csv \
    /results/fpm_stream.jsonl /results/decode_windows.tsv 1 \
    2>&1 | tee /results/decode_driver.log
fi
echo "T_decode_done=$(date +%s)" >> /results/l3_timing.log
echo DECODE-PHASE-DONE
