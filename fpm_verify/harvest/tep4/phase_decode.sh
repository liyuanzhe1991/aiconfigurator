#!/usr/bin/env bash
# decode 相:切 decode-parity 引擎 → 按 decode_plan 逐窗 bench serve
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
bash /tmp/fpm-serve/serve_run_tep4_decode.sh >/results/engine_decode.log 2>&1 &
for i in $(seq 1 200); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q MiniMax && break; sleep 5; done
echo "T_decode_ready=$(date +%s)" >> /results/l3_timing.log
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
echo "T_decode_done=$(date +%s)" >> /results/l3_timing.log
echo DECODE-PHASE-DONE
