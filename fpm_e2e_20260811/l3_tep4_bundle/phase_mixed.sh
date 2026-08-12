#!/usr/bin/env bash
# mixed 相:对每个 Bd,起池(bench serve 背景)→ 注入(mixed_driver)× kvp 两模式 × kvd 两档
set -uo pipefail
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
cd /workspace/examples/backends/vllm
echo "T_mixed_start=$(date +%s)" >> /results/l3_timing.log
for KVD_ISL in 8192 65536; do
  for BD in $(cut -d, -f3 /tmp/fpm-serve/mixed_plan.csv | tail -n +2 | sort -un); do
    vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
      --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
      --dataset-name random --random-input-len $KVD_ISL --random-output-len 6000 \
      --random-range-ratio 0 --num-prompts $((BD*3)) --max-concurrency $BD \
      --request-rate inf --ignore-eos --seed $((KVD_ISL+BD)) >/results/mx_pool_${KVD_ISL}_${BD}.log 2>&1 &
    LONG=$!
    sleep 60
    grep -E "^grp|,${BD}," /tmp/fpm-serve/mixed_plan.csv | head -1 >/dev/null
    awk -F, -v bd=$BD 'NR==1 || $3==bd' /tmp/fpm-serve/mixed_plan.csv > /tmp/fpm-serve/mixed_plan_bd.csv
    for MODE in zero chunk; do
      python3 /tmp/fpm-serve/mixed_driver.py /tmp/fpm-serve/mixed_plan_bd.csv /results/fpm_stream.jsonl /results/mixed_windows.tsv $MODE 2>&1 | tail -2
    done
    kill $LONG 2>/dev/null; wait $LONG 2>/dev/null; sleep 8
  done
done
echo "T_mixed_done=$(date +%s)" >> /results/l3_timing.log
echo MIXED-PHASE-DONE
