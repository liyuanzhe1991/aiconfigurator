#!/usr/bin/env bash
# Stack-2 prefill ground-truth sweep (prefill-cell engine params: sync sched,
# graphs<=2048, prefix caching ON, batched-tokens 8192).
# Group 1: kv=0 dense token grid via output-len=1 windows.
# Group 2: prefix axis via --random-prefix-len (cache-hit construction).
# Every window gets a fresh --seed so prefix caching never leaks across windows.
set -uo pipefail
cd /workspace/examples/backends/vllm
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
TSV=/results/prefill_windows.tsv
touch $TSV
SEED=20260811

run_window() {
  local name=$1 C=$2 ISL=$3 PREFIX=$4 NP=$5
  local s e
  SEED=$((SEED + 1))
  s=$(wc -l < /results/fpm_stream.jsonl)
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$ISL" --random-output-len 1 \
    --random-prefix-len "$PREFIX" --random-range-ratio 0 \
    --num-prompts "$NP" --max-concurrency "$C" --ignore-eos --seed "$SEED" \
    > "/results/pf_${name}.log" 2>&1
  e=$(wc -l < /results/fpm_stream.jsonl)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$C" "$ISL" "$PREFIX" "$s" "$e" >> $TSV
  echo "window $name done (C=$C ISL=$ISL prefix=$PREFIX lines $s..$e) $(date +%H:%M:%S)"
}

# --- Group 1: kv=0, dense token grid, batch 1..4 ---
for T in 128 256 512 768 1024 1536 2048 2049 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680 8192; do
  for B in 1 2 3 4; do
    run_window "b${B}_t${T}" "$B" "$T" 0 $((25*B))
  done
done

# --- Group 2: prefix axis ---
for P in 2048 8192 32768 131072 262144; do
  for T in 512 2048 4096; do
    for B in 1 2; do
      run_window "b${B}_t${T}_p${P}" "$B" "$T" "$P" $((15*B))
    done
  done
done

echo "PREFILL-SWEEP-DONE $(date +%H:%M:%S)"
