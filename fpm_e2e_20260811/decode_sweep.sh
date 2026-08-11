#!/usr/bin/env bash
# Stack-1 decode ground-truth sweep: one bench window per (C, ISL, OSL) combo,
# FPM stream positions recorded per window in /results/sweep_windows.tsv.
# Priority-ordered: B-ladder at 8.7k KV first, then KV bands, then long-KV.
set -uo pipefail
cd /workspace/examples/backends/vllm
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
TSV=/results/sweep_windows.tsv
touch $TSV

run_window() {
  local name=$1 C=$2 ISL=$3 OSL=$4 NP=$5
  local s e
  s=$(wc -l < /results/fpm_stream.jsonl)
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0 --num-prompts "$NP" --max-concurrency "$C" \
    --ignore-eos > "/results/sweep_${name}.log" 2>&1
  e=$(wc -l < /results/fpm_stream.jsonl)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$C" "$ISL" "$OSL" "$s" "$e" >> $TSV
  echo "window $name done (C=$C ISL=$ISL OSL=$OSL lines $s..$e) $(date +%H:%M:%S)"
}

# --- 1. B ladder at ISL 8192 (per-req KV ~8.2-9.2k); 40/64 already measured ---
run_window b512_kv8k   512  8192 1024 1024
run_window b1024_kv4k  1024 4096  512 2048
run_window b256_kv8k   256  8192 1024  512
run_window b128_kv8k   128  8192 1024  320
run_window b32_kv8k    32   8192 1024   96
run_window b16_kv8k    16   8192 1024   48
run_window b8_kv8k     8    8192 1024   32
run_window b1_kv8k     1    8192  512    6

# --- 2. KV bands at B in {8,40,64} ---
for C in 8 40 64; do
  run_window "b${C}_kv0p5k" $C  512  256 $((C*3))
  run_window "b${C}_kv1p5k" $C 1536  256 $((C*3))
  run_window "b${C}_kv3k"   $C 3072  256 $((C*3))
  run_window "b${C}_kv5k"   $C 5120  256 $((C*3))
done

# --- 3. long KV (capacity-checked) ---
for C in 8 40 64; do
  run_window "b${C}_kv16k" $C 16384 256 $((C*2))
  run_window "b${C}_kv32k" $C 32768 256 $((C*2))
  run_window "b${C}_kv64k" $C 65536 256 $((C*2))
done
run_window b40_kv100k 40 102400 256 80
run_window b64_kv100k 64 102400 256 96
run_window b8_kv131k  8  131072 256 16
run_window b32_kv131k 32 131072 256 64
run_window b8_kv200k  8  200448 256 16

echo "DECODE-SWEEP-DONE $(date +%H:%M:%S)"
