#!/usr/bin/env bash
# Stack-1 mixed-step sweep: a long stream holds a steady decode pool at Bd,
# a probe stream injects short-ISL requests so each injection creates mixed
# steps with chunk≈ISL. Windows marked in /results/mixed_windows_v2.tsv.
set -uo pipefail
cd /workspace/examples/backends/vllm
TOK=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
TSV=/results/mixed_windows_v2.tsv
touch $TSV
SEED=90000

bench() {  # C ISL OSL NP RATE LOG [EXTRA...]
  local C=$1 ISL=$2 OSL=$3 NP=$4 RATE=$5 LOG=$6
  SEED=$((SEED+1))
  vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
    --model MiniMaxAI/MiniMax-M2.7 --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0 --num-prompts "$NP" --max-concurrency "$C" \
    --request-rate "$RATE" --ignore-eos --seed "$SEED" > "$LOG" 2>&1
}

for BD in 8 16 32 40 64; do
  echo "=== Bd=$BD: starting long stream $(date +%H:%M:%S) ==="
  bench "$BD" 8192 1024 $((BD*14)) inf "/results/mx2_long_bd${BD}.log" &
  LONG=$!
  sleep 75   # ramp: prefill all BD requests, reach steady decode
  for CHUNK in 256 512 1024 2048 4096 6144; do
    s=$(wc -l < /results/fpm_stream.jsonl)
    bench 1 "$CHUNK" 1 20 2.0 "/results/mx2_probe_bd${BD}_c${CHUNK}.log"
    e=$(wc -l < /results/fpm_stream.jsonl)
    printf 'bd%s_c%s\t%s\t%s\t%s\t%s\n' "$BD" "$CHUNK" "$BD" "$CHUNK" "$s" "$e" >> $TSV
    echo "probe bd=$BD chunk=$CHUNK done ($s..$e) $(date +%H:%M:%S)"
  done
  kill $LONG 2>/dev/null; wait $LONG 2>/dev/null
  sleep 10   # drain
done
echo "MIXED-SWEEP-DONE $(date +%H:%M:%S)"
