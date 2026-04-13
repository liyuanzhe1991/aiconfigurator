SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server

BASE_OUT_DIR="${1:-H20-QWEN235B-FP8-BALANCE}"

request_rates=(1 2 4 6 8 12 16 20 24 28 32)
# request_rates=(1)
for rate in "${request_rates[@]}"; do
    echo "Running the rate: $rate"

    curl http://localhost:30000/flush_cache
    rm -rf /tmp/hicache/ && mkdir -p /tmp/hicache
    curl http://localhost:30000/start_profile

    OUR_DIR=`pwd`/tmp/L1/$BASE_OUT_DIR/data/$rate
    mkdir -p $OUR_DIR

    # generate cache
    python3 -m sglang.bench_serving \
        --warmup-requests 0 \
        --dataset-name random \
        --num-prompts 100 \
        --request-rate $rate \
        --random-input-len 5000 \
        --random-output-len 512 \
        --random-range-ratio 0.5 \
        --dataset-path /host/aiconfigurator/ShareGPT_V3_unfiltered_cleaned_split.json \
        --output-file $OUR_DIR/no_cache.metrics.json

    curl http://localhost:30000/start_profile
    mv $SGL_HOOK_REQ_INFO_DIR/TP0-EP0.request.jsonl $OUR_DIR/no_cache.requests.jsonl
    mv $SGL_HOOK_REQ_INFO_DIR/TP0-EP0.schedule_batch.jsonl $OUR_DIR/no_cache.schedule_batch.jsonl

done