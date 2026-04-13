export HF_ENDPOINT="https://hf-mirror.com"

SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server

BASE_OUT_DIR="${1:-H20-Qwen3-8B}"

request_rates=(1 2 4 6 8 12 16 20 24 28 32)
for rate in "${request_rates[@]}"; do
    echo "Running the rate: $rate"

    curl http://localhost:30000/flush_cache
    rm -rf /tmp/hicache/ && mkdir -p /tmp/hicache
    curl http://localhost:30000/start_profile

    OUR_DIR=`pwd`/tmp/L1/$BASE_OUT_DIR/data/$rate
    mkdir -p $OUR_DIR

    # generate cache
    python3 `pwd`/test/hisim/collect/bench_serving.py \
        --warmup-requests 0 \
        --dataset-name generated-shared-prefix \
        --request-rate $rate \
        --gsp-prompts-per-group 1 \
        --gsp-num-groups 100 \
        --gsp-question-seed 1234 \
        --gsp-system-prompt-len 1024 \
        --gsp-question-len 1024 \
        --gsp-output-len 512 \
        --random-range-ratio 0.5 \
        --output-file $OUR_DIR/no_cache.metrics.json

    curl http://localhost:30000/start_profile
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.raw_request.jsonl $OUR_DIR/no_cache.requests.jsonl
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.schedule_batch.jsonl $OUR_DIR/no_cache.schedule_batch.jsonl


    python3 `pwd`/test/hisim/collect/bench_serving.py \
        --warmup-requests 0 \
        --dataset-name generated-shared-prefix \
        --request-rate $rate \
        --gsp-prompts-per-group 1 \
        --gsp-num-groups 100 \
        --gsp-question-seed 12345 \
        --gsp-system-prompt-len 1024 \
        --gsp-question-len 1024 \
        --gsp-output-len 512 \
        --random-range-ratio 0.5 \
        --output-file $OUR_DIR/l1.metrics.json

    curl http://localhost:30000/start_profile
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.raw_request.jsonl $OUR_DIR/l1.requests.jsonl
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.schedule_batch.jsonl $OUR_DIR/l1.schedule_batch.jsonl

done