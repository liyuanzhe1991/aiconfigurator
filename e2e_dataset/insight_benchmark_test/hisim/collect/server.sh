# HF_ENDPOINT=https://hf-mirror.com \
# SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
# python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
#     --model-path="Qwen/Qwen3-8B" \
#     --disable-overlap-schedule \
#     --load-format=dummy \
#     --base-gpu-id 7


# HF_ENDPOINT=https://hf-mirror.com \
# SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
# python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
#     --model-path="Qwen/Qwen3-8B" \
#     --load-format=dummy \
#     --enable-hierarchical-cache \
#     --decode-attention-backend fa3 \
#     --max-total-tokens=200000 \
#     --page-size=64 \
#     --base-gpu-id 7 \
#     --disable-overlap-schedule


# HF_ENDPOINT=https://hf-mirror.com \
# SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
# python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
#     --model-path="Qwen/Qwen3-32B-FP8" \
#     --load-format=dummy \
#     --base-gpu-id 7 \
#     --disable-overlap-schedule


# Replay: collect schedule batch infomation
# SGL_HOOK_FETCH_BATCH_INFO=1 \
# HF_ENDPOINT=https://hf-mirror.com \
# SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
# python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
#     --model-path="Qwen/Qwen3-32B-FP8" \
#     --load-format=dummy \
#     --enable-hierarchical-cache \
#     --decode-attention-backend fa3 \
#     --max-total-tokens=200000 \
#     --page-size=64 \
#     --base-gpu-id 3 \
#     --disable-overlap-schedule


HF_ENDPOINT=https://hf-mirror.com \
SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
    --model-path=/models/Qwen3-235B-A22B-Instruct-2507-FP8/ \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --ep-size 8 \
    --cuda-graph-max-bs 256 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --max-running-requests 512


HF_ENDPOINT=https://hf-mirror.com \
SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server \
python3 `pwd`/examples/benchmark/serving_hook/sglang_launch_server.py \
    --model-path=/models/Qwen3-235B-A22B-Instruct-2507-FP8/ \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --ep-size 8 \
    --cuda-graph-max-bs 256 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --max-running-requests 512 \
    --enable-hierarchical-cache \
    --decode-attention-backend fa3 \
    --max-total-tokens=200000 \
    --page-size=64