#!/bin/bash
export SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server
export SGL_LOG_TOPK_STATS=0 # 1 Enable top-k stats logging
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /host/hisim-sglang/sglang/tools/sglang-simulator/tools/benchmark && \
python3 sgl_launch_server.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --ep-size 8 \
    --trust-remote-code \
    --max-running-requests 256 \
    --port 30000 \
    --enable-piecewise-cuda-graph \
    --disable-cuda-graph-padding \
    --cuda-graph-max-bs 256