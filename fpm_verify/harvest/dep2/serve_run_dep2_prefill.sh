#!/usr/bin/env bash
set -Eeuo pipefail

# fpm_env.sh owns every render-time collection fact (topology, rank and
# leader discovery, benchmark identity); discovery failures exit 2 here.
source "$(dirname "${BASH_SOURCE[0]}")/fpm_env.sh"

ulimit -l unlimited || true
ulimit -n 1048576 || true
export HF_HOME=/workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
export NCCL_CUMEM_ENABLE=1
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y
export UCX_LOG_LEVEL=error
export NIXL_LOG_LEVEL=ERROR
export FPM_RUN_ID=fpm-l3-dep2

# FlashInfer downloads missing cubins at first use; its default cache
# lives inside site-packages, which is read-only in the deployed image
# and crashes every engine worker with EACCES. Default the cache to the
# writable model-cache volume so pods reuse previously fetched cubins.
if [[ -z "${FLASHINFER_CUBIN_DIR:-}" && -n "${HF_HOME:-}" ]]; then
  export FLASHINFER_CUBIN_DIR="${HF_HOME}/flashinfer-cubins"
fi

export DYN_FPM_WORKER_ID="${DYN_FPM_WORKER_ID:-${FPM_RUN_ID:-fpm}-node${FPM_NODE_RANK}}"
engine_command=(python3 -m dynamo.vllm --model /workspace/model_cache/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b --served-model-name MiniMaxAI/MiniMax-M2.7 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 2 --enable-expert-parallel --kv-cache-dtype fp8 --distributed-executor-backend mp --distributed-timeout-seconds 1800 --no-enable-log-requests --no-async-scheduling --max-model-len -1 --max-num-batched-tokens 8192 --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024,1056,1088,1120,1152,1184,1216,1248,1280,1312,1344,1376,1408,1440,1472,1504,1536,1568,1600,1632,1664,1696,1728,1760,1792,1824,1856,1888,1920,1952,1984,2016,2048],"max_cudagraph_capture_size":2048}' --data-parallel-backend mp --dump-config-to /results/resolved-config-node0.json)
for index in "${!engine_command[@]}"; do
  engine_command[$index]="${engine_command[$index]//__FPM_NODE_RANK__/$FPM_NODE_RANK}"
done

if (( FPM_NODE_COUNT > 1 )); then
  if (( FPM_DATA_PARALLEL_SIZE > 1 )); then
    engine_command+=(--data-parallel-size-local "$FPM_LOCAL_DATA_PARALLEL_SIZE")
    engine_command+=(--data-parallel-start-rank "$((FPM_NODE_RANK * FPM_LOCAL_DATA_PARALLEL_SIZE))")
    engine_command+=(--data-parallel-address "$FPM_MASTER_ADDR" --data-parallel-rpc-port 29510)
    engine_command+=(--data-parallel-hybrid-lb)
  else
    engine_command+=(--nnodes "$FPM_NODE_COUNT" --node-rank "$FPM_NODE_RANK")
    engine_command+=(--master-addr "$FPM_MASTER_ADDR" --master-port 29500)
    if (( FPM_NODE_RANK > 0 )); then
      # Headless followers never write results; classifying their exit
      # against the leader's teardown belongs to the collector runtime.
      engine_command+=(--headless)
    fi
  fi
fi

# Multinode transport profiles rewrite PATH to load the fabric-patched
# libfabric first; a profile that drops /usr/local/cuda/bin silently
# starves deep_gemm's runtime nvcc JIT, which only surfaces minutes
# later as an opaque DG_HOST_ASSERT(!cubin.empty()) engine crash. Fail
# fast with an actionable message instead. Single-node runs keep the
# image default PATH and skip the check.
if (( FPM_NODE_COUNT > 1 )) && ! command -v nvcc >/dev/null 2>&1; then
  echo "run.sh: nvcc is not on PATH (PATH=$PATH); deep_gemm JIT will fail. Ensure the transport PATH keeps /usr/local/cuda/bin." >&2
  exit 2
fi

# Replace this shell so run.sh's exit code is the engine's exit code;
# setsid makes the engine its process-group leader so the collector
# runtime can terminate the whole group.
exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "${engine_command[@]}"
