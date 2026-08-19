#!/usr/bin/env bash
# Rendered by the Generator FPM target. Sourced (not executed) by both the
# collector's in-pod runtime and the generated run.sh. Exports exactly the
# variables listed in fpm_contract.FPM_ENV_EXPORTED_VARS.
export FPM_NODE_COUNT=1
export FPM_DATA_PARALLEL_SIZE=2
export FPM_LOCAL_DATA_PARALLEL_SIZE=2
export FPM_BENCHMARK_MODE=prefill
export FPM_BENCHMARK_OUTPUT_PATH=/results/benchmark.json
export FPM_WAIT_TIMEOUT_SECONDS=4200
export FPM_RESULT_SCHEMA_VERSION=2

if (( FPM_NODE_COUNT > 1 )); then
  fpm_node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"
  fpm_master_addr="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"
  if [[ -z "$fpm_master_addr" && -n "${GROVE_PCLQ_NAME:-}" && -n "${GROVE_HEADLESS_SERVICE:-}" ]]; then
    fpm_master_addr="${GROVE_PCLQ_NAME}-0.${GROVE_HEADLESS_SERVICE}"
  fi
  if [[ -z "$fpm_node_rank" || -z "$fpm_master_addr" ]]; then
    echo "Multinode FPM requires rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2
    exit 2
  fi
else
  fpm_node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"
  fpm_master_addr="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"
  if [[ -z "$fpm_master_addr" && -n "${GROVE_PCLQ_NAME:-}" && -n "${GROVE_HEADLESS_SERVICE:-}" ]]; then
    fpm_master_addr="${GROVE_PCLQ_NAME}-0.${GROVE_HEADLESS_SERVICE}"
  fi
  # Defaults apply only to a fully undiscovered environment; a partial
  # answer is a misconfigured orchestrator and must fail closed.
  if [[ -z "$fpm_node_rank" && -z "$fpm_master_addr" \
    && -z "${GROVE_PCLQ_NAME:-}" && -z "${GROVE_HEADLESS_SERVICE:-}" ]]; then
    fpm_node_rank=0
    fpm_master_addr=127.0.0.1
  elif [[ -z "$fpm_node_rank" || -z "$fpm_master_addr" ]]; then
    echo "FPM runtime requires complete rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2
    exit 2
  fi
fi
if ! [[ "$fpm_node_rank" =~ ^[0-9]+$ ]] || (( fpm_node_rank >= FPM_NODE_COUNT )); then
  echo "Invalid FPM node rank: $fpm_node_rank (node_count=$FPM_NODE_COUNT)" >&2
  exit 2
fi
export FPM_NODE_RANK="$fpm_node_rank"
export FPM_MASTER_ADDR="$fpm_master_addr"
