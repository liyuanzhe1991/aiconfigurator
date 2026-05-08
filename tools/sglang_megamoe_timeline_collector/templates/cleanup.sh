#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_env.sh"

kubectl delete job -n "${NAMESPACE}" -l "app=${APP}" --ignore-not-found
kubectl delete job "${BENCH_JOB}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete job "${RESULTS_JOB}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete configmap "${SCHEDULER_NVTX_CONFIGMAP}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete deployment "${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete statefulset "${PREFILL_STS}" "${DECODE_STS}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete svc \
  "${PREFILL_SERVICE}" \
  "${DECODE_SERVICE}" \
  "${PREFILL_LEADER_SERVICE}" \
  "${DECODE_LEADER_SERVICE}" \
  "${ROUTER_SERVICE}" \
  -n "${NAMESPACE}" --ignore-not-found
kubectl delete computedomain.resource.nvidia.com "${COMPUTE_DOMAIN}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete resourceclaimtemplate "${COMPUTE_DOMAIN_CHANNEL}" -n "${NAMESPACE}" --ignore-not-found

kubectl wait --for=delete pod -n "${NAMESPACE}" -l "app=${APP}" --timeout=10m || true

claim_names="$(kubectl get resourceclaim -n "${NAMESPACE}" -o name 2>/dev/null | grep "${COMPUTE_DOMAIN_CHANNEL}" || true)"
if [[ -n "${claim_names}" ]]; then
  while IFS= read -r claim; do
    [[ -n "${claim}" ]] && kubectl delete "${claim}" -n "${NAMESPACE}" --ignore-not-found
  done <<< "${claim_names}"
fi

kubectl get pods,svc,job,statefulset,deployment,computedomain.resource.nvidia.com,resourceclaimtemplate,resourceclaim \
  -n "${NAMESPACE}" -l "app=${APP}" -o name || true
