#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_env.sh"

cleanup_on_error() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    local diag_dir="${ARTIFACT_DIR}/diag_run_failed_$(date +%Y%m%d%H%M%S)"
    mkdir -p "${diag_dir}"
    echo "run failed with rc=${rc}; collecting diagnostics in ${diag_dir} before cleanup" >&2
    kubectl get pods,svc,job,statefulset,deployment,computedomain.resource.nvidia.com,resourceclaimtemplate,resourceclaim \
      -n "${NAMESPACE}" -l "app=${APP}" -o wide | tee "${diag_dir}/resources.txt" || true
    while IFS= read -r pod_name; do
      [[ -n "${pod_name}" ]] && kubectl logs "${pod_name}" -n "${NAMESPACE}" --all-containers=true --tail=-1 \
        >"${diag_dir}/${pod_name#pod/}.log" 2>&1 || true
    done < <(kubectl get pod -n "${NAMESPACE}" -l "app=${APP}" -o name 2>/dev/null || true)
    while IFS= read -r job_name; do
      [[ -n "${job_name}" ]] && kubectl logs "${job_name}" -n "${NAMESPACE}" --all-containers=true --tail=-1 \
        >"${diag_dir}/${job_name#job/}.log" 2>&1 || true
    done < <(kubectl get job -n "${NAMESPACE}" -l "app=${APP}" -o name 2>/dev/null || true)
    "${SCRIPT_DIR}/cleanup.sh" || true
  fi
  exit "${rc}"
}
trap cleanup_on_error EXIT

wait_job_terminal() {
  local job_name="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    local conditions
    conditions="$(kubectl get job "${job_name}" -n "${NAMESPACE}" -o jsonpath='{range .status.conditions[*]}{.type}={.status}{" "}{end}' 2>/dev/null || true)"
    if [[ "${conditions}" == *"Complete=True"* ]]; then
      return 0
    fi
    if [[ "${conditions}" == *"Failed=True"* ]]; then
      kubectl logs job/"${job_name}" -n "${NAMESPACE}" --all-containers=true --tail=-1 || true
      return 1
    fi
    sleep 30
  done
  echo "timed out waiting for job/${job_name}" >&2
  kubectl logs job/"${job_name}" -n "${NAMESPACE}" --all-containers=true --tail=-1 || true
  return 124
}

wait_backend_health() {
  local name="$1"
  local url="$2"
  local timeout_seconds="$3"
  local required_hits="${4:-3}"
  local deadline=$((SECONDS + timeout_seconds))
  local hits=0
  local last="not checked"
  local out

  while (( SECONDS < deadline )); do
    if out="$(kubectl exec deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" -- python3 -c '
import sys, urllib.error, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = resp.read().decode("utf-8", "replace")
        print(f"HTTP {resp.status} {body[:200]}", flush=True)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(f"HTTP {exc.code} {body[:200]}", flush=True)
except Exception as exc:
    print(f"ERR {exc!r}", flush=True)
' "${url}" 2>&1)"; then
      last="${out}"
      if [[ "${out}" == HTTP\ 200* ]]; then
        hits=$((hits + 1))
        echo "${name} health: ${out}; consecutive_200=${hits}/${required_hits}"
        if (( hits >= required_hits )); then
          echo "${name} health stable: ${hits}/${required_hits} consecutive HTTP 200"
          return 0
        fi
      else
        hits=0
        echo "waiting ${name} health: ${last}; consecutive_200=${hits}/${required_hits}"
      fi
    else
      hits=0
      last="kubectl exec failed: ${out}"
      echo "waiting ${name} health: ${last}; consecutive_200=${hits}/${required_hits}" >&2
    fi
    sleep 10
  done

  echo "${name} health did not become 200 at ${url}: ${last}" >&2
  return 124
}

wait_router_completion() {
  local timeout_seconds="$1"
  local base="http://${ROUTER_SERVICE}.${NAMESPACE}.svc.cluster.local:8000"
  local deadline=$((SECONDS + timeout_seconds))
  local last="not checked"
  local out

  while (( SECONDS < deadline )); do
    if out="$(kubectl exec deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" -- python3 -c '
import json, sys, urllib.error, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model, "prompt": "hello", "max_tokens": 1, "temperature": 0}
req = urllib.request.Request(
    base + "/v1/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", "replace")
        print(f"HTTP {resp.status} {body[:500]}", flush=True)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(f"HTTP {exc.code} {body[:500]}", flush=True)
except Exception as exc:
    print(f"ERR {exc!r}", flush=True)
' "${base}" "deepseek-ai/DeepSeek-V4-Pro" 2>&1)"; then
      last="${out}"
      echo "router completion probe: ${last}"
      [[ "${out}" == HTTP\ 200* ]] && return 0
    else
      last="kubectl exec failed: ${out}"
      echo "waiting router completion path: ${last}" >&2
    fi
    sleep 10
  done

  echo "router /v1/completions did not become usable: ${last}" >&2
  return 124
}

kubectl auth can-i get pods -n "${NAMESPACE}"

"${SCRIPT_DIR}/cleanup.sh" || true

if ! kubectl get secret "${IMAGE_PULL_SECRET}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "image pull secret ${IMAGE_PULL_SECRET} is missing in ${NAMESPACE}; copying from kprashanth"
  kubectl get secret "${IMAGE_PULL_SECRET}" -n kprashanth -o json \
    | python3 -c 'import json,sys; o=json.load(sys.stdin); o["metadata"]={"name":"nvcrimagepullsecret","namespace":"default"}; print(json.dumps(o))' \
    | kubectl apply --validate=false -n "${NAMESPACE}" -f -
fi

kubectl delete configmap "${SCHEDULER_NVTX_CONFIGMAP}" -n "${NAMESPACE}" --ignore-not-found
kubectl create configmap "${SCHEDULER_NVTX_CONFIGMAP}" -n "${NAMESPACE}" \
  --from-file=apply_scheduler_nvtx_patch.py="${SCRIPT_DIR}/apply_scheduler_nvtx_patch.py"
kubectl label configmap "${SCHEDULER_NVTX_CONFIGMAP}" -n "${NAMESPACE}" \
  "app=${APP}" role=scheduler-nvtx-patch --overwrite

kubectl apply --validate=false -f "${SCRIPT_DIR}/native_sglang_nsys.yaml"

echo "waiting for native SGLang pods to be created"
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  kubectl get pods -n "${NAMESPACE}" -l "app=${APP}" -o wide || true
  prefill_count="$(kubectl get pod -n "${NAMESPACE}" -l "app=${APP},component=prefill" --no-headers 2>/dev/null | wc -l)"
  decode_count="$(kubectl get pod -n "${NAMESPACE}" -l "app=${APP},component=decode" --no-headers 2>/dev/null | wc -l)"
  if [[ "${prefill_count}" -eq 2 && "${decode_count}" -eq 2 ]]; then
    break
  fi
  sleep 20
done

kubectl rollout status statefulset/"${PREFILL_STS}" -n "${NAMESPACE}" --timeout=2h
kubectl rollout status statefulset/"${DECODE_STS}" -n "${NAMESPACE}" --timeout=2h
kubectl rollout status deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=30m

wait_backend_health "prefill" "http://${PREFILL_LEADER_SERVICE}.${NAMESPACE}.svc.cluster.local:30000/health" 7200
wait_backend_health "decode" "http://${DECODE_LEADER_SERVICE}.${NAMESPACE}.svc.cluster.local:30000/health" 7200

echo "restarting router after P/D workers are ready so worker registration is fresh"
kubectl rollout restart deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}"
kubectl rollout status deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=30m
wait_backend_health "router" "http://${ROUTER_SERVICE}.${NAMESPACE}.svc.cluster.local:8000/health" 1800
wait_router_completion 1800

"${SCRIPT_DIR}/pd_profile_gate.sh"

echo "deleting native SGLang workloads so nsys can flush .nsys-rep files"
kubectl delete deployment "${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete statefulset "${PREFILL_STS}" "${DECODE_STS}" -n "${NAMESPACE}" --ignore-not-found
sleep 180
kubectl wait --for=delete pod -n "${NAMESPACE}" -l "app=${APP},component in (prefill,decode,router)" --timeout=10m || true

kubectl apply --validate=false -f "${SCRIPT_DIR}/collect_results.yaml"
wait_job_terminal "${RESULTS_JOB}" 600
kubectl logs job/"${RESULTS_JOB}" -n "${NAMESPACE}" --all-containers=true --tail=-1 | tee "${ARTIFACT_DIR}/result_files.log"

"${SCRIPT_DIR}/cleanup.sh"
trap - EXIT

echo "completed. Remote result root on PVC: ${REMOTE_RESULT_ROOT}"
