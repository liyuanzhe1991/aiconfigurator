#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_env.sh"

PD_GATE_DIR="${ARTIFACT_DIR}/pd_gate_logs"
E2E_REQUEST_MULTIPLIER="${E2E_REQUEST_MULTIPLIER:-2}"
PREFILL_PROFILE_STEPS="${PREFILL_PROFILE_STEPS:-16}"
PREFILL_PROFILE_ENABLED="${PREFILL_PROFILE_ENABLED:-1}"
PREFILL_VERIFY_MIN_TARGET_HITS="${PREFILL_VERIFY_MIN_TARGET_HITS:-4}"
DECODE_PROFILE_STEPS="${DECODE_PROFILE_STEPS:-16}"
DECODE_PROFILE_ENABLED="${DECODE_PROFILE_ENABLED:-1}"
DECODE_VERIFY_MIN_TARGET_HITS="${DECODE_VERIFY_MIN_TARGET_HITS:-4}"
E2E_GLOBAL_CONCURRENCY_OVERRIDE="${E2E_GLOBAL_CONCURRENCY_OVERRIDE:-}"

mkdir -p "${PD_GATE_DIR}" "${ARTIFACT_DIR}/profile_requests"

CASE_LOG_DIR=""
LOG_WATCH_PIDS=()

stop_log_watch() {
  if ((${#LOG_WATCH_PIDS[@]})); then
    for pid in "${LOG_WATCH_PIDS[@]}"; do
      kill "${pid}" >/dev/null 2>&1 || true
    done
    wait "${LOG_WATCH_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  LOG_WATCH_PIDS=()
}

start_component_log_watch() {
  local component="$1"
  local expected_count="$2"
  local out_dir="${CASE_LOG_DIR}/${component}"

  mkdir -p "${out_dir}"
  mapfile -t pods < <(
    kubectl get pod -n "${NAMESPACE}" -l "app=${APP},component=${component}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort
  )
  if [[ "${#pods[@]}" -ne "${expected_count}" ]]; then
    printf 'expected %s %s pods, got %s\n' "${expected_count}" "${component}" "${#pods[@]}" >&2
    return 1
  fi

  for pod in "${pods[@]}"; do
    kubectl logs -n "${NAMESPACE}" -f "${pod}" --tail=0 >"${out_dir}/${pod}.log" 2>&1 &
    LOG_WATCH_PIDS+=("$!")
  done
}

start_log_watch() {
  local case_name="$1"
  CASE_LOG_DIR="${PD_GATE_DIR}/${case_name}"
  rm -rf "${CASE_LOG_DIR}"
  mkdir -p "${CASE_LOG_DIR}"
  LOG_WATCH_PIDS=()
  start_component_log_watch prefill "${PREFILL_NODE_COUNT}"
  start_component_log_watch decode "${DECODE_NODE_COUNT}"
  sleep 2
}

parse_prefill_logs() {
  local mode="$1"
  local case_name="$2"
  local min_target_hits="${3:-1}"
  python3 - "${mode}" "${CASE_LOG_DIR}/prefill" "${case_name}" "${PREFILL_DP}" "${ISL}" "${PREFILL_LOCAL_BS}" "${min_target_hits}" <<'PY'
import pathlib
import re
import sys

mode, log_dir, case_name, prefill_dp, target_tokens, target_seqs, min_target_hits = (
    sys.argv[1],
    pathlib.Path(sys.argv[2]),
    sys.argv[3],
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
    int(sys.argv[7]),
)
expected = set(range(prefill_dp))
prefill_re = re.compile(
    r"\bDP(\d+)\b.*Prefill batch.*#new-seq:\s*(\d+),\s*#new-token:\s*(\d+)"
)
profile_start_re = re.compile(r"\bDP(\d+)\b.*Profiling starts.*" + re.escape(case_name))
profile_done_re = re.compile(r"\bDP(\d+)\b.*Profiling done.*" + re.escape(case_name))

target_seen_counts = {rank: 0 for rank in expected}
target_after_start_counts = {rank: 0 for rank in expected}
started = set()
done = set()

for path in sorted(log_dir.glob("*.log")):
    active_ranks = set()
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        continue
    for line in lines:
        m = profile_start_re.search(line)
        if m:
            rank = int(m.group(1))
            started.add(rank)
            active_ranks.add(rank)
        m = prefill_re.search(line)
        if m:
            rank = int(m.group(1))
            new_seq = int(m.group(2))
            new_token = int(m.group(3))
            if new_seq == target_seqs and new_token == target_tokens:
                target_seen_counts[rank] = target_seen_counts.get(rank, 0) + 1
                if rank in active_ranks:
                    target_after_start_counts[rank] = (
                        target_after_start_counts.get(rank, 0) + 1
                    )
        m = profile_done_re.search(line)
        if m:
            done.add(int(m.group(1)))

if mode == "gate":
    total = sum(target_seen_counts.values())
    ok = total >= 1
    print(f"{mode}: target_prefill_hits={total} counts={target_seen_counts}", flush=True)
elif mode == "done":
    missing = sorted(expected - done)
    ok = done >= expected
    print(f"{mode}: done={sorted(done)} missing={missing}", flush=True)
elif mode == "verify":
    total = sum(target_after_start_counts.values())
    ok = total >= min_target_hits
    print(
        f"{mode}: min_hits={min_target_hits} target_after_start_hits={total} "
        f"counts={target_after_start_counts}",
        flush=True,
    )
else:
    raise SystemExit(f"unknown mode: {mode}")

raise SystemExit(0 if ok else 1)
PY
}

parse_decode_logs() {
  local mode="$1"
  local case_name="$2"
  local target_bs="$3"
  local min_target_hits="${4:-1}"
  python3 - "${mode}" "${CASE_LOG_DIR}/decode" "${case_name}" "${target_bs}" "${DECODE_DP}" "${min_target_hits}" <<'PY'
import pathlib
import re
import sys

mode, log_dir, case_name, target_bs, decode_dp, min_target_hits = (
    sys.argv[1],
    pathlib.Path(sys.argv[2]),
    sys.argv[3],
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)
expected = set(range(decode_dp))
decode_re = re.compile(r"\bDP(\d+)\b.*Decode batch.*#running-req:\s*(\d+)")
profile_start_re = re.compile(r"\bDP(\d+)\b.*Profiling starts.*" + re.escape(case_name))
profile_done_re = re.compile(r"\bDP(\d+)\b.*Profiling done.*" + re.escape(case_name))

target_seen = set()
target_after_start_counts = {rank: 0 for rank in expected}
done = set()

for path in sorted(log_dir.glob("*.log")):
    active_ranks = set()
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        continue
    for line in lines:
        m = profile_start_re.search(line)
        if m:
            active_ranks.add(int(m.group(1)))
        m = decode_re.search(line)
        if m:
            rank = int(m.group(1))
            bs = int(m.group(2))
            if bs == target_bs:
                target_seen.add(rank)
                if rank in active_ranks:
                    target_after_start_counts[rank] = (
                        target_after_start_counts.get(rank, 0) + 1
                    )
        m = profile_done_re.search(line)
        if m:
            done.add(int(m.group(1)))

if mode == "gate":
    seen = target_seen
    ok = seen >= expected
    missing = sorted(expected - seen)
    print(f"{mode}: target_bs={target_bs} seen={sorted(seen)} missing={missing}", flush=True)
elif mode == "done":
    seen = done
    ok = seen >= expected
    missing = sorted(expected - seen)
    print(f"{mode}: done={sorted(seen)} missing={missing}", flush=True)
elif mode == "verify":
    seen = {rank for rank, count in target_after_start_counts.items() if count >= min_target_hits}
    ok = seen >= expected
    missing = sorted(expected - seen)
    print(
        f"{mode}: target_bs={target_bs} min_hits={min_target_hits} "
        f"counts={target_after_start_counts} seen={sorted(seen)} missing={missing}",
        flush=True,
    )
else:
    raise SystemExit(f"unknown mode: {mode}")

raise SystemExit(0 if ok else 1)
PY
}

wait_prefill_gate() {
  local case_name="$1"
  local timeout_seconds="$2"
  local load_job="${3:-}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if parse_prefill_logs gate "${case_name}"; then
      return 0
    fi
    if [[ -n "${load_job}" ]] && load_job_is_terminal "${load_job}" "${CASE_LOG_DIR}/aiperf_job_early.log"; then
      echo "E2E load job ${load_job} finished before prefill gate reached #new-seq=${PREFILL_LOCAL_BS}, #new-token=${ISL}" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "timed out waiting for prefill batch #new-seq=${PREFILL_LOCAL_BS}, #new-token=${ISL}" >&2
  return 1
}

wait_decode_gate() {
  local case_name="$1"
  local target_bs="$2"
  local timeout_seconds="$3"
  local load_job="${4:-}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if parse_decode_logs gate "${case_name}" "${target_bs}"; then
      return 0
    fi
    if [[ -n "${load_job}" ]] && load_job_is_terminal "${load_job}" "${CASE_LOG_DIR}/aiperf_job_early.log"; then
      echo "E2E load job ${load_job} finished before decode gate reached local bs ${target_bs}" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "timed out waiting for all decode DP ranks to reach local bs ${target_bs}" >&2
  return 1
}

load_job_is_terminal() {
  local load_job="$1"
  local log_path="$2"
  local conditions
  conditions="$(kubectl get job "${load_job}" -n "${NAMESPACE}" -o jsonpath='{range .status.conditions[*]}{.type}={.status}{" "}{end}' 2>/dev/null || true)"
  if [[ "${conditions}" == *"Complete=True"* || "${conditions}" == *"Failed=True"* ]]; then
    kubectl logs job/"${load_job}" -n "${NAMESPACE}" --all-containers=true --tail=-1 >"${log_path}" 2>&1 || true
    echo "load job ${load_job} terminal conditions: ${conditions}" >&2
    return 0
  fi
  return 1
}

wait_profile_done() {
  local component="$1"
  local case_name="$2"
  local target_bs="${3:-0}"
  local timeout_seconds="$4"
  local load_job="${5:-}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if [[ "${component}" == "prefill" ]]; then
      if parse_prefill_logs done "${case_name}"; then
        return 0
      fi
    else
      if parse_decode_logs done "${case_name}" "${target_bs}"; then
        return 0
      fi
    fi
    if [[ -n "${load_job}" ]] && load_job_is_terminal "${load_job}" "${CASE_LOG_DIR}/aiperf_job_early.log"; then
      echo "E2E load job ${load_job} finished before ${component} profile ${case_name} completed" >&2
      return 1
    fi
    sleep 2
  done
  echo "timed out waiting for ${component} profile ${case_name} to finish" >&2
  return 1
}

post_start_profile() {
  local component="$1"
  local case_name="$2"
  local num_steps="$3"
  local profile_by_stage="$4"
  local profile_stage="${5:-}"
  local base
  local stage_json=""
  if [[ "${component}" == "prefill" ]]; then
    base="http://${PREFILL_LEADER_SERVICE}.${NAMESPACE}.svc.cluster.local:30000"
  else
    base="http://${DECODE_LEADER_SERVICE}.${NAMESPACE}.svc.cluster.local:30000"
  fi

  local req_file="${ARTIFACT_DIR}/profile_requests/${case_name}_start.json"
  if [[ -n "${profile_stage}" ]]; then
    stage_json="$(printf ',\n    "profile_stages": ["%s"]' "${profile_stage}")"
  fi
  mkdir -p "$(dirname "${req_file}")"
  cat >"${req_file}" <<EOF
{
  "component": "${component}",
  "url": "${base}/start_profile",
  "payload": {
    "output_dir": "${REMOTE_RESULT_ROOT}/sglang_profiles/${case_name}",
    "activities": ["CUDA_PROFILER"],
    "profile_prefix": "${case_name}",
    "profile_by_stage": ${profile_by_stage},
    "num_steps": ${num_steps}${stage_json}
  }
}
EOF

  kubectl exec -i deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" -- python3 - \
    "${base}" "${component}" "${case_name}" "${num_steps}" "${REMOTE_RESULT_ROOT}" \
    "${profile_by_stage}" "${profile_stage}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

base, component, case_name, num_steps, result_root, profile_by_stage, profile_stage = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    sys.argv[5],
    sys.argv[6].lower() == "true",
    sys.argv[7],
)
payload = {
    "output_dir": f"{result_root}/sglang_profiles/{case_name}",
    "activities": ["CUDA_PROFILER"],
    "profile_prefix": case_name,
    "profile_by_stage": profile_by_stage,
    "num_steps": num_steps,
}
if profile_stage:
    payload["profile_stages"] = [profile_stage]

deadline = time.time() + 120
attempt = 0
while True:
    attempt += 1
    req = urllib.request.Request(
        base + "/start_profile",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
            print(f"start_profile {component} {case_name}: {resp.status} {body[:1000]}", flush=True)
            raise SystemExit(0)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(
            f"start_profile retry {attempt} {component} {case_name}: HTTP {exc.code} {body[:500]}",
            flush=True,
        )
        if exc.code not in (404, 409, 425, 500, 502, 503, 504) or time.time() > deadline:
            raise
    except Exception as exc:
        print(f"start_profile retry {attempt} {component} {case_name}: {exc!r}", flush=True)
        if time.time() > deadline:
            raise
    time.sleep(2)
PY
}

wait_router_completion_ready() {
  local timeout_seconds="$1"
  local base="http://${ROUTER_SERVICE}.${NAMESPACE}.svc.cluster.local:8000"
  local deadline=$((SECONDS + timeout_seconds))
  local last="not checked"
  local out

  while (( SECONDS < deadline )); do
    if out="$(kubectl exec deployment/"${ROUTER_DEPLOYMENT}" -n "${NAMESPACE}" -- python3 -c '
import json, sys, urllib.error, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model, "prompt": "ready", "max_tokens": 1, "temperature": 0}
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
      echo "router completion ready probe: ${last}"
      [[ "${out}" == HTTP\ 200* ]] && return 0
    else
      last="kubectl exec failed: ${out}"
      echo "waiting router completion ready path: ${last}" >&2
    fi
    sleep 10
  done

  echo "router /v1/completions did not become usable before load: ${last}" >&2
  return 124
}

create_e2e_load_job() {
  local local_bs="$1"
  local load_case="$2"
  local concurrency
  if [[ -n "${E2E_GLOBAL_CONCURRENCY_OVERRIDE}" ]]; then
    concurrency="${E2E_GLOBAL_CONCURRENCY_OVERRIDE}"
  else
    concurrency=$((local_bs * DECODE_DP))
  fi
  local num_requests=$((concurrency * E2E_REQUEST_MULTIPLIER))
  if [[ "${PREFILL_PROFILE_ENABLED}" == "1" ]]; then
    local min_requests_for_prefill_profile=$(((PREFILL_PROFILE_STEPS + 32) * PREFILL_DP))
    if ((num_requests < min_requests_for_prefill_profile)); then
      num_requests="${min_requests_for_prefill_profile}"
    fi
  fi
  local load_job="${BASE}-e2e-bs${local_bs}-${RUN_ID}"

  kubectl delete job "${load_job}" -n "${NAMESPACE}" --ignore-not-found
  cat <<EOF | kubectl apply --validate=false -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${load_job}
  namespace: ${NAMESPACE}
  labels:
    app: ${APP}
    role: e2e-load
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: ${APP}
        role: e2e-load
    spec:
      restartPolicy: Never
      nodeSelector:
        node.kubernetes.io/instance-type: n2d-standard-8
      tolerations:
        - key: dedicated
          operator: Equal
          value: user-workload
          effect: NoExecute
      containers:
        - name: aiperf
          image: python:3.12-slim
          imagePullPolicy: IfNotPresent
          env:
            - name: RESULT_ROOT
              value: ${REMOTE_RESULT_ROOT}
            - name: MODEL
              value: deepseek-ai/DeepSeek-V4-Pro
            - name: TOKENIZER
              value: deepseek-ai/DeepSeek-V4-Pro
            - name: ROUTER_URL
              value: http://${ROUTER_SERVICE}.${NAMESPACE}.svc.cluster.local:8000
            - name: HF_HOME
              value: /models
            - name: HF_HUB_OFFLINE
              value: "1"
            - name: TRANSFORMERS_OFFLINE
              value: "1"
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euxo pipefail
              python3 -m pip install --no-cache-dir aiperf requests
              mkdir -p "\${RESULT_ROOT}/aiperf/${load_case}" "\${RESULT_ROOT}/profile_logs"
              aiperf profile \\
                --artifact-dir "\${RESULT_ROOT}/aiperf/${load_case}" \\
                -m "\${MODEL}" \\
                --endpoint-type completions \\
                -u "\${ROUTER_URL}" \\
                --tokenizer "\${TOKENIZER}" \\
                --isl ${ISL} --isl-stddev 0 \\
                --osl ${OSL} --osl-stddev 0 \\
                --extra-inputs ignore_eos:true \\
                --extra-inputs '{"nvext":{"ignore_eos":true}}' \\
                --use-legacy-max-tokens \\
                --concurrency ${concurrency} \\
                --num-requests ${num_requests} \\
                --random-seed 100 \\
                --ui simple \\
                --streaming 2>&1 | tee "\${RESULT_ROOT}/profile_logs/${load_case}_aiperf.log"
          volumeMounts:
            - name: shared
              mountPath: /models
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: ${PVC_NAME}
EOF
}

run_prefill_profile_worker() {
  local prefill_case="$1"
  local load_job="$2"
  wait_prefill_gate "${prefill_case}" 900 "${load_job}" | tee "${CASE_LOG_DIR}/prefill_gate.log"
  post_start_profile prefill "${prefill_case}" "${PREFILL_PROFILE_STEPS}" false "" \
    | tee "${CASE_LOG_DIR}/prefill_start_profile.log"
  wait_profile_done prefill "${prefill_case}" 0 900 "${load_job}" | tee "${CASE_LOG_DIR}/prefill_profile_done.log"
  parse_prefill_logs verify "${prefill_case}" "${PREFILL_VERIFY_MIN_TARGET_HITS}" \
    | tee "${CASE_LOG_DIR}/prefill_verify.log"
}

run_decode_profile_worker() {
  local decode_case="$1"
  local local_bs="$2"
  local load_job="$3"
  wait_decode_gate "${decode_case}" "${local_bs}" 900 "${load_job}" | tee "${CASE_LOG_DIR}/decode_gate.log"
  post_start_profile decode "${decode_case}" "${DECODE_PROFILE_STEPS}" false "" \
    | tee "${CASE_LOG_DIR}/decode_start_profile.log"
  wait_profile_done decode "${decode_case}" "${local_bs}" 900 "${load_job}" | tee "${CASE_LOG_DIR}/decode_profile_done.log"
  parse_decode_logs verify "${decode_case}" "${local_bs}" "${DECODE_VERIFY_MIN_TARGET_HITS}" \
    | tee "${CASE_LOG_DIR}/decode_verify.log"
}

run_pd_profile_case() {
  local local_bs="$1"
  local concurrency
  if [[ -n "${E2E_GLOBAL_CONCURRENCY_OVERRIDE}" ]]; then
    concurrency="${E2E_GLOBAL_CONCURRENCY_OVERRIDE}"
  else
    concurrency=$((local_bs * DECODE_DP))
  fi
  local load_case="e2e_pd_dp${DECODE_DP}_ep${DECODE_EP}_prefill_bs${PREFILL_LOCAL_BS}_decode_local_bs_${local_bs}_global_concurrency_${concurrency}_isl${ISL}_osl${OSL}"
  local prefill_case="prefill_dp${PREFILL_DP}_ep${PREFILL_EP}_bs${PREFILL_LOCAL_BS}_isl${ISL}_during_decode_local_bs_${local_bs}"
  local decode_case="decode_dp${DECODE_DP}_ep${DECODE_EP}_local_bs_${local_bs}_global_concurrency_${concurrency}_isl${ISL}_osl${OSL}"
  local load_job="${BASE}-e2e-bs${local_bs}-${RUN_ID}"
  local prefill_pid
  local decode_pid
  local rc=0

  echo "starting E2E PD profile case ${load_case}"
  wait_router_completion_ready 600
  start_log_watch "${load_case}"
  trap 'stop_log_watch; kubectl delete job "'"${load_job}"'" -n "'"${NAMESPACE}"'" --ignore-not-found >/dev/null 2>&1 || true' RETURN

  create_e2e_load_job "${local_bs}" "${load_case}"

  if [[ "${PREFILL_PROFILE_ENABLED}" == "1" ]]; then
    run_prefill_profile_worker "${prefill_case}" "${load_job}" &
    prefill_pid=$!
  else
    prefill_pid=""
  fi
  if [[ "${DECODE_PROFILE_ENABLED}" == "1" ]]; then
    run_decode_profile_worker "${decode_case}" "${local_bs}" "${load_job}" &
    decode_pid=$!
  else
    decode_pid=""
  fi

  if [[ -n "${prefill_pid}" ]]; then
    wait "${prefill_pid}" || rc=1
  fi
  if [[ -n "${decode_pid}" ]]; then
    wait "${decode_pid}" || rc=1
  fi

  kubectl logs job/"${load_job}" -n "${NAMESPACE}" --all-containers=true --tail=-1 >"${CASE_LOG_DIR}/aiperf_job.log" 2>&1 || true
  kubectl delete job "${load_job}" -n "${NAMESPACE}" --ignore-not-found
  stop_log_watch
  trap - RETURN

  if [[ "${rc}" -ne 0 ]]; then
    echo "PD profile case ${load_case} failed; see ${CASE_LOG_DIR}" >&2
    return "${rc}"
  fi

  sleep 30
}

for local_bs in ${DECODE_LOCAL_BS_LIST}; do
  run_pd_profile_case "${local_bs}"
done
