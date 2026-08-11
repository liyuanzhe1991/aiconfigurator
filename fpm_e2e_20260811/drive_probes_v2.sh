#!/usr/bin/env bash
# Probe repeats 2..5 via the in-pod executor (handles the no-auto-exit engine).
set -uo pipefail
TP=$1
POD=$2
CTX=nv-prd-dgxc.teleport.sh-dynamo-nebius-2
NS=yuanli-aic
OUT=fpm_e2e_20260811/probes/tep$TP
mkdir -p "$OUT"

for r in 2 3 4 5; do
  echo "tep$TP repeat $r: start $(date +%H:%M:%S)"
  kubectl --context=$CTX -n $NS exec $POD -- bash /tmp/fpm-probe/probe_exec.sh /results/probe_r$r.json
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "tep$TP repeat $r: EXEC-FAIL exit=$rc"
    kubectl --context=$CTX -n $NS cp "$POD:/results/probe_r$r.stdout.log" "$OUT/probe_r$r.stdout.log" >/dev/null 2>&1
    exit $rc
  fi
  kubectl --context=$CTX -n $NS cp "$POD:/results/probe_r$r.json" "$OUT/probe_r$r.json" || { echo "HARVEST-FAIL r$r"; exit 1; }
  echo "tep$TP repeat $r: OK $(date +%H:%M:%S)"
done
echo "tep$TP REPEATS-DONE"
