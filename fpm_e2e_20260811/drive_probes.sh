#!/usr/bin/env bash
# L2 probe driver: stage the bundle into the keepalive pod, run the explicit
# -points benchmark 3 times, harvest results. Stops on first failure.
set -uo pipefail
TP=$1
POD=$2
CTX=nv-prd-dgxc.teleport.sh-dynamo-nebius-2
NS=yuanli-aic
B=fpm_e2e_20260811/probe_bundle_tep$TP
OUT=fpm_e2e_20260811/probes/tep$TP
mkdir -p "$OUT"
K="kubectl --context=$CTX -n $NS"

$K exec $POD -- mkdir -p /tmp/fpm-probe || { echo "STAGE-FAIL mkdir"; exit 1; }
for f in run.sh fpm_env.sh points.json; do
  $K cp "$B/$f" "$POD:/tmp/fpm-probe/$f" || { echo "STAGE-FAIL cp $f"; exit 1; }
done
echo "staged tep$TP into $POD"

for r in 1 2 3 4 5; do
  echo "tep$TP repeat $r: engine start $(date +%H:%M:%S)"
  $K exec $POD -- bash -c "PROBE_OUT=/results/probe_r$r.json bash /tmp/fpm-probe/run.sh > /results/probe_r$r.stdout.log 2>&1"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "tep$TP repeat $r: ENGINE-FAIL exit=$rc"
    $K cp "$POD:/results/probe_r$r.stdout.log" "$OUT/probe_r$r.stdout.log" >/dev/null 2>&1
    tail -5 "$OUT/probe_r$r.stdout.log" 2>/dev/null
    exit $rc
  fi
  $K cp "$POD:/results/probe_r$r.json" "$OUT/probe_r$r.json" || { echo "tep$TP repeat $r: HARVEST-FAIL"; exit 1; }
  $K cp "$POD:/results/probe_r$r.stdout.log" "$OUT/probe_r$r.stdout.log" >/dev/null 2>&1
  n=$(python3 -c "import json;d=json.load(open('$OUT/probe_r$r.json'));print(len(d) if isinstance(d,list) else len(d.get('points',d.get('results',[]))))" 2>/dev/null || echo "?")
  echo "tep$TP repeat $r: OK, harvested $n entries, done $(date +%H:%M:%S)"
done
echo "tep$TP ALL-REPEATS-DONE"
