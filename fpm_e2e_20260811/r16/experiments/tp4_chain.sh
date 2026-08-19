#!/usr/bin/env bash
# tp4 最小机制实验链:守卫 -> tep4 节点探针 -> tp4 fake -> sed 放行 moe_tp -> tp4 warm
SP=/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py
bash /tmp/clock_guard.sh || { echo TP4EXP-FAIL-CLOCKGUARD; exit 9; }

run_arm() {
  bash "$2" "$1" &
  local APID=$!
  for i in $(seq 1 320); do
    grep -q "merged results" /results/$1/engine.log 2>/dev/null && break
    sleep 15
  done
  pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
  wait $APID 2>/dev/null
  echo "ARM-$1-FINISHED bench=$(ls -l /results/$1/benchmark.json 2>/dev/null | awk '{print $5}')"
}

# 节点效应探针:tep4 decode 171 点(与前两台 pod 同点单,唯一变量=节点)
run_arm arm1 /tmp/regime/regime_arm.sh

run_arm fake /tmp/tp4exp/tp4_arm.sh
python3 - <<'PY'
import json
d = json.load(open("/results/fake/benchmark.json"))
kw = d.get("kvwarm", {})
print("FAKE-ARM kvwarm:", kw.get("warm_eligible"), kw.get("skip_reason"))
assert kw.get("warm_eligible") is False and kw.get("skip_reason") == "moe_tp_balanced_by_construction", "fake 臂制度断言失败"
PY
[ $? -eq 0 ] || { echo TP4EXP-FAIL-FAKE-ASSERT; exit 8; }

cp "$SP" /tmp/tp4exp/scheduler.bak
grep -q "elif not ep_enabled:" "$SP" || { echo TP4EXP-FAIL-PATCH-ANCHOR; exit 8; }
sed -i 's/elif not ep_enabled:/elif False and not ep_enabled:/' "$SP"
grep -q "elif False and not ep_enabled:" "$SP" || { echo TP4EXP-FAIL-PATCH-VERIFY; exit 8; }
echo "PATCH-APPLIED (moe_tp skip disabled)"

run_arm warm
python3 - <<'PY'
import json
d = json.load(open("/results/warm/benchmark.json"))
kw = d.get("kvwarm", {})
print("WARM-ARM kvwarm:", kw.get("warm_eligible"), kw.get("skip_reason"))
PY
echo "TP4EXP-ALL-DONE"
