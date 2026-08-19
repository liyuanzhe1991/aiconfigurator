#!/usr/bin/env bash
# QoS 定罪链:守卫 -> arm1 复跑(Guaranteed pod,171 点)-> 内容对拍 ABA
bash /tmp/clock_guard.sh || { echo QOS-FAIL-CLOCKGUARD; exit 9; }
bash /tmp/regime/regime_arm.sh arm1 &
APID=$!
for i in $(seq 1 240); do
  grep -q "merged results" /results/arm1/engine.log 2>/dev/null && break
  sleep 15
done
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
wait $APID 2>/dev/null
echo "QOS-ARM1-DONE bench=$(ls -l /results/arm1/benchmark.json 2>/dev/null | awk '{print $5}')"
bash /tmp/fpm-serve/content_chain.sh
E=$?
echo "REGIME2-ALL-DONE content_exit=$E"
