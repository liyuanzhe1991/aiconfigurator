#!/bin/bash
# 时钟守卫 v2:轻负载 boost 检测。健康卡轻载升满频(~1980);
# 锁频卡(nvidia-smi -lgc 残留类)停在锁定值。满载功率墙降频是正常行为,不检
# (v1 教训:满载探针本身是功耗炸弹,健康四卡均匀降到 ~1700 造成误报)。
TH=${1:-1900}
FAIL=0
N=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python3 -c "
import torch, time
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 6:
    for _ in range(20): a = (a @ a).clamp(-1, 1)
    torch.cuda.synchronize(); time.sleep(0.05)" &
  BURN=$!
  PEAK=0
  for _ in 1 2 3 4 5; do
    sleep 1
    C=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i $i)
    [ "${C:-0}" -gt "$PEAK" ] && PEAK=$C
  done
  wait $BURN
  echo "GPU$i peak_sm=${PEAK}MHz"
  [ "$PEAK" -lt "$TH" ] && { echo "CLOCK-GUARD-FAIL GPU$i peak ${PEAK} < ${TH}"; FAIL=1; }
done
[ "$FAIL" = 0 ] && echo CLOCK-GUARD-PASS || exit 9
