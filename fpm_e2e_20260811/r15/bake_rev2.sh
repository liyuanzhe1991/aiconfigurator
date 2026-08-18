#!/usr/bin/env bash
# rev2 镜像层制作(pod 侧,验证 PASS 后执行):
# 基底 gc-realcontent-20260818(kvwarm+content_v4+argsdump 已烘焙),
# 增量层 = 叠加 timing_patch 后的 instrumented_scheduler.py 单文件。
# 产物:/workspace/model_cache/fpm_layers/layer_rev2.tar + sha256。
set -euo pipefail
W=/tmp/fpm-kvwarm
SP=/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py
OUT=/workspace/model_cache/fpm_layers
mkdir -p "$OUT"
# 从烘焙态快照重建 + 叠加 timing(不信任验证跑后的现场文件,重做一遍手术)
cp "$W/baked_scheduler.py" "$SP"
python3 "$W/timing_patch.py"
python3 -c "compile(open('$SP').read(), 'sp', 'exec'); print('compile OK')"
grep -c "__dyn_timing_patch__" "$SP" | grep -qx 1 || { echo BAKE-FAIL-marker; exit 1; }
STAGE=$(mktemp -d)
mkdir -p "$STAGE/usr/local/lib/python3.12/dist-packages/dynamo/vllm"
cp "$SP" "$STAGE/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py"
tar -C "$STAGE" -cf "$OUT/layer_rev2.tar" usr
sha256sum "$OUT/layer_rev2.tar" | tee "$OUT/layer_rev2.tar.sha256"
echo BAKE-LAYER-DONE
