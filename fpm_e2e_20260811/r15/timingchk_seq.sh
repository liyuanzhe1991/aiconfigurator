#!/usr/bin/env bash
# timing 验证编排(pod 侧,自持续):A=烘焙基线 → B=+timing → C=+timing 重复。
# 产物落 PVC:/workspace/model_cache/fpm_timingchk_{A,B,C}/benchmark_on.json
set -uo pipefail
W=/tmp/fpm-kvwarm
MC=/workspace/model_cache
SLOG=$MC/fpm_timingchk_seq.log
mark() { echo "[chkseq $(date -u +%H:%M:%S)] $*" >> "$SLOG"; }
waitfile() { for _ in $(seq 1 "$2"); do [ -s "$1" ] && return 0; sleep "$3"; done; return 1; }

mark "CHK-START"
for ARM_OUT in "base:A" "timing:B" "timing:C"; do
  ARM=${ARM_OUT%%:*}; TAGC=${ARM_OUT##*:}
  cd "$W" && cp fpm_env_tep4.sh fpm_env.sh
  rm -rf "$MC/fpm_timingchk_$TAGC"
  bash timingchk_run.sh serve_r11_tep4_bench.sh points_timingchk.json \
    "$MC/fpm_timingchk_$TAGC" "$ARM" >"$MC/fpm_timingchk_${TAGC}_launch.log" 2>&1 \
    || { mark "CHK-FAIL $TAGC-launch"; exit 1; }
  waitfile "$MC/fpm_timingchk_$TAGC/benchmark_on.json" 40 60 \
    || { mark "CHK-FAIL $TAGC-timeout(40min)"; exit 1; }
  mark "CHK-$TAGC-DONE"
done
pkill -9 -f "dynamo[.]vllm" 2>/dev/null; pkill -9 -f "VLLM[:]:" 2>/dev/null
pkill -9 -f "dynamo[.]frontend" 2>/dev/null
pkill -9 -f "etcd --data-dir /tmp/fpm-forward-[e]tcd" 2>/dev/null
pkill -9 -f "nats-serve[r]" 2>/dev/null
mark "CHK-ALL-DONE"
