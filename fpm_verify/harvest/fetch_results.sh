#!/usr/bin/env bash
# 分块取件(固定方法论;缺陷4 教训制度化):16MB/块 + 逐块 sha256 + 重组校验。
# 用法: fetch_results.sh <topo> <ctx> <outdir> [ns]
set -uo pipefail
TOPO=${1:?}; CTX=${2:?}; OUT=${3:?}; NS=${4:-yuanli-aic}
POD="fpm-l3-$TOPO-agg"
K() { kubectl --context "$CTX" "$@"; }
mkdir -p "$OUT"
CHUNK=$((16*1024*1024))

fetch_one() { # $1=pod侧路径
  local f base size nchunks ref got i lo tries
  f="$1"; base=$(basename "$f")
  size=$(K exec -n $NS $POD -- stat -c%s "$f" 2>/dev/null) || { echo "SKIP $base(不存在)"; return 0; }
  ref=$(K exec -n $NS $POD -- sha256sum "$f" | cut -d" " -f1)
  nchunks=$(( (size + CHUNK - 1) / CHUNK ))
  : > "$OUT/$base"
  for i in $(seq 0 $((nchunks-1))); do
    lo=$((i*CHUNK))
    local cref cgot
    cref=$(K exec -n $NS $POD -- bash -c "tail -c +$((lo+1)) '$f' | head -c $CHUNK | sha256sum" | cut -d" " -f1)
    for tries in 1 2 3 4 5; do
      K exec -n $NS $POD -- bash -c "tail -c +$((lo+1)) '$f' | head -c $CHUNK" > "$OUT/.chunk" 2>/dev/null
      cgot=$(shasum -a 256 "$OUT/.chunk" | cut -d" " -f1)
      [ "$cref" = "$cgot" ] && break
      sleep $((tries*2))
    done
    [ "$cref" = "$cgot" ] || { echo "FETCH-FAIL $base chunk$i"; return 1; }
    cat "$OUT/.chunk" >> "$OUT/$base"
  done
  rm -f "$OUT/.chunk"
  got=$(shasum -a 256 "$OUT/$base" | cut -d" " -f1)
  [ "$ref" = "$got" ] && echo "OK $base ($size B, $nchunks 块)" || { echo "FETCH-FAIL $base 整文件"; return 1; }
}

for f in fpm_stream.jsonl decode_windows.tsv burst_windows.tsv mixed_windows.tsv \
         l3_timing.log burst_driver.log resolved-config-node0.json; do
  fetch_one "/results/$f" || exit 1
done
shasum -a 256 "$OUT"/* > "$OUT/SHA256SUMS.txt" 2>/dev/null
echo "FETCH-DONE $TOPO -> $OUT"
