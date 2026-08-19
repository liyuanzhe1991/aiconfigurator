#!/usr/bin/env bash
# 分块取件(固定方法论;缺陷4 教训制度化,dep4 968MB 两次断点实战版):
# 4MB gzip 块 + 逐块 sha256 + 跨次断点续传 + 重组校验。
# 续传语义:本地已有前缀视为有效(其每块当时已过 sha),从其末尾接着拉;
# 远端文件必须只增不改(收割产物均为追加型)。远端若缩短则报错重拉。
# 用法: fetch_results.sh <topo> <ctx> <outdir> [ns]
set -uo pipefail
TOPO=${1:?}; CTX=${2:?}; OUT=${3:?}; NS=${4:-yuanli-aic}
POD="fpm-l3-$TOPO-agg"
K() { kubectl --context "$CTX" "$@"; }
mkdir -p "$OUT"
CHUNK=$((4*1024*1024))

fetch_one() { # $1=pod侧路径
  local f base size ref got pos n tries cref cgot
  f="$1"; base=$(basename "$f")
  size=$(K exec -n $NS $POD -- stat -c%s "$f" 2>/dev/null) || { echo "SKIP $base(不存在)"; return 0; }
  ref=$(K exec -n $NS $POD -- sha256sum "$f" | cut -d" " -f1)
  pos=$(stat -f%z "$OUT/$base" 2>/dev/null || stat -c%s "$OUT/$base" 2>/dev/null || echo 0)
  if [ "$pos" -gt "$size" ]; then
    echo "WARN $base 本地($pos)大于远端($size),重拉" >&2
    : > "$OUT/$base"; pos=0
  fi
  [ "$pos" -gt 0 ] && echo "RESUME $base 从 $pos/$size"
  while [ "$pos" -lt "$size" ]; do
    n=$CHUNK; [ $((size-pos)) -lt $n ] && n=$((size-pos))
    cgot=""
    for tries in 1 2 3 4 5; do
      cref=$(K exec -n $NS $POD -- bash -c "tail -c +$((pos+1)) '$f' | head -c $n | sha256sum" | cut -d" " -f1)
      K exec -n $NS $POD -- bash -c "tail -c +$((pos+1)) '$f' | head -c $n | gzip -c | base64 -w0" > "$OUT/.chunk.b64" 2>/dev/null
      if base64 -d "$OUT/.chunk.b64" > "$OUT/.chunk.gz" 2>/dev/null || base64 -D -i "$OUT/.chunk.b64" -o "$OUT/.chunk.gz" 2>/dev/null; then
        gunzip -f "$OUT/.chunk.gz" 2>/dev/null && cgot=$(shasum -a 256 "$OUT/.chunk" | cut -d" " -f1)
      fi
      [ -n "$cgot" ] && [ "$cref" = "$cgot" ] && break
      cgot=""
      sleep $((tries*3))
    done
    [ -n "$cgot" ] || { echo "FETCH-FAIL $base at $pos(可重跑本脚本续传)"; return 1; }
    cat "$OUT/.chunk" >> "$OUT/$base"
    pos=$((pos+n))
  done
  rm -f "$OUT/.chunk" "$OUT/.chunk.b64" "$OUT/.chunk.gz"
  got=$(shasum -a 256 "$OUT/$base" | cut -d" " -f1)
  if [ "$ref" = "$got" ]; then
    echo "OK $base ($size B)"
  else
    # 远端在取件期间继续追加时,整文件 ref 会失配但本地前缀逐块已验——
    # 用前缀 sha 与远端同长前缀比对定裁
    local pref
    pref=$(K exec -n $NS $POD -- bash -c "head -c $size '$f' | sha256sum" | cut -d" " -f1)
    if [ "$pref" = "$got" ]; then
      echo "OK $base ($size B,远端仍在增长,前缀一致)"
    else
      echo "FETCH-FAIL $base 整文件"; return 1
    fi
  fi
}

for f in fpm_stream.jsonl decode_windows.tsv burst_windows.tsv mixed_windows.tsv \
         l3_timing.log burst_driver.log decode_driver.log resolved-config-node0.json nodeName.txt; do
  fetch_one "/results/$f" || exit 1
done
shasum -a 256 "$OUT"/* > "$OUT/SHA256SUMS.txt" 2>/dev/null
echo "FETCH-DONE $TOPO -> $OUT"
