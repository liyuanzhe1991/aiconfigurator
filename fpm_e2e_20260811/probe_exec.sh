#!/usr/bin/env bash
# In-pod probe executor: launch the engine, wait for the benchmark output
# (this injection path lacks the CLI-driven auto-exit), then tear down.
OUT=$1
rm -f "$OUT"
PROBE_OUT=$OUT bash /tmp/fpm-probe/run.sh > "${OUT%.json}.stdout.log" 2>&1 &
EPID=$!
for i in $(seq 1 240); do
  if [ -s "$OUT" ] && python3 -c "import json,sys; d=json.load(open('$OUT')); sys.exit(0 if d.get('status') else 1)" 2>/dev/null; then
    sleep 3
    kill -TERM -- "-$EPID" 2>/dev/null || kill -TERM "$EPID" 2>/dev/null
    sleep 8
    kill -KILL -- "-$EPID" 2>/dev/null; kill -KILL "$EPID" 2>/dev/null
    exit 0
  fi
  if ! kill -0 "$EPID" 2>/dev/null; then
    [ -s "$OUT" ] && exit 0 || { echo "engine died without output"; exit 1; }
  fi
  sleep 5
done
echo "probe timeout after 20min"
kill -KILL -- "-$EPID" 2>/dev/null; kill -KILL "$EPID" 2>/dev/null
exit 1
