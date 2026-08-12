# Generalized probe patcher: converts a rendered cell run.sh (prefill OR
# decode cell) into an explicit-points benchmark launcher via the
# additional-config injection route. Keeps every engine flag except the
# benchmark trio (+ prefill sampling knobs), preserving the cell's config
# identity (sync/async, capture list, prefix caching, batched-tokens).
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()

m = re.search(r" --benchmark-timeout \d+ --benchmark-mode (prefill|decode) --benchmark-warmup-iterations \d+", text)
assert m, "benchmark flags not found"
mode = m.group(1)
text = text.replace(m.group(0), "")
text = re.sub(r" --prefill-max-new-token-samples \d+ --prefill-max-kv-read-token-samples \d+", "", text)
text = text.replace(" --benchmark-output-path /results/benchmark.json", "")
text = text.replace("export DYN_FPM_BENCHMARK_OUTPUT_PATH=/results/benchmark.json\n",
                    'export DYN_FPM_BENCHMARK_OUTPUT_PATH="${PROBE_OUT:-/results/benchmark.json}"\n')
assert "--benchmark" not in text and "--prefill-max" not in text

builder = '''
# Explicit-points benchmark via additional-config (historical injection route).
FPM_PROBE_ADDCFG=$(python3 - <<PYEOF
import json
points = json.load(open("/tmp/fpm-probe/points.json"))
print(json.dumps({"benchmark": {
    "mode": "agg",
    "timeout": 3600,
    "warmup_iterations": 5,
    "output_path": "${PROBE_OUT:-/results/benchmark.json}",
    "points": points,
}}))
PYEOF
)

engine_command=('''

injected = (" --scheduler-cls dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
            " --worker-extension-cls dynamo.vllm.gc_policy.FpmGcWorkerExtension"
            ' --additional-config "$FPM_PROBE_ADDCFG"')
anchor = " --max-model-len -1"
assert anchor in text
text = text.replace(anchor, anchor + injected, 1)

assert text.count("engine_command=(") == 1
text = text.replace("engine_command=(", builder, 1)

open(dst, "w").write(text)
print(f"patched {dst} (source cell mode: {mode}; engine config preserved)")
