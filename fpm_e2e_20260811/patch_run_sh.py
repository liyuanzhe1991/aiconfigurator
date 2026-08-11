# Rebuild probe run.sh from the pristine rendered artifact, injecting the
# explicit benchmark points the way the historical campaign did: via
# --additional-config '{"benchmark": {..., "points": ...}}'.
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()

builder = """
# Build the benchmark additional-config with the explicit probe points
# (historical injection route; this image predates --benchmark-points-file).
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

engine_command=("""

old_flags = "--benchmark-timeout 3600 --benchmark-mode decode --benchmark-warmup-iterations 5"
new_flags = (
    "--scheduler-cls dynamo.vllm.instrumented_scheduler.InstrumentedScheduler "
    "--worker-extension-cls dynamo.vllm.gc_policy.FpmGcWorkerExtension "
    '--additional-config "$FPM_PROBE_ADDCFG"'
)
assert old_flags in text, "benchmark flags not found in run.sh"
text = text.replace(old_flags, new_flags)

assert " --benchmark-output-path /results/benchmark.json" in text
text = text.replace(" --benchmark-output-path /results/benchmark.json", "")

assert "export DYN_FPM_BENCHMARK_OUTPUT_PATH=/results/benchmark.json" in text
text = text.replace(
    "export DYN_FPM_BENCHMARK_OUTPUT_PATH=/results/benchmark.json",
    'export DYN_FPM_BENCHMARK_OUTPUT_PATH="${PROBE_OUT:-/results/benchmark.json}"',
)

assert text.count("engine_command=(") == 1
text = text.replace("engine_command=(", builder, 1)

open(dst, "w").write(text)
print(f"patched {dst}")
