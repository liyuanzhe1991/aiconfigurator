# 引擎相位打点补丁(dynamo-fpm 台架,R16 §3 耗时地图的引擎侧数据源)。
# 应用顺序:pristine → kvwarm → content → fix_args_dump → 本补丁(有序断言,fail-closed)。
#
# 已有、无需新增的耗时(直接从输出 JSON 读):
#   inference  = timing.measured_iteration_seconds(逐点 wall 求和,既有)
#   kvwarm GPU = sum(kvwarm.stages[].build_seconds)(kvwarm 补丁既有)
#   总跨度     = timing.benchmark_elapsed_seconds + started_at/completed_at
# 本补丁只补非测量 CPU 相位:数据集加载/分词器/预热规划/链分词(按 kv 深度桶)/
# 内容生成(含首调用=池构建拆分)/前缀播种;以及 import→首次台架活动 的 boot 锚点。
# 设计红线:测量步循环零触碰(不包 _kvwarm_step_busy/_kvwarm_monitor_build/
# _bench_step 等每步路径);所有包裹目标均为每点/每 stage/一次性调用。
# 输出为 timing 块的 additive 子对象 "phases"(schema_version 不变,下游未知键容忍)。
import sys

P = "/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py"
MARK = "# __dyn_timing_patch__"
src = open(P).read()
if MARK in src:
    print("timing_patch: already applied")
    sys.exit(0)

# ---- 手术①:timing 输出块扩写(锚定唯一文本) ----
T_ANCHOR = '''            "timing": {
                "started_at": self._bench_started_at,
                "completed_at": self._bench_completed_at,
                "benchmark_elapsed_seconds": elapsed_seconds,
                "measured_iteration_seconds": measured_iteration_seconds,
            },'''
T_REPL = '''            "timing": {
                "started_at": self._bench_started_at,
                "completed_at": self._bench_completed_at,
                "benchmark_elapsed_seconds": elapsed_seconds,
                "measured_iteration_seconds": measured_iteration_seconds,
                "phases": dict(getattr(self, "_dyn_phases", {})),
            },'''
assert src.count(T_ANCHOR) == 1, f"timing_patch: timing 锚点命中 {src.count(T_ANCHOR)} 次(须恰 1)"

# 有序断言:依赖 kvwarm 与 content 补丁先行
assert "class InstrumentedScheduler" in src, "timing_patch: 类名不符,拒绝盲改"
for probe in (
    "_kvwarm_load_texts",
    "_kvwarm_tokenizer",
    "_kvwarm_prepare",
    "_kvwarm_chain_token_ids",
    "_bench_prefill_content_ids",
    "_bench_cache_fake_prefixes",
):
    assert probe in src, f"timing_patch: 依赖方法 {probe} 缺失(应用顺序错误)"

# ---- 手术②:相位累加器(尾部追加,方法替换;不触碰每步路径) ----
HOOK = f'''

{MARK}
import time as _dt_time
_DT_IMPORT_T0 = _dt_time.monotonic()

def _dt_acc(self, key, dur, bucket=None):
    ph = getattr(self, "_dyn_phases", None)
    if ph is None:
        ph = {{}}
        self._dyn_phases = ph
        # boot 锚点:模块导入→首次台架活动(引擎拉起段的引擎侧近似)
        ph["import_to_first_bench_activity_s"] = round(
            _dt_time.monotonic() - _DT_IMPORT_T0, 3
        )
    if key not in ph:
        ph[key + "_first_s"] = round(dur, 6)  # 首调用拆分(如内容池一次性构建)
    ph[key] = round(ph.get(key, 0.0) + dur, 6)
    ph[key + "_n"] = ph.get(key + "_n", 0) + 1
    if bucket is not None:
        b = ph.setdefault(key + "_buckets", {{}})
        b[bucket] = round(b.get(bucket, 0.0) + dur, 6)

def _dt_kv_bucket(depth):
    if depth <= 0:
        return "0"
    lo = 1
    while lo * 4 <= depth:
        lo *= 4
    return f"{{lo}}-{{lo * 4 - 1}}"

def _dt_wrap(name, key, bucket_from=None):
    orig = getattr(InstrumentedScheduler, name)
    def wrapped(self, *a, **k):
        t0 = _dt_time.monotonic()
        try:
            return orig(self, *a, **k)
        finally:
            bucket = None
            if bucket_from is not None:
                try:
                    bucket = _dt_kv_bucket(int(a[bucket_from]))
                except Exception:
                    bucket = None
            _dt_acc(self, key, _dt_time.monotonic() - t0, bucket)
    wrapped.__name__ = name
    setattr(InstrumentedScheduler, name, wrapped)

_dt_wrap("_kvwarm_load_texts", "dataset_load_s")
_dt_wrap("_kvwarm_tokenizer", "tokenizer_load_s")
_dt_wrap("_kvwarm_prepare", "kvwarm_plan_s")
_dt_wrap("_kvwarm_chain_token_ids", "content_tokenize_s", bucket_from=1)
_dt_wrap("_bench_prefill_content_ids", "content_gen_s")
_dt_wrap("_bench_cache_fake_prefixes", "seed_total_s")
'''

out = src.replace(T_ANCHOR, T_REPL) + HOOK
compile(out, P, "exec")
open(P, "w").write(out)
print("timing_patch: applied")
