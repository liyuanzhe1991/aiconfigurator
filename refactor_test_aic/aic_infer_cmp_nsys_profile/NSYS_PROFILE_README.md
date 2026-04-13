# nsys Profiling 使用指南

对 SGLang 推理引擎进行 **prefill** 和 **decode** 阶段的 nsys 性能分析。

## 文件说明

| 文件 | 说明 |
|------|------|
| `nsys_profile.py` | Host 侧整合脚本，自动在容器中执行 nsys profiling |
| `run_prefill.py` | Prefill replay 脚本（容器内执行） |
| `run_decode.py` | Decode replay 脚本（容器内执行） |
| `hook.py` | SGLang Hook 基类，用于注入 NVTX 标注等 |

---

## 方式一：Host 侧通过 `nsys_profile.py` 执行（推荐）

在 Host 机器上运行，脚本会自动通过 `docker exec` 在容器内完成 nsys profiling。

### Prefill Profiling

```bash
python nsys_profile.py --mode prefill --case-id 321
```

### Decode Profiling

```bash
python nsys_profile.py --mode decode --case-id 6
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | （必填） | `prefill` 或 `decode` |
| `--case-id` | （必填） | CSV 中的 case_id（即 JSONL 0-based 行号） |
| `--container` | `mry-aic-collect` | Docker 容器名 |
| `--data-dir` | `/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt` | 数据目录（容器内路径） |
| `--nsys-dir` | `{data-dir}/nsys` | nsys 报告输出目录 |
| `--model` | `/models/Qwen3-235B-A22B-Instruct-2507-FP8` | 模型路径 |
| `--tp-size` | `8` | Tensor Parallel size |
| `--ep-size` | `8` | Expert Parallel size |
| `--iters` | `3` | Decode 迭代次数（仅 decode 模式） |
| `--dry-run` | — | 只打印命令，不执行 |
| `--cleanup` | — | 清理容器中残留进程 |

### 辅助操作

```bash
# 清理容器残留进程（sglang / nsys / torch compile worker）
python nsys_profile.py --cleanup

# Dry-run：只查看完整命令，不实际执行
python nsys_profile.py --mode prefill --case-id 321 --dry-run
```

---

## 方式二：容器内直接执行 nsys 命令

进入容器后手动执行 nsys profile 命令。

### 进入容器

```bash
docker exec -it mry-aic-collect bash
```

### Prefill Profiling

```bash
cd /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile && \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  -f true \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_321 \
  python3 run_prefill.py \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt \
    --csv-case-id 321 \
    --model /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp-size 8 \
    --ep-size 8
```

### Decode Profiling

```bash
cd /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile && \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  -f true \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_decode_case_6 \
  python3 run_decode.py \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt \
    --csv-case-id 6 \
    --model /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp-size 8 \
    --ep-size 8 \
    --iters 3
```

---

## nsys 参数说明

| 参数 | 说明 |
|------|------|
| `-f true` | 覆盖已有输出文件 |
| `-t cuda,nvtx,osrt` | 采集 CUDA API、NVTX 标注、OS runtime 事件 |
| `--cuda-graph-trace=node` | 将 CUDA Graph 展开到 node 级别 |
| `--capture-range=cudaProfilerApi` | 仅在 `cudaProfilerStart/Stop` 之间采集数据 |
| `--capture-range-end=stop` | 遇到 `cudaProfilerStop` 立即结束采集 |
| `--sample=none` | 关闭 CPU 采样（减小报告体积） |
| `--cpuctxsw=none` | 关闭 CPU 上下文切换跟踪 |
| `-o <path>` | 输出文件路径（自动追加 `.nsys-rep` 后缀） |

---

## NVTX 标注内容

脚本通过 Hook 在 `Scheduler.run_batch` 调用前后插入 NVTX range，标注内容包括：

| 字段 | 说明 |
|------|------|
| `bs` | Batch size |
| `forward_mode` | 当前 forward 模式（EXTEND / DECODE） |
| `input_length` | 每个请求的新输入 token 数（prefill=实际长度，decode=1） |
| `past_kv_length` | 每个请求已有的 KV cache 长度 |

当 `bs > 8` 时，为避免 NVTX 字符串过长，改为显示摘要统计：
- `total_input_tokens`：总输入 token 数
- `past_kv_min / past_kv_max / past_kv_avg`：KV cache 长度的最小/最大/平均值

---

## 输出文件

nsys 报告输出到 `{data-dir}/nsys/` 目录，命名规则：

```
nsys_prefill_case_{case_id}.nsys-rep
nsys_decode_case_{case_id}.nsys-rep
```

使用 NVIDIA Nsight Systems GUI 打开 `.nsys-rep` 文件即可查看 timeline。

---

## 清理容器残留进程

profiling 异常中断后，容器内可能残留 sglang / torch compile_worker 进程，需要清理后才能重新执行：

```bash
# 通过 nsys_profile.py（推荐）
python nsys_profile.py --cleanup

# 或手动在容器内清理
docker exec mry-aic-collect bash -c "pkill -f sglang; pkill -f torch._inductor.compile_worker"
```
