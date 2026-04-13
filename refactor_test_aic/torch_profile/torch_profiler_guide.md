# SGLang Torch Profiler 使用指南

基于 SGLang 官方 `/start_profile` HTTP API 的 torch profile 采集工具，支持精确控制 prefill/decode batch 组成。

## 前置条件

- SGLang 安装完毕（代码路径: `/host/hisim-sglang/sglang/python`）
- 模型路径可访问（如 `/models/Qwen3-235B-A22B-Instruct-2507-FP8`）
- Python `requests` 库已安装

## 核心原理

1. SGLang server 启动时设置 `SGLANG_TORCH_PROFILER_DIR` 环境变量，启用 profiling 能力
2. 通过 `/start_profile` API 触发 profiling，`profile_by_stage=true` 时分别采集 prefill (EXTEND) 和 decode (DECODE) 的 trace
3. 通过 `/generate` 端点发送 `input_ids`，精确控制每个请求的 token 组成
4. Trace 文件按 TP rank 和 stage 分别输出，格式为 `{id}-TP-{rank}-EP-{rank}-{STAGE}.trace.json.gz`

## 快速开始

### 方式一：自动启动 Server（推荐快速测试）

```bash
# 2 层快速 profile（几分钟内完成）
python3 /host/aiconfigurator/refactor_test_aic/torch_profiler.py \
  --launch-server \
  --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tp-size 8 --ep-size 8 \
  --num-layers 2 \
  --moe-a2a-backend deepep \
  --input-len 128 --output-len 64
```

脚本会自动：启动 server → 等待就绪 → warmup → profiling → 收集 trace → 停止 server。

### 方式二：手动启动 Server

```bash
# 终端 1: 启动 server
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile
python3 -m sglang.launch_server \
  --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tp-size 8 --ep-size 8 \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --disable-overlap-schedule \
  --cuda-graph-max-bs 256 \
  --trust-remote-code \
  --json-model-override-args '{"num_hidden_layers": 2}'

# 终端 2: 运行 profiler
python3 /host/aiconfigurator/refactor_test_aic/torch_profiler.py \
  --input-len 128 --output-len 64
```

### 方式三：纯 curl 手动操作

```bash
# 1. 启动 profiling
curl -X POST http://localhost:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/sglang_profile",
    "num_steps": 5,
    "activities": ["CPU", "GPU"],
    "profile_by_stage": true,
    "with_stack": true,
    "record_shapes": true
  }'

# 2. 发送请求触发 forward steps
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [100, 200, 300, 400, 500],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0}
  }'

# 3. 等待 trace 文件生成
ls /tmp/sglang_profile/*.trace.json.gz
```

## 常用场景

### Prefill Profiling

采集不同输入长度的 prefill 性能：

```bash
# 短输入 prefill
python3 torch_profiler.py --input-len 64 --output-len 8 --num-steps 3

# 长输入 prefill
python3 torch_profiler.py --input-len 2048 --output-len 8 --num-steps 3

# 只采集 prefill 阶段
python3 torch_profiler.py --input-len 512 --output-len 32 --profile-stages EXTEND
```

### Decode Profiling

```bash
# 只采集 decode 阶段，增加 step 数
python3 torch_profiler.py --input-len 32 --output-len 128 \
  --profile-stages DECODE --num-steps 10
```

### 带 Prefix Cache

```bash
# 512 token 共享前缀 + 128 token 新输入
python3 torch_profiler.py --prefix-len 512 --input-len 128 --output-len 64
```

### 多请求并发

```bash
# 4 个并发请求
python3 torch_profiler.py --input-len 128 --output-len 64 --num-requests 4
```

### 全量层 Profile

```bash
# 不指定 --num-layers，使用模型全部层（Qwen3-235B 原始 94 层）
python3 torch_profiler.py --launch-server \
  --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tp-size 8 --ep-size 8 \
  --moe-a2a-backend deepep \
  --input-len 128 --output-len 64
```

## 参数说明

### Profiling 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-len` | 128 | 输入 token 长度 |
| `--prefix-len` | 0 | prefix cache 长度（先发一次 prefix 预填充） |
| `--output-len` | 64 | 最大输出 token 数 |
| `--num-requests` | 1 | 并发请求数 |
| `--num-steps` | 5 | 每阶段采集 forward step 数 |
| `--profile-by-stage` | true | 分阶段采集 prefill/decode |
| `--profile-stages` | 全部 | 指定阶段，如 `EXTEND,DECODE` |
| `--with-stack` | false | 记录 Python 调用栈（trace 更大但更详细） |
| `--output-dir` | /tmp/sglang_profile | trace 输出目录 |
| `--port` | 30000 | SGLang server 端口 |

### Server 启动参数（配合 `--launch-server`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--launch-server` | false | 自动启动 SGLang server |
| `--model-path` | 必填 | 模型路径 |
| `--tp-size` | 8 | tensor parallel 大小 |
| `--ep-size` | 8 | expert parallel 大小 |
| `--num-layers` | 全部 | 覆盖 `num_hidden_layers`，减层加速 |
| `--moe-a2a-backend` | deepep | MoE all-to-all 后端 |
| `--auto-shutdown` | true | profiling 后自动停止 server |
| `--launch-timeout` | 600 | 等待 server 启动超时（秒） |

## Trace 文件分析

### 输出文件格式

`profile_by_stage=true` 时，每个 TP rank 生成两个文件：
```
{timestamp}-TP-0-EP-0-EXTEND.trace.json.gz   # prefill trace
{timestamp}-TP-0-EP-0-DECODE.trace.json.gz   # decode trace
{timestamp}-TP-1-EP-1-EXTEND.trace.json.gz
{timestamp}-TP-1-EP-1-DECODE.trace.json.gz
...
```

8×TP 配置下共生成 16 个文件（8 rank × 2 stage）。

### 可视化

1. 打开 [Perfetto UI](https://ui.perfetto.dev)
2. 拖入 `.trace.json.gz` 文件（支持 gzip 格式）
3. 重点关注:
   - **CUDA kernel 耗时**: 搜索 `nccl`、`gemm`、`flash_attn` 等关键词
   - **MoE dispatch/combine**: 搜索 `deepep`、`all_to_all`
   - **Prefill vs Decode 对比**: 分别加载 EXTEND 和 DECODE trace

### 关键指标

- **Prefill (EXTEND)**: 关注 attention 和 MoE expert 计算耗时
- **Decode (DECODE)**: 关注 KV cache 访问和通信开销
- **通信**: 搜索 `ncclKernel` 查看集合通信耗时

## 注意事项

1. **必须设置 `SGLANG_TORCH_PROFILER_DIR`**: 手动启动 server 时需要 export 此环境变量，否则 `/start_profile` 虽然返回 200 但不会生成 trace
2. **`profile_by_stage=true` 是非阻塞的**: `/start_profile` 立即返回，profiling 在后续 forward step 中进行，需要发送足够的请求触发采集
3. **减层 profiling**: `--num-layers 2` 可以将启动时间从几分钟缩短到十几秒，trace 中的 kernel pattern 与全量层一致（只是层数少）
4. **端口冲突**: 确保 `--port` 指定的端口未被占用
5. **`/generate` vs `/v1/completions`**: 脚本使用 `/generate` 端点，因为它支持 `input_ids` 精确控制 token 组成；`/v1/completions` 只支持文本 `prompt`
