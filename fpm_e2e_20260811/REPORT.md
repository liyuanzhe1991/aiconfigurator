# FPM 端到端实验与精度验证 — 完整报告

日期:2026-08-11 | 集群:nebius-2(H200)| 模型:MiniMaxAI/MiniMax-M2.7(FP8-block MoE)
工况:8k/1k(ISL 8192 / OSL 1024)| 拓扑:TEP4(4 卡)/ TEP8(8 卡)

**一句话结论**:采集→预测→验证全链路打通;FPM 插值层精度达标(1.3-2.5%,复现历史 2.33%);
真实流量逐步对账暴露三个数据层/契约层问题(decode 路由熵乐观 5-9%、prefill 小步 host 地板
~6ms、mixed 配置契约断裂 -75%~+19%),全部归因闭环并落账。

---

## 一、环境与镜像准备

### 1.1 集成分支(所有 collection + modeling PR)

```bash
git fetch upstream main
git checkout -b fpm-all-20260811 upstream/main          # 基线 648aebcc
git merge --no-ff fpm-pr0-contract    # PR #1473 契约表          → 无冲突
git merge --no-ff fpm-prg-generator   # PR #1474 generator 渲染面 → 无冲突
git merge --no-ff fpm-prc-collector   # PR #1475 collector 工作流 → 无冲突
git merge --no-ff fpm-modeling-rust   # PR #1461 modeling(#1384 超集)→ 无冲突
# + 恢复 #1461 误删的 .agents/skills/aic-codeowners/SKILL.md
# + 搬入 scripts/experiments/(实验专用 driver + runbook)
```

验证:#1384 触碰的所有文件逐一确认在合并结果中;分支推送 origin,HEAD `f55ff21d`。

### 1.2 本地环境

```bash
uv sync --extra dev
```

- 建 venv 并由 maturin **从本分支 Rust 源码现编** native 扩展
  `_aiconfigurator_core.abi3.so`(含 #1461 新增的 `Op::FpmForward`;rustup 工具链在
  `/opt/homebrew/opt/rustup/bin`);
- 目标单测:contract 16 + modeling 91 + collector/generator FPM 304 + rust-step 等 126,
  共 **537 passed**;ruff check/format 全绿;
- **Rust/Python 引擎 parity 套件 335 passed**(含 `forward_model="fpm"` 整模对齐段)。

### 1.3 镜像(全程铁律:采集与验证同镜像同参数)

| 用途 | 镜像 |
|---|---|
| h200 采集 + 全部真值测量 | `nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-steady-16xfix-20260809` |

(NGC 私有仓,pod 侧 `nvcr-push-secret` 拉取;h100/gb200/b200 的 tag 见 runbook Appendix B。)

### 1.4 集群前置

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh:443   # 浏览器 SSO;登出后四集群 kube login 重建
kubectl --context=$CTX get secret nvcr-push-secret -n yuanli-aic   # ✓
kubectl --context=$CTX get pvc -n yuanli-aic                       # model-cache ✓
```

---

## 二、工作流程(步骤 → 命令 → 结果)

### Phase 1 — 正式采集(collector self-benchmark)

```bash
export FPM_PYTHON=$PWD/.venv/bin/python
scripts/experiments/fpm_collect.sh h100 4 m27            # 冒烟
scripts/experiments/fpm_collect.sh h200 4 m27 --formal   # 正式
scripts/experiments/fpm_collect.sh h200 8 m27 --formal
```

| 运行 | 结果 |
|---|---|
| h100 4卡 smoke | **失败**:分到的卡启动时仅 4.99/79.18 GiB 空闲(邻居租户欠申报占卡)。已记录,零残留,h100 顺延 |
| h200 4卡 formal(47 min) | **0 错误**,11,062 行,零残留 |
| h200 8卡 formal(29 min) | **0 错误**,11,155 行,零残留 |

产物:`fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet`(22,217 行,
schema v6,sha256 密封)+ metadata sidecar。4 个 cell = TEP4/TEP8 × prefill/decode。
**注意:prefill cell 与 decode cell 引擎参数不同**(prefill:同步调度 + CUDA graph 捕获
至 2048 token + prefix caching ON;decode:async + 无 prefill 图 + prefix caching OFF)
—— 这在 Phase 5 成为关键线索。

### Phase 2 — 数据入库与 FPM 预测

```bash
mkdir -p aic-core/src/aiconfigurator_core/systems/data/h200_sxm/vllm/0.25.1
cp fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.{parquet,metadata.json} \
   aic-core/src/aiconfigurator_core/systems/data/h200_sxm/vllm/0.25.1/

.venv/bin/aiconfigurator cli default --model-path MiniMaxAI/MiniMax-M2.7 \
  --system h200_sxm --backend vllm --backend-version 0.25.1 \
  --total-gpus 8 --isl 8192 --osl 1024 --forward-model fpm
```

| 配置 | tokens/s/gpu | tokens/s/user | TTFT |
|---|---|---|---|
| 4 卡 agg TEP4 | 336.99 | 34.20 | 448 ms |
| 8 卡 2×TEP4(rank1) | 336.99 | 34.20 | 448 ms |
| 8 卡 1×TEP8(rank2) | 308.27 | 39.11 | 394 ms |

disagg 正确拒答(需 TEP1/2 数据,FPM 不外推)。

### Phase 3 — L0 闭环 + 双引擎对齐(纯本地)

- `level0_closure.py`:全库可寻址格点逐一 `FPMForwardOp.query()` 回读
  → **13,281 点全部位精确(最差相对误差 0.000e+00)**;
- 同一 sweep 分别跑 `--engine-step-backend python|rust` → **输出逐列一致**。

### Phase 4 — 通道 B:显式点位探针(self-benchmark 机制内验证)

点位设计(`make_probe_manifest.py`,程序化断言 A 组在格上/B 组不在格上):
每拓扑 32 点 = A 组格点锚(悬崖对 2048/2049、graph 对 8/9、8k/1k 工作区锚)
+ B 组非格点(eager 稀疏段 2560-7168、KV 曲线中点、离站 batch 12/52、工作区中点)。

注入方式(历史同款;镜像无 `--benchmark-points-file` flag):

```bash
python3 -m dynamo.vllm ... \
  --scheduler-cls dynamo.vllm.instrumented_scheduler.InstrumentedScheduler \
  --worker-extension-cls dynamo.vllm.gc_policy.FpmGcWorkerExtension \
  --additional-config '{"benchmark":{"mode":"agg","timeout":3600,"warmup_iterations":5,
                        "output_path":"...","points":{...32 点内联...}}}'
```

执行:复用正式采集渲染的 `k8s_deploy.yaml` 起 keepalive pod → pod 内起 etcd(照抄
collector 参数)→ `probe_exec.sh` 包装(此注入路径引擎测完不自退,需输出文件完成检测
+ 进程组回收)→ **5 次重复 × 2 拓扑,r1-r5 全部 complete(32/32 点)**。

调试沿革(三层剥洋葱,均已根因归档):flag 不存在 → etcd 缺失 → prefix 探针点与
decode 配置冲突(删 2 点)→ 引擎不自退(pkill 误伤致脏卡一次,EngineCore 进程名
逃逸 pkill,已温柔清理并验证归零)。

### Phase 5 — 通道 A:真实流量逐步真值(FPM 流 = 记录仪,流量 = 真值源)

**架构**(复刻历史 arm A/D 设计):同镜像 pod 内 etcd + nats + `dynamo.frontend` +
worker(采集同款 run.sh **仅删 benchmark 三 flag**)+ `DYN_FORWARDPASS_METRIC_PORT=20380`
让 InstrumentedScheduler 被动发布每步 `ForwardPassMetrics`(msgpack over ZMQ PUB)
+ 监听器落 JSONL。流量由镜像内置 `vllm bench serve` 制造:

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --model MiniMaxAI/MiniMax-M2.7 --tokenizer <本地 snapshot> \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --random-range-ratio 0 --num-prompts $NP --max-concurrency $C --ignore-eos
```

**双栈对齐采集的双配置**:栈①(decode-cell 参数)测 decode/mixed;
栈②(prefill-cell 参数:`--no-async-scheduling` + graph≤2048 + prefix ON + batched-tokens 8192)测 prefill。

扫描规模:

| 扫描 | 窗口 | 覆盖 |
|---|---|---|
| decode(栈①) | 34 窗(27 有效) | B ∈ {1..512,1024} × per-req KV ∈ {0.6k..200k} |
| prefill(栈②) | 110 窗(101 有效) | b ∈ {1-4} × tokens 20 档(128..8192,含 2048/2049 悬崖对)+ prefix ∈ {2k,8k,32k,131k,262k} |
| mixed(栈①,双流法:长流稳 Bd 池 + 探针流注入指定 chunk) | 30 窗 | chunk ∈ {256..6144} × Bd ∈ {8..64} |
| 判别实验 | 温度对照窗(--temperature 0)、C=40 交叉窗 | 熵敏感性 / 跨 batch |

每步对账:流记录 (n_prefill, prefill_tokens, prefill_kv, n_decode, decode_kv, wall_time)
→ 同坐标查 `FPMForwardOp`(mixed 用组合式 prefill + [decode − baseline])→ 逐步 delta。

---

## 三、Verify 方法与结果

### 方法清单

| 方法 | 验什么 | 结果 |
|---|---|---|
| L0a 闭环回读(全格点查询) | 查表路径(loader/身份/记账) | **0.000e+00,位精确** ✅ |
| L0b Rust vs Python 引擎(真实数据) | 双引擎一致性 | **逐列一致** ✅ |
| L1 噪声地板(探针 5 重复极差) | "多准算准"的标尺 | TEP4 1.74% / TEP8 3.80% |
| L2 插值留出(B 组非格点 vs 模型) | FPM 核心主张 | **TEP4 2.51% / TEP8 1.31% median** ✅(历史 2.33%) |
| A 组格点重测 | 机制可复现性 | 大 KV 锚 ±0.3-1.8%;regime 边界点(B=9 等)+5~13% 不稳 |
| 双仪器互证(客户端 ITL vs FPM 流) | 记录仪无共模误差 | 17.23 vs 17.12 ms(+0.6%)✅ |
| 真实流量逐步对账(三类全坐标) | 数据/组合层保真 | 见下 |
| 判别实验(温度 / KV 平坦性 / B 钟形) | 偏差归因 | 路由熵三重证据 |

### 三类点位的分布统计(窗级中位对账)

| 类 | n | mean | median | p95(abs) | max(abs) | 最大 gap 位置 |
|---|---|---|---|---|---|---|
| **decode** | 27 | -6.0% | -5.3% | 10.3% | 11.1% | **B=16, kv=8.7k/req**(10.72 vs 9.53) |
| **prefill** | 101 | -0.7% | +1.2% | 29.9% | 32.8% | **b=1, tokens=128**(18.39 vs 12.36) |
| **mixed** | 27 | -27.2% | -35.3% | 72.4% | 75.0% | **Bd=8, chunk=512**(81.0 vs 20.2) |

### 结构性发现(每条均有判别证据)

1. **decode:benchmark 数据系统性乐观 5-9%,归因 MoE 路由熵**
   —— 温度单变量实验(采样 17.12 → 贪心 16.37 ms,-4.4%);偏差沿 KV 平坦
   (B=8 从 0.6k 到 200k 恒 -3~-5%);沿 B 钟形(B=1:-0.9% → B=16-64:-7~-11% →
   B=512:-1.3%,两端因无塌缩空间/全 expert 覆盖而归零)。偏差是流量熵的函数,
   记为 [-5%,-9%] 带;终审判别(dense 模型对照)未跑。
2. **prefill 小步:真实 serving 有 ~6ms/步固定 host 开销**,benchmark 行未携带
   (128t:12.4→18.4;随步长稀释);**eager 大步(2049-8192):benchmark 悲观 13-16%**
   (与 6 月 golden launch-bound 发现同向);**prefix 轴插值健康(±3-8%)**。
   两相偏差方向相反,端到端 TPOT 只差 -6.5%(部分对消)。
3. **mixed:配置契约断裂(最重要)** —— 部署引擎(decode 配置)中 chunk 512-4096
   撞 eager 地板(78-91ms),组合公式却拼入 prefill 配置(图捕获态,20-50ms)的行
   → -50~-75%;chunk≥4096 两侧同为 eager,回到 +12~19% 加法高估。
   **决策需求:采集两相与部署必须钉同一引擎配置**,公式调参无法修复数据身份错配。
4. **端到端**(参考):TPOT 预测 25.57 vs 实测 27.36 ms(-6.5%);TTFT 需受控准入
   才可比(爆发到达含排队,限速后 228ms,与 172ms 纯 prefill 地板自洽)。

### 覆盖洞(记账,均有明确修法)

- 高并发 × 长 KV 7 窗(b1024、b40/64@64k-100k、b32@131k):OSL 太短稳态池未成形;
- mixed Bd=8 尾部 2 格(长流排空);
- prefix > 262k(构造成本);h100/gb200/b200 集群未跑;dense 对照(qwen32b)未跑。

---

## 四、工件清单

```
fpm_e2e_20260811/
├── LEDGER.md                        # 战役总账(来源钉板+全部对账+挂账残差)
├── REPORT.md                        # 本报告
├── per_step_validation.csv          # 12,647 真实步逐步对账
├── decode_validation_stack1.csv     # decode B×KV 窗级
├── prefill_validation_stack2.csv    # prefill 全网格窗级
├── mixed_validation_stack1.csv      # mixed chunk×Bd 窗级
├── probes/tep{4,8}/probe_r{1..5}.json   # 通道 B 5 重复原始数据
├── serve_results/                   # FPM 流快照 + 窗口清单 + bench 结果
├── level0_closure.py / probe_analysis.py / per_step_validation.py / ...
└── probe_bundle_*/ serve_bundle/    # 全部可复现执行包
```

集群侧(截至报告时):4 个 pod 在位(fpm-dc6ae7…/fpm-c33f65… 探针 ×2、
fpm-serve-groundtruth、fpm-serve-prefill),等清理/补测决定;此外零残留。

## 五、待决事项(需要 owner 拍板)

1. mixed 配置契约:统一采集/部署引擎配置的方案(collector+generator 联动,跨模块);
2. decode 路由熵偏差的处置:dense 对照终审 → 数据层修正(采集侧真实化输入熵)或
   modeling 侧带宽表述;
3. prefill 小步 host 地板:归属哪个代码路径、是否入模;
4. 4 pod 清理 vs 留用补测;h100/其余集群补跑窗口。
