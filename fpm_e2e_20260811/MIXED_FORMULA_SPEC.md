# FPM 混合步公式改造说明书(交接给 modeling PR)

日期:2026-08-12 | 依据:fpm-all-20260811 战役全部实测证据(见文末附录)
目标 PR:#1461(fpm-modeling-rust)跟进 | 涉及:aic-core Python + Rust,不碰 collector/数据契约

## 0. 一句话

混合步的 prefill 分量改为**按"本步实际调度的总 token 数(chunk + Bd)"查 FPM 曲线**,
废除现行"整条 ISL 的行摊到每个 chunk"的定价 —— regime(CUDA graph 悬崖)由数据自带,
公式只需把查询坐标放对。

## 1. 现状(实测确认的行为)

位置:`aic-core/src/aiconfigurator_core/sdk/backends/base_backend.py`
`_get_fpm_mix_step_latency(model, database, runtime_config, ctx_tokens, gen_tokens, isl, osl, prefix)`
(~L1283 起;Rust 侧无对应,混合步走 Python 组合)。

现行组合:

```
prefill 分量 = run_static(batch=ceil(ctx/isl), isl=isl, osl=1, prefix, mode="static_ctx")
               / chunk_scale,  chunk_scale = ceil(isl / ctx_tokens)
decode 分量  = decode(B=gen_tokens, KV=avg) − query_pass_baseline(gen_tokens)   # 边际
mixed        = prefill 分量 + max(0, decode 分量)
```

两个实测缺陷:

1. **regime 盲区(主缺陷)**:真实引擎按"本步调度总 token = ctx + gen"决定 graph/eager。
   现行查询坐标是 ISL 行的坐标,与本步总量无关。配置对齐(capture-2048)后的实测网格:
   chunk=2048 行(2048+Bd 恰好跨界)误差 **-38%~-44%**(真实 76-88ms vs 模型 48-59ms)。
2. **摊薄失真**:同一请求的各 chunk 成本不同(past-KV 逐 chunk 增长),ISL 行 / chunk 数
   的平均值抹掉了这个结构;且 FPM 数据本身按 (batch, total_new_tokens, past_kv) 索引,
   完全有能力精确定价单个 chunk。

## 2. 新设计

### 2.1 新查询接口(Python + Rust 镜像)

```python
# aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py
class FPMForwardOp:
    def query_totals(self, database, *, batch_size: int,
                     total_prefill_tokens: int,
                     total_kv_read_tokens: int) -> PerformanceResult:
        """按原始总量坐标查询(绕过 per-request (b, s, prefix) 的整除约束)。
        内部直通现有 _resolve(cell, coords) 路径;prefill 相位 coords =
        (batch_size, total_prefill_tokens, total_kv_read_tokens)。"""
```

必要性:chunk + Bd 一般不能被 batch 整除,现有 `query(b, s, prefix)` 表达不了。
Rust 侧在 `aic-core/rust/aiconfigurator-core/src/operators/fpm_forward.rs` 加同名入口,
engine runtime 的混合步路径同步(两引擎必须同一轮改完,由 parity 测试锁住)。

### 2.2 混合步组合(替换 _get_fpm_mix_step_latency 的 prefill 分量)

```python
step_total = ctx_tokens + gen_tokens          # ← regime 与 GEMM 宽度的真实决定量
pre = prefill_op.query_totals(
        database,
        batch_size=1,                         # 常态:一个 chunk 属于一个请求
        total_prefill_tokens=step_total,      # ← 关键:含 gen_tokens
        total_kv_read_tokens=chunk_past_kv)   # 该 chunk 的已算上下文(见 2.3)
mixed = pre + max(0, dec_marginal)            # decode 边际项不变
```

**无重复计费论证**(已用 v2 实测网格离线验证):
`query_totals(1, ctx+gen, kv)` 含 ctx+gen 个 token 的 GEMM/权重/固定开销;
`dec_marginal = decode(B,KV) − baseline(B)` 经 baseline 减法只剩 **KV-attention 边际**
(权重读、GEMM(B)、每步固定开销都在 baseline 里被减掉)。gen token 的 GEMM 恰好只在
pre 里计一次。残余的高估仅为"gen 个 token 在 chunk 上下文内的自注意力",量级可忽略。

### 2.3 多 chunk 请求(ctx_tokens < isl 的连续步)

逐 chunk 精确求和,替代平均:

```
for k in chunks(isl, chunk_size):             # past_kv_k = prefix + Σ_{<k} chunk
    cost_k = query_totals(1, chunk_k + gen_tokens_k, past_kv_k)
TTFT 聚合 = Σ cost_k(调用方聚合接口保持不变)
```

### 2.4 边界情形

- `ctx_tokens == 0`:纯 decode 整步,维持现行(full decode query);
- `gen_tokens == 0`:纯 prefill chunk → `query_totals(1, ctx_tokens, prefix)`(同样废除摊薄);
- 域外:沿用 FPM 铁律,`query_totals` 域外照常抛 PerfDataNotAvailableError,不外推。

### 2.5 前提声明(写进 docstring 与用户文档)

本公式的正确性依赖 FPM 契约前提:**部署引擎配置(尤其 cudagraph 捕获面)与采集一致**。
悬崖位置不进模型,由数据的悬崖对(如 2048/2049)编码;配置错配时误差不可由公式挽救
(实测:错配部署 chunk=512 行 -75%)。

## 3. 测试与验收

单元测试(Python + Rust parity):
1. 跨界:chunk=2048, gen ∈ {1..64} 必须落 eager 侧(断言 > 图侧值;用悬崖对 fixture);
2. 图内:chunk+gen ≤ 捕获界 → 图侧值;
3. 无重复计费:synthetic fixture 上 |mixed(ctx,gen) − pure(ctx+gen) − dec_marginal| ≤ tol;
4. 多 chunk 求和 == 各 chunk 独立查询之和;
5. gen=0 / ctx=0 退化路径。

回归验收(harness 已存在,可直接复放):
- 数据:`fpm_e2e_20260811/mixed_validation_v2_cap2048.csv`(30 窗真实逐步网格)
  + `serve_results/stream_stack1_v3.jsonl`(原始步流);
- 门限:**跨界行(chunk=2048)从 -38~-44% 收到 ±15% 以内(旧数据)**;
  配合重采后的新 parquet(随机 token 版)全网格 median|δ| ≤ 8%。
- 已做过的离线预演:v2 网格上仅改坐标即达 |δ| median 13.7%,叠加数据修复后 5.4%。

## 4. 证据附录(全部可复算)

- 跨界实测:`mixed_validation_stack1.csv`(v1, -75%)/ `mixed_validation_v2_cap2048.csv`(v2, -44% 行);
- 悬崖对:parquet tp8/b1/kv0:2048→47.24ms, 2049→99.48ms;
- 摊薄定价现状:base_backend.py `run_static(... ) / chunk_scale`(~L1318-1345);
- 配料修复(另行进行,与本改造独立):dynamo 全零输入随机化已实测使 prefill 全域
  (128-8192)对齐真实流量 ≤±1.5%(probe_rand*.json vs 真实流);
- 战役总账:`fpm_e2e_20260811/LEDGER.md` / `REPORT.md`。

## 5. 增补(2026-08-12,v2 探针实证):decode 批轴同病,一并修

§2.1 的 regime 感知不能只管 mixed —— **decode 单相查询的批轴插值也在跨 regime 桥接**。
v2 离网探针(randtok2 数据,decode 配置对齐,TEP4)实测:

| 点 | 实测 | 模型现值 | 偏差 | 机制 |
|---|---|---|---|---|
| b=600, kv=65400 | 88.21 ms | 44.37 ms | **+98.8%** | regime 投票稀释(机制精确复现,见下) |
| b=768, kv=65280 | 89.90 ms | 57.70 ms | **+55.8%** | 同上 |
| b=12, kv=8184 | 11.91 ms | 11.10 ms | +7.3% | 引擎把 batch 向上 pad 到下一 capture size(16),代价是阶梯而非线性;线性插值 (9→16) 低估 |

关键事实(parquet 本身没问题,悬崖对都在):
- tp4/kv65536: b=512→36.27ms(graph),**b=513→98.72ms(eager)**,b=1024→89.10ms;
  tp8 同构(26.50 / 92.82 / 91.92)。精确命中 b=513 走 own-site 路径,返回行值正确。
- **确切机制(已数值复现到小数点后两位)**:`perf_interp` 的 ScatteredSites 对
  未采集 batch 用 **nn_sites=4 最近邻的 util 迁移**(log2(batch) 距离平方反比加权,
  engine.py `_resolve_scattered`)。距离度量对 graph/eager regime 全盲:查 b=600 时
  4 邻居是 {513(w19.6, eager 98.7), 512(w19.1, graph 36.3), 497(w13.5, 35.9),
  496(w13.3, 35.5)}——采集器的悬崖对设计使悬崖附近永远是"1 张 eager 票对 3 张
  graph 票",调和混合后得 44.37ms(op.query 实际返回值,分毫不差);b=1024 距离
  0.77 octave 排第 5,根本不在邻居集里。b=768/900 时 1024 逐渐进入并主导,
  这就是误差随 batch 递减的原因(+97%→+51%→+12%)。
- 修法与 §2.1 同一把刀:**站点先按 regime 分区,再做 k-NN**——graph 区
  (b ≤ max capture size,本模型 512)与 eager 区(b > 512)互不为邻;查询侧
  regime 按同一规则判定(边界可从悬崖对或 sidecar 记录的 capture 列表导出)。
  分区后 b=600 锚在 {513, 1024},给 ~97ms(vs 实测 88,余差即路由带)。
  graph 段内用"向上 pad 到下一 capture size"的阶梯语义(悬崖对 (8,9)、(16,17)…
  正是采集器留给你的段边界标记;注意 pad-up 不是保守上界,见 §5 增补告示 1)。
- 若正确用 eager 段 (513→1024) 插值,b=600 预期 ~97ms vs 实测 88.2ms(-9%),
  余差即已知的 decode 路由分布带(-5~-8%),与全局一致。

验收补充:`fpm_e2e_20260811/probes_v2_scores.csv`(v2 探针全点位打分,含上表);
修后 b=600/768 两点应从 +56~+99% 收敛到路由带内(|δ| ≤ 10%)。

### §5 增补(TEP8 复核 + 两条边界告示)

- TEP8 复核通过:eager 高原平坦(513→101.1ms, 900→95.2ms @kv≈131k),
  桥接错误同构(b=600 +163%, b=768 +95%);分段禁桥修法不变。
- **告示 1(pad-up 非上界)**:tp8 实测 live(12)=11.32 > live(16)=10.12——
  ragged batch 在 pad 图里可能比 pad 目标更贵。阶梯语义按"取 pad 目标行值"
  实现即可,但不要宣称它是保守上界;残差归入路由带。
- **告示 2(振荡口袋)**:prefix 轴与 decode KV 轴存在 kernel-plan 振荡区
  (实测复现,非坏行;见 LEDGER round-2)。插值在口袋内有 ~±25% 局部误差
  地板,格点命中不受影响。公式层不处理;文档标注即可。

### §5 增补 2:regime 边界从哪来(回应"inference 侧没有 cudagraph 参数")

这个依赖不是新增的——parquet 每一行的数值本来就是被采集引擎的 capture 面
决定的,FPM 的配置一致性教义(部署 == 采集)意味着数据只对同 capture 面
有效。修法只是让插值器尊重数据里已有的结构。**查询接口不变**,边界来源
按优先级:

1. **纯数据驱动(modeling PR 用这个,对 v6 数据立即生效)**:扫相邻批次对
   (b, b+1),latency 跳变 ≥2×(实测悬崖 2.6-3.5×,远超坏行 ±12% 污染)即
   regime 边界;悬崖对 (512,513)、(2048,2049) 正是采集器留下的标记。
   prefill token 轴在 §2.1 已用同一约定。
2. **sidecar 记录(collector PR,schema v7)**:把 resolved-config dump 里的
   `cudagraph_capture_sizes` 写入 sidecar(dump 已存在于采集工件,零成本)。
   两者都在时 sidecar 为准、数据推导作校验——校验不一致即部署/采集漂移,
   直接报错,这本身是教义的运行时守卫。

边界情形:悬崖对缺失 → eager 区 = 域外,FPM 不外推,行为不变;
过渡行污染不影响 ≥2× 跳变检测。

### §5 增补 3:实现边界(回应"插值器是共享的")

核查:op-level 算子(gemm/moe/attention/mla…)全部用 `Grid()`;`ScatteredSites`
目前仅 fpm_forward.py 使用。但无论共用与否,**`perf_interp/engine.py` 一行不改**
——共享插值引擎不得引入 cudagraph 领域逻辑:

- regime 分区在 **FPM op 层建表时**完成:fpm_forward.py 把 decode 数据字典切成
  graph(b ≤ B*)/eager(b > B*)两张子表,查询按同规则路由;每张子表交给
  原封不动的 ScatteredSites。引擎只看到两份普通数据。
- Rust 同构:operators/fpm_forward.rs 同样在 op 层切表,parity 测试锁定。
- prefill token 轴无需任何切表:悬崖对 (2048,2049) 是曲线轴相邻整数,之间不存在
  可查询的整数点,数学上不可能桥接——病只在批轴(站点稀疏、崖间有大量整数)。
- op-level 预测因此不受任何影响(Grid 路径零接触;ScatteredSites 语义零变化)。
