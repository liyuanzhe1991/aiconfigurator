# L3 精度评测 — MiniMax-M2.7 / H200 / dep4(dp=4, tp=1, moe_ep=4)

日期:2026-08-13。数据:`l3_harvest_dep4_live/`(burst 162/162 零失败,
三相全程 3h32m,pod 已按零残留铁律回收)。模型侧:分析专用 scratch 库
(`dep4_analysis_root/`,由 4df9b0f2 计划的两个 dep4 cell 原始 artifact 经
collector 自身 `aggregate_cell` 校验聚合;**非正式发布**,发布路线另行决定)。
打分器:`score_l3_dep4.py`(方法见 §2,与 tep 版不通用)。

## 1. 结论(headline)

| 相 | 层 | n | MAPE | P95 APE | MAX APE(坐标) |
|---|---|---|---|---|---|
| prefill | A(格点)| 8 | **4.41%** | 6.30% | 6.82% @(1,512,0) |
| prefill | B(域内离网)| 7 | **4.83%** | 7.87% | 9.10% @(2,1024,kv98304) |
| prefill | C(bs 外插 1.5-4x)| 8 | **3.94%** | 4.83% | 4.83% @(12,88,0) |
| mixed | C锚 | 4 | 5.67% | 6.45% | 6.56% @(1,1024,Bd40) |
| decode | A | 55 | 12.53% | 24.47% | 31.29% @(kv16,Bd8) |
| decode | B | 198,098 | 11.35% | 18.57% | 45.19% @(kv33209,Bd512) |

- **prefill(含 bs 外插)达标**:误差全域 3-9%,C 层随外插倍数(1.5x→4x)
  **不增长**(1x-2x 3.94% / 2x-4x 3.93%)——batch clamp 外插在 dep4 成立。
  全部点 model 略高(+3~7%),部分归因于独跑估计器的残余稀释(§2.3)。
- **decode 与 tep4 同分布**:tep4 decode B 为 9.24%/P95 17.4%,dep4 11.35%/
  P95 18.6%——这是跨战役已知残差(randtok 路由带 -5~-9% + 巨 KV 口袋
  15-40% + regime 过渡),**不是 dep4 新问题**。方向一致:model 偏快
  (中位 model/meas=0.88)。decode A 的 12.5% 集中在玩具坐标
  (kv≤32,亚毫秒级步,调度开销主导)。
- **采集数据本身被真实流量证实**:干净均衡步 vs parquet 格点 = -3~-7%
  (例 (4,1024,0):独跑 84.3ms vs 采集 87.8ms)。

## 2. dep4 特有的方法论发现(本次最大产出)

### 2.1 balanced_v1 坐标系与非均衡步不可通约

采集坐标 = per-rank totals,且采集时**四个 rank 同形状同步测量**(EP
all-to-all 满负载)。真实流量步大多不满足:

- **EP 稀释**:只有 1 个 rank 活跃时,其 MoE token 摊到 4 rank(负载 1/4)。
  deep_gemm 分组 GEMM 在小 token 数有延迟平坦区,故 per-rank 总 token
  ≤2048 时稀释无感(实测 -2~-4%),≥4096 时实测比均衡坐标**快 ~1.9x**
  ((1,4096,0) 实测 147.5 vs 格点 283.8;(1,8192,0) 301 vs 580.9)。
- **lockstep 拖拽**:与更重的并发步(如 burst blocker 的 8192-token 步)
  同步的测量步,墙钟被抬到 ~blocker 级(87.8ms 的步报 258ms,+194%)。

初版按 tep 语义打分得到 prefill A "45% MAPE"——全部是这两种伪影,
不是模型误差。

### 2.2 有效样本判据(本打分器实现)

1. **均衡簇**:到达序相邻的 4 行、每 rank 一行、形状全同、
   **墙钟彼此 ≤3%**(拖拽表现为 2-3x 离群,自动否决)→ 取 max(采集语义)。
2. **blocker 后独跑**:blocker 所在 rank 在 blocker 结束后独自补跑测量步,
   无拖拽;窗口 per-rank 总 token ≤2048(平坦区)时稀释可忽略 → 有效。
   判据被 5 个 rep 全验证(blocker rank == 独跑 rank,-3.9% vs 格点)。
3. 其余(bp 不整除 DP、单 rank 长请求分块、bp≤2 mixed 探针)=
   **结构性不可通约**,列账不计分:本次 186 窗(prefill 139 + mixed 47)。

### 2.3 覆盖代价与残余偏差

- kv 深梯子(长请求 prefix 阶梯)在 dep4 下全部不可通约——单 rank 分块
  必然非均衡。本次 prefill kv 覆盖上限仅 98,304(9.1%)。
- 独跑估计器带轻微稀释低估(实测侧偏低),表现为 prefill 全点 model 略高;
  A 层独跑点(3.5-5.3%)与均衡簇点(2.9-6.8%)同带,未见额外失真。

## 3. 对后续 dep L3 驱动器的设计要求(交 L3 kit 维护者)

1. **burst 预乘 DP**:像 decode/mixed pool 一样发 bp×DP 个请求,让每个
   rank 恰好收到计划形状(bp 为 per-rank 目标);router 摊派不可靠时用
   数量冗余 + 打分侧均衡簇判据兜底。
2. **弃用单 blocker**:一个 8192-token blocker 会拖拽全部并发测量步;
   要么每 rank 一个等重 blocker(保持均衡),要么改为时间隔离
   (blocker 完成后再发测量 burst)。
3. **长请求 ×DP 并发**:4 条同长请求同时发(每 rank 一条),分块步
   即为均衡步,kv 深梯子恢复可通约。
4. mixed 探针同理:bp 取 DP 的倍数。

## 4. 复现

```bash
.venv/bin/python fpm_e2e_20260811/build_dep4_analysis_db.py
.venv/bin/python fpm_e2e_20260811/score_l3_dep4.py \
  --dir fpm_e2e_20260811/l3_harvest_dep4_live \
  --plans fpm_e2e_20260811/l3_dep4_bundle \
  --root fpm_e2e_20260811/dep4_analysis_root
```

逐点明细:`l3_harvest_dep4_live/l3_scores.csv`(198,180 行,decode 占绝对
主体)。窗口解剖证据(bp16/n256 的 blocker/拖拽/独跑三态、bp4/n512 的
干净簇与 99ms 拖拽线)见本文件 git 历史所在会话记录。
