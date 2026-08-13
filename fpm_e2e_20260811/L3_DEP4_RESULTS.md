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
- **decode 的 11.35% 不是一个 gap,是三个成分**(分解见 §5):
  Bd=512 的 DP graph 边界跨骑(2,062 点,34%,确定性,机制已判)+
  小墙钟 serving 常数开销(17 万点,绝对差中位仅 +3.2ms)+
  真实建模质量(墙钟≥50ms 且剔 512:**5.63% / P95 11.9%**)。
  tep4 decode B 9.24% 是同配方(无 DP,不踩 512 跨骑)。
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

## 5. decode 三成分归因(2026-08-13 探针战役)

### 5.1 Bd=512:DP graph 边界跨骑(确定性,非抽签)

| 证据 | 数值 |
|---|---|
| 采集 batch=512(balanced,全 rank 精确 512)| 62ms(FULL graph,`expected_cudagraph_mode=FULL, capture_size=512`)|
| 采集 batch=513(超出默认 capture 表)| 103-106ms(eager)|
| 真实流量 C=512 池(L3 + 探针 boot1/2/3,4 个独立 boot)| **106-112ms,全部 eager 档,零翻档** |
| 真实流量 C=513 池 | 111ms(eager,与采集 eager 值 +6% serving 开销吻合)|

机制:decode 引擎(两侧同配置)默认 capture 表 `[1..512]`;DP 下 router
摊派不均衡(rank 持有实测 497-546,偏斜可达 +34),池 ~2048 时必有 rank
越过 512 → 该 lockstep 步全组落到 eager 水平。自基准的 balanced_v1(全
rank 精确 512)命中 graph,真实部署永远跨骑 → 62 vs 111 的 80% 错位。
旁证:C=300 池(全员 graph)差 +0.27ms;C=600/900/1024(两边都 eager)
差 ±3ms;窗口排水段(全员降到 ≤512)墙钟从 112 骤降 76.8;tep4(无 DP)
的 C=512 无此现象。

**修法**(验证实验 `probe_fix_capture.sh` 待 GPU):decode 显式
compilation-config 把 capture 表覆盖到最大并发(prefill 配置早已到 2048)
→ 边界移出工作区,真实性能本身提升 ~80%,且与采集值重新对齐;上游
issue:DP padding-to-max 使 ±1 请求不均衡把全组打下 graph。
注意采集侧 (512, kv=16384)=75.2ms 鼓包是另一独立小异常——按 merge-test
(large kv) session 的判决(该 kv 区异常率 0%,跨启动散布 0.1-0.3%,引擎
确定性),它既非"口袋"也未必是边界机制,最可能是自基准单样本测量伪影
(wall_time 步配对错位,详见其 HANDOVER §6.1 修订);同因,采集库中
kv≳2M 的 decode 单点(473 行)在其修法落地前不可作归因支点。

### 5.2 小墙钟 serving 常数开销

剔 512 后墙钟<50ms 的 17 万点:MAPE 11.9% 但绝对差中位 +3.2ms、95.8%
实测偏慢——自基准安静稳态 vs serve 语境(lockstep 木桶抖动 + 同 pod
frontend/bench/listener CPU 竞争)的固定开销,小步上放大为 10-20% 相对差。
逐点等权 MAPE 被小墙钟步主导,故总均值 11.35% 高估了建模误差。

### 5.3 结论口径

对吞吐建模要紧的口径(墙钟≥50ms,剔 512 跨骑):**decode 5.63% / P95
11.9%**。512 跨骑修复(capture 扩表)落地后应全量重测。

## 6. 复现

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
