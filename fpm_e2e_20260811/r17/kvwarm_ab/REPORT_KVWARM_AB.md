# kvwarm 开/关 A/B 判决(tep4 × M2.7 × H200,2026-08-20/21)

## 实验设计

同日串行重采 tep4 两臂,与 R17 原始采集逐字同配置(镜像 gc-vocabfix-20260820、
代码树 r16-wt、同 collector 命令),**唯一变量 = `DYN_BENCH_KV_WARMUP`**
(with=默认开;without=off,补丁保证 off 与打补丁前逐字节等价):

- with 臂:prefill cell 25.1 min(9,505 点)+ decode cell 89.1 min(1,659 点),0 错
- without 臂:prefill cell 25.1 min(9,505 点)+ decode cell 19.9 min(1,659 点),0 错
- 节点:prefill 双臂同落 0477z;decode with=0477z / without=e01sbam(健康节点,
  时钟守卫 1980MHz 全过闸;节点地板差 ±3-4%,远小于被测效应)
- 运维差异(不进测量物理):default-scheduler(kai bin-packing 抢占规避)、
  admission.datadoghq.com/enabled=false(挡 pod 内 ddtrace 注入,保持进程内环境
  与 R17 一致)

判决真值:R17 tep4 同机真值(96,074 坐标,fpm_stream.jsonl + decode_windows.tsv,
锁步收割、滚动中位、一坐标一票),两臂同一份、同一打分器
(fpm_verify/scoring/score_decode.py,FPM_REPLACE_FAKE_EXTRAP=1 值替换语义,
with 臂 102 个 fake_fallback 顶格行被站内外插替换;without 臂全行 skip:flag_off,
替换层不触碰)。

## 判决一:精度 gap(decode,96,074 坐标)

| | with kvwarm | without kvwarm |
|---|---|---|
| 全量 MAPE | **1.97%** | **14.70%** |
| 全量 P95 | 5.15% | 54.98% |
| 捕获带 b≤512(n=83,162) | 1.70%(P95 3.86%) | 15.83%(P95 60.94%) |
| eager 带 b≥513(n=12,912) | 3.69% | 7.40% |

**7.5× 差距,且 without 的误差是系统性、随深度换向的**(签名中位,正=虚高):

| kv 深度 | without MAPE | without 签名 | with MAPE | with 签名 |
|---|---|---|---|---|
| 0-10k | 6.6% | -4.0% | 1.7% | -1.3% |
| 10k-50k | 11.0% | -11.3% | 1.9% | -1.6% |
| 50k-100k | 12.1% | -11.9% | 1.7% | -1.4% |
| 100k-300k | 11.5% | -12.9% | 1.6% | -1.4% |
| 300k-700k | 15.5% | -4.9% | 1.8% | -1.3% |
| 700k-1.5M | 30.1% | **+28.2%** | 3.1% | -0.6% |
| 1.5M-3M | 35.5% | **+15.7%** | 2.7% | +1.4% |

浅中深三段病(浅虚低 → 中虚低加深 → 深翻向虚高 30%+),不是常数偏置,
校准系数救不了。结论:**kvwarm 不能关**——关掉后库不可用作模型。

## 判决二:耗时 gap

| 段 | with | without |
|---|---|---|
| prefill cell | 25.1 min | 25.1 min(分毫不差,内置对照兑现) |
| decode cell | 89.1 min | 19.9 min |
| 整臂 | ~120 min | ~51 min |

kvwarm 税 = **69 min/decode cell(4.5×)**,与 R17 七段分解(kvwarm 预热 72.4 min,
占 decode cell 83%)吻合。税的去向:每个 batch 档用 chunked prefill 真实建整池 KV 链。

## 判决三:prefill 健全性对照(burst,113 坐标)

with 5.71% vs without 5.47%——统计同水平(kvwarm 不触 prefill,单变量成立)。

## 附带闭环:Datadog gpu-nodes-agent 时代判决

with 臂新库 vs R17 原库(同仪器、同配置、同坐标 1,557 行配对;唯一差异=采集时代,
R17 在 agent 部署前 1 小时收官):

- 捕获带 b≤512:中位比 **1.003**(n=1,532)——图回放免疫,完全复刻
- eager 带 b≥513:中位比 **1.306**(n=25)——同坐标慢 31%
- 双 real_kv 行同样 1.308/1.003,排除 regime 构成差异

第三条独立证据:with 臂(agent 时代采)对 agent 时代真值打分,eager 带 3.69%
——R17 库(agent 前时代)同带的时代错配病消失,全量从 4.25% 降到 1.97%。
verify 侧(1.22-1.27×)、采集侧(1.306×)、同时代重采(误差消失)三链闭环:
**eager 带 95→118ms 是节点环境在 agent 部署后整体变慢,非我方采集/验证任何环节**。

## 工件

- `AB_tep4_decode_with_vs_nowarm.csv.gz` / `AB_tep4_prefill_*.csv.gz`:逐坐标分数
  (side=new 为 with 臂,side=old 为 without 臂)
- `witharm_/nowarm_fpm_forward_perf.parquet(+metadata)`:两臂库原件
- `AB_*.log`:打分日志;`SHA256SUMS.txt`:封印
- 发射器:scratchpad regime/full_recollect_tep4{,_nowarm}.sh(差异仅 ckpt/db/log/
  KV_WARMUP env 四处)

## 遗留

- kvwarm 税优化提案(挂账):稀疏真实锚点 + 站内外插校准 fake 行(值替换机器
  反向复用),目标把 69 min 税砍到 ~15-20 min 且保 <3% MAPE;需专门实验验证。
- eager 带跨时代口径:库若与真值不同时代,eager 带必须分带处理/标注。
