# Handover — dynamo-FPM(自基准运行时)改动全记录:randtok2

交给 dynamo-fpm dev。内容:改了什么、为什么、问题出在哪、怎么改的、结果、
以及还欠什么。所有数字均来自 fpm-all-20260811 战役实测(证据索引见文末)。

## 0. 一句话

dynamo FPM 自基准的合成输入是 `[0]*n`(全零 token),使 MoE 路由退化,采回的
数据带系统性偏差(小 M 偏快 -5~-9%,大 M 偏慢 +15%);改成 **salt 配对的随机
token** 后,prefill 全域(128-8192)与真实流量对齐到 ≤±1.6%。第一版随机化
(randtok)因破坏 prefix 播种曾报废一轮采集,教训在 §3。

## 1. 症状(问题怎么被发现的)

精度战役用"真实流量下的 FPM 流逐步实测"做 ground truth(不是端到端客户端
指标),对账 parquet 时发现:

- decode 稳态:parquet@(64, 531k..582k)=15.43..15.72ms,真实稳态步
  16.94..17.22ms —— **采集数据系统性偏快 8.7-9.3%**,且对 KV 均匀(排除噪声);
- prefill:小 token 档偏快(128 档差 -6ms 量级),8192 档偏慢 +15% ——
  **符号随 M 翻转**,单一常数解释不了;
- 决定性线索:KV 阶梯倒挂(同 batch 下更大 KV 反而更快的采集行),指向
  "被测的计算本身和真实计算不是同一种"。

## 2. 根因

自基准构造合成请求时 token 全为 0:

- MoE 路由器对全零输入产生**退化路由**(所有 token 走同一小撮 expert):
  小 M 时命中的 expert 权重驻留 cache → 偏快;大 M 时 EP 各 rank 负载极度
  不均 → 偏慢(TEP 下木桶效应);
- 判别实验(同引擎、同点位、只换输入):随机 token 后 prefill 128-8192 与
  真实流量差收敛到 ≤±1.5%;换回全零立即复现偏差。归因三重证据
  (temperature 实验 / KV 平坦性 / B 钟形)详见 LEDGER。

## 3. 改动(镜像层,instrumented_scheduler.py 三处替换)

文件:`/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py`
(dynamo vllm 运行时的 FPM 自基准调度器)。三处 `[0]*N` 替换:

```python
# 1) prefix 播种请求(为 total_kv_read_tokens>0 的点构造已缓存前缀)
- prompt_token_ids=[0] * prefix_tokens,
+ prompt_token_ids=__import__("random").Random(cache_salt).choices(range(1, 199000), k=prefix_tokens),

# 2) 测量请求本体
- prompt_token_ids=[0] * prompt_len,
+ prompt_token_ids=__import__("random").Random(cache_salts[index] if cache_salts is not None else req_id).choices(range(1, 199000), k=prompt_len),

# 3) padding 请求
- prompt = [0] * padded_len
+ prompt = __import__("random").Random(padded_len).choices(range(1, 199000), k=padded_len)
```

### 关键设计:salt 配对(第一版就栽在这)

**randtok v1(废)**:播种方和消费方各自独立随机 → 前缀内容不同 → vLLM 的
block hash(含 cache_salt 与 token 内容)对不上 → 所有 kv>0 点
`fake_prefix_cache_validation_failed`,当轮 **8,914 点阵亡**。教训:我的
判别实验只测过 kv=0 点,盲区放过了它。

**randtok2(现役)**:播种方与消费方用**同一个 salt** 作 RNG 种子,依赖性质
`Random(salt).choices(k=prefix) == Random(salt).choices(k=longer)[:prefix]`
(同种子前缀一致,已单测验证),消费方生成的 prompt 前缀与播种内容逐 token
相同 → block hash 命中。上镜像前在真机验证 8/8 prefix 点通过。

## 4. 镜像与发布

| tag(nvcr.io/0980761089281446/dynamo-fpm-frozen)| digest | 状态 |
|---|---|---|
| `gc-steady-randtok2-20260812` | sha256:3067293d… | **现役**(x86 h100/h200)|
| `gc-steady-randtok-20260812` | sha256:940e6bd4… | **废弃勿用**(prefix 播种破坏)|
| `gc-steady-16xfix-20260809` | (基底) | superseded,其 parquet 带路由塌缩偏差 |

发布方式:in-cluster crane pod(`crane append -f layer.tar -b <base> -t <tag>`,
nvcr-push-secret 只挂载在 pod 内,凭证不落本机)。runbook Appendix B 已同步。

## 5. 结果

- **正式重采(randtok2)**:h200 TEP4/TEP8 × prefill/decode 4/4 cell 通过,
  0 错误,22,217 行,L0 闭环位精确(13,281 点 0.000e+00);
- **prefill 修复达成**:全坐标对齐真实流量(128 档:18.10 vs 真实 18.37;
  8192 档:150.09 vs 150.7);v2 探针离网中位 1.1/2.8%(tp4/tp8);
- **暴露了误差抵消**:旧数据的 TPOT "-6.5% 精度"是 prefill 偏慢与 decode
  偏快互相抵消;数据修对后诚实差距为 -11.9%,全部来自下一条;
- **decode 残余 -5~-9% 带(本改动不解决,如实记账)**:均匀随机 token 的
  路由分布 ≠ 真实文本的偏斜路由分布。判别实验(qwen32b dense 对照:dense
  模型无路由,若带消失则归因闭环)尚未做。**不要用常数补偿。**

## 6. 战役中发现、尚未修的 dynamo-FPM 侧问题(按优先级)

1. **巨点 KV 播种静默失败 → 坏行**:3 条物理不可能的行((256, 6.56M)=11.22ms
   真值 57.46;(481, 2.1M)=12.35 真值 39.77;(496, 1.05M)=17.29 真值 30.97)。
   播种失败后基准照常测量(实测的是近乎空 KV 的步)且结果标记 complete。
   **修法**:播种后、测量前校验实际生效的 KV 块数,达不到点位要求就把该点
   标记 failed(宁缺毋假)。这是自基准的正确性守卫,优先级最高。
2. **regime 过渡坐标 warmup 不足**:b=513(graph→eager 过渡后第一个批次)
   行 +12% vs 真值(98.72 vs 87.90);(256, 4096) 行 +18%(24.72 vs 20.88,
   3 遍稳定)。疑似过渡点的 5 次 warmup 不够(新 regime 首点)。
   **修法**:对 regime 过渡坐标加大 warmup 或复测取中位。
3. **巨 KV 档位抽签(引擎问题,基准可缓解)**:kv 总量 ≳3M 区域,步延迟落在
   2-3 个相距 15-40% 的离散档,同形状跨启动/同启动跨微形状都会换档;七次
   启动判别:噪声排除(对照 ≤1%)、autotune 排除(关掉照翻)、CUDA graph
   捕获态为主嫌(eager 对照 ≤4.4%,非唯一,768 点 +12.7% 存疑)。
   **基准侧缓解**:标记坐标多启动重复采集取中位(采集编排层落地);
   **根治**:给 vLLM 报 determinism issue(七启动数据表在
   `probes_v2/pocket/`)。

## 7. 上游化建议(dynamo 仓)

randtok2 目前只活在冻结镜像的单文件层里。建议向 dynamo 提 PR:

1. 合成输入默认随机化(带 salt 配对语义,§3 的三处);
2. §6.1 的播种校验守卫;
3. 可选:暴露每点 warmup 次数配置(§6.2)。

提交时引用本战役数据作 motivation(全零输入的偏差量化:-9%~+15%)。

## 8. 证据索引(fpm_e2e_20260811/)

| 文件 | 内容 |
|---|---|
| `LEDGER.md` | 战役总账:根因判定链、randtok v1 事故、round-2 重采、七启动判别 |
| `probes/tep8/probe_rand*.json` + `probe_zero_small.json` | 全零 vs 随机判别实验原始数据 |
| `per_step_validation.csv` | 12,647 真实步逐步对账(修后数据) |
| `probes_v2_scores.csv` | v2 探针全点位(prefill 对齐的最终证明) |
| `probes_v2/pocket/pocket_b{1..7}.json` | 巨 KV 抽签七启动判别 |
| `../scripts/experiments/README.md` Appendix B | 镜像谱系与规则 |

## §6 补充(2026-08-12 多并行战役新发现):第 4 个待修问题

4. **永久不可行 cell 阻塞整个计划的发布**:发布门是"计划内全 cell 通过"
   (runner.py `all_passed and covers_full_plan`),而内存裁决无法证明不可行的
   拓扑(如 M2.7 的 2 卡形状:估算器在 moe 整除性上报错 → 交运行时验证)在
   真机上永久失败后,该计划永远无法发布——8 个通过 cell 的数据被 4 个
   注定失败的 cell 扣为人质,只能收窄形状重跑(全部重采,GPU 时间翻倍)。
   **修法**:运行时验证失败且分类为拓扑不可行的 cell 应落 `infeasible` 终态,
   发布门接受 `passed ∪ infeasible` 覆盖全计划;或允许计划带审计记录地重冻结。
   (归 collector;实测案例:fpm_forward_artifacts/4df9b0f29115c63d)
