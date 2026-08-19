# r15 与 R16 的 verify 全流程与规则(含真值数据如何产生)

(2026-08-19。写给复核者:每条规则标脚本出处;r15/R16 差异逐项点名。
"verify" = 对齐验证:独立 serving 台架实测真值 vs 库预测,与采集链解耦。
本文全部事实取自仓库内脚本原文,四路并行提读后合成;与报告数字交叉核对。)

---

## 0. 验证对象与四条铁则

- **验什么**:一个 perf 库(旧=手工链,新=产品链)对真实 serving 引擎**每个
  调度步 wall time** 的预测精度(MAPE/P95)。
- **铁则一 score-what-forms**:不按"计划打了什么"打分,按"调度器实际形成了
  什么批"打分;真值坐标从引擎内部遥测流重建,不从客户端侧推。
- **铁则二 配对**:两库在同一批坐标、同一份真值上对打;任一侧查不到,坐标
  两侧同弃(score_decode.py:169-173)。
- **铁则三 一坐标一票**:同坐标重复测量先取中位,再进 MAPE。
- **铁则四 考卷/练习册分离**:采集侧用 ShareGPT **偶数池**(sha256(正文)[0]%2==0),
  真值侧用**奇数池**——模型被测的内容与建库内容零重叠(sharegpt_ids.py:34)。

## 1. 真值数据是怎么产生的

### 1.1 遥测源(两代相同)

引擎里的 InstrumentedScheduler(r15 时代现场打补丁,R16 烘进 gc-timing 镜像;
serve 模式下只透传遥测,不改调度)把每个调度步发布到 ZMQ 端口
(`DYN_FORWARDPASS_METRIC_PORT=20380`;dep4 每 rank 一口,20380-20387)。
`fpm_listener.py` 订阅后逐行落 `fpm_stream.jsonl`,打分只依赖五个字段:
`dp_rank`、`wall_time`、`scheduled_requests.{num_prefill_requests,
sum_prefill_tokens, sum_prefill_kv_tokens, num_decode_requests}`
(dep4 版监听整体包 try/except——r14 教训:监听器死=全链静默失明)。

**窗口 = 流的行号区间**:每个驱动动作前后各数一次 `wc -l`,`[s0, s1)` 写进
窗口表。打分器只信"窗口内的流",驱动的计划参数只是注释。

### 1.2 decode 层真值:固定并发池

原理:固定并发 C、`ignore_eos` 长产出,decode 期形成稳定"池即批";一窗扫出
一条 (b, kv) 射线(kv 每步涨)。**两代的驱动工具不同**:

| | r14-native 真值(r15 报告用的这份) | R16 kit 真值(本役新采) |
|---|---|---|
| 驱动 | `l3v2_decode_driver.py`(aiohttp 直发 /v1/completions token ids) | `vllm bench serve --dataset-name random`(phase_decode.sh 包一层窗口记账) |
| 内容 | ShareGPT 奇数池(seed `f'{tag}:{rep}:{j}'`) | 随机 token(bench 内置 random,seed=$RANDOM) |
| 进场 | DP 拦路石(每 rank 一块 8192-token 石头)+ isl==1 行做锁步进场校验(窗内纯 decode 步 kv==C×isl 恰好 C 个;不过重试 3 次换 seed,仍不过标 ragged) | 无拦路石、无进场校验 |
| 窗口表 | v3 九列:tag,C,isl,osl,rep,s0,s1,ok,mark(mark∈lockstep/ragged/na) | v1 七列:grp,C,isl,osl,kind,s0,e0 |
| 计划 | r11 v2 格点 ⊕ 填洞 addendum ⊕ 锁步浅池(COMBINE_V2,l3v3_run.sh) | decode_plan.csv 41 行(tep4):浅扫段 isl=1、跳扫@16384/@65536、顶端扫;dep4 30 行(容量守卫裁剪) |
| 引擎 | serve_r11_*_serve.sh(pristine 调度器,serving 模式,无 kvwarm) | serve_run_*_decode.sh:decode-parity 引擎,**显式 --no-enable-prefix-caching**(防 bench 随机请求前缀命中物理挤占 KV 池;README 注明与采集侧 kvwarm 制度不冲突,r15 实测稳态等价 1.70/1.61%) |

后果:R16 真值在打分器眼里全部是"非锁步窗"(v1 七列),浅端走前 4 步剔除
+ ±20 平滑;r14 真值的 lockstep 窗逐坐标直录。两份真值喂同一把尺,r15 库
分别读 1.70%(r14 份)与 2.82%(R16 份)——差里既有环境也有驱动代差。

dep4 专项:plan 的 C 是**每 rank 目标**,实际并发 = C×DP;容量守卫
POOL×(isl+osl) > 9,000,000 tokens 的格跳过(phase_decode.sh dep4:18-19)。

### 1.3 prefill 层真值:burst(两代同一驱动家族)

`burst_driver.py`(kit 与 r14/r15 战场版同源):
- 一格 = (grp, bp, n, kv, repeats=5):bp 路并发、每路 n 个新 token、kv 长
  共享前缀;`max_tokens=1, temperature=0`,一次 burst 一窗;
- **前缀预热**:`prefix_ids(kv)=sharegpt_ids.ids(10000+kv, kv)`,先发
  前缀+1 token 把前缀打进 KV cache(prefill 引擎 prefix caching 保持开启,
  这是 kv>0 格点的物理前提);
- **拦路石**:tep4 在 bp≥6 时先扔一块 8192-token 石头(提前 0.05s),让整个
  burst 在同一步进场;dep4 扔 **DP 块**(每 rank 一块,提前 0.1s,单块只堵
  一个 rank 会双峰),且门槛降到 bp≥2;
- **驱动器现场校验**(把废票挡在采集时而非打分时):单步格(n≤8192)扫窗内
  流,要求存在一步恰好 (bp, bp·n, bp·kv);dep4 加**四 rank 镜像门**
  `dp_balanced`:DP 个 rank 各有一步恰等于每 rank 目标 (bp/DP, …),且组内
  wall max ≤ min×1.03;不过换 salt 重试(上限 reps×3);
- **熔断**:连续 5 格全灭→探引擎健康;活着=判"结构不可构造区"记档继续,
  死了=BURST-DRIVER-ABORT 退 7;dep4 版还带断点续采(已有 ≥reps 窗的格跳过);
- 计划契约(dep4):每行 bp%DP==0 且 bp≥DP,否则 fail-loud(防 0 目标被
  空闲步冒充)。
- 计划规模:tep4 188 格(A42/B45/C39/Cpfx6/长请求56;bp 1~64;n 16~163840
  含 √2 格点;kv 0~98304;eager 111 / graph 77),dep4 162 格(bp 全为 4 的
  倍数,全 graph)。

**r14→r15 的内容修迭代**:r14 burst 有 randtok/realtok 两个变体(唯一差异=
被测请求内容:随机 token vs ShareGPT 奇数池),r14 正采用 realtok;R16 kit
与 dep4 战场版一致(ShareGPT + 镜像门)。

### 1.4 mixed 相位(补充口径,非主判据)

`phase_mixed.sh`:外层 kvd 两档(decode 池 isl∈{8192,65536},
`vllm bench serve random` osl=6000 ignore-eos 常驻),内层按 Bd 分组;
`mixed_driver.py` 每行注入 15 次,单步恰形 (bp, chunk, kvp)+Bd 背景 decode;
zero/chunk 两种 kvp 模式;NONCE=时间戳防前缀缓存吞探针(tep4 曾因复用 salt
丢 27/51 窗)。dep4:POOL=Bd×DP,容量守卫 POOL×kvd ≤ 8M tokens。

### 1.5 (对照)采集侧的 kvwarm 是什么

verify 的对面——被验证的库,其 decode 数据由 kvwarm 采集:按批量档位用
ShareGPT **偶数池**真实文本经分块 prefill 构好一组互异链并常驻,每个测点
(B, kv) 用影子请求 1:1 借链的 block table(只读零分配)走原双拍测量;巨点
(kv≥1e6)重复稳态步取中位;`DYN_BENCH_KV_WARMUP=off` 与原文件逐字节行为
等价(kvwarm_patch.py 头部保证)。random-KV 原始库(9.99%)= kvwarm 之前
用伪造 KV 直测的产物。

## 2. r15 的 verify 是怎么跑的(考古核实版)

关键事实:**r15 没有采过任何新真值**。

1. **真值 = r14-native**:`pod_sequencer_r14.sh` 在 fpm-r13-agg pod 上六段
   连跑:S1/S2 采集(t1_run.sh,agg 模式,kvwarm on)→ S3/S4 decode 真值
   (l3v3_run.sh,单会话/拓扑,预算 5h)→ S5/S6 prefill burst(burst_run.sh;
   tep4 中途熔断 3 次,resume 续采,所以 tep4 burst 真值实际跨多个 boot)。
2. **r15 库**:prefill 用 r15 重采(真实内容 v4 补丁 + 带内 3-boot 中位:
   1 全量 boot + 2 补充 boot,`merge_r15_boots.py` 按物理坐标配对取中位,
   明确不按 benchmark_id);**decode 行原封复用 r14 采集**
   (combine_prefill_decode.py:"内容修复不涉及 decode,复用")。
3. **打分**:decode 分数直接沿用 r14(r15_finish.sh:4);prefill 用
   score_r13_burst.py 对 r14 burst 真值重打(--new-root r15 库
   --old-root r14 库)。
4. 所以 r15 报告里 tep4 decode 1.70% 的完整供应链是:
   **r14 采集(kvwarm)→ r14 单会话真值(l3v2 驱动)→ score_t3_decode.py**,
   同机同期同节点世代;这与 R16"跨会话新真值"天然不同尺度。

r15 时代没有强制时钟守卫/排空守卫(r15 末期因病卡事故补了 clock_guard 的
重采 db_*_clean 变体);staging 已有 per-file sha256(stage_r15.sh)。

## 3. 打分规则全集(带阈值;两代差异点名)

### 3.1 decode(score_t3_decode.py ≡ fpm_verify/scoring/score_decode.py)

**diff 实证:两文件除以下四点外逐字节相同**——(a) --topo 加 tp4;
(b) model/system/backend 参数化(原硬编码 M2.7/h200_sxm/vllm/0.25.1);
(c) tp4 的 MOE_TP/MOE_EP 开关;(d) `native_identity`(新库侧 dep4 按原生
tp1/dp4/ep4 查询)。**全部真值构建规则同一**:

1. 坐标 = (num_decode_requests, sum_decode_kv_tokens)/纯 decode 步
   (num_prefill==0 且 num_decode≥1),wall×1000 进 ms;
2. dep 弃窗:rank 步数差 N > n×1.25+2 → 整窗弃(计数上报);
3. dep 配速者:各 rank 第 k 步同拍,步 wall=组内**中位**;组内 max>min×1.2
   只计数不弃(咨询性);坐标取定拍 rank,定拍键按 cudagraph 捕获桶
   (1,2,4,8 的倍数至 512;>512 eager)——裸字典序在桶边界会选错
   (实测桶效应 +26%,跨 eager 界 +146%);
4. 非锁步窗前 4 步弃(进场碎批);lockstep 窗(v3 标记 + isl==1)全程保留;
5. 同批量窗内总步数 <50 → 整池弃;
6. ±20 邻域滚动中位,邻域 <9 弃;lockstep 窗例外:逐坐标直录不平滑
   (极浅端 wall 对 kv 有真实斜率,平滑会把 kv=1 抬向 kv≈11);
7. 跨窗同坐标全部票取中位 = 唯一真值;
8. F 层:kv < 2b(低于网格地板 per-req<2)按钳位语义查 2b 行,单列不进主口径;
9. A/B 层:坐标恰在该侧 parquet decode 网格 = A(纯插值),否则 B(泛化);
10. 配对同弃;MAPE=坐标 APE 均值,一坐标一票;输出 side,b,kv,truth,model,ape,stratum。

多段真值:--stream/--windows 成对可重复,多役窗口并入同一坐标中位。

### 3.2 prefill(burst 口径)——**这里两代规则确实不同**

坐标 = 每步三元组 (bp, sum_prefill_tokens, sum_prefill_kv_tokens);
纯 prefill 步过滤(num_decode==0);burst 打分对准设计形
(bp, bp·n, bp·kv),dep 对准每 rank 形 (bp/DP, …)。

| 规则 | r15 原版 score_t3_prefill.py | R16 score_prefill_burst.py |
|---|---|---|
| dep 取票 | **连续同形簇**:相邻步、每 rank 一步、同形、wall≤min×1.03,簇值=max;一窗可多簇多票 | **按 rank 首匹配**(不要求连续):每 rank 第一条恰等目标形的步,凑满 DP 个且 max≤min×1.03,**一窗一票**,票值=max |
| 结构跳过 | n>8192 或 bp%DP≠0 整格跳过(结构不可构造) | 无显式跳过(依赖 bp//DP 取整+精确形匹配自然失配) |
| 拦路石孤儿回退 | 有(跑过 8192 石头的 rank 单独出票,限 target·n≤2048) | **删除** |
| 配对同弃 | 无(各侧独立,None 静默跳) | 有 |
| 汇总行 | 只按层报 | 加全量行 |

tep(单调度器)语义两代一致:精确形匹配、跨 rep 取中位。
分层:C 层 = bp 超出该侧 prefill 网格 bp 上限(外插),其余 A/B 同 decode。

### 3.3 chunks 口径(R16 新增,补充)

score_prefill_chunks.py:真值=收割流里**全部**纯 prefill 步(含长请求分块
续步),score-what-forms;dep 用连续同形簇(承袭 r15 语义,非 burst 新规)。
**已知不一致(挂账)**:native_identity 定义了但调用点没传——dep4 两侧都按
tp4 形查询;--topo 无 tp4。作为补充口径未纳主判据。

### 3.4 验收阈值(fpm_verify/VERIFY_RUNBOOK.md §4)

decode MAPE ≤2.0%(基线 1.70/1.61%);tep4 prefill ≤5.5%(基线 4.76%);
dep4 prefill 照出不判(2026-08-18 产品裁定出范围);新 cell 首跑建档。

## 4. R16 的固定 kit(fpm_verify/harvest/)执行面

`stage_and_run.sh <topo> <ctx> [ns] [PHASES=all|prefill]`:

1. apply k8s_deploy.yaml(4×H200、kai-scheduler、病节点
   computeinstance-e01cywzf37n5mecwvv nodeAffinity 拉黑、/results 挂 PVC
   subPath `_yuanli_l3_results/r16_<topo>` 防抢占)→ 等 Running(600s 上限);
2. **clock_guard.sh 强制闸**:每卡 6s 轻载烤机(1024² fp16 matmul),1s×5 采
   clocks.sm 取峰,峰值 <1900MHz 即整链退 9 并提示拉黑节点(v1 教训:满载
   探针会把健康卡功率墙降频误报,故用轻载);
3. staging:kit 全部 *.sh/*.py/*.csv/*.json 逐文件 exec-cat 注入 + 双端
   sha256 比对,5 次重试(kubectl cp 明令禁用);
4. 单 nohup 链:serve_stack(排空守卫→etcd/nats→frontend:8000→listener→
   prefill-parity 引擎,就绪轮询 1200s)→ phase_decode(杀引擎家族
   `dynamo[.]vllm`+`VLLM[:]:` →排空→decode-parity 引擎→逐行 bench serve 记窗)
   → phase_prefill_entry(换 prefill 引擎→burst→mixed)→ HARVEST-ALL-DONE。
   守护进程必须活在链内(exec 会话断则死)。
5. 引擎双 parity 面:decode 引擎关 prefix caching、无 batched-tokens 上限;
   prefill 引擎 `--no-async-scheduling --max-num-batched-tokens 8192` + 密集
   capture 表(1..2048)且 prefix caching 开(kv 前缀预热的物理前提)。
6. 取件 fetch_results.sh:16MiB 分块、pod 侧逐块 sha、5 次重试退避、整文件
   sha 终验、SHA256SUMS.txt 落档(治 teleport 会话级流劣化;>4h 战役取件前
   重登 tsh)。固定清单:fpm_stream.jsonl、三张窗口表、l3_timing.log、
   burst_driver.log、resolved-config-node0.json。
7. 已知瑕疵(挂账):dep4/phase_mixed.sh 尾部链了一次 phase_decode——dep4
   在 PHASES=all 下 decode 会跑两遍(实际操作模式为 PHASES=prefill 使 decode
   恰好一遍,README 未写明);tep4 burst_driver 里 BLOCKER 全局变量是死代码。

## 5. 头条数字的完整供应链(每个数字用的哪份真值哪把尺)

| 数字 | 库 | 真值 | 打分器 |
|---|---|---|---|
| 1.70%(r15 报告) | r15 库(decode 行=r14 采集) | r14-native 单会话(l3v2 驱动,ShareGPT,v3 窗) | score_t3_decode.py |
| 2.82% | 同上 r15 库 | R16 新采(kit,bench-random,v1 窗,守卫节点) | 统一尺(≡原版) |
| 2.97% | 同上 r15 库 | r15-era 存档(l3_harvest_tep4) | 两把尺读数一致 |
| 4.00% | R16 产品库 | R16 新采 | 统一尺 |
| 4.24% | R16 产品库 | r14-native | r15 原版尺 |
| 9.99% | random-KV 原始库 | R16 新采 | 统一尺 |

结论重申:打分器无差(decode 逐字节同,已交叉校准);真值有代差(驱动工具
+会话环境,合计 ~1pp,两库同担);R16 产品库对 r15 手工库的 ~2.5pp 缺口是
采集侧真实回归,集中在小坐标段(signed −6.1% vs −1.1%),判别实验
(新鲜态 vs 锻炼态,fpm-regime-tep4)进行中。

## 6. 复核入口

- kit:`fpm_verify/harvest/`(README=方法论与故障恢复);
- 打分:`fpm_verify/scoring/{score_decode, score_prefill_burst, score_prefill_chunks}.py`;
- r15 原版:`fpm_e2e_20260811/kvwarm_patch/score_t3_{decode,prefill}.py`;
- r14 真值生产:`fpm_e2e_20260811/{r14/pod_sequencer_r14.sh, r13/l3v3_run.sh, r11/l3v2_decode_driver.py}`;
- 数字复算:`fpm_e2e_20260811/r16/R16_CROSSCHECK_R15TRUTH.html` 附全表。
