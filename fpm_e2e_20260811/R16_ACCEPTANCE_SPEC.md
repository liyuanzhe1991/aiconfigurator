# R16 验收规格 — H200×MiniMax M2.7 四卡全并行"纯官方命令"端到端

目标:任何人可复现"采集→建库→预测→对齐"全链,零临时脚本。
范围切分(用户裁定):**采集/建库/预测 = aic 官方命令;对齐(verify)不属于
aic 产品范围**,是独立的验证套件(单独工作项:版本化、文档化、可复现,
但不集成进 aic 命令面)。
开发依据:HANDOVER_R15_{FPM_SELFBENCHMARK, AIC_COLLECTOR, GENERATOR}.md。
本页只定验收判据。审查合并按各模块规则执行。

## 1. 命令面(四条,名字可议,语义不可议)

### C1 采集
`aic fpm collect --model <hf_id> --system h200_sxm --topos tep4,dep4,tp4 ...`
- 自动:边界封顶网格(引擎网格发生器实现;collector 仅对外部覆盖点单
  复用同一校验)、CPU/内存 Guaranteed pod(**模板由 generator 渲染**,
  collector 只消费模板)、镜像默认 `gc-realcontent-20260818`、
  `DYN_BENCH_PREFILL_CONTENT=sharegpt` 透传(内容源可配,分池语义固定:
  采集偶数池)。归属矩阵见三本 handover。
- **不含**(用户裁定:环境健康属运维责任,非产品):节点健康检查/时钟
  守卫、坏节点重试、GPU 状态采样——脚本留档供运维参考
  (r15/clock_guard.sh、r15/gpu_sampler.sh),不进命令面与模板。
- 产物:benchmark JSON(含 §3 耗时字段)+ resolved-config + 采样器 csv +
  run manifest;全部 sha256 清单落盘。
- 判据:tep4/dep4/tp4 三拓扑 0 人工干预跑通;9k+ 点 0 跳点;
  重跑同参数(同节点)带内坐标中位差 ≤3%。

### C2 建库
`aic fpm build-db --runs <...> --require "tp=4,moe_ep=4" ...`
- 语义:物理坐标配对(禁 benchmark_id 入键)、dep 逐坐标多 rank 中位、
  可拼装(prefill/decode 可来自不同 run)。
- 判据:产出 parquet + 元数据(来源 run 的 sha、镜像 digest、协议版本);
  与 r15 手工链在同输入上逐行 diff = 0。

### C3 预测
既有 aic CLI。判据:能消费 C2 产物,输出与 r15 库预测一致。

### C4 对齐(独立验证套件,不属于 aic)
独立工具集(暂名 fpm-verify,归属/仓库位置由用户定),固定 CLI + 文档,
不挂 aic 命令面。
- 语义:parity serving 拉起(stock 捕获、奇数池真值)、decode 锁步收割 +
  prefill burst(DP 拦路石/镜像门)、score-what-forms 配对打分
  (A/B/C 分层、配对弃行、配速者合并)。
- 产物:分数 CSV + 散点 + 对齐报告(口径自动生成)。
- 判据见 §4。参考实现:l3v3/burst/score 工装(fpm_e2e_20260811/)。

## 2. 环境要求

产品义务(进代码):
- pod:requests==limits(cpu/mem/gpu),Guaranteed QoS(launch-bound 弥散
  根因,F 臂终审 3.3%)——由 generator 模板渲染;
- 传输:exec-cat + 双端 sha256(禁 kubectl cp);
- 陈旧产物:任何 wait-for-file 闸门前必须清空输出目录。

运维前提(用户责任,不进产品;验收环境须满足):
- 节点健康:无锁频卡(自查参考 r15/clock_guard.sh,轻载 boost ≥1900MHz;
  勿用满载探针)、无坏 GPU;运行存证需要时可自挂 r15/gpu_sampler.sh。

## 3. 耗时 breakdown(用户硬性要求)

**目的:为采集提速战役画地图——统计的是速度,不是精度。** 每个分量都要能
回答"这块时间能不能压、怎么压",因此除层级汇总外,大头分量必须带
可优化归因:

- `kvwarm_warmup_s` 加按 **kv 深度桶** 细分(优化方向:链跨点复用、
  砍宽续深摊销——现状 tep4 预热 ~80 分钟是全场最大肥肉);
- `engine_launch_s` 加 {权重加载 / 引擎初始化 / cudagraph 捕获} 三段
  (优化方向:热启动、捕获裁剪);
- `seeding_s` 按 kv 桶(优化方向:前缀播种复用);
- `inference_s` 不是优化对象(它就是被测量本身),只作占比参照。

C1 产出如下层级的耗时清单(JSON + 报告表格两种形态):

```
run_total_s                      # 本次采集全部拓扑合计
└─ 每拓扑 <topo>:
   ├─ topo_total_s               # 该拓扑总墙钟
   ├─ engine_launch_s            # 引擎拉起(进程起→台架首点;多 boot 则逐 boot 列出+合计)
   ├─ kvwarm_warmup_s            # KV 预热总时长(未启用则 0 并标注)
   ├─ inference_s                # 真正被测 forward 的合计(= 被测步 wall 之和)
   ├─ seeding_s                  # kv>0 前缀播种、内容池构建等测量前置
   └─ other_s                    # 注入/簿记/落盘等其余(= topo_total − 上四项)
```

- 守恒判据:各分量之和 = topo_total ±5%;各拓扑之和 = run_total ±5%;
- 打点用 time.monotonic,禁在测量路径上加同步/落盘(冻结令:各拓扑总时长
  不得超过现行基线 ±2%);
- 数据来源:台架内相位打点(细粒度可保留 per-point,聚合成上表)+
  collector run manifest(pod 调度/布置等台架外段,单列不进 topo_total)。

**参考基线(r14/r15 实测,验收时对照)**:

| topo | topo_total | engine_launch | kvwarm_warmup | inference | 备注 |
|---|---|---|---|---|---|
| tep4(decode+prefill 全量) | ~103 min | ~5-8 min | **~80 min** | ~13 min | 预热是大头 |
| dep4(同) | ~75 min | ~5-8 min | ~45 min | ~22 min | per-rank KV 浅,预热短 |
| prefill 单独段 | 24/26 min | ~5 min | 0(无 decode 点) | ~13 min | 含内容池 ~2 min |

## 4. 端到端复现判据(验收仗)

| cell | 判据(对 C4 全新真值) |
|---|---|
| tep4 decode | MAPE ≤2.0%(r15 基准 1.70%) |
| dep4 decode | MAPE ≤2.0%(基准 1.61%) |
| tep4 prefill | MAPE ≤5.5%(基准 4.76%;CPU 治理后带内弥散 ~3%) |
| dep4 prefill | 范围外,不设判据;分数照出、口径照标 |
| tp4(纯 tp/moe_tp)等新 cell | **无既有基准**:首跑出数 + 记档,发现新病走正常立案,不阻塞验收 |

## 5. 交付物

一条命令序列(文档化,可复制粘贴;采集/建库/预测为 aic 命令,对齐为
独立验证套件命令)+ 全部产物 sha 清单 + 耗时地图 + 对齐报告。任何一步
出现"需要手工 exec/临时补丁/临时脚本"即验收失败。
