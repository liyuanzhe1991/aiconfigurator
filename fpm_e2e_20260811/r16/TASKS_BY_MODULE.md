# R16 判别实验日修正任务清单(按模块归属)

> 所属 session:`d07354d7-72f0-47a1-aeab-be8fd346e942`(2026-08-19/20,
> R16 验收仗判别实验日)。每条任务的实证依据见 R16_FINAL_REPORT.html 第 7 章
> 与 experiments/EXPERIMENTS_STEPBYSTEP.md(E1~E9)。
> 用户裁定记录:C 组只保留 C1;老库 backfill 暂缓;nodeName 记录与机器层
> 结论不归 aic(归验证方法论/工作记忆);真值与采集必须同机。

## 一、引擎/镜像(dynamo-fpm kvwarm 补丁层 → 烘新 frozen 镜像)

| ID | 任务 | 实证 | 状态 |
|---|---|---|---|
| A1 | kvwarm 谓词删 `elif not ep_enabled: skip`(moe_tp 放行;dense 的 skip 保留) | E7:同机 fake/warm −15.3%;E8:tp4 全网格 10.43%→4.80% | sed 版已验证;待改补丁源码 + 烘镜像 |
| A2 | 渲染联动:pure_tp decode cell 撤 `--no-enable-prefix-caching` pin(warm-eligible 拓扑全开 prefix caching;缺陷1 条件化的修订) | warm 谓词前提;E7/E8 均以此运行 | 待 PR(渲染代码在 collector runner) |

## 二、collector(aic 仓库 `collector/fpm_forward/`,挂 #1475 v6 schema 链)

| ID | 任务 | 实证 | 状态 |
|---|---|---|---|
| B1 | v6 加 `kv_seed_regime` 列:`native_artifact.py` 暴露 kvwarm 元数据(补丁已写:experiments/collector_kvwarm_meta.patch)+ `database.aggregate_cell` 行推导。**推导必须结合 cell 级 skip_reason**——warm-ineligible 拓扑(如 tp4 旧采)逐点标记全是 fake_fallback,只看点标会误杀整库(实测 1659/1659 全中) | E9 回填版全链验证 | 元数据补丁已写;行推导待写;待 PR |

已裁撤:~~B2 老库 backfill 工具~~(用户:暂时不用);~~B3 manifest 记
nodeName~~(用户:不让 aic 做,归验证侧自记);~~B4 planner 顶格回撤~~
(被"记录+排除"路线替代)。

## 三、aic-core SDK(`aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py`)

| ID | 任务 | 实证 | 状态 |
|---|---|---|---|
| C1 | 装载器排除 `kv_seed_regime == fake_fallback` 行(加载后、建插值索引前;9 行) | E9:tep4 4.00→3.72 / 2.82→2.50,tp4 终态 4.80,MAX 60.7→16.5;快路径无损 | **已实现入分支**(env 门控 `FPM_EXCLUDE_FAKE_FALLBACK`);转正式定默认值时 Rust 移植同步 |

已裁撤(用户:只保留 C1):~~C2 上边缘查询语义~~(维持现状 fail-closed,
排除后顶格上方 1.4~1.5% 坐标无数据即无数据);~~C3 平滑性验尸闸~~;
~~C4 插值档选择 bug~~;~~C5 单列 Rust 项~~(并入 C1)。

## 四、fpm_verify(验证套件,本人域,不在 aic wheel)

| ID | 任务 | 实证 | 状态 |
|---|---|---|---|
| D1 | kit decode 真值驱动换 ShareGPT 版(奇数池 token 直发 + DP 拦路石 + isl==1 锁步校验,v3 窗口;保留 bench-random 回退开关) | E6:random 池偏快 −1.51%(ABA,漂移对照 ×10 信噪) | 驱动与 phase 改造已写好,**本地未提交待用户点头**;GPU 未验证,下次收割冒烟先行 |
| D2 | **同机协议**:采集开跑时自记 cell pod nodeName;收割 `PIN_NODE=<node>` 钉同节点;nodeName 落 /results/nodeName.txt | E3/E4:健康节点间 4-6%,守卫盲区;CPU 争抢因果复现 | **已实现并推送**(6040ecbc) |
| D3 | fetch_results.sh 收编断点续传 + gzip 小块(968MB 实战版) | dep4 取件两次断点实战 | 待收编(同 D1 一批) |
| D4 | dep4 kit PHASES=all 双跑 decode 隐患(phase_mixed 尾链拆除) | 提读时发现 | 改动已写好,**同 D1 待点头** |

## 五、验收协议(spec 文档,本人)

| ID | 任务 | 状态 |
|---|---|---|
| E1 | 门槛绑定节点条件:同机判 2.0%,跨机只作参考(±3-4% 地板);写入 R16_ACCEPTANCE_SPEC/VERIFY_RUNBOOK | 待写 |

## 六、挂账(不阻塞,均有档)

- GLM dep8 decode CUDA device-side assert(b=497/kv=763k,DSA 架构疑点;复现入口在 OPEN_ISSUES);
- tp4 正式库重采:等 A1 进镜像后走纯产品链(本次实验库为节点C 单机产物,仅作修复实证);
- dep4 b=34/100/101 误差口袋(~0.5pp);
- T4 正式发布(等用户批白名单)、上游 PR #1473/#1475 合并(等 maintainer)。

## 依赖关系(动工顺序)

A1 → A2(渲染跟引擎走)→ 烘镜像 → tp4 正式重采;B1 → C1 转正式
(列先存在,过滤才有依据);D 组与 E1 独立,随时可做。
