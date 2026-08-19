# R16 终审后修正任务表(按模块,2026-08-20 用户裁定版)

> 所属 session:`d07354d7-72f0-47a1-aeab-be8fd346e942`。
> 每项均有实验实证背书,证据指针:R16_FINAL_REPORT.html 第 7 章、
> experiments/EXPERIMENTS_STEPBYSTEP.md(E1-E9)、r16/scores/*.csv.gz。
> 本表是分工派单的唯一权威版本(v2,取代 4c8ef871 的 v1);裁撤项列在文末,不要复活。
> 分工现状:B1 已派单(由本 session 承接实现);A 组由 merge-test session 承接;
> C1 已实现入分支;D 组已全部完成(3dbc65e8/a2e5056a/6040ecbc),**勿重做**。

## A. 引擎/镜像(dynamo-fpm kvwarm 补丁层)

**A1|kvwarm 删 moe_tp skip**
`_kvwarm_warm_eligible` 里 `elif not ep_enabled: skip("moe_tp_balanced_by_construction")`
分支删除(dense 的 skip 保留——dense 无 expert,真物理免疫)。
实证:同机 fake/warm 24 点 −15.3%(E7);全网格重采 tp4 decode 10.43%→4.80%
(E8,含中段 12%→3.8%、巨段 1.51%)。交付=烘进正式镜像。

**A2|渲染撤 pure_tp 的 prefix-caching pin(A1 联动)**
缺陷1 条件化修复把 pure_tp 归入"禁 prefix caching"侧;开 warm 后 tp4 必须
prefix caching ON(warm 谓词前提)。渲染规则改为"warm-eligible 拓扑全开"。
没有 A2,A1 无效。位置:collector fpm_forward runner 的引擎参数渲染。

## B. collector(挂 #1473/#1475 PR 链)

**B1|v6 schema 加 `kv_seed_regime` 列**(派单已发)
- native_artifact.py:NativeCollection 增 `kvwarm_meta`(rank payload 顶层
  "kvwarm" 块;跨 rank 校验 warm_eligible/skip_reason 一致);半成品 diff:
  experiments/collector_kvwarm_meta.patch;
- database.py aggregate_cell:每行推导 kv_seed_regime——
  **cell 级 skip_reason 非空 → `skip:<reason>` 优先**;否则点级
  kvwarm_real_kv→`real_kv` / kvwarm_fake_fallback→`fake_fallback`;
  无元数据→legacy;prefill→`n/a`。
  【实测陷阱】warm-ineligible 拓扑的逐点标记全部是 fake_fallback
  (tp4 1659/1659),点级判定必须让位于 cell 级,否则下游误杀整库;
- write_formal_database:加性列(不进 _ROW_KEY),老行合并为 null,
  验证 pyarrow 混合行写出与 row_count 校验;
- 验收:存档回归 76923ff0498bfc2a(tep4:decode 102 fallback/其余 real)、
  511c5e49d1331154(tp4:全 skip:moe_tp_balanced_by_construction);
  collector 单测全绿;遵守 layer_permissions(记录是数据,collector 不过滤)。

## C. aic-core SDK(modeling)

**C1|装载器排除 `kv_seed_regime == "fake_fallback"` 行**(唯一保留的 modeling 项)
已实现入分支(fpm_forward.py load_fpm_forward_data,env 门控
`FPM_EXCLUDE_FAKE_FALLBACK=1`,加载后建索引前过滤,内存自洽不掉慢路径)。
实证:tep4 4.00→3.72 / 2.82→2.50,尖刺带 26.5→0.2%,覆盖代价 1.4-1.5%
顶格坐标(物理不可温区,fail-closed)。转正式:定默认值 + Rust 移植同步。
注意:`skip:*` 行保留(拓扑级合法制度),只排 fake_fallback。

## D. fpm_verify(验证方,已自理完成大半)

- D1 ✅ decode 真值驱动默认 ShareGPT(decode_driver.py,`L3_DECODE_DRIVER=bench`
  回退;v3 九列窗口,isl==1 锁步校验)——实证依据:random 池偏快 −1.51%(E6);
- D2 ✅ 同机协议:`PIN_NODE=<采集节点> bash stage_and_run.sh ...`,nodeName
  落产物;跨机 ±3-4% 地板只作参考(E3/E4 定案);
- D3 ✅ fetch_results.sh 断点续传+gzip 块收编(a2e5056a:4MB gzip 块+跨次续传+增长文件前缀定裁);
- D4 ✅ dep4 kit phase_mixed 尾链双跑隐患已拆。

## E. 验收协议(spec)

decode ≤2.0% 门槛绑定"采集与真值同机"条件;跨机验证判据 ~5%(地板 ±3-4%)。
依据:r15 的 1.70% 为同机同 boot 产物;三节点探针 + 噪声注入因果(E3/E4)。

## 裁撤/暂缓(用户拍板,勿复活)

- B2 老库 backfill 工具(暂缓;逻辑已验证,见 E9);
- nodeName 记录进 aic manifest(机器域信息归验证方自记,不入 aic);
- C2 上边缘查询语义扩展(维持 fail-closed 现状)、C3 平滑性验尸闸、
  C4 插值档选择案、C5 单独排期的 Rust 项(并入 C1 转正式);
- B4 planner 顶格点回撤/引擎跳过路线(被「记录+排除」路线替代)。

## 挂账(不阻塞本批)

GLM dep8 decode CUDA device-side assert(复现入口见 OPEN_ISSUES.md);
tp4 正式库重采(等 A1 镜像后走纯产品链);dep4 b=34/100/101 口袋(~0.5pp);
T4 正式发布(等用户批白名单);上游 #1473/#1475 合并(等 maintainer)。
