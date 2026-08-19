# R16 判别实验日:逐步复现手册(2026-08-19/20)

本目录 = 六案判决与两项修复实证的全部脚本、点单、pod 规格与关键补丁。
通用件:`k8s_regime.yaml`(4×H200 pod 模板,病节点拉黑)、时钟守卫
`fpm_verify/harvest/clock_guard.sh`(1900MHz 轻载闸)、staging 一律
exec-cat + 双端 sha256、取件一律分块 sha(大件 gzip)。打分统一
`fpm_verify/scoring/score_decode.py`,真值见 fpm_verify/harvest 方法论。

## E1 锻炼态 vs 新鲜态(判决:排除,+0.21%)

1. 点单:`../points_regime_arm1.json`(171 小坐标)/`arm2`(=arm1 ⊕ 9505 prefill);
2. pod = k8s_regime.yaml;staging 后单 nohup 链:守卫 → `regime_arm.sh arm1`
   (--benchmark-mode decode,复刻 R16 decode cell 逐字命令)→ 杀驻留引擎
   (bench 引擎写完 benchmark.json 不自退,轮询 "merged results" 后 pkill)
   → `regime_arm.sh arm2`(--benchmark-mode agg,复刻 r15 锻炼态);
3. 分析:两臂 benchmark.json 逐坐标中位 wall 配对,(arm1−arm2)/arm2。

## E2 CPU 配额(判决:排除,+0.03%)

同 E1 的 arm1,唯二差异:pod 规格 `k8s_qos.yaml`(cpu 56/256Gi
requests=limits,与产品 cell 同规格)+ nodeAffinity 钉死同一节点。
链:`qos_chain.sh`。对照 = E1 的 arm1(同节点无配额)。

## E3 节点异质性(判决:坐实,4-6%)

同一份 arm1 点单在第三台节点重跑(tp4_chain.sh 链首集成),三节点
两两对照 + 各自对 r14 库/R16 库的存值(parquet latency_ms 直读)。
判据:实测贴哪个库、批量形状(小批重、随批量衰减)。

## E4 噪声注入因果(判决:CPU 争抢机制坐实)

`noise_arm.sh noisy`:引擎启动前在**本 pod 内**起 48 个 shell 自旋 +
4 个 dd(记录 /proc/loadavg 首尾佐证噪声在场),同点单重测;对照 =
同节点 quiet 臂。签名复现判据:小批膨胀(b=1 +8.3%)随批量衰减归零。
机器域结论只入工作记忆,不入 aic(用户裁定)。

## E5 池边毒行边界二分(判决:制度翻转=值跳变)

1. 点单 `../points_edge_bisect.json`:b=512/32 在 2.2M→2.375M 铺 14 档;
2. `edge_arm.sh edgewarm`(warm 默认开)/`edgefake`(DYN_BENCH_KV_WARMUP=off);
3. 判据:warm 臂逐点 sample_reasons(real_kv/fake_fallback)与 wall 的
   跳变位置是否逐档重合(实测 kv 差 0.2% 处 63.1→125.0ms 同步翻转);
   fake 臂全段读数对照(105-127ms,连 2.2M 都虚高)。
4. 历史互验:全零时代库(aic-core 树内,08-12 前)同坐标 48-85ms 且非单调
   → fake 巨 kv 跨时代不可复现。

## E6 真值池内容 ABA(判决:random 偏快 −1.51%)

1. serve 栈(etcd/nats/frontend/listener + kit decode-parity 引擎,
   prefix caching 关)= `content_chain.sh`;
2. 驱动 `content_decode_driver.py`(l3v2 驱动 + L3_CONTENT 开关:
   sharegpt=奇数池 / random=均匀 token;拦路石两臂同为 ShareGPT,单变量);
3. 计划 `content_plan.csv`(异常点 6 格:b6 浅/深、b20@16k、b513 eager、
   b512 容量边、b32 对照);三遍 ABA(sharegpt→random→sharegpt2);
4. 分析:窗口 tag 带遍名,逐坐标配对 (random−sharegpt)/sharegpt,
   A-A 漂移为对照(−0.14%,信噪 ×10)。产物:content_windows.tsv。

## E7 tp4 fake vs warm 最小机制实验(判决:−15.3%)

1. 点单 `../points_tp4_minexp.json`(24 个鼓形带网格点);
2. `tp4_chain.sh`:fake 臂(tp4 cell 逐字命令,唯一受控偏离=去掉
   --no-enable-prefix-caching,两臂同开)→ 制度断言
   (warm_eligible=False/moe_tp_balanced_by_construction)→ **一行 sed**
   `s/elif not ep_enabled:/elif False and not ep_enabled:/` 放行 →
   warm 臂 → 断言 warm_eligible=True;
3. 同机逐坐标配对。修复收益 14 真值配对点:19.94%→2.52%
   (`../tp4_fix_beforeafter.csv`)。

## E8 tp4 全网格 kvwarm 重采(修复终态 10.43%→4.80%)

1. 点单 `../points_tp4_fullgrid.json`(现库全部 1557 个 decode 格点);
2. E7 的 sed + `tp4_arm.sh warm` 全网格(预热链 ~102 档,~2h);
   产出统计:real_kv 1455 / fake_fallback 102(全部为各档顶格点);
3. 建修复库:同网格值替换 + kv_seed_regime 列 + sidecar parquet_sha256
   重算(网格不变→快路径保留);
4. 打分:vs 现库同真值(../scores/scores_tp4_decode_fixed*.csv.gz)。

## E9 "collector 记录 + modeling 排除"试验(tep4 4.00→3.72)

1. 列回填:从存档 benchmark.json 推导 kv_seed_regime——**推导必须结合
   cell 级 skip_reason**(warm-ineligible 拓扑的点全标 fake_fallback,
   直接按点标过滤会误杀整库;tp4 旧库 1659 点全中);
2. SDK 过滤:`sdk_fake_fallback_filter.patch`(9 行,FPM_EXCLUDE_FAKE_FALLBACK=1
   门控;加载后建索引前过滤,内存自洽不掉慢路径);
3. collector 侧配套:`collector_kvwarm_meta.patch`(NativeCollection 暴露
   kvwarm 元数据,v6 列的写方前置);
4. 陷阱记录:直接删 parquet 行会毁 runtime_grid_digests → 全部查询掉
   ~20× Python 慢路径(实测 5 分钟变 2 小时不完);值替换或加载时过滤免疫。

## 关键结果数据(../scores/*.csv.gz)

dep4 终审(2.42/2.75)、tp4 修复(fixed/fixed_filtered)、tep4 列过滤
(colfilter)、外推替换(repaired)、全零时代库(randomKV——历史命名,
实为 all-zero 输入时代,见报告勘误)、r14-native 终审(4.24/1.70 校准)。
