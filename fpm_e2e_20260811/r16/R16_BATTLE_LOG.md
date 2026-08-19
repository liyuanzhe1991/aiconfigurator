# R16 验收仗 — 作战序列与战报(H200×MiniMax-M2.7 四卡全并行,纯官方命令)

代码树:r16-acceptance 分支 @ dfc7896d(= 最新 main + #1473 delta @82d83e50 +
#1475 collector 栈;origin/main 为其祖先,无需三方合并)。
镜像:nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818
(digest adcd28a9…8c48;含 kvwarm+内容池 v4+argsdump+timing.phases)。
集群:nebius-2(H200),namespace yuanli-aic,PVC model-cache。
规格:../R16_ACCEPTANCE_SPEC.md;命令面出处:集成分支 scripts/experiments/README.md
(开发 session 撰写的字面命令,本文仅改 4 卡形状)。

## 0. 环境(一次)

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh   # 用户 SSO
PY=<repo>/.venv/bin/python                   # 带 collector 依赖的 venv
cd <r16-wt>                                  # dfc7896d 检出
export PYTHONPATH=$PWD/src:$PWD/aic-core/src
$PY -c "import aiconfigurator.sdk.common"    # 需 native .so 就位
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-nebius-2"
```

## C1 采集(三条命令,复跑合并进同一 parquet)

公共尾部(三条共用):
```
  --model-path MiniMaxAI/MiniMax-M2.7 --gpu h200_sxm \
  --namespace yuanli-aic \
  --model-cache model-cache:/workspace/model_cache:models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b \
  --image-pull-secret nvcr-push-secret \
  --generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818 \
  --generator-set 'K8sConfig.extra_env=[{"name":"DYN_BENCH_PREFILL_CONTENT","value":"sharegpt"}]' \
  --generator-set 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}' \
  --generator-set 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0}}' \
  --fpm-database-root "$PWD/fpm_formal_database"
```

形状头部(plan-only 已验:每条恰一形):
```bash
# smoke 门(新场景先冒烟:去掉 --fpm-database-root,加 --smoke --limit 1)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep --fpm-tp-sizes 4 <公共尾部去database-root> --smoke --limit 1

# 正式三连(顺序执行,单卡位 4 GPU)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep --fpm-tp-sizes 4 <公共尾部>       # tep4: tp4/ep4
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets dep --fpm-dp-sizes 4 <公共尾部>       # dep4: dp4/ep4
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets pure_tp --fpm-tp-sizes 4 <公共尾部>   # tp4: tp4/moe_tp4(新 cell,首跑记档)
```

判据:0 人工干预;产物 = parquet + run-manifest + benchmark JSON(timing.phases)
+ resolved-config,sha256 清单落盘。

## C2 建库

按产品设计,发布即建库:write_formal_database 在采集尾部把 passed cell 聚合进
`fpm_formal_database/h200_sxm/vllm/<ver>/fpm_forward_perf.parquet`(sha 封印 +
first-publisher-wins)。三次 run 合并同一 parquet,row-key 唯一性在发布时校验。
判据:parquet + commit metadata(run sha/镜像 digest/schema v6)。

## C3 预测(官方命令,铁律:不写自有脚本)

`aiconfigurator cli --forward-model fpm`(cli/main.py:157-162:whole-model
forward 预测,要求 exact model/system/backend/version 的 fpm_forward 数据);
指向新库用官方环境变量 `AICONFIGURATOR_SYSTEMS_PATH`(engine.py:1191 /
rust_engine_step.py:883 的官方解析链)。判据:消费 C2 parquet 出预测,与
r15 库同坐标对拍一致。

## C4 对齐(fpm_verify,独立套件,不属于 aic)

奇数池真值,parity serving(stock 捕获),per ../../fpm_verify/VERIFY_RUNBOOK.md:
- decode:make_l3_plan → l3_decode_driver(tep4/dep4)
- prefill:burst_driver(tep4;dep4 范围外照出不判)
- 打分:score_decode / score_prefill_burst(--new-root=C2 parquet root,
  --old-root=r15 库)→ 散点 + 报告
判据:tep4/dep4 decode ≤2.0%,tep4 prefill ≤5.5%;tp4 首跑记档。

## 耗时地图(§3)

来源:run-manifest(collector 外测段)+ benchmark JSON timing.phases(引擎内相位,
嵌套树:content_gen ⊃ 池构建 ⊃ dataset/tokenizer;kvwarm.stages ⊃ content_tokenize)。
四段拼法:engine_launch=进程起→started_at;kvwarm=Σstages.build_seconds;
inference=measured_iteration_seconds;other=余项。守恒 ±5%。

## 战报(滚动)

- [2026-08-18] 树/镜像/计划就绪;plan-only 三形验证通过;等 teleport SSO。
- [2026-08-18 15:07] smoke tep4(--limit 1)6.3min 过;0 错误,pod 自清理。
- [2026-08-18 15:16] 正式 tep4 首轮 36min "过"——**缺陷1**:渲染含
  `--no-enable-prefix-caching`(collector runner.py `_FPM_VLLM_DECODE_ARGS`
  有意 pin,有测试钉),kvwarm 全跳(skip_reason=prefix_caching_disabled),
  1661 decode 点全 fake 回退(r15 定罪的低估制度)。**缺陷2**:默认
  `--benchmark-timeout 3600` 装不下预热(需 10800)。首轮数据隔离
  (quarantine_fakeregime_*),序列停,pod 清。
- 考古:原 pin 在 fake 协议下是对的(逐点重 admit 全上下文,prefix on 时
  Request.__init__ 块 hash 落被测窗,实测 (256,2.1M) 26.5→121ms;serving
  稳态无此 hash,禁=serving-faithful)。kvwarm 制度把 hash 移到播种期+借链
  免 admit → 必须开。故修法=按 kvwarm 资格**条件化**,非全反转。
- 修复:8b5d4399(decode 不禁+timeout 10800)→ 8552f522(条件化:tep/dep
  开、pure_tp/dense 禁回,渲染判定与引擎 _kvwarm_warm_eligible 同谓词;
  测试参数化五例;collector 997 绿)。
- [2026-08-18 16:34] 修复版 smoke(--limit 2):**kvwarm 复活确认**
  (warm_eligible=True/stages=2/real_kv=3),但 decode cell 被 collector
  校验器错杀——**缺陷3**:native_artifact.py:293 断言文件序=ID 序,而
  kvwarm 重排契约(执行序与 benchmark_id 解耦)使产物乱序落盘(3,1,4,2,
  集合连续)。开发期 decode 永远禁 prefix → 重排从未发生 → 校验器从未被打;
  缺陷1/2 的修复解锁了这条路径。修复请求已发(顺序无关校验+乱序 fixture)。
- 挂账(引擎侧,我):kvwarm meta real+fake 计数 5 > 4 点,疑多拍/重入
  双计,meta-only 不影响测量。
- 教训沉淀:验收仗的价值即在此——三个缺陷全是"手工链正确、产品化漂移/
  错杀"类,只有纯官方命令端到端才暴露。
- [2026-08-18 17:06→19:00] **tep4 修复态正式收官,exit=0,产物全绿**:
  decode 1659 点(102 warm stages,real_kv 1559 / fake 102 = 94% 真实 KV,
  fake 仅池不可行巨点,与 r15 制度一致);prefill 9505 点;topo 113.6min
  (decode cell 82.3 = 预热 72.2 + inference 1.0 + 其他;prefill cell 17.0);
  parquet 11062 行发布(旧 fake 库隔离于 quarantine_fakeregime_db)。
- [2026-08-18 19:37] **缺陷4(基建级)**:dep4 prefill cell 引擎侧完整
  (远程 sha=d342c6d2…,size=52,359,646),取件 3/3 EOFError(gz 断点
  464/526/546KB)——52MB 撞 teleport exec 流上限(tep4 最大成功件 25.7MB;
  dep4 dp=4 四 rank 数据翻倍)。cell 记 failed,pod 回收数据丢。runner 的
  gz+sha+3retry 纪律在但不够。产品语义正确接管:run 尾将按 B2/B3 拒发布+
  campaign_incomplete+非零退出;恢复走官方 resume(重跑同命令,只补失败
  cell,顺带端到端验收 resume)。修法建议(follow-up):分块传输(≤16MB/块
  逐块 sha)或 PVC 出件通道;重试加指数退避。dev session 离线,回线即转达。

## C1-C3 收官(2026-08-18 深夜)

- C1:三拓扑全收(tep4 113.5m / dep4 91.3m / tp4 35.4m,合计 4.00h),
  32,797 行单 parquet;dep4 中途遭 teleport 会话级流劣化(缺陷4,重登自愈,
  重跑 91m);tp4 无预热验证了"预热=一半采集时长"的结构。
- 耗时地图:守恒差 0.0%;kvwarm 109.6m(46%)> 带内簿记 33.1m > 拉起 33.0m;
  inference 43.8m 为被测本体。
- C2:发布内建;坐标合并语义实证(tep4 1659 样本→1557 唯一坐标,102 个
  巨点钳位重复按设计合并)。
- 真值计划:r15 计划复用(tep4/dep4)+ tp4 同构(truth_plans/,含来源裁定)。
- C3:官方 `cli estimate --forward-model fpm` 三拓扑出数(bs32/ctx2048:
  tep4 59.7 tok/s/u;dep4 34.5×128 并发;tp4 75.3)。需重编 native 核
  (schema 7→11 偏斜);上游发现:agg 仿真 ramp 段查询低于采集包络
  (kv=31@batch 25/32 桥)被 FPM fail-closed 拒绝——仿真需 clamp 或网格补
  ctx=1 档;发布布局 vs V3 op 轴布局两条 cosmetic warning。
- 状态:C4(全新真值,三拓扑)待用户指令。

### 耗时地图·按 cell 分列(prefill/decode,用户要求口径)

| topo | cell | cell总 | 拉起+退场 | kvwarm | inference | 内容+播种 | 带内其他 | 调度取件 |
|---|---|---|---|---|---|---|---|---|
| tep4 | prefill | 24.9m | 6.2m | — | 9.9m | 3.2m | 3.9m | 1.7m |
| tep4 | decode | 88.6m | 4.4m | 72.2m(81%) | 1.0m | 0.2m | 8.9m | 2.0m |
| dep4 | prefill | 37.5m | 7.2m | — | 21.3m | 2.7m | 4.3m | 2.1m |
| dep4 | decode | 53.8m | 5.6m | 37.4m(70%) | 1.1m | 0.2m | 7.6m | 1.9m |
| tp4 | prefill | 24.1m | 5.4m | — | 9.8m | 3.3m | 3.9m | 1.7m |
| tp4 | decode | 11.4m | 4.3m | — | 0.8m | 0.2m | 4.5m | 1.6m |

结构结论:prefill cell 大头=inference 本体(不可压,可压项=拉起);
decode cell 大头=预热(72/37m 换 ~1m 测量的 KV 真实性,tp4 免预热 11.4m
反衬);带内其他=每点注入拍+簿记(decode ~0.3s/点,prefill ~25ms/点)。
提速分道:decode 攻预热摊销,prefill 攻拉起,inference 双双不碰。

### C3 预测全景(官方 estimate,fpm forward)

prefill(static_ctx, bs=1, isl=4096):tep4 9.45 seq/s(~106ms);
dep4 14.54 合计/4 副本(~275ms/副本);tp4 9.01(~111ms)。
decode(static_gen, bs=32, ctx=2048):tep4 59.7 tok/s/u;dep4 34.5×128 并发
(seq/s 17.3 最高);tp4 75.3 tok/s/u 单步最快。
物理自洽:dep prefill 单请求慢×并发补吞吐(印证"没人用 dep 做 prefill");
tep4≈tp4 prefill(计算主导);decode 三形排序符合通信/路由结构。

## 缺陷5(用户抓获)+ dep4 decode 40% 案(2026-08-19)

- 现象:R16 dep4 decode 对干净真值 MAPE 40.8%(旧库 2.85%),signed 全段
  −37~−46%;制度指纹翻转(r15 采集 eager 66% → R16 采集 capture 97%)。
- **用户指出真凶**:产品渲染丢失 parity 超参 `--max-num-seqs`。三方对证:
  r15 采集 resolved-config = 512,r15 真值 serve resolved-config = 512,
  R16 产品采集 = null。violate R11-P0 parity 规格。我此前把根因猜到同步器,
  已更正——参数缺失才有硬证据。
- 验证性本地补丁:runner.py dep 策略 decode/prefill 渲染补
  `--max-num-seqs 512`(注明正式修法=按模型 parity 策略层,归 collector dev);
  dep4 重采至独立 verify 库根(病节点已拉黑),打分与制度指纹双验证中。
- 真值污染案(tep4):+7.65% 均匀系数 = 锁频病卡(clock-guard 现场抓获
  GPU0@1590MHz);守卫已烤进采收方法论,病节点拉黑,tep4 全量重采中。
- dep4/tp4 真值节点健康(守卫过),dep4 真值干净(旧库 2.85% 佐证)。

## 缺陷5 案终审(2026-08-19 午后)——40% 是打分器假伤

- 控制变量实验:补 `--max-num-seqs 512` 重采 dep4(resolved-config 实证生效)
  → decode 制度指纹不变(capture 737/0)、分数同量级(2.72% vs 2.55%)。
  **参数是真实 parity 卫生缺陷但非精度原因**;eager/capture 叙事系误导,撤回。
- 真凶 = 打分器身份错配:r15 手工库把 dep4 发布为 tp4 形身份(旧惯例),
  scorer 沿用该口径查 R16 合并库时匹配到 tep4 行 → "40.8%" 实为 tep4 曲线
  对 dep4 真值。scorer 已修(产品库按原生 dp4 身份查询)。
- **修正后 dep4 decode:R16 产品库 2.55%,r15 旧库 2.85%(同真值同口径)
  ——产品链复现且略优于手工链。** 2.0% 门槛两库同超 ~0.6-0.9%,指向真值侧
  残余环境浮动(采收节点不可考);安排守卫过的 dep4 pod 重采 decode 真值终审。
- r15 制度指纹对比(66% eager)系 agg 模式合并档与产品分 cell 档的
  错位比较,不构成证据;dep4 eager 常态档案维持为 serving 观察,不再驱动
  采集制度改动。

## 小坐标偏快案(2026-08-19 下午,用户抓获抽签解释不成立)

- 事实:R16 两轮独立 boot(fake 轮/预热轮)decode 小坐标均偏快 6-8%;
  r15 旧库对同一新真值仅偏 −1%。两轮同向 → 结构性,非单 boot 彩票,
  缺陷6(多 boot 中位)最多解释波动幅度,不解释方向一致。
- r15 报告对照确认:当年 1.70/1.60% 为"同会话真值"(r14-native
  single-session,库与真值共享节点/日期/boot 生态);跨环境后旧库为
  2.82-2.97%,红利 ~1.1pp。real-vs-fake 压倒性在中大 kv 主场复现
  (3×,signed −8.75%→−0.94%)。
- 头号嫌疑(候选缺陷8):产品 cell 拆分使 decode 网格跑在新鲜引擎上
  (r14/r15 agg 模式 decode 跑在 9505 prefill 之后的锻炼态),主机敏感的
  小步在新鲜态系统性偏快;真值 serving 恒为长跑态。次嫌:板卡修订差
  (-000 vs -001,resolved-config gpu_info 实录)。
- 判别实验(待用户拍板):同 boot 内"9505 prefill 后 decode"(复刻 agg 态)
  vs 新鲜态 decode-only 同坐标对拍,~1.5h。
- 附:resolved-config 全字段 diff 闸首跑即产出(737 字段,剔噪 14 差异),
  另暴露 R16 未显式设 DYN_BENCH_KV_WARMUP(依赖引擎默认开)与 UCX/NIXL
  env 缺失(单节点无害,进 parity 清单)。
