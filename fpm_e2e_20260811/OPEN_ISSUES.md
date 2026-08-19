# Open 问题清单(2026-08-13,跨 session 汇总)

状态口径:仅收技术问题,不收 campaign 待办。解法栏标注:已验证待落地 / 方案明确未做 / 有待执行 / 没有。

| # | 问题 | 描述 | 出现的地方 | 解法 | 相关 |
|---|---|---|---|---|---|
| 1 | 深 KV 点的耗时读数偶发错误 | decode 每点只测一次,耗时=两次输出到达时刻之差,被测步紧挨 1.3s 注入窗口;CPU 侧抖动使读数错 15%~5 倍(kv≳1M 的点 10-30% 中招)。GPU 实际执行稳定(CUDA event 实测 0.4-2.9%),错的只是计时。 | 全部 decode 深 KV 区;parquet kv>1M 673 行(21%)不可信 | 已验证待落地:N 步 steady 链、丢前两步、取中位(净 13 行;104-179%→0.2-1.1%);根治 CUDA event(原型已验证)。详见 HANDOVER_DYNAMO_FPM.md §6.3 | fpm-selfbenchmark;影响 collect 数据 / modeling 可用行 |
| 2 | decode 采集值系统性比 serving 快 | Bd32-256 快 11-17%(小 KV 最深 -39%)。来源:①假 KV 内容(零/垃圾,attention 无真实上下文)②运行环境差;采样参数已排除。RANDKV 收回 2-11%,化妆式修补到不了 serving 值。 | 全部 decode 数据 | 部分:RANDKV 一行(已验证未落地);(更新 08-20)kvwarm 全网格重采已落地实证(tp4 decode 10.43%→4.80%,中段 12%→3.8%);fake_fallback 行以 v6 kv_seed_regime 列记录、SDK env 门控排除(已入分支);残余归因闭环=节点异质性 4-6% + 真值内容 -1.51%,采集代码无回归;dense 对照裁撤。详见 HANDOVER_DYNAMO_FPM.md §5 | fpm-selfbenchmark;决定 modeling decode 精度上限 |
| 3 | dep 拓扑 batch≈512 数据与部署不符 | 采集全 rank 精确 512 命中 graph(62ms);部署 router 摊派不均必有 rank 越界→全组 eager(111ms),80% 错位。 | dep4/dep8 decode,capture 表边界 | 方案明确待验证:decode capture 表扩到最大并发;前置:owner 拍板部署配置同步扩(配置一致性)。详见 L3_DEP4_RESULTS §5.1 | fpm-collect(引擎配置);vLLM issue |
| 4 | 少数 cell 失败扣住整个计划发布 | 发布门要求全 cell 通过;注定失败的拓扑让通过 cell 数据发不出。M2.7 9 cell + B200 GLM 5 cell 卡着。 | collector 发布流程 | 有待执行:合入 PR #1475 部分发布 + 撤本地临时补丁 + 执行补发。详见 HANDOVER_DYNAMO_FPM.md §6.4 | fpm-collect |
| 5 | 发布前无数据质量检查 | 物理上不可能的行以 complete 落库,零报错,只能事后人工扫。 | collector 发布 + 自基准落库 | 方案明确未做:落库前耗时下限(同 batch 最小 KV 实测 + 0.5×KV字节÷带宽);发布前单调性粗筛。详见 FIX_DESIGN_KV_GUARD.md | fpm-selfbenchmark + fpm-collect |
| 6 | mixed(chunked prefill)公式缺陷 | batch 轴跨 512 台阶桥接错误(b=600 +97%);pad-up 台阶语义缺失(+20.9%)。 | modeling mixed/decode 离网查询 | (更新 08-20)batch 轴 regime 分区 + bracket 已实现并验收(#1461,b=600 +97%→-10.6%,LEDGER 'Acceptance replay');剩余仅可选的 pad-up snap 语义。 | fpm-modeling |
| 7 | prefill 2049-4096 段格点太疏 | 真实曲线下凹,线性插值高估 9-11%。 | prefill 离网查询 | 明确未做:该段加密到 ~512 间隔 | fpm-collect(点阵) |
| 8 | serving 流步计时偶发借-还 | 相邻两步一慢一快(总量守恒),19 对/96k 步;MAPE 无感,MAX/P95 被假值占据(370% 假 MAX)。 | 所有以 FPM 流做真值的评测 | 缓解已落地(score_l3.py 配对滤波);根治 CUDA event 未做。详见 HANDOVER_DYNAMO_FPM.md §6.5 | fpm-selfbenchmark |
| 9 | regime 边界行偏差未重验 | b=513 行 +12%、(256,4096) +18%,扫程内稳定偏离孤立实测;疑似计时家族,未重验。 | tp4/tp8 decode 个别坐标 | 无独立解法;N 步中位落地重采后顺带验证。详见 HANDOVER_DYNAMO_FPM.md §6.2 | fpm-selfbenchmark |
| 10 | 两个残余机制未闭环 | ①步1 被额外停顿撑大(15.7%),停顿在 worker→调度器交付链路哪一环未指认;②步10 在 (496,6.55M) 3/10 异常原因不明。 | 自基准测量链 | 绕过已验证(丢前两步+中位);根因没有。低优先级 | fpm-selfbenchmark |
| 11 | prefill 为何显式关 async | 两侧一致故内部有效;若生产默认 async-on,prefill 代表性可能 1-2% 偏差。 | prefill 引擎参数 | 没有(待查配置来历) | fpm-collect(引擎配置) |
| 12 | dep 拓扑 L3 工具不可通约 | DP 下单请求/单 blocker 产生不均衡步,与采集坐标系不可比;深 KV 阶梯测不了(prefill kv 只到 98k)。 | dep4/dep8 L3 验证 | (更新 08-20)kit 已更新并跑通 dep4:decode 真值驱动默认 ShareGPT(decode_driver.py,bench 回退);同机协议 PIN_NODE;dep4 尾链双跑已拆;dep4 decode 终审 2.42%(产品)/2.75%(手工)。 | 评测工具(L3 kit) |

(kvwarm=decode 采集用 prefix caching 以真实内容逐段预热 KV 的机制,替代假 KV;warm 不可达点回退 fake 并记 kv_seed_regime)

优先级:#1/#4 解锁被卡数据(解法已备好);(更新 08-20)#2 已决策落地(10.43%→4.80% 实证);仅 #3(dep capture 表扩展)仍待 owner 决策;其余为质量加固与机制尾巴。

## 附:timing.phases 消费契约(rev2 镜像,2026-08-18 通报存档)

镜像 `gc-timing-20260818`(digest sha256:adcd28a9…)的 benchmark JSON 在
timing 块新增 additive 子对象 `timing.phases`(schema_version 仍为 2),
键形如 dataset_load_s / tokenizer_load_s / kvwarm_plan_s /
content_tokenize_s(带 *_buckets 按 kv 深度桶)/ content_gen_s /
seed_total_s / import_to_first_bench_activity_s,以及每键 *_n(调用次数)
与 *_first_s(首调用拆分)。

**相位是嵌套树,聚合时严禁平铺求和**:
- content_gen_s ⊃ 池构建(首调用)⊃ dataset_load_s + tokenizer_load_s;
- kvwarm.stages[].build_seconds ⊃ content_tokenize_s。

run-manifest 四段拼法(r16/R16_FINAL_REPORT.md §3 的落地口径;r15/R16=第 15/16 轮 H200 采集-验证战役,档案 fpm_e2e_20260811/r16/):
- engine_launch_s = 进程拉起 → started_at(collector 外测段);
- kvwarm_warmup_s = Σ kvwarm.stages[].build_seconds;
- inference_s = timing.measured_iteration_seconds;
- other_s = benchmark_elapsed_seconds − 顶层互斥相位
  (content_gen_s + Σkvwarm_stages + seed_total_s + kvwarm_plan_s)
  − measured_iteration_seconds。

collector 侧 run-manifest 聚合按此树实现(未知键容忍已具备);落点=
#1475 合并后的紧随 commit,与 DYN_BENCH_PREFILL_CONTENT 产品侧透传同批。

## 附:R16 验收首轮产品缺陷(2026-08-18;缺陷1/2 已提前修复:@8b5d4399 全反转 → @8552f522 按验收方更正条件化(decode+tep/dep 不禁 prefix caching,decode+pure_tp/dense 维持禁——fake 制度下 admission 哈希污染真实存在,禁=serving-faithful(终审更正 08-20:tp4 fake 制度坐实 -15.3%,kvwarm 重采实证 10.43%→4.80%——pure_tp 维持禁的方案作废;A2=对 warm-eligible 拓扑全面撤 pin,dense 维持禁。);timeout 无条件 10800),挂 #1475 栈顶——验收方证实 CLI 白名单只放 K8sConfig、无用户侧引擎参数逃生口,绕行不存在,默认值是唯一修法)

归属更正:两项均为 collector 侧渲染参数(#1475 代码),非 generator 模板。

1. **decode pin 死 prefix caching(高危,parity 面漂移)**:
   `_FPM_VLLM_DECODE_ARGS = ("--no-enable-prefix-caching",)`(runner.py:91,
   仅 decode cell 注入,test_decode_render_pins_prefix_caching_off_* 钉着)。
   kvwarm 增量续深依赖 prefix caching → tep4 正式跑 1661 decode 点全部
   fake 回退(skip_reason=prefix_caching_disabled),即 r15 已定罪的低估
   制度;r15 parity 基准不传该 flag(vLLM 默认开)。
   修法:decode 渲染不再禁 prefix caching;原 pin 测试反转为断言渲染命令
   **不含** --no-enable-prefix-caching + kvwarm 视角 warm_eligible 冒烟断言。
   注意:改动需带动机注释(为何当初 pin、为何 r15 证据推翻)。
2. **--benchmark-timeout 默认 3600 太短**:DEFAULT_BENCHMARK_TIMEOUT_SECONDS
   =3600;kvwarm 生效后 tep4 decode 段 ~80min 预热必超时(r15 基准 10800)。
   显式 extra_cli_args 已可覆盖(argparse 后者胜出,验收方即此绕行)。
   修法:默认抬到 10800(或随 kvwarm 生效放大),保持可覆盖。
3. **FLASHINFER_CUBIN_DIR 未设**(记档,不影响精度):r15 用 PVC 缓存
   加速 boot;提速战役素材。

4. **校验器错杀乱序产物(缺陷3,已修 @68707ffb)**:native_artifact.py 曾
   断言文件顺序==benchmark_id 顺序;引擎契约是执行顺序与 ID 解耦(kvwarm
   按批量/kv 深度降序重排、fake 殿后)。decode 禁 prefix caching 时代该
   路径不可达,前两修复解锁后首个 warm 产物即被错杀("not contiguous",
   实际 ID 集 {1,2,3,4} 完整)。修法=ID 集合断言(1..N 无重无缺)+ 按 ID
   规范化排序(points/groups/跨 rank 对比三处);乱序+缺号双测试钉住。
5. **引擎侧 minor(评审方自记账)**:kvwarm meta 计数 real3+fake2=5>4,
   疑似巨点双计;meta-only 不影响测量。

验收方绕行(留档):--generator-set 'Workers.agg.extra_cli_args=
["--enable-prefix-caching","--benchmark-timeout","10800"]';坏制度首轮
数据已隔离(quarantine_fakeregime_*)。

## GLM-5.2 dep8 decode cell CUDA device-side assert(2026-08-19,B200)

- 现象:纯命令采集 dep8(plan 07414edaf017a4dd)prefill cell 通过(产物已存
  artifacts),decode cell 跑至 per-rank b=497 / kv_reads=763,124 处
  inductor 编译核触发 `cudaErrorAssert`(Worker_DP2_EP2,异步显形于 shutdown),
  8 rank 全体 dump scheduler output;dump 中含 1535-token 预热链注入请求。
- 环境:gc-timing-20260818,GlmMoeDsaForCausalLM(index_topk=2048 稀疏索引),
  NVFP4 NaN-clamp 与 quack FP8 patch 已装。
- 疑点方向:DSA indexer 核在 dep8 warmed/影子 KV 特定形状(b≈497 奇数批,
  kv 763k)上的断言;或 fp8/nvfp4 patch 与该形状交互。
- 复现入口:单点点单 {batch 497, kv 763124} + CUDA_LAUNCH_BLOCKING=1;
  证据:scratch r16-wt/fpm_forward_artifacts/07414edaf017a4dd/cells/
  fpm-da9afd202a49f3a8/logs/。
- 状态:dep8 decode 未采成;tp8 亦待重试。B200 集群已零残留。

### 机制深挖更新(2026-08-20 夜,session d07354d7)

- **断言真身**:`index out of bounds: 0 <= tmp5 < 154880`(×728;154880=GLM 词表)
  ——非法 token id 进 embedding 查表;
- **崩溃请求出生地定位**:引擎 instrumented_scheduler.py **line 2788**
  (fake-decode 请求制造:`prompt = Random(padded_len).choices(range(1,199000), k=padded_len)`)
  ——dump 中影子请求 prompt_token_ids_len=1536 与 padded_len=ctx+1=1536 咬合;
  (497,763124) 为该 tier 的 fallback 点(kvwarm 链不可达);
- **randtok 199000 硬编码族谱**:0812"全零→随机输入"修复引入(按 M2.7 词表
  手工校准),content v4 只换掉两个主位点,line 2788(fake 制造车间)与
  line 3477(分派器回退)仍存活;GLM 词表 154880 → 该上界 22% 越界;
- **但直接凶器判定为合流机制**:本地确定性复算证明崩溃点/对照点的
  prompt 末位 id 均在词表内(beat1 输入无罪),且 GLM tep8 同样有 102 个
  fallback 点却通过——当前结论:**fake 路径读未写垃圾 KV × GLM fp8/DSA
  数值病 → NaN logits → multinomial 垃圾索引 → beat2 embedding 越界**;
  tep8 免死疑为 TP8 编译图无该断言(静默读垃圾,数据同样不可信);
- **修法(与主线闸门天然汇合)**:①fake fallback 行本就该被 B1/C1 排除
  (GLM 上它连行都产不出,直接炸);②line 2788/3477 上界改运行时读
  vocab_size(消雷);③根治=fallback 点跳过+记账(B4 路线,GLM 需要它);
- **逻辑加固(2026-08-20,升格为排他结论)**:多项式采样返回的是 logits
  分布的索引,维度=词表——**有限 logits 下采样索引不可能 ≥154880**。故
  embedding 越界的充要前置 = logits 非有限(NaN/Inf 破坏采样核不变量)。
  结合崩溃步 dump(每请求恰调度 1 token、prompt 末位 id 已验在词表内),
  "垃圾 KV → NaN logits → 采样垃圾索引 → 越界"为**排他结论**;line 2788
  的 199000 上界确认为休眠雷(修法②属预防性)。
- **复现降级为盖章**:单点复现(对照 512,753533 + 崩溃 497,763124 +
  CUDA_LAUNCH_BLOCKING=1,点单 r16/points_glm_repro.json、臂脚本已备)
  因 B200 白天满载(kai:27 节点 GPU 不足)撤下排队,**排深夜错峰档**;
  Pending pod 已清(防夜间无人驾驶孤儿)。

