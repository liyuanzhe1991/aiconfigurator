# R17 预测 vs R16 真值:先行对比 + 采集耗时 breakdown

> 所属 session:`d07354d7-72f0-47a1-aeab-be8fd346e942`(R17 完整实验,2026-08-20)。
> 性质:**跨机参考口径**的先行体检——真值沿用 R16 终审的三份 decode 真值,
> R17 库的采集节点与真值节点不同机(节点异质性地板 ±3-4%,E3/E4 定案),
> 正式判定(≤2.0% 门)须等同机真值收割(PIN_NODE 到 R17 各 cell 节点)。
> 2 卡三形(tep2/dep2/tp2)从未有过真值,本轮不可打分,待收割后补。

## 一、结论(TL;DR)

R17 库在三份 R16 终审真值上**全面不劣于、两形显著优于** R16 库:

| 形 | R16 库 | R17 库(裸) | **R17 库 + fake_fallback 过滤** | 变化 |
|---|---|---|---|---|
| tep4 | 4.00% | 2.80% | **2.42%**(P95 4.92,MAX 16.3) | −1.58pp |
| dep4 | 2.42% | 2.59% | **2.58%**(P95 5.84) | +0.16pp(带内) |
| tp4  | 10.43% | 2.76% | **2.43%**(P95 5.59,MAX 18.7) | **−8.00pp** |

- **tp4 是本轮修复的主战果**:kvwarm 放行(A1 镜像 + A2 渲染)+ C1 过滤,
  10.43% → 2.43%,比判别实验日手工重建库的 4.80%(E8)更好——产品链
  原生采集 + 原生 `kv_seed_regime` 列的完整效果。
- **tep4** 2.42%:修复叠加后与 dep4/tp4 收敛到同一水平;MAX 从 51% 压到 16%
  (毒行出库的直接效果)。
- **dep4** 2.58% vs R16 的 2.42%:+0.16pp,远小于跨机地板,判平。dep4 本来
  就没有 tp4/tep4 那两类病(kvwarm 原生可用、毒行占比小),符合预期。
- 三形滤后收敛在 **2.4–2.6%**,恰好落在"跨机真值参考判据 ~5%、地板
  ±3-4%"以内且彼此一致——残余差异形状符合纯环境浮动,无单形异常凸起。
- `FPM_EXCLUDE_FAKE_FALLBACK=1` 的边际贡献:tep4 −0.38pp、tp4 −0.33pp、
  dep4 −0.01pp;MAX 的压制更显著(tep4 55%→16%,tp4 73%→19%)——
  与 C1 立项判断一致(治长尾,不治均值)。

## 二、口径(与 R16 终审逐位可比)

- 打分器:`fpm_verify/scoring/score_decode.py`(score-what-forms,滚动中位
  ±20 邻域,一坐标一票;dep4 配速者口径)。
- 真值(与 R16 终审同三份,ShareGPT serving 收割):
  - tep4:`truth2/tep4`(R16 新真值,守卫节点,41 窗)
  - dep4:`truth2/dep4_final`(终审真值,150 窗)
  - tp4:`truth/tp4`(41 窗)
- 身份:产品库(R16/R17)一律 native identity(dep4 = tp1/dp4/ep4)。
  交叉校验:R16 库本轮复打 4.00(tep4)/2.42(dep4)/10.39-10.43(tp4),
  与终审存档逐位一致,口径无漂移。
- R17 库 = `r17/db/fpm_forward_perf.parquet`(53,483 行,六形,原生
  `kv_seed_regime` 列);过滤臂 = 装载时排除 `fake_fallback` 行
  (tep4 decode 102 行、tp4 decode 102 行、dep4 decode 100 行等,
  `skip:*` 行保留)。
- 分数文件:scratch `scores_r17/`(r17_{tep4,dep4,tp4}[_filtered].csv、
  r16base_dep4.csv)。
- 注意:dep4 对比里 scorer 的 old 列(40.67%)是身份错配的废数
  (旧惯例 tp4 形身份查 native 库,缺陷5 案已定性),R16 dep4 基线
  取自单独一发 native 重打(2.42%)。

## 三、采集耗时 breakdown(R17 全程)

**全程 wall:2 小时 38 分 39 秒**(本地 00:12:52 发射 → 02:51:31 最后一流
收官,2026-08-20;三流并行)。折合 GPU 时 ≈ **24.5 GPU·h**。

### 三条流时间线(本地时间)

| 流 | cell 序列 | 起止 | 流 wall |
|---|---|---|---|
| tep(tep2+tep4) | tep2 prefill → tep4 prefill → tep2 decode → tep4 decode | 00:12:52 → 02:51:31 | **2:38:39**(关键路径) |
| dep(dep2+dep4) | dep2 prefill → dep4 prefill → dep2 decode → dep4 decode | 00:12:54 → 02:20:53 | 2:07:59 |
| tp(tp2+tp4) | tp2 prefill(败)→ tp4 prefill ∥ tp2 decode → tp4 decode | 00:12:58 → 02:34:08(r3 部分发布 02:38:06) | 2:21:10 +resume 4 分 |

### 逐 cell 相位秒数(run manifest `collector_phase_seconds` / `engine_phase_seconds`)

| cell | exec_wall(s) | 其中纯测量 inference(s) | 测量占比 | collect(s) |
|---|---|---|---|---|
| tep2 prefill | 1,318 | 724 | 55% | 32 |
| tep2 decode | 1,188 | 59 | 5% | 28 |
| tep4 prefill | 1,393 | 593 | 43% | 29 |
| tep4 decode | **5,196** | 60 | 1.1% | 50 |
| dep2 prefill | 1,343 | 850 | 63% | 40 |
| dep2 decode | 953 | 57 | 6% | 34 |
| dep4 prefill | 2,124 | 1,278 | 60% | 50 |
| dep4 decode | **3,114** | 64 | 2.1% | 47 |
| tp2 prefill | (败,烧 ~22 分钟,双 pod 尝试) | — | — | — |
| tp2 decode | 1,121 | 48 | 4% | 25 |
| tp4 prefill | 1,343 | 586 | 44% | 30 |
| tp4 decode | **5,254** | 46 | 0.9% | 26 |
| (schedule/render 每 cell <8s,忽略) | | | | |

### 耗时结构判读

1. **decode 相的 wall 几乎全是 kvwarm 预热链 + 引擎 boot,不是测量**:
   tep4 decode 86.6 分钟里纯测量只有 1 分钟(1.1%);tp4 decode 87.6 分钟
   里 46 秒(0.9%)。这是 kvwarm 修复的直接代价——tp4 decode 在 R16
   fake 制度下只要 ~10 分钟(旧存档 manifest exec_wall 587s),开 warm 后
   ×9。换来的是 10.43%→2.43%。若要压 decode 采集时长,方向是 warm 链
   复用/顺序优化(引擎域,非本轮范围)。
2. **prefill 相测量占比健康**(43-63%),大头是真实推理,无明显浪费。
3. **关键路径 = tep 流**(2:38:39),因 tep4 decode 的 warm 链最长;
   dep 流最短(2:08)。三流并行的排布已接近最优——单流串行估算 ~6.7h,
   并行压到 2.6h。
4. **tp2 prefill 物理不可行**烧了 ~22 分钟(两次 pod 尝试,引擎 init
   `Cannot auto-fit max_model_len`,114GB 权重 + 密集捕获图放不进 2×H200
   预算),随后 resume + `--fpm-publish-partial` 6 秒完成 3/4 发布。
   缺席是诚实的(IncompleteCampaign 语义 + partial 旋钮验证可用)。

### 归档卫生附注

`r17/manifests/` 里混入了非本轮的 manifest(511c5e49=R16 tp4 存档、
76923ff0=R16 tep4 存档、GLM tp8/dp8 若干)——系 full_db 根沿途累积。
读数时以 cell id 台账(12 个 R17 cell)过滤;tp4 decode 的 exec_wall
双值(587 vs 5,254)即由此而来,5,253.8s(a4ac3d05 原始 manifest)为准。

## 四、下一步(既定)

同机真值收割六路(PIN_NODE 台账:tep4→0477z、dep4/dep2→e01ryp、
tp4/tp2→e01qqz、tep2→e01sbam;mixed 相已改默认跳过),先 4 台节点
并行发 tep4/dep4/tp4/tep2,双占节点串行补 dep2/tp2;2 卡 kit 首窗健康
检查后放行全程。滤后同机分数出来才走 ≤2.0% 正式判定。
