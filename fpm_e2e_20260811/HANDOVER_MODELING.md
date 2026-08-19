# Modeling Session 交接书 — FPM regime 感知改造(2026-08-12)

> **状态更新(2026-08-12 晚)**:任务 A 已在 #1461(`fpm-modeling-rust`)完成并推送
> (公式 commit `74de0ff6`,随后 `0ba8dff2` 合入了含 #1384/#1474/#1496 的最新 main,
> CI 19/19 绿)。**剩余工作 = 任务 B + 任务 A 的验收网格复放**(§1.5)。
> (更新 08-20:两者均已落地并过验收——任务 B regime 分区 #1461 landed,
> 悬崖点全 PASS;验收复放见 LEDGER 'Acceptance replay' 与 'Strict
> three-phase validation'。本文档余下价值=设计与机理记录。)
> 分支拓扑已变:#1384(Python modeling)与 #1474(generator)已合入 upstream main;
> #1461 现在是纯 Rust 增量,基于最新 main。后续改动**直接在 `fpm-modeling-rust`
> 分支做**(fpm-all-20260811 集成分支已落后,仅作实验复放用)。

执行本书即可,不需要读完整战役档案;需要证据时按文末索引取。
详细公式推导在 `MIXED_FORMULA_SPEC.md`(本书是它的执行版,冲突时以本书为准)。

## 0. 你在哪、改什么、不改什么

- 分支:`fpm-modeling-rust`(#1461,head `0ba8dff2`,已含最新 main)。任务 B 在此跟进。
- **只改**:`aic-core/src/aiconfigurator_core/sdk/`(operations/fpm_forward.py、backends/base_backend.py)
  与 `aic-core/rust/aiconfigurator-core/`(operators/fpm_forward.rs 及 engine runtime 混合步路径)。
- **禁改**:Python `sdk/perf_interp/engine.py` **与 Rust
  `perf_database/perf_interp.rs` 的 resolver 实现**(共享插值引擎一行不动——注意
  Rust 侧 GEMM raw 表在 `perf_database/gemm.rs:463` 就用着 `Resolver::ScatteredSites`,
  改 resolver = 动 op-level 预测);`collector/**`;`src/aiconfigurator/generator/**`;
  数据 schema(sidecar v7 是 collector PR 的活)。模块边界规则见
  `.claude/rules/repo-guide.md`,review 也按它执行。
- 数据:`fpm_formal_database_randtok/h200_sxm/vllm/0.25.1/`(22,217 行,schema v6,
  randtok2 镜像重采,已 staged 进 `aic-core/src/aiconfigurator_core/systems/data/`)。
  L0 闭环位精确(13,281 点 0 误差),数据可信;已知 3 条坏行见 §4 注意事项。

### 0.5 影响面论证(op-level 为什么零影响——PR 描述可直接引用)

插值引擎是纯函数 `query(cfg, data, *coords)`;影响 op-level 只有四条通道,逐条已验证切断:

| 通道 | 事实 | 结论 |
|---|---|---|
| 共享代码 | engine.py / perf_interp.rs 一行不改 | 函数本体不变 |
| 调用图 | `_get_fpm_mix_step_latency` 唯一调用点 base_backend.py:1147,被 `if model.forward_model == "fpm"`(L1140)守卫;`query_totals` 只挂在 FPMForwardOp 上 | op-level 控制流到不了改动 |
| 数据 | 各算子自带 cfg+data 进纯函数;任务 B 只切 `load_fpm_forward_data` 产出的 FPM decode 字典,gemm/moe 表由各自 loader 装载 | op-level 的 (cfg, data, coords) 三者全不变 → 输出逐位不变(引用透明) |
| 共享可变状态 | 仅站点索引 LRU(`_SITE_INDEX_CACHE`,上限 32,键 `id(data)`,immutable 契约见 engine.py L135-142)| 被挤占只损失一次 O(N) 重建,不可能产生错误数值;切表 = 新 dict = 新 id,无 stale-index 风险 |

机器证明 = §2 的 op-level 回归门(测试全绿零 diff)。

## 1. 任务 A — 混合步公式:按"本步调度总 token"定价 【已完成,见 §1.5】

现状缺陷与新公式全文见 SPEC §1-§2。要点:

1. 新接口 `FPMForwardOp.query_totals(db, batch_size, total_prefill_tokens,
   total_kv_read_tokens)`——直通总量坐标,绕过 per-request 整除约束(SPEC §2.1);
2. `_get_fpm_mix_step_latency` 的 prefill 分量改为
   `query_totals(1, ctx_tokens + gen_tokens, chunk_past_kv)`(SPEC §2.2,含无重复计费论证);
3. 多 chunk 逐个求和替代摊薄(SPEC §2.3);边界情形 SPEC §2.4。
4. Python 与 Rust 同一轮改完,parity 测试锁住。

### 修改方法(代码级,锚点为当前分支行号)

`query()`(fpm_forward.py L650)已经是"组坐标 → `_resolve`(L623 域门+插值)"
两段式,query_totals 只是跳过组坐标那步:

已落地版本(比草稿严一点):`total_kv_read_tokens` 为必填关键字;prefill 要求
`total_prefill_tokens >= 1`;decode 传非零 prefill 直接 ValueError。域门/不外推
语义原样继承(先门后插值,包围盒外 `PerfDataNotAvailableError`)。

落地位置(0ba8dff2 后):Python 组合在 `_get_fpm_mix_step_latency`(FPM 路由统一走
`run_mixed`:rust 路由优先,Python 组合是其后的显式分支——**wrapper 级 FPM 早退已
删除**,别加回来,它会让 sweep 静默绕过 Rust 引擎);Rust 组合在
`engine/runtime.rs::fpm_mixed_step_components`,并且 #1496 的 per-op FFI 落地后
`mixed_step_breakdown_per_op` 有独立的 FPM 分支(标量核心的 sink 喂不到它)——
任务 B 若动 Rust 混合路径,两处都要看。SPEC §3 五类单测已双侧落地
(悬崖跨界/图内/无重复计费/多 chunk 平均/退化),parity 368 绿。

### 1.5 任务 A 剩余:验收网格复放(已做,见下注)

(复放已完成:跨界行 -39~-46% → +16~+25%,grid median|δ| 13.22%,与 SPEC
预演一致;门限全绿仍依赖 collector eager 加密。)

`mixed_validation_v2_cap2048.csv` 复放与"新 parquet 全网格 median|δ| ≤ 8%"
尚未在真数据上复现(单测用的是 synthetic 悬崖夹具)。做法:randtok parquet
staged 后按 §2 的验收命令跑;门限:跨界行从 -38~-44% 收到 ±15%(旧数据)/
median|δ| ≤ 8%(新数据,离线预演 5.4%)。注意 §3 的联动依赖:eager 段格点
加密(collector 侧)未做前,mixed 会继承 -10% 段误差。

## 2. 任务 B — decode 批轴 regime 分区(新,机制已钉死)

(已于 2026-08-12 落地并通过验收门,LEDGER 'Acceptance replay — #1461':
悬崖点 |δ|≤12% 全 PASS;本节保留为设计记录。)

### 病灶(数值复现分毫不差,勿再猜)

`perf_interp` ScatteredSites 对未采集 batch 做 nn_sites=4 的 k-NN util 迁移,
距离 = |log2(b_site/b_query)|,权重 1/d²。距离度量对 CUDA graph regime 全盲:
查 b=600(kv=65400, tp4)时邻居是 {513(eager,98.7ms), 512(graph,36.3), 497(36.9),
496(35.5)} —— 1 张 eager 票被 3 张 graph 票稀释,调和混合 = 44.37ms(op.query
实际返回值),真值 88.2ms。b=1024 距离 0.77 octave 排第 5,进不了邻居集。
实测危害:b=600 +97%、b=768 +51~95%、b=900 +12%(tp4/tp8 同构)。

### 修法(全部在 FPM op 层,引擎零接触)

在 `fpm_forward.py` **建表时**把每张 decode 表(per cell×phase)切成两张子表:

1. **边界检测(数据驱动,v6 数据即用)**:对相邻采集批次对 (b, b+1),在重叠 KV
   坐标上比较曲线值;中位 ratio ≥ 2 判为 regime 悬崖(实测 512/513 跳变 2.6-3.5×;
   pad-up 对 (8,9)(16,17)… 只有 5-15%,阈值 2× 干净分离)。要求跳变在 ≥3 个 KV 点
   上成立(抗坏行);期望恰好一个悬崖,检出多个 → 报错留人工。B* = 悬崖左值(512)。
2. **切表**:graph 子表 = {b ≤ B*},eager 子表 = {b > B*}(本数据 = {513, 1024})。
3. **路由**:query b ≤ B* → graph 子表;b > B* → eager 子表。每张子表交给
   **原封不动**的 ScatteredSites。b > 1024 的 scale-up frontier 语义在 eager
   子表内自然保留。
4. Rust 同构(op 层切表 + 路由),parity 锁定。
5. (可选,收益小)graph 段内 pad-up 阶梯语义:snap 到下一 capture 批次行值。
   注意它**不是保守上界**(实测 live(12)=11.32 > live(16)=10.12),做不做都行,
   做了把 off-capture 批次从 -21% 收到约 -11%(仍在路由带内)。

### 修改方法(代码级锚点)

decode 表在 `load_fpm_forward_data`(fpm_forward.py,行号已因 query_totals /
部署身份门漂移,按函数名定位)组装,cell 结构为 `cell["tables"][phase]` +
`cell["domains"][phase]`。注意文件里现已有 `_validate_deployment_identity`
(vLLM pinned 旋钮 fail-closed 门,#1384 合入前加的)——与任务 B 无交互,别动。
Rust 侧对应物:loader 在 `perf_database/fpm_forward.rs`(cell 含 per-phase `Node`
+ 预建 `SiteIndex`,切表 = 两组 Node+SiteIndex+boundary 字段),resolve 路由在
`operators/fpm_forward.rs`(注意上游已把 `SiteIndex::resolve` 改名
`resolve_value`)。改三处:

```python
# 1) load 阶段(表组装完、进 _data_cache 前——保住 immutable-after-load 契约):
def _detect_decode_regime_boundary(table) -> int | None:
    batches = sorted(table)
    hits = []
    for b in batches:
        if b + 1 not in table: continue
        ratios = [table[b+1][kv] / _curve_interp(table[b], kv)   # b 曲线上分段线性求值
                  for kv in table[b+1] if _covers(table[b], kv)]
        if len(ratios) >= 3 and statistics.median(ratios) >= 2.0:
            hits.append(b)
    if len(hits) > 1: raise PerfDataNotAvailableError("ambiguous decode regime cliffs: %s" % hits)
    return hits[0] if hits else None

bstar = _detect_decode_regime_boundary(dec_table)
cell["decode_regime_boundary"] = bstar
if bstar is not None:   # 两个新 dict = 两个新 id(data),站点索引缓存天然不串
    cell["tables"]["decode_graph"] = {b: c for b, c in dec_table.items() if b <= bstar}
    cell["tables"]["decode_eager"] = {b: c for b, c in dec_table.items() if b > bstar}

# 2) _resolve(L623)选表处(域门仍用全量 decode domain,先门后路由):
table = cell["tables"][self._phase]
if self._phase == "decode" and cell.get("decode_regime_boundary") is not None:
    key = "decode_graph" if coords[0] <= cell["decode_regime_boundary"] else "decode_eager"
    table = cell["tables"][key]

# 3) query()/query_pass_baseline() 不改——它们只组坐标,路由在 _resolve 统一生效。
```

悬崖缺失(`bstar is None`,如未来某模型没采 eager 点)→ 不切表,行为与今天
完全一致。Rust 侧:loader 在 `perf_database` 模块组装 FPM 表处同构切表,
`operators/fpm_forward.rs` 的 resolve 同构路由。

### 为什么边界能从数据来、不需要 inference 传参

capture 面本来就烙在每一行数据里(FPM 配置一致性教义:部署 == 采集),悬崖对
(512,513) 是采集器特意留下的边界标记。sidecar 记录 capture 列表是 collector PR
的 schema v7 项,落地后"sidecar 为准、数据推导作校验",校验不一致 = 配置漂移
直接报错。你这轮只做数据驱动检测。

### 验收(复放 harness 已就绪)

```
PYTHONPATH=aic-core/src .venv/bin/python fpm_e2e_20260811/probe_analysis_v2.py
```

对照 `probes_v2_scores.csv`(修前基线)。门限:

| 点(decode 配置) | 实测 | 修前模型 | 修后要求 |
|---|---|---|---|
| tp4 b=600 kv=65400 | 88.21 | 44.37 (+98.8%) | \|δ\| ≤ 12%(eager 段内 k-NN 应给 ~95-98) |
| tp4 b=768 kv=65280 | 89.90 | 57.70 (+55.8%) | \|δ\| ≤ 12% |
| tp8 b=600 kv=130800 | 90.68 | 34.45 (+163%) | \|δ\| ≤ 12% |
| tp8 b=768 kv=130560 | 94.33 | 48.29 (+95%) | \|δ\| ≤ 12% |
| 全部 53×2 拓扑 decode 点 | — | MAPE 7.1/14.3% | **除坏行毒化点外** MAPE ≤ 6%,graph 侧点位回归零劣化(±1%) |

单元测试:悬崖检测 fixture(含"多悬崖报错""跳变<2× 不切")、路由正确性、
b>1024 frontier 保留、Rust parity。
**op-level 回归门**:GEMM(Rust raw 表与 FPM 共用 `Resolver::ScatteredSites`)
与全部 Grid 算子的既有测试/parity 套件必须零 diff 通过——这是"resolver 未被
扰动"的机器证明,单独列在 PR 验证清单里。

## 3. 对 prefill:**不要做同样的分区**(已论证,别过度工程)

1. **两条轴走的是不同求值路径,这是全部不对称的根源**:
   - 曲线轴(prefill 的 token 轴)= `_eval_curve` 的 bisect **2 点括住**。
     悬崖对 (2048, 2049) 是相邻整数 → 任何 q ≥ 2049 的左括号必然 ≥ 2049,
     graph 行在数学上进不了括号。例:查 (1, 3084, 0) 的括号是 {2049, 4096},
     全 eager(反向验证:3328 的模型值 106.88 精确等于该两点连线)。
     段内点(如 3084)受的是格点稀疏病(见上一条,-10%),不是混票病。
   - 站点轴(decode 的 batch 轴)= **k=4 最近邻投票**,无括住纪律 →
     b=600 的邻居集 {513,512,497,496} 三张 graph 票压一张 eager 票 → 必须分区。
   - 即使 prefill 查未采集站点(离网 kv),跨站 k-NN 的每个邻居曲线也先在
     查询 token 坐标处括住求值再投票,token regime 由各曲线自己的悬崖对钉死,
     混不了。
2. prefill 图是 PIECEWISE、按总 token 数 pad,**没有批轴悬崖**。探针证据:
   b=2/3328 (-9.1%) 与 b=1/3328 (-10.4%) 偏差同源同幅(都是 token 轴格点缺口),
   无批次效应。
3. v2 探针 prefill 离网已是个位数(MAPE 2.6/3.6%,中位 1.1/2.8%)。剩余两个残差
   都不归 modeling:
   - **eager 段内缺口(2049-8192,当前只有 3 个格点)**:真实曲线段内非单调
     (2049 是 2048+1 补齐尖点,之后先降后升),且凹陷窗口随拓扑漂移
     (TEP4 在 2304-3328 下凹 -10~-11%;TEP8 那里 ±4% 但 4864-5376 下凹
     -10~-11%)——没有可编码的先验,**唯一正确修法是 collector 把 eager 段
     加密到 ~512 步长**(至少 kv=0 全 batch;kv>0 常用档抽查)。
     **与任务 A 联动**:mixed 的 chunk=2048+Bd 行恰好落进此段,加密前 mixed
     会继承 -10% 段误差(SPEC 预演"eager 行 +15-20%"即此)——mixed 验收门
     全绿依赖 collector 加密,PR 描述里要写明这个依赖。
   - 振荡口袋(kernel-plan 形状敏感,实测可复现非坏行):文档化的插值地板,
     数据无法修。
   modeling 侧**不要**试图用特殊锚点规则(如"排除 2049")缓解——窗口随拓扑
   漂移,先验编不出来,且违反"数据自带形状"教义。
4. prefill 的 regime 正确性由任务 A 保证:mixed 用 chunk+Bd 总量查曲线后,
   曲线自带的悬崖对自动把 regime 放对。

## 4. 注意事项(会咬人的)

- **3 条坏行还在数据里**(tp8 decode (256,6557530)=11.22 / (481,2097152)=12.35 /
  (496,1048576)=17.29,真值 57.5/39.8/31.0;tp4 (256,4096) 偏高 +18%):QA 门是
  collector PR 的活,你不清洗数据。但验收统计时**排除被它们毒化的点**
  (`probes_v2_scores.csv` 里 tp8 (256, 5375744) 一点),并在 PR 描述里注明。
- decode 全局有 -5~-9% 路由分布带(真实文本路由偏斜 vs benchmark 均匀随机;
  dense 对照未做)。**不要**为它加任何常数补偿——它是数据层问题,留给后续。
  (终审归因 08-20:该带主体=假 KV 内容〔kvwarm 修复实证 tp4
  10.43%→4.80%〕+ 节点异质性 4-6% + 真值内容 -1.5%;dense 对照裁撤;
  「不加常数补偿」结论仍成立。)
- `FPMForwardOp.clear_cache()` 会清 perf_interp 站点索引缓存;切表后确认缓存键
  仍按 id(data) 正确失效(两张子表是两个 dict,天然不同 key,应无事,但测一下)。
- 本地 CI 清单(过了再推):codeowners strict / import contract / public-api
  contract / workspace doctests / DCO;Rust 侧 parity 套件(0ba8dff2 时 368 通过,
  上游 #1496 扩了用例)。
- 环境坑(实测可用的版本):工具链二进制在
  `~/.rustup/toolchains/stable-aarch64-apple-darwin/bin`(`rustup run` 代理不
  可靠,直接 export PATH);**改任何 .rs 后必须 `cd aic-core &&
  .venv/bin/maturin develop` 重建 .so 再跑 Python/parity**——`cargo test` 只重编
  测试二进制,venv 里的 .so 不会更新(本轮已有一次"新 Python 对旧 Rust"的
  parity 假分叉,教训);Python 用 `.venv/bin/python`(#1461 worktree 的 venv
  editable 指向该 worktree,PYTHONPATH 不必设);ruff 用 `uvx ruff@0.14.1`。

## 5. 证据索引(全部可复算)

| 文件 | 内容 |
|---|---|
| `MIXED_FORMULA_SPEC.md` | 公式推导全文 + §5 机制/边界论证(本书的依据) |
| `probes_v2_scores.csv` | 168 点修前基线(tp4/tp8 × decode/prefill,A/B 分组) |
| `probes_v2/tep{4,8}/*.json` | 探针原始数据(decode ×3/×2 遍、prefill ×2 遍、解剖轮) |
| `mixed_validation_v2_cap2048.csv` | 任务 A 验收网格(30 窗真实逐步) |
| `per_step_validation.csv` | 12,647 真实步 vs 新 parquet 逐步对账 |
| `LEDGER.md` | 战役总账(round-2 章:坏行判定、振荡判定、全部实测) |
| `probe_analysis_v2.py` | 复放打分器(即验收 harness) |
