# FPM 端到端流程：从库采集到 AgentX 真值对齐（tep4 实例版）

状态：v2.1（2026-08-24）。每个阶段带**可逐字执行的 tep4 例子**——tep4 =
TP4+EP 形，每个引擎占 4 张 H200；1p1d 即 P=tep4（GPU 0-3）+ D=tep4
（GPU 4-7）同节点 8 卡 Guaranteed pod。这就是 agx_p1_tep4 首战的原口径。
低容量兜底：tep2_1p1d 变体（每引擎 2 卡，共 4 卡）流程完全同构，仅作
链路验证/参考读数。

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐
│ A 库采集     │ → │ B 建库/装载  │ → │ C AgentX 采收 │ → │ D 对齐打分   │ → │ E 审计报告│
│ (GPU,一次性) │   │ (离线)      │   │ (GPU,一次性)  │   │ (离线,可重复)│   │ (离线)   │
└─────────────┘   └─────────────┘   └──────────────┘   └─────────────┘   └──────────┘
        真值录一次 = 固定考卷;此后库怎么改都只重跑 D/E,分钟级、零 GPU(阶段 F)
```

---

## A. 库采集（collection）

**干什么**：在真实引擎上按格点逐 cell 测 forward-pass 延迟。decode cell =
(batch 档 × KV 锚点)，prefill cell = (bp, chunk tokens, kv_read)。库 = 这些
测量值的 parquet。

**tep4 例子**：采 tep4 形（TP4+EP，MiniMax-M2.7 / H200 / vLLM 0.25.1，
单引擎 4 卡）。cell 引擎参数与下文 C 阶段 `serve_run_1p1d_prefill.sh`
**逐字节相同**（这就是 parity 的含义：库和真值活在同一个引擎世界）——
`--tensor-parallel-size 4 --enable-expert-parallel`、kv fp8、
`--max-num-batched-tokens 8192`、cudagraph capture 表至 2048、prefill 侧
前缀缓存默认开 / decode 侧 `--no-enable-prefix-caching`。R17 六形库
（tep4/dep4/tp4/tep2/dep2/tp2，53,483 行）就是这么按形逐 cell 采出来的，
tep4 采集节点 = 0477z。
collector 实跑命令（入口 = `aic-joint-test` worktree 的 `collector/collect.py`；
fpm_forward 必须单独采，不能与其他 op 混跑）：

```bash
cd <aic-joint-test worktree>

# 1) 先看计划,不占 GPU:打印 cell 清单与 plan 指纹
python3 -m collector.collect --backend vllm --gpu h200_sxm \
  --model-path MiniMaxAI/MiniMax-M2.7 \
  --ops fpm_forward \
  --fpm-gpu-counts 4 --fpm-tp-sizes 4 --fpm-moe-ep-sizes 4 \
  --fpm-max-prefill-isl 16384 \
  --fpm-database-root <发布root> \
  --namespace yuanli-aic --model-cache <PVC名> \
  --plan-only

# 2) 正式采集:同命令去掉 --plan-only;断点续采必带 --resume
python3 -m collector.collect ...同上... --resume
```

多形 campaign 先例（`fpm/campaigns/fpm_collector_minimax_*`，含
FINAL-REPORT/PROVENANCE）用 `--fpm-gpu-counts 1,2,4,8` +
`--fpm-parallel-presets` 一次铺六形；R17 六形库即此路数。

**注意要点**：

1. **kvwarm 种子是 decode 深 KV 的生命线**：深 KV 锚点必须先真实造出 KV 再测。
   种子欠分配时行标 `kv_seed_regime=fake_fallback`（假 KV 测量，值不可信，
   B 阶段修）；正常行标 `real_kv`。没有这一列的老库无法做 v5 修复。
   dynamo 侧 fail-closed 守卫：`yuanli/fpm-kv-seeding-guard`。
2. **内容分池**：采集用 ShareGPT **偶数池**；真值线用奇数池/AgentX 第三池。
   混池 = 考题污染。
3. **同机 + 环境冻结**：PIN_NODE 钉死采集节点（R17=0477z）;Guaranteed QoS
   requests==limits;轻载 clock guard 6s 探频 ≥1900MHz（**禁全载探针**——
   功耗病毒假阳性）;库与真值必须同镜像世代（eager 带曾漂 1.306×）。
4. **collector 状态机三陷阱**：`--resume` 不是默认，不带会全量重测；
   发布目录动过任何文件都会翻 plan sha、孤儿化 checkpoint；
   `--fpm-database-root` 换 root = 换 plan 命名空间。
5. **等长构造的原罪（已实锤）**：采集批是等长构造（CV≈0），真实流量是参差批
   ——GPU 对参差度有离散分支，等长锚点对真实流量系统性偏慢 +20~28%
   （病带正半边根因）。下一代采集应按真实参差度（CV 0.4-0.8）构造深区锚点。
6. **锚点密度**：深区每档仅 ~3 个 KV 锚点，单锚点踩到快分支就是整格系统性
   偏差（锚点轮盘，病带负半边根因）；1.8-2.0M 有格洞。加密锚点 + N-step
   median 重测是既定修法。

**产物**：`<root>/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet` +
`fpm_forward_perf.metadata.json`。R17 root 已存
`fpm/artifacts/r17-systems/`。

---

## B. 建库与装载（modeling / loader）

**干什么（人话版）**：这一步发生在"采集完"和"能用"之间，核心一件事——
**给库里的坏数据打补丁，但不改原始文件**。A 阶段有 605 行是 KV 种子没铺够
就硬测的（标 `fake_fallback`），测量值虚高 2–3 倍，是毒数据。处理原则像
财务记账：原始凭证（parquet）永不涂改，补丁贴在"程序把库读进内存"的那一
瞬间——纯内存动作，不产生新文件，每次加载自动重做一遍。

**具体怎么替换（v5 算法，全自动，
`aic-core/.../sdk/operations/fpm_forward.py:326-358`，Rust 版同公式）**：

1. **分组**：所有 decode 行按"站点"分组。站点 = 同一套配置（17 个身份列：
   模型/机型/拓扑形/量化等）+ **同一个 batch 档**——直观说就是
   "b=32 这一档从浅到深的整条 KV 测量曲线"。
2. **找参照**：站点内把 `real_kv`（好行）按 KV 深度排序。
3. **外推覆写**：对每个坏行，取其 KV 下方最近的两个好行连直线，
   延长到坏行的 KV 位置，用算出的值覆写坏行 latency：
   `新值 = 好2延迟 + (坏KV − 好2KV) × (好2延迟 − 好1延迟)/(好2KV − 好1KV)`。
   下方好行不足两个则放弃替换、保留原值（宁可留毒不瞎编）。
4. **记账**：替换几行打一行日志。

数字例子：b=32 站点，好锚在 1.05M（27.6ms）和 2.10M（~48ms），坏行在
2.38M（实测虚高 ~130ms）。连线延长到 2.38M ≈ 53ms，覆写掉 130。
残差的来源也在这：真实曲线在池顶上弯，直线延长追不上——所以修复后
cap 带仍剩 −8%~−26%（b32/33 最深），根治 = 把 2.38M 锚点实测掉。

**"自动"的两个边界**：① Python 打分工具里**不设
`FPM_REPLACE_FAKE_EXTRAP=1` 就整个跳过**，静默用毒值、无任何日志；
产品 Rust 路径才是默认开。② 只认 `kv_seed_regime` 标记列，老库没这列
= 空转（无害也无修复）。

**验证修复生效的观察点**——D 阶段打分时 stderr 出现：

```
fpm_forward: replaced 605 fake_fallback value(s) with in-site extrapolation from .../fpm_forward_perf.parquet
```

出现 `excluded ... row(s)` 说明开错了 flag（那是被否掉的删行方案——
删行后查询会错位匹配到别的行，76.5% 误差）。

**注意要点**：

1. **两套实现，默认值相反**：Rust loader（`aic-fpm-modeling`@
   `fpm-fake-fallback-extrap`）**默认开**（`AIC_FPM_FAKE_FALLBACK_RAW=1` 退回）；
   Python 打分工装**默认关**。打分忘开 = decode overall 从 4.25% 变 22%
   （cap 带虚高全暴露）——本阶段最容易踩的坑。
2. **均未合上游 main**：上游 aiconfigurator 连 `kv_seed_regime` 列都不认识，
   产品默认路径仍吃 fake 原值。合入前所有对外数字必须注明口径。
3. **语义分叉隐患**：Python 版只用坏行下方的锚点，Rust 版取全站点 KV 最高
   的两锚——坏行不在站点顶格时两版会贴出不同的值（修复任务已开）。
   当前数据坏行全在顶格，两版等价。
4. **发布身份**：R16 起产品库按真实拓扑发布（dep4=tp1/dp4/ep4）。打分侧对
   产品库要 `native_identity=True`，对 r15 手工库不要。配错身份 = 拿错
   考卷对答案——数字全错但不报错。

---

## C. 真值采收（AgentX 回放 × 1p1d）

**干什么**：aiperf 回放真实 Claude Code 会话（AgentX 256k 语料），打进
Dynamo 1p1d 分离式部署。P/D 两个引擎的 FPM 流分别落盘——P 纯 prefill、
D 纯 decode，天然纯相位。

**tep4 例子**（kit：`fpm_verify/harvest/tep4_1p1d/`，pod `fpm-agx-tep4-1p1d`，
cpu112/mem512Gi/gpu8，镜像 `gc-vocabfix-20260820` = R17 采集镜像；
agx_p1_tep4 首战即此命令，e01sbam 跨机参考口径，181k 条记录零丢包）：

前提：tsh/kubectl 已登录目标集群；PVC `_yuanli_l3_results` 存在；
aiperf venv 与 AgentX 语料已在 PVC（首次由 `bootstrap_aiperf.sh` 就位，
`fpm/artifacts/aiperf-venv/`、`hf-cache/` 有本地备份可传）；
镜像拉取 secret 就位。pod/QoS/标签由 kit 的 `k8s_deploy.yaml` 负责。

```bash
cd fpm_verify/harvest/tep4_1p1d

# 冒烟:不钉节点,c=4 × 300s × 单跑
bash stage_and_run_1p1d.sh <kubectl-ctx> yuanli-aic 300 4 1

# 正式:钉节点,三档 × 25min × 同 seed 双跑(噪声下限)
PIN_NODE=<节点全名> bash stage_and_run_1p1d.sh <kubectl-ctx> yuanli-aic 1500 4,16,64 1,2

# 取件 + 全套分析(audit/包络/双相打分/分层/噪声下限/config diff)
bash fetch_and_analyze.sh <kubectl-ctx> <outdir>
```

内部结构（单 nohup 链，全部自动）：etcd/nats → frontend（disagg，探活 =
真实 completions 请求）→ fpm_listener ×2（P 段 20380-/D 段 20480-）→
P 引擎（`CUDA_VISIBLE_DEVICES=0,1,2,3`，TP4+EP，`--no-async-scheduling`，
前缀缓存开）→ D 引擎（`CUDA_VISIBLE_DEVICES=4,5,6,7`，TP4+EP，
`--no-enable-prefix-caching`，async 默认开）
→ aiperf 档位序列（`replay_tiers.sh` 在每档前后快照两条 stream 的行号写
windows v3 九列）。相位标记：`DISAGG-PROBE`/`ENGINE-READY` →
`AIPERF-READY`/`CORPUS-READY` → `TIER-c{N}-DONE` → `HARVEST-ALL-DONE`。
结果落 PVC `_yuanli_l3_results/agx_tep4_1p1d`。

**注意要点**：

1. **pod 必须带 `admission.datadoghq.com/enabled=false` 标签**——Datadog
   注入 webhook 会让引擎 import 即死。从修复后 kit 派生，勿手抄旧 yaml。
2. **容量三资源联查**（GPU+CPU+mem）再发射——只数 GPU 会空放。0477z 长期
   只有 7 张 allocatable，8 卡 pod 上不去：降级顺序 = 换健康 8 卡节点
   （跨机参考口径，首战 e01sbam 即此）→ tep2_1p1d 4 卡变体（链路验证）。
   非同机读数全程标"参考口径"。
3. **disagg flag 两代不同**：`--is-prefill-worker`（老树）vs
   `--disaggregation-mode`（新树）。kit 用运行时探针自动适配，探不到就
   失败关闭并把 `--help` 落盘——别手改脚本猜 flag。
4. **首跑必做 resolved-config 全字段 diff**（P 对采集 prefill cell、D 对
   decode cell）：除 disagg 增量外应零漂移。这个 diff 抓过 max-num-seqs
   缺失、UCX env 缺失、2 卡 capture 表单边 8.6% 三次事故。
5. **frontend 必须真探活**（发一条真 completions），端口通了 ≠ 能服务。
6. **监听器先于引擎拉起且异常全免疫**——监听器死 = 全链静默失明。
   FPM 条数必须与引擎步数对账（publisher 队列满会静默丢，流内不可检）。
7. **并发档 ≤64**：保证 decode 全落 cudagraph capture 带，结构性避开
   eager 带的时代漂移。
8. **传输纪律**：一律 exec-cat + 双端 sha256，禁 `kubectl cp`；取件用分块
   断点续传（>4h 会话先重登 tsh）。
9. **故障恢复**：引擎重启必须全家杀（含 `VLLM::` worker，漏杀残留 130+GiB
   显存）、显存 <1GB 轮询后再拉、stale 产物先清。
10. **语料预下载到 PVC**、aiperf 独立 uv venv——pod 内不出网、不污染镜像
    python 环境。

**产物**：`stream_{prefill,decode}.jsonl`、`windows_*.tsv`（双跑双份）、
resolved-config ×2 + diff、nodeName.txt、aiperf profile。

---

## D. 对齐打分（score-what-forms）

**干什么**：每条真值记录按**实际形成的坐标**入账（decode (b, Σkv)；
prefill (bp, Σtok, Σkv)），同坐标 ±20 滚动中位、一坐标一票，然后拿同坐标
问库要预测值，配对出误差。

**tep4 例子**（本机离线，worktree venv；agx_p1_tep4 首战原命令）：

```bash
cd <worktree>
FPM_REPLACE_FAKE_EXTRAP=1 .venv/bin/python fpm_verify/scoring/score_decode.py \
  --topo tep4 \
  --stream <outdir>/fpm_stream_decode.jsonl \
  --windows <outdir>/agx_windows_decode.tsv \
  --new-root /Users/yuanzhe/Desktop/codex_workspace/fpm/artifacts/r17-systems \
  --old-root aic-core/src/aiconfigurator_core/systems \
  --out scores_decode.csv

# prefill(chunk 口径,无 windows 参数,--stream 可给多段)
FPM_REPLACE_FAKE_EXTRAP=1 .venv/bin/python fpm_verify/scoring/score_prefill_chunks.py \
  --topo tep4 \
  --stream <outdir>/fpm_stream_prefill.jsonl \
  --new-root /Users/yuanzhe/Desktop/codex_workspace/fpm/artifacts/r17-systems \
  --old-root aic-core/src/aiconfigurator_core/systems \
  --out scores_prefill.csv
```

（两脚本共同默认：`--model-id MiniMaxAI/MiniMax-M2.7 --system h200_sxm
--backend vllm --backend-version 0.25.1`，换模型/机型时显式传。）

stdout 直接给三方汇总（old/new × MAPE/P95/MAX）；逐坐标明细在 CSV
（列:side,b,kv,truth,model,ape,stratum）。

**注意要点**：

1. **`FPM_REPLACE_FAKE_EXTRAP=1` 必须带**（见 B-1）；产品库加
   `native_identity`（dep 形才生效，tep2/tep4 不受影响但别删）。
2. **`--topo` 要和采收拓扑一致**（tep4 kit 就是 tep4）——它决定形参数表
   (tp/dp/moe_tp/moe_ep) 即查库身份。首战实测参考：decode overall
   MAPE 4.25% / P95 24.0%（new=R17），old 树内库 5.99%。
3. 内建过滤别关：纯相位过滤、掐头 4 步、<50 步池弃、配对弃行。D 流含少量
   条件分离本地 prefill 步（首战 8.4%）属正常，单列不判。
4. **解读尾部要小心**：MAPE 对借-还伪影免疫，但 MAX/P95 可能被单点劫持
   （如 b41 那个 82% 单点）。
5. **可选 op 级第三方对照**（脚本已收编 `scoring/score_op_level.py`；
   EVAL/ROOT 路径在脚本头改）：

   ```bash
   # 参数1=限量(0=全量) 参数2=输出 csv
   .venv/bin/python fpm_verify/scoring/score_op_level.py 0 op_pred_full.csv
   ```

   口径：decode `static_gen` 单步 / prefill `static_ctx` 单 chunk、等分假设、
   `engine_step_backend="python"`、算子库取现存最近版本（vLLM 只到 0.24.0）。
   已知结论：op 路线 decode 带 ~+5.3ms/步恒定固定开销（孤立 kernel 基准 vs
   CUDA graph 整步执行），b≤8 减去该常数后 2.0%——引用 op 数字必须带口径注。

---

## E. 审计、分层与报告

**tep4 例子**（`fetch_and_analyze.sh` 已自动串，手动重跑单件时）：

```bash
# 数据健康门:心跳/相位违例/counter 缺口
.venv/bin/python fpm_verify/report/smoke_audit.py \
  --stream-prefill <outdir>/fpm_stream_prefill.jsonl \
  --stream-decode  <outdir>/fpm_stream_decode.jsonl \
  --windows-prefill <outdir>/agx_windows_prefill.tsv \
  --windows-decode  <outdir>/agx_windows_decode.tsv

# 决策门:真值坐标 vs 库网格包络重合度(A/B/OUT 三档 + 散点图)
.venv/bin/python fpm_verify/report/envelope_overlap.py \
  --parquet /Users/yuanzhe/Desktop/codex_workspace/fpm/artifacts/r17-systems/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet \
  --stream-prefill <outdir>/fpm_stream_prefill.jsonl \
  --stream-decode  <outdir>/fpm_stream_decode.jsonl \
  --topo tep4 --out-prefix envelope_tep4

# 散点与汇总报告
.venv/bin/python fpm_verify/report/make_scatter.py \
  --scores scores_decode.csv --title "tep4 decode: new vs old" --out scatter_decode.png
.venv/bin/python fpm_verify/report/gen_report.py --manifest verify_manifest.json
```

看什么：audit 三件套先过（数据健康门）→ 包络重合度是**决策门**（首战
decode 100%/prefill 96.5% → 无需扩采直进正式；过低则先议网格扩采）→
噪声下限 = 同 seed 双跑坐标桶配对 spread（首战 c4/c16/c64 =
0.14/0.89/2.96%）。

**注意要点**：

1. 结论句式固定："预测误差 X% vs 噪声下限 Y%"——没有噪声下限的精度数字
   不能下判断。
2. 分层固定列：capture/eager 带、idle 后首步（agentic think-time 触发计时
   回退，单列不判）、in/out-hull、D 侧 onboarding 步、深 KV 带。
3. **绝不引用复合精度当数据健康度**（误差相消教训）；分布级
   median/P95/max 分层报告。
4. 判据绑定环境：同机 decode ≤2.0% / tep4 prefill ≤5.5%；跨机口径只作
   参考（跨机地板 4-6%，首战 e01sbam 属此类）；新口径首跑一律记档不判。
   tep2 4 卡变体没有正式判据基准——链路验证与低容量兜底。

---

## F. 固定 eval set 与离线重评

真值录一次后 `stream+windows` 即固定考卷。已有资产：
`fpm/artifacts/agx_p1_tep4/`（tep4 首战，decode 109,708 + prefill 4,335 坐标，
scores CSV、报告、op 对照 `op_pred_full.csv` 齐全）。

**tep4 例子**——库换代后重评（分钟级、零 GPU）：

```bash
FPM_REPLACE_FAKE_EXTRAP=1 .venv/bin/python fpm_verify/scoring/score_decode.py \
  --topo tep4 \
  --stream fpm/artifacts/agx_p1_tep4/fpm_stream_decode.jsonl \
  --windows fpm/artifacts/agx_p1_tep4/agx_windows_decode.tsv \
  --new-root <新一代库root> \
  --old-root aic-core/src/aiconfigurator_core/systems \
  --out scores_decode_R18.csv
```

修复 A/B = 同命令开关环境变量各跑一次（fake-KV 修复 22.02%↔4.25% 就是
这么定量的）。**唯一需要重录真值的情形：引擎/镜像换代**（真值世界变了）。

---

## 资产索引

| 路径 | 内容 |
|---|---|
| `fpm/artifacts/r17-systems/` | R17 六形库 root（被验对象） |
| `fpm/artifacts/agx_p1_tep4/` | 固定 eval set + 全部报告（含 kvwarm_vs_nokvwarm_compare.html） |
| `fpm/artifacts/aiperf-venv/`、`hf-cache/` | aiperf 环境 + AgentX 256k 语料 |
| `fpm_verify/harvest/tep4_1p1d/` | **tep4 正式 kit（本文档示例，8 卡 pod）** |
| `fpm_verify/harvest/tep2_1p1d/` | 4 卡兜底变体（链路验证） |
| `fpm_verify/scoring/`、`report/` | 打分器与报告器 |
| `AGENTIC_VERIFY_PLAN.md` / `VERIFY_RUNBOOK.md` | 方案全文 / 运行手册 |
