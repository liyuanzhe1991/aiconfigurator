# Handover — fpm-self-benchmark(dynamo FPM 台架)r14-r15 轮改动全记录

交给 dynamo-fpm dev。承接上一轮《HANDOVER_DYNAMO_FPM.md》(randtok2:全零 token
→ salt 随机 token)。本轮把台架合成输入的**最后两处不真实**修掉,并烘焙成新镜像。
所有数字来自 fpm-all-20260811 战役 r14-r15 段实测(证据路径见文末)。

## 0. 一句话

台架捏出来的请求,decode 侧 KV 缓存里是垃圾内容、prefill 侧 prompt 是随机
token——MoE 模型对两者都路由敏感,采集数据带系统性偏差;修复后 decode
MAPE 7.66/7.39% → **1.70/1.61%**(tep4/dep4),prefill 18.17% → **4.76%**
(tep4,A 层 2.95%)。全部修复已烘焙:
**`nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-realcontent-20260818`**
(digest `sha256:981ab853…12bc`,层 tar sha256 `3ac0025e…9898`)。

## 1. 修复清单(引擎侧,全部在 instrumented_scheduler.py)

### 1.1 kvwarm:decode 测量前用真实文本预热 KV(11 处手术)

- **病**:decode 注入的假 KV 块内容不真实,注意力读到退化数据,
  on/off 对照实测 +2~6% 系统性偏差(方向随形状变)。
- **修**:测量前按 ShareGPT(偶数池,按对话 sha256 奇偶分池)真实文本跑
  prefill,把真 KV 写进缓存,再借块注入给被测请求。要件:进程内 tokenizer、
  确定性拼链(seed=网格digest:dp_rank:链号)、砍宽续深阶梯(重用已预热链)、
  倒带测量、dense/moe_tp 无预热收益自动跳过。
- **证**:T0 验收 off 零 diff(旧路径逐字节不变)+ warmed 验收;终版 decode 分数。

### 1.2 块表按需截断(巨 KV 点可真实测量)

- **病**:按请求名义长度建块表/占块,巨 KV 点(百万 token)超缓存池 →
  假回退,深段失真。
- **修**:只建实际触达的块:`need = ceil((ctx+1+max(2,巨点拍数))/block_size)+1`。
- **证**:kv≥2M 深段从假回退变真值;r13 深段毒化带(最差 56.8%)三件套之一。

### 1.3 巨点(kv≥1e6)引擎内 3 拍取中位

单拍巨点噪声大;引擎内连续 3 拍稳态取中位,repeats 如实写进产物。

### 1.4 prefill 真实内容注入 v4(本轮新增,最重要)

- **病**:prefill prompt 用 `Random(salt).choices(range(1,199000))` 随机 token,
  MoE 路由均匀化 → 全专家激活 → 访存带(每步 700-3000 token)采集值比真实
  流量**偏慢 5~12%**(同 boot A/B 实测);随机对随机打分互相抵消,偏差蛰伏。
- **修**:`_bench_prefill_content_ids` 分派器,env `DYN_BENCH_PREFILL_CONTENT`:
  - `sharegpt`:**扁平池窗口**——偶数池对话按固定种子洗牌拼接 ~240 万 token
    一次性分词(内存 ~20MB,构建 1-2 分钟),每请求按种子取确定性连续窗口;
    窗口起点只由种子决定(与长度无关,越界回绕)→ 同 salt 任意两长度严格
    前缀一致;
  - `sharegpt_chain`:真值同款逐请求拼链(诊断用);
  - 未设:原随机路径逐字节不变。
- **两个生成位必须同改**(v2 只改一处的血泪):被测请求(k=prompt_len)与
  kv 前缀播种请求(k=prefix_tokens)靠同 salt 保证前 kv token 逐位相同,
  只改一处 → `fake_prefix_cache_validation_failed` 全线崩。
- **证**:带内签名偏差 +5.3%→+0.6%;三臂对撞池≡链(±1%);池化 vs 拼链
  取证统计相同(唯一 token 率 0.234/0.229)。

### 1.5 网格发生器:边界封顶 + 幂阶梯(归属更正,自 collector 书移入)

- **病**:台架网格发生器产出越过 max_model_len 的点(上限须读模型
  config.json,禁 hardcode),引擎拒收或假回退;r13 深段毒化带(最差
  56.8%)成因之一;深段档位断档使插值失控。
- **修**:逐点封顶 `kv ≤ b×(max_model_len−8)`(prefill 同理);深段按
  相邻挡位比 ≤2 的幂阶梯补齐。战役期以 `DYN_BENCHMARK_POINTS_FILE`
  外部点单实现(r14/make_points_r14.py 为参考实现),产品化应内建于
  网格发生器本体;外部点单覆盖路径保留并复用同一校验。

### 1.6 args.py dump 修复(存证)

烘焙时错挂的装饰器使 resolved-config 转储全线失败;函数搬位修复。只影响存证。

## 2. 镜像谱系与烘焙记录

| tag | 内容 | 状态 |
|---|---|---|
| gc-steady-randtok2-20260812 | 随机 token 修复(上一轮) | 曾任基底 |
| gc-realcontent-20260818 | + kvwarm 11 手术 + 真实内容 v4 + args dump 修复 + 原件备份(instrumented_scheduler_pristine.py) | rev2 基底 |
| **gc-timing-20260818** | + timing_patch(相位耗时仪表,timing.phases;层=调度器单文件,sha256 ed383f95…dbaf) | **现役**(digest adcd28a9…8c48) |

烘焙法:in-cluster crane pod(`crane append -f layer.tar -b <基底> -t <tag>`,
nvcr-push-secret 只挂 pod 内,凭证不落本机)。验证:全新拉取 → digest 比对 +
标记 grep(kvwarm×9 / v4×6 / timing×1 / 备份×1 / argsdump 正则)+ compile 通过。
**B200 适用**:层为纯 Python(架构无关);基底 torch arch list 含 sm100,
GLM5.2×B200 冒烟已实证(kvwarm 零改动工作、4/4 real-kv)。

### 2.1 timing_patch(rev2 增量,r15/timing_patch.py)

设计红线 = 测量步循环零包裹(不碰 _kvwarm_step_busy/monitor/每步路径),只包
每点/每 stage/一次性调用。benchmark JSON `timing` 块新增 additive 子对象
`phases`(schema_version 仍 2,下游未知键容忍):
`dataset_load_s / tokenizer_load_s / kvwarm_plan_s / content_tokenize_s(+
_buckets 按 kv 深度 4 的幂分桶)/ content_gen_s / seed_total_s /
import_to_first_bench_activity_s`,每键附 `*_n`(调用数)与 `*_first_s`
(首调用拆分——content_gen 首调用≈内容池一次性构建)。
**相位嵌套树(消费方勿平铺求和)**:content_gen ⊃ 池构建 ⊃ dataset_load +
tokenizer_load;kvwarm.stages[].build_seconds ⊃ content_tokenize。
既有字段继续负责:kvwarm GPU 预热 = `kvwarm.stages[].build_seconds` 逐段和;
真正 inference = `timing.measured_iteration_seconds`。四段拼法(collector 侧):
engine launch = 进程拉起→started_at(外测);kvwarm = stages 和;inference =
measured;other = elapsed − 顶层互斥相位 − measured。
零开销验证(2026-08-18,0477z Guaranteed pod,5 boot):带 timing 的两 boot
在全部 10 坐标落入基线三 boot 包络并居快侧(measured 0.6976/0.6970 vs 基线
0.6893~0.7205);相位守恒余量 +0.87s;基线臂带内瞬态 +13.7% 系共享节点邻居
(内存带宽,cgroup 不防),再证正式采集须保留带内 3-boot 中位。
工装:r15/{timingchk_run.sh,timingchk_seq.sh,timingchk_score.py,
points_timingchk.json,bake_rev2.sh,k8s_timingchk_pod.yaml}。

## 3. 已立规格、待实现(不在本镜像)

1. ~~逐点细分耗时仪表~~ **已落地为 rev2 timing_patch(见 §2.1)**,与原规格的
   差异:不做逐点 timings 数组(逐点 wall 既有),相位粒度按调用位而非逐点。
2. **每步 cudagraph 模式字段**(每 rank 提议值 + 组同步值)——dep4 双制度
   档案的遥测债。
3. **每步专家路由直方图**——内容效应机制债。
4. prof_patch.py(诊断工具,已有):env 门控 torch profiler 包裹
   execute_model,导出逐内核 chrome trace——本轮 launch-bound 定罪的仪器,
   建议收编为正式诊断开关。

## 4. 证据索引

r15 报告 `fpm_e2e_20260811/r15/R15_REPORT.{md,html}`;分数
`r15/tep4_clean/scores_*`、`r14/*/scores_*_decode_r14.csv`;三臂/矩阵原始件
`/workspace/model_cache/fpm_dmx_*`(PVC);取证脚本 `r15/content_forensics.py`;
补丁源 `kvwarm_patch/kvwarm_patch.py`、`r15/prefill_content_patch_v2.py`、
`r13/fix_args_dump.py`。
