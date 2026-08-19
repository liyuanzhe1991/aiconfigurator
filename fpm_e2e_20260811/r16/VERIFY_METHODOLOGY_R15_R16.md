# FPM 库对齐验证(verify)方法论 — 面向外部读者的完整说明

> v2,2026-08-20 终审版(v1 写于判别实验前,结论已更新)。
> 所属 session:`d07354d7-72f0-47a1-aeab-be8fd346e942`。
> 配套材料:验收报告 `R16_FINAL_REPORT.md/.html`、判别实验逐步手册
> `experiments/EXPERIMENTS_STEPBYSTEP.md`、可执行套件 `fpm_verify/`。

## 0. 这份文档在讲什么

aiconfigurator 的 FPM(forward-pass model)库,是"GPU 上每个推理调度步
要花多少毫秒"的查询表:采集器在目标硬件上按网格逐点测量,建成 parquet;
上层用它预测部署延迟。**verify = 用一套独立于采集链的真实 serving 测量
("真值"),给库的预测精度打分。** 本文说明真值怎么生产、分数怎么算、
判据怎么定,以及每条规则背后的实验依据。

先定义贯穿全文的术语:

| 术语 | 含义 |
|---|---|
| 调度步(step) | 引擎调度器的一次迭代;decode 步同时为一批请求各生成 1 个 token |
| 坐标 | decode 步的物理身份 `(batch=并发请求数, total_kv=本步读取的 KV 总 token 数)`;prefill 步为 `(批数, 新 token 总数, 复用 KV 总数)` |
| wall | 一个调度步的墙钟耗时(ms),真值与库预测的比较对象 |
| kvwarm | 采集侧机制:用真实文本 prefill 出真内容 KV 链,decode 测点借链读取——替代早期"伪造块表(零上下文)"的 fake 测量 |
| fake_fallback | kvwarm 链在物理上放不下时(每批量档最深的"顶格点")回退到 fake 测量的行;实测值不可信(虚高 ×2-3.7 且跨会话不可复现) |
| MAPE | 平均绝对百分比误差;本文一切分数均为"一坐标一票"的 MAPE |

## 1. 真值是怎么生产的(fpm_verify/harvest)

### 1.1 遥测:唯一的数据源

引擎内的 InstrumentedScheduler(已烘入 frozen 镜像;serve 模式只透传
不改调度)把**每个调度步**发布到 ZMQ 端口(`DYN_FORWARDPASS_METRIC_PORT=20380`,
DP 拓扑每 rank 一口)。监听器([`fpm_listener.py`](../../fpm_verify/harvest/tep4/fpm_listener.py),dep4 为多端口版 [`dep4/fpm_listener.py`](../../fpm_verify/harvest/dep4/fpm_listener.py))逐行落 `fpm_stream.jsonl`。打分只用五个字段:
`dp_rank`、`wall_time`、`scheduled_requests.{num_prefill_requests,
sum_prefill_tokens, sum_prefill_kv_tokens, num_decode_requests}`。

**窗口 = 流的行号区间**:每个驱动动作前后各数一次流行数,`[s0,s1)` 写进
窗口表。打分器只认"窗口内的流"——这叫 **score-what-forms**:按调度器
实际形成的批打分,不按客户端计划打分。

### 1.2 decode 真值:固定并发池

固定并发 C、`ignore_eos` 持续生成 → decode 期形成稳定的"池即批";
kv 每步增长,一窗扫出一条 (C, kv) 射线;多窗多次重复 = 同坐标多票。

驱动为 [`decode_driver.py`](../../fpm_verify/harvest/tep4/decode_driver.py)(kit v2 默认,ShareGPT 真实文本奇数池;由 [`phase_decode.sh`](../../fpm_verify/harvest/tep4/phase_decode.sh) 调起,内容池实现见 [`sharegpt_ids.py`](../../fpm_verify/harvest/tep4/sharegpt_ids.py)):
- prompt 为 ShareGPT 奇数池 token id(与采集侧偶数池零重叠——考卷与
  练习册分离);实验依据:随机内容池会让真值系统性偏快 −1.51%
  (同机 ABA 三遍对拍,漂移对照 −0.14%,手册 E6);
- DP 拦路石:每 rank 先压一条 8192-token 大请求,让整池同一拍进场;
- isl==1 的浅池窗做锁步进场校验,通过则标 `lockstep`(打分器对锁步窗
  逐坐标直录,不平滑);
- 窗口表 v3 九列:`tag,C,isl,osl,rep,s0,s1,ok,mark`;
- 回退开关 `L3_DECODE_DRIVER=bench` 可切回旧 vllm-bench-random 路径。

> **版本注意(重要,防口径混淆)**:decode 真值驱动有两代——
> ①**R16 战役当时(08-19)用的是旧路径**:`vllm bench serve random`,
> 随机内容、**无拦路石、无进场校验**、v1 七列窗;**本报告终审记分牌的
> 全部 decode 数字(2.42/2.75、4.00/2.82、10.43/4.80 等)都是对这份
> 旧真值打的分**。复现这些数字必须 `L3_DECODE_DRIVER=bench`。
> ②**kit v2(08-20 起默认)**即上述 decode_driver.py(ShareGPT+DP
> 拦路石+锁步校验+v3 窗),依据内容 ABA 实验(E6,random 偏快 −1.51%);
> 语义链已验证,首次正式使用前需 GPU 冒烟。两代不可混打:同一批数字
> 必须出自同一代真值。
> (prefill burst 不受此代差影响——它从 R16 战役起就一直带拦路石与
> 现场校验,两代同款。)

引擎为 decode-parity 配置([`serve_run_tep4_decode.sh`](../../fpm_verify/harvest/tep4/serve_run_tep4_decode.sh)):与采集 cell 同镜像、同拓扑、同 kv dtype;
显式 `--no-enable-prefix-caching`(防 bench 请求前缀命中挤占物理 KV 池;
与采集侧 kvwarm 制度不冲突,r15 实测稳态等价)。

### 1.3 prefill 真值:burst

[`burst_driver.py`](../../fpm_verify/harvest/tep4/burst_driver.py)(dep4 的 DP 拦路石+镜像门版:[`dep4/burst_driver.py`](../../fpm_verify/harvest/dep4/burst_driver.py)):一格 = (bp 路并发 × 每路 n 个新 token × kv 共享前缀),
`max_tokens=1`,一次 burst 一窗;先预热前缀进 KV cache(prefill 引擎保持
prefix caching 开启);单步格现场校验"流里存在恰好 (bp, bp·n, bp·kv) 的步";
dep 拓扑加四 rank 镜像门(各 rank 同形且 wall 差 ≤3%,不达标换 seed 重试
——废票挡在采集时而非打分时)。

### 1.4 环境门禁(每条都由踩坑实验烙成)

按 [`stage_and_run.sh`](../../fpm_verify/harvest/stage_and_run.sh) 的执行序:

1. **同机协议(最重要)**:真值必须采在与采集**同一台物理机**上——
   `PIN_NODE=<采集时记下的节点名> bash stage_and_run.sh <topo> <ctx>`。
   依据:账面完全相同的健康节点之间,小坐标段测速天然差 4-6%(三节点
   同点单实测;CPU 噪声注入因果复现了该签名,手册 E3/E4)。跨机采的
   真值只作参考口径。实际节点名写入 `/results/nodeName.txt`;
2. **时钟守卫**([`clock_guard.sh`](../../fpm_verify/harvest/clock_guard.sh)):逐卡轻载烤机读 boost 峰值,<1900MHz 整链失败(曾当场
   抓获锁频病卡造成 +7.65% 假真值)。注意:守卫只能挡病卡,挡不住上述
   节点间离散——守卫通过 ≠ 环境同质;
3. **staging 完整性**:全部文件 exec-cat 注入 + 双端 sha256(kubectl cp
   禁用——曾静默截断);
4. **单 nohup 链**:守护进程(etcd/nats/frontend/监听器)必须活在链内,
   exec 会话断链即死;相位切换处有显存排空守卫(轮询 <1GB)与引擎家族
   清杀清单(含 `VLLM::` worker);
5. **取件**:[`fetch_results.sh`](../../fpm_verify/harvest/fetch_results.sh) 分块 + 逐块 sha + gzip + 断点续传
   (teleport 会话级流劣化会静默截断大文件,968MB 实战验证)。

## 2. 分数是怎么算的(fpm_verify/scoring)

### 2.1 decode([`score_decode.py`](../../fpm_verify/scoring/score_decode.py);r15 原版对照:[`score_t3_decode.py`](../kvwarm_patch/score_t3_decode.py))

真值重建(规则与 r15 时代原版打分器逐字节相同,已交叉校准):

1. 窗口内只取纯 decode 步(无 prefill 混入);
2. dep 弃窗:窗内各 rank 步数差 >1.25×+2 → 整窗弃;
3. dep 配速者归并:各 rank 第 k 步同拍,wall 取组内中位;坐标取按
   cudagraph 捕获桶排序的最大负载 rank(裸字典序在桶边界会选错,
   实测桶效应 +26%、eager 断崖 +146%);
4. 非锁步窗剔前 4 步(进场碎批);锁步窗全程保留;
5. 窗内同批量总步数 <50 → 整池弃(爬坡/收尾残段);
6. 沿 kv 轴 ±20 邻域滚动中位(邻域 <9 弃);锁步窗例外直录;
7. 同坐标跨窗全部票取中位 = 唯一真值——**一坐标一票**;
8. 每坐标问两库各一个预测,`|预测−真值|/真值` 入 MAPE;任一侧查不到
   两侧同弃(配对纪律,禁止单侧吃难点);
9. 分层:A=坐标恰在该库网格(纯插值)/B=网格外(泛化)/F=低于网格
   地板的钳位坐标(单列不进主口径)。

### 2.2 prefill([`score_prefill_burst.py`](../../fpm_verify/scoring/score_prefill_burst.py);补充口径 [`score_prefill_chunks.py`](../../fpm_verify/scoring/score_prefill_chunks.py))

坐标 = 步的 (bp, 新 token 总数, 复用 KV 总数),对准设计形;dep 按每
rank 形取票(凑满 DP 个 rank、wall 差 ≤3%,一窗一票取 max);
跨窗中位、配对同弃、A/B/C 分层(C=批数超出库网格上限的外插)。

### 2.3 查询身份

r15 时代的手工库把 dep 拓扑发布成 tp 形身份;产品库按原生拓扑发布
(dep4 = tp1/dp4/ep4)。打分器用 `native_identity` 开关对齐——身份口径
错配曾造成 40.8% 的假伤。

## 3. 判据:数字什么时候有资格判 PASS/FAIL

**核心结论(六案判别实验,手册 E1-E9):打分尺子与采集代码都无罪,
决定分数的隐藏变量是"库和真值是否同机"。**

- 同一个库对三份不同真值读 1.70% / 2.97% / 2.82%——差异 = 真值采收
  会话的节点 + 驱动内容(random −1.5%);
- 判据规则:**decode ≤2.0% 门槛只在"采集与真值同机"条件下可判**
  (r15 的 1.70% 正是同机同 boot 产物);跨机验证判据 ~5%
  (地板 ±3-4%),或改用多机中位;
- 库侧已知需排除项:`kv_seed_regime == fake_fallback` 的行(每批量档
  顶格点,值不可信)——SDK 装载器 `FPM_EXCLUDE_FAKE_FALLBACK=1` 排除(实现:[`fpm_forward.py`](../../aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py) 的 load_fpm_forward_data,行数见 git blame;collector 写列侧在 PR 树 f2a2b61b),
  实证 tep4 4.00→3.72%、尖刺带 26.5→0.2%(手册 E5/E9)。

## 4. 快速上手(字面命令)

```bash
# 1) 采集开跑时,自己记下 cell pod 的节点:
kubectl get pod <cell-pod> -n <ns> -o jsonpath='{.spec.nodeName}'

# 2) 真值收割(同机钉死;kit 自带守卫/staging/单链/相位序):
PIN_NODE=<上一步的节点> bash fpm_verify/harvest/stage_and_run.sh tep4 <kubectl-context>
#    等 /results/phase_all.log 出现 HARVEST-ALL-DONE

# 3) 取件(分块 sha + 断点续传):
bash fpm_verify/harvest/fetch_results.sh tep4 <kubectl-context> <本地目录>

# 4) 打分(排除已知不可信行;两库同真值配对):
FPM_EXCLUDE_FAKE_FALLBACK=1 PYTHONPATH=src:aic-core/src python \
  fpm_verify/scoring/score_decode.py --topo tep4 \
  --stream <本地目录>/fpm_stream.jsonl --windows <本地目录>/decode_windows.tsv \
  --new-root <被验库根> --old-root <基线库根> --out scores.csv
```

## 5. 关键数字与出处(全部可从 r16/scores/*.csv.gz 复算)

| 数字 | 含义 | 出处 |
|---|---|---|
| tep4 decode:全零时代库 9.99% → kvwarm 产品库 4.00%(排除后 3.72%)→ 手工库 2.82% | 三代采集制度同真值天梯 | R16_FINAL_REPORT §7.8 |
| tp4 decode:10.43% → 4.80% | kvwarm 修复(删 moe_tp skip)全网格实证 | §7.4,手册 E7/E8 |
| dep4 decode:2.42% / 2.75%(26.7 万坐标) | 终审真值,产品优于手工 | §7.6 |
| 节点间 4-6%、CPU 噪声注入复现签名 | 同机协议的依据 | §7.2,手册 E3/E4 |
| random 池 −1.51% | ShareGPT 驱动的依据 | 手册 E6 |

## 6. 已知边界(交接必读)

1. 跨机比较地板 ±3-4%:任何低于此的精度声明必须注明"同机"条件;
2. decode 真值单 boot(r15/R16 两代皆然);prefill 曾用 3-boot 中位;
3. kit v2 的 ShareGPT decode 驱动已按窗口/打分链验证语义,首次正式
   使用前建议做一次 GPU 冒烟(与 bench 路径同点对拍);
4. mixed 相位(混布批)窗口照采,但按用户裁定不进主判据;
5. 顶格坐标(每档最深 ~0.2% kv 区间)在排除 fake_fallback 后无库值,
   查询 fail-closed——这是"不假装知道没测过的区域"的既定语义。

## 附录:机制 → 实现文件总索引(相对本文档路径,GitHub 可点击)

| 机制/规则 | 实现 |
|---|---|
| 收割总编排(部署/守卫/staging/单链/相位序/PIN_NODE 同机) | [`fpm_verify/harvest/stage_and_run.sh`](../../fpm_verify/harvest/stage_and_run.sh) |
| 时钟守卫(1900MHz 轻载闸) | [`clock_guard.sh`](../../fpm_verify/harvest/clock_guard.sh) |
| 取件(分块 sha/gzip/断点续传) | [`fetch_results.sh`](../../fpm_verify/harvest/fetch_results.sh) |
| serve 栈(etcd/nats/frontend/监听器/prefill 引擎) | [`tep4/serve_stack.sh`](../../fpm_verify/harvest/tep4/serve_stack.sh) |
| decode 相编排(引擎切换/驱动选择) | [`tep4/phase_decode.sh`](../../fpm_verify/harvest/tep4/phase_decode.sh) / [`dep4/phase_decode.sh`](../../fpm_verify/harvest/dep4/phase_decode.sh) |
| decode 真值驱动 v2(ShareGPT/拦路石/锁步校验) | [`tep4/decode_driver.py`](../../fpm_verify/harvest/tep4/decode_driver.py) |
| prefill burst 驱动(拦路石/现场校验/镜像门) | [`tep4/burst_driver.py`](../../fpm_verify/harvest/tep4/burst_driver.py) / [`dep4/burst_driver.py`](../../fpm_verify/harvest/dep4/burst_driver.py) |
| prefill/mixed 相编排 | [`tep4/phase_prefill_entry.sh`](../../fpm_verify/harvest/tep4/phase_prefill_entry.sh) / [`tep4/phase_mixed.sh`](../../fpm_verify/harvest/tep4/phase_mixed.sh) |
| mixed 探针驱动 | [`tep4/mixed_driver.py`](../../fpm_verify/harvest/tep4/mixed_driver.py) |
| ShareGPT 奇/偶池与确定性取 id | [`tep4/sharegpt_ids.py`](../../fpm_verify/harvest/tep4/sharegpt_ids.py) |
| 遥测监听器(单口/多口) | [`tep4/fpm_listener.py`](../../fpm_verify/harvest/tep4/fpm_listener.py) / [`dep4/fpm_listener.py`](../../fpm_verify/harvest/dep4/fpm_listener.py) |
| decode-parity 引擎配置 | [`tep4/serve_run_tep4_decode.sh`](../../fpm_verify/harvest/tep4/serve_run_tep4_decode.sh) |
| prefill-parity 引擎配置(capture 表/8192 预算) | [`tep4/serve_run_tep4_prefill.sh`](../../fpm_verify/harvest/tep4/serve_run_tep4_prefill.sh) |
| pod 规格(4×H200/PVC 出件/病节点拉黑) | [`tep4/k8s_deploy.yaml`](../../fpm_verify/harvest/tep4/k8s_deploy.yaml) |
| decode 打分(五条过滤/配速者/A-B-F/配对) | [`fpm_verify/scoring/score_decode.py`](../../fpm_verify/scoring/score_decode.py) |
| prefill 打分(burst 口径) | [`score_prefill_burst.py`](../../fpm_verify/scoring/score_prefill_burst.py) |
| r15 原版打分器(交叉校准对照) | [`kvwarm_patch/score_t3_decode.py`](../kvwarm_patch/score_t3_decode.py) |
| SDK 排除 fake_fallback 行(C1) | [`aic-core .../fpm_forward.py`](../../aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py) |
| collector 写 kv_seed_regime 列(B1) | PR 树 f2a2b61b:collector/fpm_forward/{native_artifact,database}.py |
| 采集侧 kvwarm 机制(铺链/借表/倒带/巨点中位) | 引擎补丁源:[`kvwarm_patch/kvwarm_patch.py`](../kvwarm_patch/kvwarm_patch.py)(烘焙于 gc-timing/gc-warmtp 镜像) |
| 计划表(decode/prefill/mixed 网格) | [`tep4/decode_plan.csv`](../../fpm_verify/harvest/tep4/decode_plan.csv) 等,kit 目录内同名 |
| 判别实验 E1-E9 全部脚本与原始数据 | [`experiments/`](experiments/)(手册 [`EXPERIMENTS_STEPBYSTEP.md`](experiments/EXPERIMENTS_STEPBYSTEP.md)) |
