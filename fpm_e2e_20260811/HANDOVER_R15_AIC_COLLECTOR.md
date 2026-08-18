# Handover — aic-fpm collector(采集编排/点单/建库发布)r14-r15 轮改动全记录

交给 aiconfigurator collector dev。范围:引擎台架**之外**的一切采集侧改动
(台架见《HANDOVER_R15_FPM_SELFBENCHMARK.md》)。落点为 collector/** 的
#1473/#1474/#1475 PR 链(v6 schema 写方在 #1475)。全部改动零采集耗时增加
(用户冻结令),唯一例外(带内补测)已被根因修复取代、标记删除。

## 1. 点单/网格(归属更正:主体在 dynamo-fpm)

网格生成器活在引擎台架内(benchmark config 的 *_samples 参数族),
**边界封顶(kv ≤ b×(max_model_len−8),上限读模型 config.json)与幂阶梯
延伸(相邻挡位比 ≤2)应修在引擎的网格发生器**——见
《HANDOVER_R15_FPM_SELFBENCHMARK.md》§1.6。
collector 侧职责(本书范围):走 `DYN_BENCHMARK_POINTS_FILE` 覆盖路径时,
对外部点单做同一套封顶校验(双保险,共享校验逻辑,勿抄两份)。

## 2. 采集运行环境(pod 规格)——本轮最大发现

### 2.1 CPU 治理(根因修复,已终审)
- **病**:采集 pod 只申请 GPU、零 CPU 配额(BestEffort 档)→ 进程线程安置
  与邻居争抢由 OS 随机决定且进程期内锁定 → **launch-bound 区间**(cudagraph
  外、内核短的 700-3000 token 带)主机发射速度每 boot 抽签,带内测量弥散
  ±8~18%、重尾 +17%,且随节点上邻居噪声恶化(实测同节点从 82 退化到 104)。
- **定罪链**:硬件排除(采样器逐拍存证)→ 内容排除(固定种子)→ 通信排除
  (NCCL 焊死 B 臂、IPC 整体禁用 D 臂,弥散原样)→ 逐内核 trace 对撞:
  **GPU 内核逐个 ≤0.03ms 相同,差额全在内核间隙;主机侧全线均匀 +5~17%**。
- **修**:pod 规格 requests==limits(cpu 64、memory 256Gi、gpu 4)→
  Guaranteed QoS。**终审(F 臂)**:最吵的病机节点上 4 boot 带内极差
  **3.3%**(内含 ±1.5% 测量噪声),重尾绝迹;带外 0.2~1.2%。
- **落点**:采集 pod 模板(k8s spec);同时通报 generator(见其 handover)。

### 2.2 时钟守卫(运维附录——用户裁定不进产品)
- **病**:机队存在单卡锁频节点(实证:GPU0 锁 1590MHz,其余 1980;计算带
  被抬 2~4%,换干净节点后 (b2,tot8192) 192.9→184.4,对真值 +1.0%)。
- **修**:采集前逐卡 6 秒**轻负载 boost 检测**(峰值 ≥1900MHz 才准采)。
  **勿用满载探针**——它是功耗炸弹,健康四卡会均匀降到 ~1700 造成误报(v1
  教训)。脚本 `r15/clock_guard.sh`。
- 换节点手段:pod 反亲和性 NotIn 病节点(kai 会摇回同节点)。

### 2.3 GPU 状态采样器(运维附录——用户裁定不进产品)
5 秒粒度 csv(温度/时钟/功率/事件);本轮病卡与 CPU 定罪均靠它。
脚本 `r15/gpu_sampler.sh` 留档,运维/排障需要时自挂,不进命令面。

## 3. 采集协议

### 3.1 带内 3-boot 中位 —— **已被 2.1 取代,标记删除**
CPU 治理坐实前的统计对冲(带内 512<tot<4096 补测 ×2 boot 逐点中位,
+10~12 分钟/拓扑)。曾实证救回单点 +17% 错(97.4→83.4)。**下次全量采集
以 Guaranteed pod 跑通后正式移除**;merge_r15_boots.py 保留作历史工具。

### 3.2 内容源(归属澄清:机制在引擎,collector 只透传)
偶数池筛选与内容分派全在引擎(kvwarm loader / content dispatcher)。
collector 侧职责:①配置透传(`DYN_BENCH_PREFILL_CONTENT`、数据集路径);
②数据集分发到 PVC(带 sha256 校验);③文档化分池语义(采集偶数池/
真值奇数池,防考题污染),不实现机制。

## 4. 建库/发布链

- 输入拼装:`combine_prefill_decode.py`(prefill 与 decode 可来自不同采集
  轮,按 point 全字段配对);`merge_r15_boots.py`(跨 boot 逐点中位,派生
  工件、原件不动)。配对键**只用物理坐标**(point_type,b,tot,kv,dp_rank)——
  benchmark_id 等台架自编字段随点单变,入键必配空(实错教训)。
- dep4:逐坐标 4-rank 中位(×4 rank 文件)。
- 落点:v6 发布链(#1475)。

## 5. 部署/传输纪律(工具链)

- **kubectl cp 禁用**:teleport 通道上静默丢文件/截断且报成功(单日三次
  实锤:31 文件丢 5、60MB JSON 截断)。一律 `exec cat` 单流 + 两端 sha256
  校验循环(`stage_r15.sh` 模式)。
- pkill 自杀陷阱:清场与发射永不同一条命令行(pkill -f 按 cmdline 匹配,
  发射行的进程名明文会被自家 pkill 命中)。
- 陈旧产物陷阱:waitfile 类闸门前必须清空/改名旧产物(单日两次实锤)。

## 6. 已立规格、待实现

**run manifest(编排级机器可读耗时清单)**:每次采集运行落一份 JSON,
含 pod 调度耗时、布置耗时、引擎拉起耗时(launch→bench 首点)、每 boot
分段墙钟、时钟守卫与采样器产物路径——与台架侧 per-point timings(见
selfbenchmark handover §3.1)合成完整耗时地图,服务提速战役(冻结令解除后
启动;当前实测:tep4 全量 102.6 分钟其中 kvwarm 预热约 83 分钟,dep4 75.1
分钟;prefill 单独段 24/26 分钟)。

## 7. 证据索引

F 臂判决 `/workspace/model_cache/fpm_dmx_F1..4`(PVC);病卡对照
`r15/tep4_clean/`(干净)vs `sicknode_*`(PVC 归档);走廊与矩阵数据
`r14/p3_*`、`fpm_dmx_*`;报告 `r15/R15_REPORT.md`。
