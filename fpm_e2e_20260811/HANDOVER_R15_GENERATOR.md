# Handover — generator(部署配置渲染)r15 轮相关发现与开发项

交给 generator dev。generator 本轮**零代码改动**;本书移交实证发现与
由此产生的开发项。**范围裁定(用户):k8s 规格渲染是 generator 本职——
采集 pod 模板(Guaranteed 资源块、PVC/镜像引用)由 generator 渲染,
collector 只消费模板并执行运行时流程。节点健康检查类内容不进模板
(环境健康属运维责任)。**与 serving
配置的 CPU 治理同一处实现、两处受益。按仓库规矩,generator/** 的改动须走
该模块规则(generator-development.md)。

## 1. k8s 部署规格必须带 CPU/内存 Guaranteed 配额(强烈建议,证据完备)

### 发现
vLLM/dynamo 的 MoE serving 在 **launch-bound 区间**(cudagraph 捕获外、
内核短的负载——MiniMax M2.7 tep4 为每步 700~3000 token 的 prefill)性能
由**主机发射速度**决定;而 pod 若不申请 CPU 配额(当前 generator 渲染的
k8s_deploy.yaml 与我们采集 pod 同病:只申请 GPU),进程线程安置与邻居
争抢由节点当时状态决定,该区间性能**每次启动抽签 ±8~18%,并随同节点
邻居负载恶化**(实测同节点同配置从 82ms 退化到 104ms)。

### 定罪与修复证据(全实测)
- 逐内核 trace 对撞:GPU 内核逐个相同(≤0.03ms),差额全在内核间隙;
  主机侧调用全线均匀 +5~17%;
- 修复 = requests==limits(cpu 64 / memory 256Gi / gpu 4,Guaranteed QoS):
  最吵节点上 4 boot 极差 **3.3%**(修复前同族节点 5.9~17.5%),重尾绝迹。

### 建议
1. generator 渲染的所有 k8s 部署配置默认带 CPU/内存 requests==limits
   (建议基线 **16 核/GPU、64Gi/GPU**,随 GPU 数缩放;可配置);
2. 文档标注:不带配额的部署,launch-bound 负载存在跨重启 ±8~18% 的
   性能抽签与邻居干扰暴露——这直接影响 SLA 与容量规划的可信度;
3. 若集群 kubelet 启用 static CPU manager,整数 CPU 的 Guaranteed pod
   可获独占核,效果最佳;未启用时配额仍提供 cgroup 权重保护(本轮实测
   即此情形,已足够)。

## 2. 机队健康信息(仅移交给运维,不进产品)

机队存在**单卡锁频节点**(实证:四卡 H200 之一被前租户锁在 1590MHz,
TP 锁步下计算主导负载被拖慢 2~4%,常规监控不可见——温度功率全正常)。
**用户裁定:此类环境健康保障属运维责任,不进 generator 模板/产品代码。**
运维自查脚本参考 `fpm_e2e_20260811/r15/clock_guard.sh`(轻载 boost 检测,
勿用满载探针——功耗墙会误伤健康卡)。

## 3. 仅供参考的口径信息

- dep4(attention-DP)prefill 已由产品裁定为范围外(真实部署 prefill 跑
  在 TP/EP worker 上);若 generator 的拓扑推荐涉及 dp 配置的 prefill
  角色,建议与该裁定对齐;
- dep4 小步在同步锁步服役下以 eager 制度为常态(组模式最小值同步,
  ×1.78 于捕获态),对 dp 部署的 TTFT 估算有含义,档案见
  `HANDOVER_R15_AIC_COLLECTOR.md` 与 r15 报告 §5。

## 4. 证据索引

F 臂终审 `/workspace/model_cache/fpm_dmx_F1..4`(PVC);trace 对撞
`fpm_e2e_20260811/r15/traces/boot{1,4}.json`;报告
`fpm_e2e_20260811/r15/R15_REPORT.{md,html}`;CPU 治理 pod 样例
`fpm_e2e_20260811/r15/k8s_f_pod.yaml`。
