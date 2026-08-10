# FPM 采集实验手册(step by step)

目标:**在 H100 / H200 / GB200 / B200 上,用一条命令跑 FPM 自基准采集。**
本分支 = 最新 upstream/main + 三个 FPM PR(#1473/#1474/#1475)的合并结果。

跟着下面 7 步走,每步都有"敲什么"和"应该看到什么"。

---

## Step 1:登录 Teleport(每 ~7 小时一次)

**敲:**

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh:443
for C in dynamo-aws-dev-02 dynamo-aws-dev-01 dynamo-nebius-2 dynamo-nscale-dev-cluster; do tsh kube login $C; done
tsh status | grep 'Valid until'
```

**应该看到:**浏览器弹出 SSO 完成登录;最后一行显示有效期(约 7 小时)。

> 注意:`tsh logout` 或过期会清空所有 kube 上下文,重新登录后必须重跑上面的 for 循环。

---

## Step 2:指定 Python 环境(每个终端一次)

**敲:**

```bash
export FPM_PYTHON=/path/to/venv/bin/python    # 换成装齐 collector 依赖的 venv
$FPM_PYTHON -c "import kubernetes, yaml; print('venv OK')"
```

**应该看到:**`venv OK`。

---

## Step 3:确认集群基建(每个集群只需第一次)

**敲**(以 h200 为例,context 名见附录 A):

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nebius-2
kubectl --context=$CTX get secret nvcr-push-secret -n yuanli-aic
kubectl --context=$CTX get pvc -n yuanli-aic
```

**应该看到:**secret 存在;PVC 列表里有该集群的模型缓存盘(h100/b200 叫 `shared-model-cache`,h200/gb200 叫 `model-cache`)。

**如果缺**:secret 需要复制凭证创建;模型没下载过则起一个下载 Job(`python:3.12-slim` + `pip install hf_transfer` + `HF_HUB_CACHE=/cache` 指向 PVC)。这两步做过一次就永久有效。

---

## Step 4:跑实验

**先 dry-run 看命令**(不真跑):

```bash
scripts/experiments/fpm_collect.sh h200 8 m27 --dry-run
```

**确认无误后去掉 --dry-run 真跑:**

```bash
scripts/experiments/fpm_collect.sh h200 8 m27
```

**应该看到:**开头打印 `== 集群 h200 | 8卡 | m27 | --smoke ==`,随后是渲染日志、pod 建立、引擎启动;整个过程 20-40 分钟(16 卡多机首跑更久)。

### 所有合法组合(直接复制)

```bash
scripts/experiments/fpm_collect.sh h100  4  m27
scripts/experiments/fpm_collect.sh h100  8  m27
scripts/experiments/fpm_collect.sh h100  16 m27                      # 2 台机器
scripts/experiments/fpm_collect.sh h200  4  m27
scripts/experiments/fpm_collect.sh h200  8  m27
scripts/experiments/fpm_collect.sh h200  16 m27                      # 2 台机器
scripts/experiments/fpm_collect.sh gb200 8  m27 --imex-workaround    # ARM;2 台机器
scripts/experiments/fpm_collect.sh gb200 16 m27 --imex-workaround    # ARM;4 台机器
scripts/experiments/fpm_collect.sh b200  8  glm-nvfp4
scripts/experiments/fpm_collect.sh b200  16 glm-nvfp4                # 2 台机器
```

规则(脚本会自动拦错误组合):

- `glm-nvfp4` 只能跑 b200;`m27` 跑 h100/h200/gb200;
- `--imex-workaround` 只在 GB200 集群 NVLink 故障期(2026-08)需要,集群修好后去掉;
- 默认是**冒烟档**(1 个 cell、4 个基准点、不写正式库);加 `--formal` 变**正式采集**(全网格、写 parquet)。永远先冒烟后正式。

---

## Step 5:盯进度(另开一个终端,可选)

**敲**(context 换成对应集群):

```bash
watch -n 30 "kubectl --context=$CTX get pods -n yuanli-aic | grep fpm-"
```

**应该看到:**pod 从 `Pending` → `Running`;多机时 leader/follower 各一个 pod;结束后全部消失。

---

## Step 6:看结果

**敲:**

```bash
grep '"status"' .collector_checkpoint/fpm_forward_smoke.json
```

**应该看到:**`"status": "passed"` ← 成功收工。

其他产物位置:

| 想看什么 | 路径 |
|---|---|
| 失败时的引擎日志 | `fpm_forward_artifacts/<run>/smoke/cells/<cell>/raw/<pod>/engine.std{out,err}.log` |
| 渲染的 K8s 配置/启动脚本 | 同目录 `k8s_deploy.yaml`、`run.sh` |
| 采集汇总(错误分类) | `all_<时间戳>/collection_summary_vllm.json` |

---

## Step 7:确认清理(铁律:零残留)

脚本正常结束会自动打印 `零残留 ✓`。**异常中断**(Ctrl-C、断网、超时)后手动兜底:

```bash
kubectl --context=$CTX get pods,podcliquesets,computedomains -n yuanli-aic
# 有残留就按名字删:
kubectl --context=$CTX delete podcliqueset <名字> -n yuanli-aic
kubectl --context=$CTX delete computedomain <名字> -n yuanli-aic    # 仅 GB200 有这种资源
```

用过 `--imex-workaround` 的话,再还原被就地修改的配置:

```bash
git checkout -- src/aiconfigurator/generator/facts/hardware.yaml
```

---

## 出错了?对照这张表

| 症状 | 原因 | 怎么办 |
|---|---|---|
| `timed out waiting for N FPM pods` | 集群没空位(16 卡要 2-4 台整机同时空闲) | 纯排队;30 分钟后重跑 Step 4 |
| pod 长期 Pending | 同上(群调度凑不齐整机) | 同上 |
| `NCCL error: unhandled cuda error`(GB200) | 集群 NVLink/IMEX 故障 | 确认带了 `--imex-workaround`;仍崩=抽中坏节点,重跑 |
| 引擎跑几分钟后 `!cubin.empty()` 断言 | 旧代码的 PATH bug | 确认你在本分支(已修复) |
| pod 无故消失/退出码 137 | 绕过 KAI 队列被回收 | 用本脚本跑(已带合规标签),别手改调度参数 |
| `unrecognized arguments: --prefill-...` | 镜像里 dynamo 版本不带 FPM 参数 | 别改脚本里的镜像 tag(已锁定实测可用版本) |

---

## 附录 A:速查表

| 集群 | kubectl context | 架构 | PVC | 16 卡 = 几台机器 |
|---|---|---|---|---|
| h100 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02` | x86 | shared-model-cache | 2 |
| h200 | `nv-prd-dgxc.teleport.sh-dynamo-nebius-2` | x86 | model-cache | 2 |
| gb200 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01` | **ARM** | model-cache | 4(每台 4 卡) |
| b200 | `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster` | x86 | shared-model-cache | 2 |

| 模型 key | 实际模型 | 大小 |
|---|---|---|
| m27 | MiniMaxAI/MiniMax-M2.7(FP8 MoE) | 222GB |
| glm-nvfp4 | nvidia/GLM-5.2-NVFP4 | sm100 专用 |
| qwen32b | Qwen/Qwen3-32B(稠密) | 需先下载到目标集群 |

脚本内置(不用你操心):各集群镜像选择、KAI 队列合规标签、已知坏节点黑名单、16 卡 TEP16/DP1 钉死、结束后残留检查。
