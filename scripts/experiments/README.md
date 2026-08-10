# FPM 解耦栈 · 异构集群采集实验 Runbook

本分支(`fpm-decoupled-merged-20260810`)= 最新 `upstream/main` + 按序合并的三连 PR:

```
upstream/main
  └─ merge #1473 fpm-pr0-contract   (FPM 契约表 + 契约测试)
      └─ merge #1474 fpm-prg-generator (generator FPM 三件套渲染 + CUDA PATH 修复 + nvcc 自检)
          └─ merge #1475 fpm-prc-collector (collector 采集工作流 + follower 看门狗)
```

树与 `fpm-prc-collector` 栈顶逐字节一致;合并提交仅记录脉络。
`scripts/experiments/` 为实验附加件,**不随任何 PR 上游**。

---

## 一、环境准备(每台操作机一次)

1. **Teleport 登录 + 四集群上下文**(SSO 需人在浏览器旁;`tsh logout` 会清空 kubeconfig,登录后必须立刻恢复全部上下文):

   ```bash
   tsh login --proxy=nv-prd-dgxc.teleport.sh:443
   for C in dynamo-aws-dev-02 dynamo-aws-dev-01 dynamo-nebius-2 dynamo-nscale-dev-cluster; do tsh kube login $C; done
   ```

2. **Python 环境**:任一装齐 collector 依赖的 venv;通过 `FPM_PYTHON=/path/to/venv/bin/python` 传给脚本。

3. **Rust 原生扩展**:单测需要 `aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so`(构建产物,不入库)。本机有 cargo 则 `maturin develop`;否则从已构建的工作树拷贝。**采集本身不需要它。**

4. **每个目标集群的一次性基建**(namespace 默认 `yuanli-aic`,可用 `FPM_NAMESPACE` 覆盖):
   - imagePullSecret `nvcr-push-secret`(nvcr.io 私仓拉取);
   - 模型缓存 PVC + 预下载(见下表);下载 job 模板参考 hf_transfer + `HF_HUB_CACHE=/cache`。

## 二、集群/模型/卡数矩阵

| 集群 | 上下文简名 | PVC | 网络 | 编排 | 队列合规 | 镜像 |
|---|---|---|---|---|---|---|
| **h100**(aws-dev-02) | dynamo-aws-dev-02 | `shared-model-cache` | EFA | Grove | KAI 未强制 | `gc-steady-16xfix-20260809`(x86) |
| **h200**(nebius-2) | dynamo-nebius-2 | `model-cache` | IB | Grove(无 LWS) | `kai-scheduler` + queue=`dynamo` | 同上 |
| **gb200**(aws-dev-01) | dynamo-aws-dev-01 | `model-cache` | NVLink/MNNVL | Grove + ComputeDomain | `kai-scheduler` + queue=`default-queue` | `gc-steady-arm64-schedonly-20260810`(ARM) |
| **b200**(nscale) | dynamo-nscale-dev-cluster | `shared-model-cache` | NVLink | LWS(默认) | `kai-scheduler` + queue=`dynamo` | `d719cca-gc-steady-20260729`(x86) |

| 模型 key | HF 路径 | 适用集群 | 说明 |
|---|---|---|---|
| `m27` | MiniMaxAI/MiniMax-M2.7 | h100 / h200 / gb200 | FP8 块量化 MoE,222GB,主力多机模型 |
| `glm-nvfp4` | nvidia/GLM-5.2-NVFP4 | **仅 b200** | NVFP4=sm100;GB200 arm 基底缺 FP4 kernel |
| `qwen32b` | Qwen/Qwen3-32B | h200(已缓存) | 稠密对照组,允许 tp 预设 |

卡数:`4`(单机)/ `8`(h/b 单机;gb200 为 2 节点)/ `16`(全部为多机;脚本自动钉 TEP16/DP1)。

## 三、跑实验

```bash
# 冒烟(默认,--smoke --limit 1,不写正式库):
scripts/experiments/fpm_collect.sh h100 16 m27
scripts/experiments/fpm_collect.sh h200 8 m27
scripts/experiments/fpm_collect.sh gb200 8 m27 --imex-workaround
scripts/experiments/fpm_collect.sh b200 16 glm-nvfp4

# 正式采集(全网格采样,写数据库/parquet):
scripts/experiments/fpm_collect.sh h200 8 m27 --formal --limit 2

# 先看命令不执行:
scripts/experiments/fpm_collect.sh gb200 16 m27 --dry-run
```

结果判定:checkpoint `.collector_checkpoint/fpm_forward_smoke.json` 中 cell `status: passed`;
失败救捞产物在 `fpm_forward_artifacts/<run>/smoke/cells/<cell>/`(渲染件、run.sh、双 pod 引擎日志)。

## 四、已知坑(2026-08 战役实录)

1. **多机 PATH 事故(已修,防回归)**:EFA 传输块曾覆盖 PATH 丢掉 `/usr/local/cuda/bin`,饿死 deep_gemm 运行时 nvcc JIT,表象为 `DG_HOST_ASSERT(!cubin.empty())`。本分支已修 + run.sh 启动自检(缺 nvcc 秒退并报可读错误)。
2. **follower 空烧(已修)**:leader 完工/崩溃后 headless follower 曾无界等待烧满 4h exec 预算;本分支的看门狗会探测 leader etcd 消失并主动收尾。
3. **GB200 IMEX 病态(集群侧,2026-08-10 起)**:症状 = NVLS 组播绑定 `CUDA error 400` 或加载完成瞬间 worker 被静默杀。绕行 = `--imex-workaround`(就地关 NVLS/cumem/symm,**实验后 `git checkout -- src/.../hardware.yaml` 还原**)。集群修复后不要再用。
4. **坏节点黑名单**(脚本已内置,集群修复后可删):GB200 `ip-100-64-148-63` / `ip-100-64-173-248`;B200 `…-prctr-xmhbj` / `…-prctr-7wrxm`。
5. **KAI 队列执法**:nebius/nscale 绕过队列的 GPU pod 会被静默回收(表象 137/pod 消失)——脚本已带合规标签,别删。
6. **容量排队**:16 卡需整机 ×2(h/b)或 ×4(gb200);等位失败表象为 `timed out waiting for N FPM pods`,重试即可(可参考会话中的 watcher 循环模式:凭证护栏 + flock 防重 + 间隔重试)。
7. **凭证过期**:Teleport ~7h;跨夜实验前先续期,watcher 类脚本务必带"余量 < 阈值即自停"护栏。

## 五、清理铁律

**任何集群任何实验后必须零残留。**collector 自带 verified teardown;异常中断后手动兜底:

```bash
kubectl --context=<ctx> get pods,podcliquesets,computedomains -n yuanli-aic
kubectl --context=<ctx> delete podcliqueset <cell>-agg -n yuanli-aic   # Grove
kubectl --context=<ctx> delete computedomain <cell>-agg-compute-domain -n yuanli-aic  # GB200
```

PVC/密钥属可复用基建,战役整体结束时请示 owner 再处置。
