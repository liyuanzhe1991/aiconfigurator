#!/usr/bin/env bash
# FPM 采集实验驱动器 — 覆盖 4 类 GPU × 3 档卡数 × 多模型。
#
# 用法:
#   scripts/experiments/fpm_collect.sh <h100|h200|gb200|b200> <4|8|16> <m27|glm-nvfp4|qwen32b> [选项]
#
# 选项:
#   --formal          正式采集(写数据库/parquet;默认 --smoke 冒烟档)
#   --limit N         限制 cell 数(默认 1)
#   --imex-workaround GB200 集群 IMEX 病态期间关闭 NVLS/cumem/symm(就地改 hardware.yaml,勿提交)
#   --dry-run         只打印将执行的命令
#
# 前置条件见同目录 README.md(Teleport 登录、PVC、密钥、模型缓存、venv)。
set -Eeuo pipefail

CLUSTER=${1:?用法: fpm_collect.sh <h100|h200|gb200|b200> <4|8|16> <m27|glm-nvfp4|qwen32b> [选项]}
GPUS=${2:?缺少卡数 (4|8|16)}
MODEL_KEY=${3:?缺少模型 (m27|glm-nvfp4|qwen32b)}
shift 3

MODE_FLAGS=(--smoke)
LIMIT=1
DRY_RUN=0
IMEX_WORKAROUND=0
while (( $# )); do
  case "$1" in
    --formal) MODE_FLAGS=() ;;
    --limit) LIMIT=$2; shift ;;
    --imex-workaround) IMEX_WORKAROUND=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "未知选项: $1" >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${FPM_PYTHON:-python3}
NAMESPACE=${FPM_NAMESPACE:-yuanli-aic}
IMAGE_REPO=nvcr.io/0980761089281446/dynamo-fpm-frozen

# ---------- 模型注册表 ----------
case "$MODEL_KEY" in
  m27)
    MODEL_PATH=MiniMaxAI/MiniMax-M2.7
    SNAPSHOT=models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b
    PRESET=tep ;;
  glm-nvfp4)
    MODEL_PATH=nvidia/GLM-5.2-NVFP4
    SNAPSHOT=models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
    PRESET=tep ;;
  qwen32b)
    MODEL_PATH=Qwen/Qwen3-32B
    SNAPSHOT=models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
    PRESET=tp ;;   # dense model -> dense TP family
  *) echo "未知模型: $MODEL_KEY" >&2; exit 2 ;;
esac

# ---------- 集群注册表(战役实测配置)----------
EXTRA_SETS=()
case "$CLUSTER" in
  h100)
    CTX=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02
    GPU_PROFILE=h100_sxm
    CACHE=shared-model-cache:/workspace/model_cache:$SNAPSHOT
    IMAGE=$IMAGE_REPO:gc-steady-16xfix-20260809
    ORCH=(--fpm-orchestrator grove --transport efa)
    # aws-dev-02 的 GFD 标签与 hardware.yaml 默认不同;KAI 未强制
    EXTRA_SETS+=("--generator-set" 'K8sConfig.worker_extra_pod_spec={"nodeSelector":{"nvidia.com/gpu.product":"NVIDIA-H100-80GB-HBM3"},"securityContext":{"runAsUser":0,"runAsGroup":0}}')
    ;;
  h200)
    CTX=nv-prd-dgxc.teleport.sh-dynamo-nebius-2
    GPU_PROFILE=h200_sxm
    CACHE=model-cache:/workspace/model_cache:$SNAPSHOT
    IMAGE=$IMAGE_REPO:gc-steady-16xfix-20260809
    ORCH=(--fpm-orchestrator grove --transport ib)
    EXTRA_SETS+=("--generator-set" 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}')
    EXTRA_SETS+=("--generator-set" 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0}}')
    ;;
  gb200)
    CTX=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01
    GPU_PROFILE=gb200
    CACHE=model-cache:/workspace/model_cache:$SNAPSHOT
    IMAGE=$IMAGE_REPO:gc-steady-arm64-schedonly-20260810   # ARM 专用;含双签名 scheduler 修复
    ORCH=(--fpm-orchestrator grove --transport nvlink)
    # 已知病节点黑名单(2026-08-10 IMEX 事故;集群修复后可移除)
    EXTRA_SETS+=("--generator-set" 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"default-queue"}')
    EXTRA_SETS+=("--generator-set" 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0},"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"NotIn","values":["ip-100-64-148-63.ec2.internal","ip-100-64-173-248.ec2.internal","ip-100-64-174-195.ec2.internal","ip-100-64-226-152.ec2.internal"]}]}]}}}}')
    ;;
  b200)
    CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
    GPU_PROFILE=b200_sxm
    CACHE=shared-model-cache:/workspace/model_cache:$SNAPSHOT
    IMAGE=$IMAGE_REPO:d719cca-gc-steady-20260729
    ORCH=()   # nscale 的 LWS 正常,默认编排即可
    EXTRA_SETS+=("--generator-set" 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}')
    # xmhbj/7wrxm 为已知脏 GPU 节点
    EXTRA_SETS+=("--generator-set" 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0},"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"NotIn","values":["cluster-0967a26d-pool-14bee067-prctr-xmhbj","cluster-0967a26d-pool-14bee067-prctr-7wrxm"]}]}]}}}}')
    ;;
  *) echo "未知集群: $CLUSTER" >&2; exit 2 ;;
esac

# ---------- 模型×集群合法性 ----------
if [[ "$MODEL_KEY" == glm-nvfp4 && "$CLUSTER" != b200 ]]; then
  echo "GLM-5.2-NVFP4 仅支持 B200(NVFP4=sm100 x86;GB200 arm 缺 FP4 kernel)" >&2; exit 2
fi

# ---------- GB200 IMEX 绕行(可选,就地改 facts,勿提交)----------
HW=$REPO_ROOT/src/aiconfigurator/generator/facts/hardware.yaml
if (( IMEX_WORKAROUND )) && [[ "$CLUSTER" == gb200 ]]; then
  sed -i.bak -e 's/NCCL_NVLS_ENABLE: "1"/NCCL_NVLS_ENABLE: "0"/' \
             -e 's/NCCL_CUMEM_ENABLE: "1"/NCCL_CUMEM_ENABLE: "0"/' \
             -e 's/VLLM_USE_NCCL_SYMM_MEM: "1"/VLLM_USE_NCCL_SYMM_MEM: "0"/' "$HW"
  echo "!! 已就地关闭 NVLS/cumem/symm($HW,备份 .bak)——实验后 git checkout 还原,勿提交" >&2
fi

# ---------- 组装命令 ----------
CMD=("$PYTHON" collector/collect.py
  --backend vllm --ops fpm_forward
  --model-path "$MODEL_PATH"
  --gpu "$GPU_PROFILE"
  --fpm-max-gpus "$GPUS" --fpm-parallel-presets "$PRESET" --fpm-tp-sizes "$GPUS"
  --namespace "$NAMESPACE"
  --model-cache "$CACHE"
  --image-pull-secret nvcr-push-secret
  --generator-set "K8sConfig.k8s_image=$IMAGE"
  ${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"} ${ORCH[@]+"${ORCH[@]}"} ${MODE_FLAGS[@]+"${MODE_FLAGS[@]}"})
# --limit is a smoke-only knob: formal collections must run their full plan
# (the CLI rejects the combination), and the pinned preset/tp keeps the
# formal plan to a single cell anyway.
(( ${#MODE_FLAGS[@]} )) && CMD+=(--limit "$LIMIT")
(( GPUS == 16 )) && CMD+=(--fpm-dp-sizes 1)   # 16 卡钉死 TEP16/DP1

echo "== cluster=$CLUSTER | gpus=$GPUS | model=$MODEL_KEY | ${MODE_FLAGS[*]:-formal} =="
if (( DRY_RUN )); then printf '%q ' "FPM_KUBECTL=kubectl --context=$CTX" "${CMD[@]}"; echo; exit 0; fi

cd "$REPO_ROOT"
export FPM_KUBECTL="kubectl --context=$CTX"
export PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT/aic-core/src${PYTHONPATH:+:$PYTHONPATH}
"${CMD[@]}"
RC=$?

# ---------- 收尾验证(集群清理铁律)----------
echo '== residue check =='
kubectl --context="$CTX" get pods -n "$NAMESPACE" 2>/dev/null | grep fpm- || echo 'zero residue OK'
exit $RC
