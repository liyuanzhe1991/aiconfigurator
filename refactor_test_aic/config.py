"""
统一配置模块
============
集中管理路径、模型参数、AIC SDK 配置以及各阶段输出子目录。
修改此文件即可切换到不同模型 / 不同数据目录。
"""

import os
from aiconfigurator.sdk.common import (
    CommQuantMode,
    DatabaseMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
# ============================================================================
# 数据目录（容器内路径；宿主机映射路径请自行调整）
# ============================================================================
# DATA_DIR = os.environ.get(
#     "AIC_DATA_DIR",
#     "/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_piecewise_cudagraph/",
# )

# DATA_DIR = os.environ.get(
#     "AIC_DATA_DIR",
#     "/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/",
# )

DATA_DIR = os.environ.get(
    "AIC_DATA_DIR",
    "/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_piecewise_cudagraph/",
)

# JSONL 文件名（由 hook.py 生成）
SCHEDULE_JSONL_FILENAME = "TP0-EP0_schedule_batch.jsonl"

# ============================================================================
# 各阶段输出子目录
# ============================================================================
SUBDIR_CSV = "csv"                  # 阶段 1 输出
SUBDIR_ESTIMATION = "estimation"    # 阶段 2 输出
SUBDIR_ACCURACY = "accuracy"              # 阶段 3: MAPE 统计输出
SUBDIR_SIGNED_ERROR = "signed_error"      # 阶段 3: 误差桶分析输出

# ============================================================================
# 模型 / 后端参数
# ============================================================================
MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507-FP8"
MODEL_PATH = f"/models/{MODEL_NAME}"
BACKEND_NAME = "sglang"

# AIC 性能数据库参数
AIC_SYSTEM = "h20_sxm"
AIC_BACKEND = "sglang"
AIC_VERSION = "0.5.9"

# ============================================================================
# ModelConfig 参数（对应 aiconfigurator.sdk.config.ModelConfig）
# ============================================================================
# MODEL_CONFIG_KWARGS = dict(
#     pp_size=1,
#     tp_size=8,
#     moe_tp_size=1,
#     moe_ep_size=8,
#     attention_dp_size=1,
#     moe_backend="deepep_moe",
#     enable_wideep=False,
#     workload_distribution="power_law_1.2",
# )

MODEL_CONFIG_KWARGS = dict(
    pp_size=1,
    tp_size=8,
    moe_tp_size=1,
    moe_ep_size=8,
    attention_dp_size=1,
    enable_wideep=False,
    # workload_distribution="power_law_1.01",
    workload_distribution="power_law_1.2",
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.fp8_block,
    kvcache_quant_mode=KVCacheQuantMode.float16,
    fmha_quant_mode=FMHAQuantMode.float16,
    comm_quant_mode=CommQuantMode.half,
)

# ============================================================================
# SGLang Server 启动命令（用于 nsys_profiler 自动解析为 ServerArgs）
# 直接粘贴你部署 sglang 时用的命令即可
# ============================================================================
SGLANG_LAUNCH_CMD = """
python3 /host/aiconfigurator/refactor_test_aic/hook_dataset_collector/sglang_launch_server.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --ep-size 8 \
    --cuda-graph-max-bs 256 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --max-running-requests 512
"""

# ============================================================================
# 估算校正系数（来自 refactor_test_aic/stage2_run_aic_estimation.py ）
# ============================================================================
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.0

# ============================================================================
# 辅助函数：获取各阶段的绝对输出目录
# ============================================================================

def get_output_dir(data_dir: str, subdir: str) -> str:
    """返回子目录的绝对路径，不存在则创建。"""
    path = os.path.join(data_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path
