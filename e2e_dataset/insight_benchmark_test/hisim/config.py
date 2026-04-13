from pathlib import Path

cur_dir = Path(__file__).parent
DEFAULT_OUTPUT_DIR = cur_dir / "output"


def get_schedule_config_from_sglang_server_info(
    server_info: dict,
    dtype: str | None = None,
    kv_dtype: str | None = None,
    backend_version: str | None = None,
) -> dict:
    if dtype is None:
        if (
            server_info["quantization"] is not None
            or "FP8" in server_info["model_path"].upper()
        ):
            dtype = "FP8"  # FIXME: the value of quantization might be fp8, fp8_m5e3...
        else:
            dtype = "FP16"

    if kv_dtype is None:
        if server_info["kv_cache_dtype"] in ["auto", "bfloat16"]:
            kv_dtype = "FP16"
        else:
            kv_dtype = "FP8"

    if backend_version is None:
        backend_version = server_info["version"]

    return {
        "tp_size": server_info["tp_size"],
        "ep_size": server_info["ep_size"],
        "dp_size": server_info["dp_size"],
        "data_type": dtype,
        "kv_cache_data_type": kv_dtype,
        "backend_name": "sglang",
        "backend_version": backend_version,
    }


def get_prefictor_config(
    predictor_name: str,
    db_path: str,
    extra_config: dict = {},
    device_name: str = "h20_sxm",
) -> dict:
    if predictor_name == "aiconfigurator":
        return {
            "name": "aiconfigurator",
            "database_path": db_path,
            "device_name": device_name,
            "prefill_scale_factor": extra_config.get("prefill_scale_factor", 1),
            "decode_scale_factor": extra_config.get("decode_scale_factor", 1),
            "xgb_model_path": extra_config.get("xgb_model_path"),
            "workload_distribution": extra_config.get("workload_distribution", "balanced"),
        }
    elif predictor_name == "step_benchmark":
        return {
            "name": "step_benchmark",
            "database_path": db_path,
        }
    elif predictor_name == "mixture":
        return {
            "name": "mixture",
            "aic_database_path": extra_config.get("aic_database_path"),
            "step_database_path": extra_config.get("step_database_path"),
            "device_name": device_name,
        }
    raise ValueError("Unknow predictor")


def get_platform_config(
    device_name: str,
    disk_read_bandwidth_gb: float = 4,
    disk_write_bandwidth_gb: float = 4,
) -> dict:
    return {
        "accelerator": {"name": device_name},
        "disk_read_bandwidth_gb": disk_read_bandwidth_gb,
        "disk_write_bandwidth_gb": disk_write_bandwidth_gb,
        "memory_read_bandwidth_gb": 64,
        "memory_write_bandwidth_gb": 64,
        "num_device_per_node": 8,
    }
