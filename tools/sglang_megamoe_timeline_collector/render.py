#!/usr/bin/env python3
"""Render a reproducible native SGLang PD MegaMoE timeline collection run."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
TEMPLATE_DIR = PROJECT_DIR / "templates"


class LiteralDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.Dumper, data: str) -> yaml.nodes.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralDumper.add_representer(str, _represent_str)


DEFAULT_CONFIG: dict[str, Any] = {
    "run_id": None,
    "namespace": "default",
    "name": "gb200_native_sglang_megamoe_scheduler_nvtx",
    "app_prefix": "aic-native-sglang-megamoe-sched-nvtx",
    "base_prefix": "aic-sgl-native",
    "artifact_root": str(REPO_ROOT / "artifacts"),
    "remote_root_base": "/models/aic_native_sglang_megamoe_scheduler_nvtx",
    "pvc_name": "shared-model-cache",
    "images": {
        "sglang": "lmsysorg/sglang:deepseek-v4-grace-blackwell",
        "nsys": "nvcr.io/nvidia/pytorch:25.04-py3",
        "aiperf": "python:3.12-slim",
        "copy": "python:3.12-slim",
        "image_pull_secret": "nvcrimagepullsecret",
        "image_pull_secret_source_namespace": "kprashanth",
    },
    "cluster": {
        "gb200_nodepool": "customer-gpu-w0e",
        "gb200_clique": "60efb90c-3713-4488-abf7-0da90d2eaa64.2",
        "cpu_node_instance_type": "n2d-standard-8",
    },
    "model": {
        "model_path": "deepseek-ai/DeepSeek-V4-Pro",
        "served_model_name": "deepseek-ai/DeepSeek-V4-Pro",
        "tokenizer_path": "deepseek-ai/DeepSeek-V4-Pro",
        "trust_remote_code": True,
        "dsv4_mode": "2604",
    },
    "profile": {
        "isl": 8192,
        "osl": 1024,
        "prefill_local_bs": 1,
        "decode_local_bs_list": [16],
        "prefill_profile_enabled": True,
        "decode_profile_enabled": True,
        "e2e_global_concurrency_override": 256,
        "e2e_request_multiplier": 2,
        "prefill_profile_steps": 16,
        "prefill_verify_min_target_hits": 4,
        "decode_profile_steps": 16,
        "decode_verify_min_target_hits": 4,
        "aiperf_random_seed": 100,
    },
    "scheduler_nvtx": {
        "enabled": True,
        "log_forward_iters": True,
    },
    "deepgemm": {
        "enable_jit": True,
        "precompile": True,
        "use_megamoe": True,
        "fix_hash_megamoe": True,
        "fix_megamoe_memory": True,
        "fix_nextn_megamoe": True,
        "print_compiler_command": True,
        "prefill_cache_dir": "/models/aic_native_sglang_megamoe_timeline/20260507200011/deep_gemm_cache/prefill",
        "decode_cache_dir": "/models/aic_native_sglang_megamoe_timeline/20260507200011/deep_gemm_cache/decode",
        "prefill_max_tokens_per_rank": 8320,
        "decode_max_tokens_per_rank": 512,
        "deepep_num_max_dispatch_tokens_per_rank": 0,
    },
    "sglang_common": {
        "hf_home": "/models",
        "offline": True,
        "warmup_timeout": 1800,
        "profile_with_stack": 0,
        "profile_record_shapes": 0,
        "set_cpu_affinity": True,
        "kv_cache_dtype": "fp8_e4m3",
        "context_length": 32768,
        "page_size": 64,
        "watchdog_timeout": 3600,
        "dist_timeout": 3600,
        "moe_dense_tp_size": 1,
        "moe_a2a_backend": "deepep",
        "moe_runner_backend": "deep_gemm",
        "disaggregation_transfer_backend": "nixl",
        "disaggregation_ib_device": "mlx5_0,mlx5_1,mlx5_2,mlx5_3",
        "disaggregation_bootstrap_port": 8998,
    },
    "prefill": {
        "nodes": 2,
        "tp": 8,
        "pp": 1,
        "dp": 8,
        "ep": 8,
        "gpus_per_pod": None,
        "deepep_mode": "normal",
        "mem_fraction_static": 0.8,
        "chunked_prefill_size": 65536,
        "max_prefill_tokens": 8192,
        "prefill_max_requests": 1,
        "max_running_requests": 1024,
        "disable_radix_cache": True,
        "extra_args": [],
        "extra_env": {},
    },
    "decode": {
        "nodes": 2,
        "tp": 8,
        "pp": 1,
        "dp": 8,
        "ep": 8,
        "gpus_per_pod": None,
        "deepep_mode": "low_latency",
        "mem_fraction_static": 0.849,
        "chunked_prefill_size": 65536,
        "max_prefill_tokens": 32768,
        "max_running_requests": 128,
        "decode_log_interval": 1,
        "cuda_graph_max_bs": 16,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "disable_radix_cache": False,
        "extra_args": [],
        "extra_env": {},
    },
    "nsys": {
        "tools_subdir": "NsightSystems-cli-2025.2.1/target-linux-sbsa-armv8",
        "trace": "cuda,nvtx,osrt",
        "cuda_graph_trace": "node",
        "capture_range": "cudaProfilerApi",
        "capture_range_end": "repeat:3",
        "trace_fork_before_exec": True,
        "sample": "none",
        "cpuctxsw": "none",
    },
    "download": {
        "chunk_bytes": "32M",
        "local_port": 18082,
        "remote_http_port": 8001,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def q(value: Any) -> str:
    return shlex.quote(str(value))


def shell_word(value: Any) -> str:
    """Quote a shell word while preserving intentional runtime env expansion."""
    text = str(value)
    if "$" in text:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'
    return shlex.quote(text)


def bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"


def bool_str(value: Any) -> str:
    return "true" if bool(value) else "false"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return deep_merge(DEFAULT_CONFIG, data)


def derive(cfg: dict[str, Any], config_path: Path) -> dict[str, str]:
    run_id = str(cfg.get("run_id") or time.strftime("%Y%m%d%H%M%S"))
    short_id = str(cfg.get("short_id") or run_id[-6:])
    base = f"{cfg['base_prefix']}-{short_id}"
    app = f"{cfg['app_prefix']}-{run_id}"
    artifact_dir = Path(cfg["artifact_root"]) / f"{cfg['name']}_{run_id}"
    remote_root = f"{cfg['remote_root_base'].rstrip('/')}/{run_id}"
    names = {
        "RUN_ID": run_id,
        "NAMESPACE": cfg["namespace"],
        "APP": app,
        "BASE": base,
        "PREFILL_STS": f"{base}-prefill",
        "DECODE_STS": f"{base}-decode",
        "PREFILL_SERVICE": f"{base}-prefill",
        "DECODE_SERVICE": f"{base}-decode",
        "PREFILL_LEADER_SERVICE": f"{base}-prefill-leader",
        "DECODE_LEADER_SERVICE": f"{base}-decode-leader",
        "ROUTER_DEPLOYMENT": f"{base}-router",
        "ROUTER_SERVICE": f"{base}-router",
        "COMPUTE_DOMAIN": f"{base}-domain",
        "COMPUTE_DOMAIN_CHANNEL": f"{base}-channel",
        "BENCH_JOB": f"{base}-profile-{run_id}",
        "RESULTS_JOB": f"{base}-results-{run_id}",
        "SCHEDULER_NVTX_CONFIGMAP": f"{base}-scheduler-nvtx-patch",
        "COPY_POD": f"{base}-copy-{run_id}",
        "COPY_APP": f"{cfg['app_prefix']}-copy-{run_id}",
        "ARTIFACT_DIR": str(artifact_dir),
        "REMOTE_RESULT_ROOT": remote_root,
        "CONFIG_PATH": str(config_path.resolve()),
    }
    return names


def role_gpus(role: dict[str, Any]) -> int:
    if role.get("gpus_per_pod") is not None:
        return int(role["gpus_per_pod"])
    world = int(role["tp"]) * int(role.get("pp", 1))
    nodes = int(role["nodes"])
    if world % nodes != 0:
        raise ValueError(f"tp*pp={world} is not divisible by nodes={nodes}")
    return world // nodes


def validate_config(cfg: dict[str, Any]) -> None:
    decode_bs_list = [int(x) for x in cfg["profile"]["decode_local_bs_list"]]
    if not decode_bs_list:
        raise ValueError("profile.decode_local_bs_list must contain at least one target local bs")

    max_decode_bs = max(decode_bs_list)
    if max_decode_bs > int(cfg["decode"]["cuda_graph_max_bs"]):
        raise ValueError(
            "decode.cuda_graph_max_bs must be >= max(profile.decode_local_bs_list)"
        )

    required_running = max_decode_bs * int(cfg["decode"]["dp"])
    if int(cfg["decode"]["max_running_requests"]) < required_running:
        raise ValueError(
            "decode.max_running_requests must be >= max_decode_local_bs * decode.dp "
            f"({required_running})"
        )

    concurrency_override = cfg["profile"].get("e2e_global_concurrency_override")
    if concurrency_override not in (None, "") and int(concurrency_override) < required_running:
        raise ValueError(
            "profile.e2e_global_concurrency_override must be empty or >= "
            f"max_decode_local_bs * decode.dp ({required_running})"
        )

    required_prefill_tokens = int(cfg["profile"]["isl"]) * int(cfg["profile"]["prefill_local_bs"])
    if int(cfg["prefill"]["max_prefill_tokens"]) < required_prefill_tokens:
        raise ValueError(
            "prefill.max_prefill_tokens must be >= profile.isl * profile.prefill_local_bs "
            f"({required_prefill_tokens})"
        )

    if not bool(cfg["deepgemm"]["use_megamoe"]):
        raise ValueError("deepgemm.use_megamoe must remain true for MegaMoE timeline collection")
    if cfg["sglang_common"]["moe_runner_backend"] != "deep_gemm":
        raise ValueError("sglang_common.moe_runner_backend must be deep_gemm for this collector")
    if cfg["sglang_common"]["moe_a2a_backend"] != "deepep":
        raise ValueError("sglang_common.moe_a2a_backend must be deepep for this collector")

    role_gpus(cfg["prefill"])
    role_gpus(cfg["decode"])


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run_env(cfg: dict[str, Any], names: dict[str, str]) -> str:
    profile = cfg["profile"]
    concurrency_override = profile.get("e2e_global_concurrency_override")
    env: dict[str, Any] = {
        **names,
        "PVC_NAME": cfg["pvc_name"],
        "IMAGE": cfg["images"]["sglang"],
        "NSYS_IMAGE": cfg["images"]["nsys"],
        "AIPERF_IMAGE": cfg["images"]["aiperf"],
        "COPY_IMAGE": cfg["images"]["copy"],
        "IMAGE_PULL_SECRET": cfg["images"]["image_pull_secret"],
        "IMAGE_PULL_SECRET_SOURCE_NAMESPACE": cfg["images"][
            "image_pull_secret_source_namespace"
        ],
        "GB200_NODEPOOL": cfg["cluster"]["gb200_nodepool"],
        "GB200_CLIQUE": cfg["cluster"]["gb200_clique"],
        "CPU_NODE_INSTANCE_TYPE": cfg["cluster"]["cpu_node_instance_type"],
        "MODEL_PATH": cfg["model"]["model_path"],
        "SERVED_MODEL_NAME": cfg["model"]["served_model_name"],
        "TOKENIZER_PATH": cfg["model"]["tokenizer_path"],
        "HF_HOME": cfg["sglang_common"]["hf_home"],
        "PREFILL_DP": cfg["prefill"]["dp"],
        "PREFILL_EP": cfg["prefill"]["ep"],
        "PREFILL_TP": cfg["prefill"]["tp"],
        "PREFILL_PP": cfg["prefill"]["pp"],
        "PREFILL_NODE_COUNT": cfg["prefill"]["nodes"],
        "PREFILL_GPUS_PER_POD": role_gpus(cfg["prefill"]),
        "DECODE_DP": cfg["decode"]["dp"],
        "DECODE_EP": cfg["decode"]["ep"],
        "DECODE_TP": cfg["decode"]["tp"],
        "DECODE_PP": cfg["decode"]["pp"],
        "DECODE_NODE_COUNT": cfg["decode"]["nodes"],
        "DECODE_GPUS_PER_POD": role_gpus(cfg["decode"]),
        "PREFILL_LOCAL_BS": profile["prefill_local_bs"],
        "DECODE_LOCAL_BS_LIST": " ".join(map(str, profile["decode_local_bs_list"])),
        "E2E_GLOBAL_CONCURRENCY_OVERRIDE": ""
        if concurrency_override in (None, "")
        else concurrency_override,
        "E2E_REQUEST_MULTIPLIER": profile["e2e_request_multiplier"],
        "PREFILL_PROFILE_ENABLED": 1 if profile.get("prefill_profile_enabled", True) else 0,
        "DECODE_PROFILE_ENABLED": 1 if profile.get("decode_profile_enabled", True) else 0,
        "PREFILL_PROFILE_STEPS": profile["prefill_profile_steps"],
        "PREFILL_VERIFY_MIN_TARGET_HITS": profile["prefill_verify_min_target_hits"],
        "DECODE_PROFILE_STEPS": profile["decode_profile_steps"],
        "DECODE_VERIFY_MIN_TARGET_HITS": profile["decode_verify_min_target_hits"],
        "AIPERF_RANDOM_SEED": profile["aiperf_random_seed"],
        "ISL": profile["isl"],
        "OSL": profile["osl"],
        "DOWNLOAD_CHUNK_BYTES": cfg["download"]["chunk_bytes"],
        "DOWNLOAD_LOCAL_PORT": cfg["download"]["local_port"],
        "DOWNLOAD_REMOTE_HTTP_PORT": cfg["download"]["remote_http_port"],
    }
    lines = ["# Generated by tools/sglang_megamoe_timeline_collector/render.py"]
    for key, value in env.items():
        lines.append(f"{key}={q(value)}")
    return "\n".join(lines) + "\n"


def export_lines(env: dict[str, Any]) -> list[str]:
    return [f"export {key}={q(value)}" for key, value in env.items()]


def bool_arg(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def launch_script(role_name: str, cfg: dict[str, Any], names: dict[str, str]) -> str:
    role = cfg[role_name]
    common = cfg["sglang_common"]
    deepgemm = cfg["deepgemm"]
    nsys = cfg["nsys"]
    model = cfg["model"]
    is_prefill = role_name == "prefill"
    component = "prefill" if is_prefill else "decode"
    service = names["PREFILL_SERVICE"] if is_prefill else names["DECODE_SERVICE"]
    master = f"{names['PREFILL_STS']}-0.{service}.${{POD_NAMESPACE}}.svc.cluster.local" if is_prefill else f"{names['DECODE_STS']}-0.{service}.${{POD_NAMESPACE}}.svc.cluster.local"
    torch_profile_dir = f"{names['REMOTE_RESULT_ROOT']}/sglang_profiles/default_{component}"
    dg_cache = deepgemm["prefill_cache_dir"] if is_prefill else deepgemm["decode_cache_dir"]
    max_tokens_per_rank = (
        deepgemm["prefill_max_tokens_per_rank"]
        if is_prefill
        else deepgemm["decode_max_tokens_per_rank"]
    )

    env: dict[str, Any] = {
        "HF_HOME": common["hf_home"],
        "HF_HUB_OFFLINE": "1" if common["offline"] else "0",
        "TRANSFORMERS_OFFLINE": "1" if common["offline"] else "0",
        "SGLANG_DSV4_MODE": model["dsv4_mode"],
        "SGLANG_WARMUP_TIMEOUT": common["warmup_timeout"],
        "SGLANG_PROFILE_WITH_STACK": common["profile_with_stack"],
        "SGLANG_PROFILE_RECORD_SHAPES": common["profile_record_shapes"],
        "SGLANG_TORCH_PROFILER_DIR": torch_profile_dir,
        "SGLANG_SET_CPU_AFFINITY": bool_str(common["set_cpu_affinity"]),
        "SGLANG_ENABLE_JIT_DEEPGEMM": bool_env(deepgemm["enable_jit"]),
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": bool_env(deepgemm["precompile"]),
        "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": bool_env(deepgemm["use_megamoe"]),
        "SGLANG_OPT_FIX_HASH_MEGA_MOE": bool_env(deepgemm["fix_hash_megamoe"]),
        "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": bool_env(deepgemm["fix_megamoe_memory"]),
        "SGLANG_OPT_FIX_NEXTN_MEGA_MOE": bool_env(deepgemm["fix_nextn_megamoe"]),
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": max_tokens_per_rank,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": deepgemm[
            "deepep_num_max_dispatch_tokens_per_rank"
        ],
        "SGLANG_DG_CACHE_DIR": dg_cache,
        "DG_JIT_PRINT_COMPILER_COMMAND": bool_env(deepgemm["print_compiler_command"]),
    }
    if cfg["scheduler_nvtx"]["log_forward_iters"]:
        env["SGLANG_LOG_FORWARD_ITERS"] = "1"
    if cfg["scheduler_nvtx"]["enabled"]:
        env["SGLANG_SCHEDULER_NVTX"] = "1"
    env.update(role.get("extra_env") or {})

    launch_args = [
        "--model-path",
        model["model_path"],
        "--served-model-name",
        model["served_model_name"],
    ]
    bool_arg(launch_args, "--trust-remote-code", bool(model["trust_remote_code"]))
    launch_args.extend(
        [
            "--disaggregation-ib-device",
            common["disaggregation_ib_device"],
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--dist-init-addr",
            "${MASTER_ADDR}:29500",
            "--nnodes",
            str(role["nodes"]),
            "--node-rank",
            "${RANK}",
            "--tensor-parallel-size",
            str(role["tp"]),
            "--pipeline-parallel-size",
            str(role["pp"]),
            "--data-parallel-size",
            str(role["dp"]),
            "--expert-parallel-size",
            str(role["ep"]),
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            str(common["moe_dense_tp_size"]),
            "--moe-a2a-backend",
            common["moe_a2a_backend"],
            "--deepep-mode",
            role["deepep_mode"],
            "--moe-runner-backend",
            common["moe_runner_backend"],
            "--kv-cache-dtype",
            common["kv_cache_dtype"],
            "--mem-fraction-static",
            str(role["mem_fraction_static"]),
            "--context-length",
            str(common["context_length"]),
            "--page-size",
            str(common["page_size"]),
            "--chunked-prefill-size",
            str(role["chunked_prefill_size"]),
            "--max-prefill-tokens",
            str(role["max_prefill_tokens"]),
        ]
    )
    bool_arg(launch_args, "--disable-radix-cache", bool(role.get("disable_radix_cache")))
    if is_prefill:
        launch_args.extend(
            [
                "--prefill-max-requests",
                str(role["prefill_max_requests"]),
                "--max-running-requests",
                str(role["max_running_requests"]),
                "--watchdog-timeout",
                str(common["watchdog_timeout"]),
                "--dist-timeout",
                str(common["dist_timeout"]),
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-decode-tp",
                str(cfg["decode"]["tp"]),
                "--disaggregation-decode-dp",
                str(cfg["decode"]["dp"]),
                "--disaggregation-bootstrap-port",
                str(common["disaggregation_bootstrap_port"]),
                "--disaggregation-transfer-backend",
                common["disaggregation_transfer_backend"],
            ]
        )
    else:
        launch_args.extend(
            [
                "--max-running-requests",
                str(role["max_running_requests"]),
                "--decode-log-interval",
                str(role["decode_log_interval"]),
                "--cuda-graph-max-bs",
                str(role["cuda_graph_max_bs"]),
                "--cuda-graph-bs",
                *[str(x) for x in role["cuda_graph_bs"]],
                "--watchdog-timeout",
                str(common["watchdog_timeout"]),
                "--dist-timeout",
                str(common["dist_timeout"]),
                "--disaggregation-mode",
                "decode",
                "--disaggregation-transfer-backend",
                common["disaggregation_transfer_backend"],
            ]
        )
    launch_args.extend(map(str, role.get("extra_args") or []))

    nsys_args = [
        "--force-overwrite=true",
        f"--trace={nsys['trace']}",
        f"--cuda-graph-trace={nsys['cuda_graph_trace']}",
        f"--capture-range={nsys['capture_range']}",
        f"--capture-range-end={nsys['capture_range_end']}",
        f"--trace-fork-before-exec={str(nsys['trace_fork_before_exec']).lower()}",
        f"--sample={nsys['sample']}",
        f"--cpuctxsw={nsys['cpuctxsw']}",
        f"--output={names['REMOTE_RESULT_ROOT']}/nsys/{component}/${{HOSTNAME}}",
    ]

    lines = [
        "set -euxo pipefail",
        "ulimit -l unlimited",
        "ulimit -n 1048576",
        f"export PATH=\"/nsys-tools/{nsys['tools_subdir']}:${{PATH}}\"",
    ]
    if cfg["scheduler_nvtx"]["enabled"]:
        lines.append("python3 /sglang_nvtx_patch/apply_scheduler_nvtx_patch.py")
    lines.extend(
        [
            'export RANK="${HOSTNAME##*-}"',
            f'export MASTER_ADDR="{master}"',
            'until getent hosts "${MASTER_ADDR}"; do',
            '  echo "waiting for ${MASTER_ADDR}"',
            "  sleep 2",
            "done",
        ]
    )
    lines.extend(export_lines(env))
    lines.extend(
        [
            'export DG_JIT_CACHE_DIR="${SGLANG_DG_CACHE_DIR}"',
            "nsys --version",
            'mkdir -p "${DG_JIT_CACHE_DIR}"',
            'if [ -d "${DG_JIT_CACHE_DIR}/cache" ]; then',
            '  deep_gemm_cache_kernels=$(find "${DG_JIT_CACHE_DIR}/cache" -maxdepth 1 -type d -name \'kernel.*\' | wc -l)',
            "else",
            "  deep_gemm_cache_kernels=0",
            "fi",
            'echo "DeepGEMM cache dir: ${DG_JIT_CACHE_DIR}; cached kernel dirs: ${deep_gemm_cache_kernels}"',
            f"mkdir -p {q(names['REMOTE_RESULT_ROOT'] + '/nsys/' + component)}",
            "exec nsys profile \\",
        ]
    )
    lines.extend([f"  {shell_word(arg)} \\" for arg in nsys_args])
    lines.append("  python3 -m sglang.launch_server \\")
    lines.extend([f"    {shell_word(arg)} \\" for arg in launch_args[:-1]])
    lines.append(f"    {shell_word(launch_args[-1])}")
    return "\n".join(lines) + "\n"


def mutate_native_yaml(docs: list[dict[str, Any]], cfg: dict[str, Any], names: dict[str, str]) -> list[dict[str, Any]]:
    total_nodes = int(cfg["prefill"]["nodes"]) + int(cfg["decode"]["nodes"])
    for doc in docs:
        if not doc:
            continue
        kind = doc.get("kind")
        meta = doc.setdefault("metadata", {})
        meta["namespace"] = names["NAMESPACE"]
        labels = meta.setdefault("labels", {})
        labels["app"] = names["APP"]

        if kind == "ComputeDomain":
            meta["name"] = names["COMPUTE_DOMAIN"]
            doc["spec"]["numNodes"] = total_nodes
            doc["spec"]["channel"]["resourceClaimTemplate"]["name"] = names[
                "COMPUTE_DOMAIN_CHANNEL"
            ]
        elif kind == "Service":
            old_name = meta.get("name", "")
            if "prefill-leader" in old_name:
                meta["name"] = names["PREFILL_LEADER_SERVICE"]
                doc["spec"]["selector"] = {
                    "statefulset.kubernetes.io/pod-name": f"{names['PREFILL_STS']}-0"
                }
            elif "decode-leader" in old_name:
                meta["name"] = names["DECODE_LEADER_SERVICE"]
                doc["spec"]["selector"] = {
                    "statefulset.kubernetes.io/pod-name": f"{names['DECODE_STS']}-0"
                }
            elif "prefill" in old_name:
                meta["name"] = names["PREFILL_SERVICE"]
                doc["spec"]["selector"]["app"] = names["APP"]
            elif "decode" in old_name:
                meta["name"] = names["DECODE_SERVICE"]
                doc["spec"]["selector"]["app"] = names["APP"]
            elif "router" in old_name:
                meta["name"] = names["ROUTER_SERVICE"]
                doc["spec"]["selector"]["app"] = names["APP"]
        elif kind == "StatefulSet":
            component = meta["labels"]["component"]
            role = cfg[component]
            meta["name"] = names["PREFILL_STS"] if component == "prefill" else names["DECODE_STS"]
            doc["spec"]["serviceName"] = (
                names["PREFILL_SERVICE"] if component == "prefill" else names["DECODE_SERVICE"]
            )
            doc["spec"]["replicas"] = int(role["nodes"])
            doc["spec"]["selector"]["matchLabels"]["app"] = names["APP"]
            tmpl = doc["spec"]["template"]
            tmpl["metadata"]["labels"]["app"] = names["APP"]
            spec = tmpl["spec"]
            spec["imagePullSecrets"] = [{"name": cfg["images"]["image_pull_secret"]}]
            spec["nodeSelector"]["cloud.google.com/gke-nodepool"] = cfg["cluster"][
                "gb200_nodepool"
            ]
            spec["nodeSelector"]["nvidia.com/gpu.clique"] = cfg["cluster"]["gb200_clique"]
            spec["initContainers"][0]["image"] = cfg["images"]["nsys"]
            spec["resourceClaims"][0]["resourceClaimTemplateName"] = names[
                "COMPUTE_DOMAIN_CHANNEL"
            ]
            for volume in spec["volumes"]:
                if volume["name"] == "scheduler-nvtx-patch":
                    volume["configMap"]["name"] = names["SCHEDULER_NVTX_CONFIGMAP"]
                if volume["name"] == "shared-model-cache":
                    volume["persistentVolumeClaim"]["claimName"] = cfg["pvc_name"]
            container = spec["containers"][0]
            container["image"] = cfg["images"]["sglang"]
            container["args"] = [launch_script(component, cfg, names)]
            container["resources"]["limits"]["nvidia.com/gpu"] = str(role_gpus(role))
            if "requests" in container["resources"]:
                container["resources"]["requests"]["nvidia.com/gpu"] = str(role_gpus(role))
        elif kind == "Deployment":
            meta["name"] = names["ROUTER_DEPLOYMENT"]
            doc["spec"]["selector"]["matchLabels"]["app"] = names["APP"]
            tmpl = doc["spec"]["template"]
            tmpl["metadata"]["labels"]["app"] = names["APP"]
            spec = tmpl["spec"]
            spec["imagePullSecrets"] = [{"name": cfg["images"]["image_pull_secret"]}]
            spec["nodeSelector"]["cloud.google.com/gke-nodepool"] = cfg["cluster"][
                "gb200_nodepool"
            ]
            container = spec["containers"][0]
            container["image"] = cfg["images"]["sglang"]
            container["command"] = [
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--prefill",
                f"http://{names['PREFILL_LEADER_SERVICE']}.{names['NAMESPACE']}.svc.cluster.local:30000",
                str(cfg["sglang_common"]["disaggregation_bootstrap_port"]),
                "--decode",
                f"http://{names['DECODE_LEADER_SERVICE']}.{names['NAMESPACE']}.svc.cluster.local:30000",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
    return docs


def collect_results_yaml(cfg: dict[str, Any], names: dict[str, str]) -> str:
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {names['RESULTS_JOB']}
  namespace: {names['NAMESPACE']}
  labels:
    app: {names['APP']}
    role: results
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: {names['APP']}
        role: results
    spec:
      restartPolicy: Never
      nodeSelector:
        node.kubernetes.io/instance-type: {cfg['cluster']['cpu_node_instance_type']}
      tolerations:
        - key: dedicated
          operator: Equal
          value: user-workload
          effect: NoExecute
      containers:
        - name: list
          image: {cfg['images']['copy']}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euxo pipefail
              root={q(names['REMOTE_RESULT_ROOT'])}
              echo "RESULT_ROOT=${{root}}"
              find "${{root}}" -maxdepth 5 -type f -printf '%p %s bytes\\n' | sort
          volumeMounts:
            - name: shared
              mountPath: /models
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: {cfg['pvc_name']}
"""


def copy_pod_yaml(cfg: dict[str, Any], names: dict[str, str]) -> str:
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: {names['COPY_POD']}
  namespace: {names['NAMESPACE']}
  labels:
    app: {names['COPY_APP']}
    role: pvc-copy
spec:
  restartPolicy: Never
  nodeSelector:
    node.kubernetes.io/instance-type: {cfg['cluster']['cpu_node_instance_type']}
  tolerations:
    - key: dedicated
      operator: Equal
      value: user-workload
      effect: NoExecute
  containers:
    - name: copy
      image: {cfg['images']['copy']}
      command: ["/bin/bash", "-lc"]
      args:
        - |
          set -euo pipefail
          while true; do sleep 3600; done
      volumeMounts:
        - name: shared
          mountPath: /models
  volumes:
    - name: shared
      persistentVolumeClaim:
        claimName: {cfg['pvc_name']}
"""


def patch_script_text(path: Path, cfg: dict[str, Any], names: dict[str, str]) -> str:
    text = path.read_text()
    text = text.replace('"deepseek-ai/DeepSeek-V4-Pro"', '"${SERVED_MODEL_NAME}"')
    text = text.replace("deepseek-ai/DeepSeek-V4-Pro", "${SERVED_MODEL_NAME}")
    text = text.replace("python:3.12-slim", "${AIPERF_IMAGE}")
    text = text.replace("n2d-standard-8", "${CPU_NODE_INSTANCE_TYPE}")
    text = text.replace("-n kprashanth -o json", '-n "${IMAGE_PULL_SECRET_SOURCE_NAMESPACE}" -o json')
    text = text.replace("value: ${SERVED_MODEL_NAME}\n            - name: TOKENIZER\n              value: ${SERVED_MODEL_NAME}", "value: ${SERVED_MODEL_NAME}\n            - name: TOKENIZER\n              value: ${TOKENIZER_PATH}")
    text = text.replace("value: /models", "value: ${HF_HOME}")
    text = text.replace("--random-seed 100", "--random-seed ${AIPERF_RANDOM_SEED}")
    text = text.replace(
        'o["metadata"]={"name":"nvcrimagepullsecret","namespace":"default"}',
        'o["metadata"]={"name":sys.argv[1],"namespace":sys.argv[2]}',
    )
    text = text.replace(
        "| python3 -c 'import json,sys; o=json.load(sys.stdin); o[\"metadata\"]={\"name\":sys.argv[1],\"namespace\":sys.argv[2]}; print(json.dumps(o))' \\\n"
        "    | kubectl apply --validate=false -n \"${NAMESPACE}\" -f -",
        "| python3 -c 'import json,sys; o=json.load(sys.stdin); o[\"metadata\"]={\"name\":sys.argv[1],\"namespace\":sys.argv[2]}; print(json.dumps(o))' \"${IMAGE_PULL_SECRET}\" \"${NAMESPACE}\" \\\n"
        "    | kubectl apply --validate=false -n \"${NAMESPACE}\" -f -",
    )
    text = text.replace(
        'if [[ "${prefill_count}" -eq 2 && "${decode_count}" -eq 2 ]]; then',
        'if [[ "${prefill_count}" -eq "${PREFILL_NODE_COUNT}" && "${decode_count}" -eq "${DECODE_NODE_COUNT}" ]]; then',
    )
    return text


def download_results_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/run_env.sh"

LOCAL_ROOT="${1:-${ARTIFACT_DIR}/local_timelines}"
CHUNK_DIR="${LOCAL_ROOT}/nsys_chunks"
mkdir -p "${LOCAL_ROOT}/nsys" "${CHUNK_DIR}"

pf_pid=""
cleanup_download() {
  [[ -n "${pf_pid}" ]] && kill "${pf_pid}" >/dev/null 2>&1 || true
  kubectl delete pod "${COPY_POD}" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup_download EXIT

kubectl apply --validate=false -f "${SCRIPT_DIR}/copy_pvc_pod.yaml"
kubectl wait --for=condition=Ready pod/"${COPY_POD}" -n "${NAMESPACE}" --timeout=10m

kubectl exec "${COPY_POD}" -n "${NAMESPACE}" -- bash -lc \
  "cd ${REMOTE_RESULT_ROOT} && find nsys -type f -name '*.nsys-rep' -print0 | sort -z | xargs -0 -r sha256sum" \
  | tee "${ARTIFACT_DIR}/remote_nsys_sha256.txt"

kubectl exec "${COPY_POD}" -n "${NAMESPACE}" -- bash -lc \
  "rm -rf /tmp/timeline_chunks && mkdir -p /tmp/timeline_chunks && cd /tmp/timeline_chunks && nohup python3 -m http.server ${DOWNLOAD_REMOTE_HTTP_PORT} >/tmp/timeline_chunks_http.log 2>&1 </dev/null & echo \$! >/tmp/timeline_chunks_http.pid"

kubectl port-forward pod/"${COPY_POD}" "${DOWNLOAD_LOCAL_PORT}:${DOWNLOAD_REMOTE_HTTP_PORT}" -n "${NAMESPACE}" &
pf_pid="$!"
sleep 3

mapfile -t rel_files < <(
  kubectl exec "${COPY_POD}" -n "${NAMESPACE}" -- bash -lc \
    "cd ${REMOTE_RESULT_ROOT} && find nsys -type f -name '*.nsys-rep' -printf '%P\n' | sort"
)
if ((${#rel_files[@]} == 0)); then
  echo "no remote .nsys-rep files found under ${REMOTE_RESULT_ROOT}/nsys" >&2
  exit 1
fi

for rel in "${rel_files[@]}"; do
  echo "downloading ${rel}"
  out="${LOCAL_ROOT}/nsys/${rel}"
  mkdir -p "$(dirname "${out}")"
  rm -rf "${CHUNK_DIR:?}"/*
  kubectl exec "${COPY_POD}" -n "${NAMESPACE}" -- bash -lc \
    "rm -rf /tmp/timeline_chunks/* && split -b ${DOWNLOAD_CHUNK_BYTES} -d -a 4 '${REMOTE_RESULT_ROOT}/nsys/${rel}' /tmp/timeline_chunks/chunk_ && cd /tmp/timeline_chunks && sha256sum chunk_* > chunks.sha256"
  curl --noproxy '*' --fail --show-error --location \
    "http://127.0.0.1:${DOWNLOAD_LOCAL_PORT}/chunks.sha256" \
    -o "${CHUNK_DIR}/chunks.sha256"
  awk '{print $2}' "${CHUNK_DIR}/chunks.sha256" | while read -r chunk; do
    curl --noproxy '*' --fail --show-error --location --connect-timeout 10 --max-time 180 \
      "http://127.0.0.1:${DOWNLOAD_LOCAL_PORT}/${chunk}" \
      -o "${CHUNK_DIR}/${chunk}"
  done
  (cd "${CHUNK_DIR}" && sha256sum -c chunks.sha256)
  cat "${CHUNK_DIR}"/chunk_* > "${out}"
done

(cd "${LOCAL_ROOT}" && find nsys -type f -name '*.nsys-rep' -print0 | sort -z | xargs -0 -r sha256sum) \
  | sort | tee "${ARTIFACT_DIR}/local_nsys_sha256.txt"
diff -u "${ARTIFACT_DIR}/remote_nsys_sha256.txt" "${ARTIFACT_DIR}/local_nsys_sha256.txt"
echo "downloaded timelines to ${LOCAL_ROOT}/nsys"
'''


def verify_nvtx_script(cfg: dict[str, Any]) -> str:
    decode_bs_list = ", ".join(str(x) for x in cfg["profile"]["decode_local_bs_list"])
    prefill_new_token = int(cfg["profile"]["isl"])
    prefill_required = "True" if cfg["profile"].get("prefill_profile_enabled", True) else "False"
    decode_required = "True" if cfg["profile"].get("decode_profile_enabled", True) else "False"
    script = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path


DEFAULT_DECODE_BS_LIST = [__DECODE_BS_LIST__]
DEFAULT_PREFILL_NEW_TOKEN = __PREFILL_NEW_TOKEN__
DEFAULT_PREFILL_REQUIRED = __PREFILL_REQUIRED__
DEFAULT_DECODE_REQUIRED = __DECODE_REQUIRED__


def ensure_sqlite(rep: Path) -> Path:
    db = rep.with_suffix(".sqlite")
    if db.exists():
        return db
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "--output", str(db), str(rep)],
        check=True,
    )
    return db


def count_like(db: Path, pattern: str) -> int:
    con = sqlite3.connect(db)
    try:
        return con.execute("select count(*) from NVTX_EVENTS where text like ?", (pattern,)).fetchone()[0]
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--decode-bs", action="append", type=int, default=[])
    parser.add_argument("--prefill-new-token", type=int, default=DEFAULT_PREFILL_NEW_TOKEN)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    decode_bs_list = args.decode_bs or DEFAULT_DECODE_BS_LIST
    reps = sorted((args.artifact_dir / "local_timelines" / "nsys").glob("**/*.nsys-rep"))
    if not reps:
        raise SystemExit("no local .nsys-rep files found")
    decode_totals = {bs: 0 for bs in decode_bs_list}
    prefill_total = 0
    for rep in reps:
        db = ensure_sqlite(rep)
        print(f"{rep.relative_to(args.artifact_dir)}")
        print(f"  sgl_sched: {count_like(db, '%sgl_sched%')}")
        print(f"  sgl_decode_stats: {count_like(db, '%sgl_decode_stats%')}")
        print(f"  sgl_prefill_stats: {count_like(db, '%sgl_prefill_stats%')}")
        for bs in decode_bs_list:
            count = count_like(db, f'%sgl_sched mode=decode%bs={bs}%')
            decode_totals[bs] += count
            print(f"  decode bs={bs} labels: {count}")
        prefill_count = count_like(
            db,
            f'%sgl_sched mode=extend%new_token={args.prefill_new_token}%',
        )
        prefill_total += prefill_count
        print(f"  prefill new_token={args.prefill_new_token} labels: {prefill_count}")
    if args.strict:
        missing_decode = [bs for bs, count in decode_totals.items() if count == 0]
        if (DEFAULT_DECODE_REQUIRED and missing_decode) or (
            DEFAULT_PREFILL_REQUIRED and prefill_total == 0
        ):
            raise SystemExit(
                f"missing target NVTX labels: decode_bs={missing_decode}, "
                f"prefill_new_token_hits={prefill_total}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return (
        script.replace("__DECODE_BS_LIST__", decode_bs_list)
        .replace("__PREFILL_NEW_TOKEN__", str(prefill_new_token))
        .replace("__PREFILL_REQUIRED__", prefill_required)
        .replace("__DECODE_REQUIRED__", decode_required)
    )


def render(config_path: Path, run: bool) -> Path:
    cfg = load_config(config_path)
    validate_config(cfg)
    names = derive(cfg, config_path)
    artifact_dir = Path(names["ARTIFACT_DIR"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_text(artifact_dir / "run_env.sh", run_env(cfg, names))
    write_text(artifact_dir / "apply_scheduler_nvtx_patch.py", (TEMPLATE_DIR / "apply_scheduler_nvtx_patch.py").read_text())

    for script_name in ["cleanup.sh", "run_after_login.sh", "pd_profile_gate.sh"]:
        text = patch_script_text(TEMPLATE_DIR / script_name, cfg, names)
        write_text(artifact_dir / script_name, text, executable=True)

    docs = list(yaml.safe_load_all((TEMPLATE_DIR / "native_sglang_nsys.yaml").read_text()))
    docs = mutate_native_yaml(docs, cfg, names)
    write_text(
        artifact_dir / "native_sglang_nsys.yaml",
        yaml.dump_all(docs, Dumper=LiteralDumper, sort_keys=False),
    )
    write_text(artifact_dir / "collect_results.yaml", collect_results_yaml(cfg, names))
    write_text(artifact_dir / "copy_pvc_pod.yaml", copy_pod_yaml(cfg, names))
    write_text(artifact_dir / "download_results.sh", download_results_script(), executable=True)
    write_text(artifact_dir / "verify_nvtx.py", verify_nvtx_script(cfg), executable=True)
    write_text(artifact_dir / "config.rendered.yaml", yaml.safe_dump(cfg, sort_keys=False))

    for path in ["run_after_login.sh", "cleanup.sh", "pd_profile_gate.sh", "download_results.sh"]:
        subprocess.run(["bash", "-n", str(artifact_dir / path)], check=True)
    list(yaml.safe_load_all((artifact_dir / "native_sglang_nsys.yaml").read_text()))
    list(yaml.safe_load_all((artifact_dir / "collect_results.yaml").read_text()))
    list(yaml.safe_load_all((artifact_dir / "copy_pvc_pod.yaml").read_text()))

    print(f"rendered artifact: {artifact_dir}")
    print(f"remote result root: {names['REMOTE_RESULT_ROOT']}")
    if run:
        log = artifact_dir / f"run_after_login_{names['RUN_ID']}.log"
        with log.open("w") as f:
            proc = subprocess.Popen(
                ["bash", str(artifact_dir / "run_after_login.sh")],
                cwd=artifact_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            rc = proc.wait()
        if rc:
            raise SystemExit(rc)
    return artifact_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="run the rendered artifact immediately")
    args = parser.parse_args()
    render(args.config, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
