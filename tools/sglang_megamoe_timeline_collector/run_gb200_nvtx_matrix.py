#!/usr/bin/env python3
"""Run the GB200 MegaMoE NVTX timeline matrix as independent native SGLang PD cases."""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
RENDER = PROJECT_DIR / "render.py"


def utc8_now_id() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d%H%M%S")


def graph_bs_list(target: int) -> list[int]:
    return [bs for bs in [1, 2, 4, 8, 16, 32, 64, 128] if bs <= target]


def common_overrides(batch_id: str, run_id: str, short_id: str, name: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "short_id": short_id,
        "name": name,
        "app_prefix": "aic-sgl-megamoe-nvtx",
        "base_prefix": "aic-sgl-nvtx",
        "artifact_root": str(
            REPO_ROOT / "artifacts" / f"gb200_megamoe_nvtx_matrix_{batch_id}"
        ),
        "remote_root_base": f"/models/aic_native_sglang_megamoe_nvtx_matrix/{batch_id}",
        "profile": {
            "osl": 1024,
            "prefill_local_bs": 1,
            "e2e_request_multiplier": 2,
            "prefill_profile_steps": 16,
            "prefill_verify_min_target_hits": 4,
            "decode_profile_steps": 16,
            "decode_verify_min_target_hits": 4,
            "aiperf_random_seed": 100,
        },
        "prefill": {
            "nodes": 2,
            "tp": 8,
            "pp": 1,
            "dp": 8,
            "ep": 8,
            "deepep_mode": "normal",
            "mem_fraction_static": 0.849,
            "prefill_max_requests": 1,
            "max_running_requests": 1024,
            "disable_radix_cache": True,
        },
        "deepgemm": {
            "enable_jit": True,
            "precompile": True,
            "use_megamoe": True,
            "prefill_max_tokens_per_rank": 8320,
            "decode_max_tokens_per_rank": 512,
        },
        "sglang_common": {
            "moe_a2a_backend": "deepep",
            "moe_runner_backend": "deep_gemm",
            "disaggregation_transfer_backend": "nixl",
            "watchdog_timeout": 3600,
            "dist_timeout": 3600,
        },
        "scheduler_nvtx": {
            "enabled": True,
            "log_forward_iters": True,
        },
        "nsys": {
            "trace": "cuda,nvtx",
            "capture_range": "cudaProfilerApi",
            "capture_range_end": "repeat:12",
            "sample": "none",
            "cpuctxsw": "none",
        },
        "download": {
            "chunk_bytes": "32M",
            "local_port": 18082,
            "remote_http_port": 8001,
        },
    }


def prefill_case(batch_id: str, idx: int, isl: int) -> tuple[str, dict[str, Any]]:
    case_id = f"prefill_ep8_isl{isl}_bs1"
    run_id = f"{batch_id}-{idx:02d}p{isl // 1024}k"
    cfg = common_overrides(batch_id, run_id, f"{idx:02d}p{isl // 1024}k", case_id)
    cfg["profile"].update(
        {
            "isl": isl,
            "decode_local_bs_list": [8],
            "prefill_profile_enabled": True,
            "decode_profile_enabled": False,
            "e2e_global_concurrency_override": 256,
        }
    )
    cfg["prefill"].update(
        {
            "chunked_prefill_size": isl,
            "max_prefill_tokens": isl,
        }
    )
    cfg["decode"] = decode_role(ep=8, target_local_bs=8)
    return case_id, cfg


def combined_prefill_decode_case(
    batch_id: str, idx: int, isl: int, decode_ep: int, target_local_bs: int
) -> tuple[str, dict[str, Any]]:
    case_id = f"prefill_ep8_isl{isl}_bs1_decode_ep{decode_ep}_bs{target_local_bs}"
    run_id = f"{batch_id}-{idx:02d}p{isl // 1024}kd{decode_ep}b{target_local_bs}"
    cfg = common_overrides(
        batch_id,
        run_id,
        f"{idx:02d}p{isl // 1024}kd{decode_ep}b{target_local_bs}",
        case_id,
    )
    required = decode_ep * target_local_bs
    cfg["profile"].update(
        {
            "isl": isl,
            "decode_local_bs_list": [target_local_bs],
            "prefill_profile_enabled": True,
            "decode_profile_enabled": True,
            "e2e_global_concurrency_override": max(256, required),
        }
    )
    cfg["prefill"].update(
        {
            "chunked_prefill_size": isl,
            "max_prefill_tokens": isl,
        }
    )
    cfg["decode"] = decode_role(ep=decode_ep, target_local_bs=target_local_bs)
    return case_id, cfg


def decode_role(ep: int, target_local_bs: int) -> dict[str, Any]:
    return {
        "nodes": ep // 4,
        "tp": ep,
        "pp": 1,
        "dp": ep,
        "ep": ep,
        "deepep_mode": "low_latency",
        "mem_fraction_static": 0.849,
        "chunked_prefill_size": 65536,
        "max_prefill_tokens": 32768,
        "max_running_requests": ep * target_local_bs,
        "decode_log_interval": 1,
        "cuda_graph_max_bs": target_local_bs,
        "cuda_graph_bs": graph_bs_list(target_local_bs),
        "disable_radix_cache": False,
    }


def decode_case(batch_id: str, idx: int, ep: int, target_local_bs: int) -> tuple[str, dict[str, Any]]:
    case_id = f"decode_ep{ep}_bs{target_local_bs}_isl8192_osl1024"
    run_id = f"{batch_id}-{idx:02d}d{ep}b{target_local_bs}"
    cfg = common_overrides(batch_id, run_id, f"{idx:02d}d{ep}b{target_local_bs}", case_id)
    required = ep * target_local_bs
    cfg["profile"].update(
        {
            "isl": 8192,
            "decode_local_bs_list": [target_local_bs],
            "prefill_profile_enabled": False,
            "decode_profile_enabled": True,
            "e2e_global_concurrency_override": max(256, required),
        }
    )
    cfg["prefill"].update(
        {
            "chunked_prefill_size": 8192,
            "max_prefill_tokens": 8192,
        }
    )
    cfg["decode"] = decode_role(ep=ep, target_local_bs=target_local_bs)
    return case_id, cfg


def build_cases(batch_id: str) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    cases.append(prefill_case(batch_id, 1, 4096))
    cases.append(combined_prefill_decode_case(batch_id, 2, 8192, 8, 8))
    idx = 3
    for ep in [8, 16, 32]:
        for bs in [8, 32, 64, 128]:
            if ep == 8 and bs == 8:
                continue
            cases.append(decode_case(batch_id, idx, ep, bs))
            idx += 1
    return cases


def run_logged(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return proc.wait()


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def render_case(config_path: Path, log_dir: Path) -> Path:
    rc = run_logged(
        ["python3", str(RENDER), "--config", str(config_path)],
        REPO_ROOT,
        log_dir / f"{config_path.stem}.render.log",
    )
    if rc:
        raise RuntimeError(f"render failed for {config_path} with rc={rc}")
    cfg = yaml.safe_load(config_path.read_text())
    return Path(cfg["artifact_root"]) / f"{cfg['name']}_{cfg['run_id']}"


def run_case(case_id: str, cfg: dict[str, Any], batch_dir: Path, continue_on_error: bool) -> bool:
    config_path = batch_dir / "configs" / f"{case_id}.yaml"
    log_dir = batch_dir / "batch_logs"
    write_yaml(config_path, cfg)
    artifact_dir = render_case(config_path, log_dir)
    ok = False
    try:
        print(f"=== RUN {case_id} ===", flush=True)
        rc = run_logged(
            ["bash", str(artifact_dir / "run_after_login.sh")],
            artifact_dir,
            log_dir / f"{case_id}.run.log",
        )
        if rc:
            raise RuntimeError(f"run_after_login failed rc={rc}")
        print(f"=== DOWNLOAD {case_id} ===", flush=True)
        rc = run_logged(
            ["bash", str(artifact_dir / "download_results.sh")],
            artifact_dir,
            log_dir / f"{case_id}.download.log",
        )
        if rc:
            raise RuntimeError(f"download_results failed rc={rc}")
        print(f"=== VERIFY {case_id} ===", flush=True)
        rc = run_logged(
            [str(artifact_dir / "verify_nvtx.py"), str(artifact_dir), "--strict"],
            artifact_dir,
            log_dir / f"{case_id}.verify_nvtx.log",
        )
        if rc:
            raise RuntimeError(f"verify_nvtx failed rc={rc}")
        ok = True
        return True
    except BaseException as exc:
        print(f"CASE_FAILED {case_id}: {exc}", file=sys.stderr, flush=True)
        cleanup = artifact_dir / "cleanup.sh"
        if cleanup.exists():
            run_logged(["bash", str(cleanup)], artifact_dir, log_dir / f"{case_id}.cleanup_after_failure.log")
        if isinstance(exc, KeyboardInterrupt):
            raise
        if not continue_on_error:
            raise
        return False
    finally:
        status_path = batch_dir / "batch_status.tsv"
        with status_path.open("a") as status:
            status.write(f"{case_id}\t{'ok' if ok else 'failed'}\t{artifact_dir}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=utc8_now_id())
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Case id to run. Can be repeated. Defaults to the whole matrix.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    cases = build_cases(args.batch_id)
    selected = set(args.only)
    if selected:
        cases = [(case_id, cfg) for case_id, cfg in cases if case_id in selected]
        missing = selected - {case_id for case_id, _ in cases}
        if missing:
            raise SystemExit(f"unknown --only case(s): {sorted(missing)}")

    batch_dir = REPO_ROOT / "artifacts" / f"gb200_megamoe_nvtx_matrix_{args.batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "matrix_manifest.txt"
    manifest.write_text("\n".join(case_id for case_id, _ in cases) + "\n")
    print(f"batch_dir={batch_dir}")
    print(f"cases={len(cases)}")
    if args.plan_only:
        for case_id, _ in cases:
            print(case_id)
        return 0

    all_ok = True
    for case_id, cfg in cases:
        all_ok = run_case(case_id, cfg, batch_dir, args.continue_on_error) and all_ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
