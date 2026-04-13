#!/usr/bin/env python3
"""
SGLang Torch Profiler — 基于官方 /start_profile API 的性能采集工具
=================================================================

通过 SGLang 原生 HTTP API 进行 torch profile 采集，支持:
  - 精确控制 input_len / prefix_len / output_len 构造 prefill/decode batch
  - 分阶段 profiling (prefill + decode 分别采集)
  - 减少模型层数快速 profiling (--num-layers)
  - 可选自动启动 SGLang server (--launch-server)

前置条件:
  1. SGLang server 需要带 SGLANG_TORCH_PROFILER_DIR 环境变量启动
  2. 或使用 --launch-server 让脚本自动启动

快速开始:
  # 方式1: 手动启动 server, 然后运行 profiler
  SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile python3 -m sglang.launch_server \\
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 ...

  python3 torch_profiler.py --input-len 128 --output-len 64

  # 方式2: 脚本自动启动 server (推荐快速测试, 2层快速profile)
  #   所有 profiler 不认识的参数会自动透传给 sglang
  python3 torch_profiler.py --launch-server \\
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \\
    --num-layers 2 \\
    --input-len 128 --output-len 64 \\
    --tp-size 8 --ep-size 8 --trust-remote-code
"""

import argparse
import glob
import json
import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional


# ============================================================================
# HTTP 工具函数
# ============================================================================

try:
    import requests as _requests
except ImportError:
    _requests = None


def _check_requests():
    if _requests is None:
        print("错误: 需要 requests 库。运行: pip install requests")
        sys.exit(1)


def _health_check(server_url: str, timeout: float = 5.0) -> bool:
    try:
        resp = _requests.get(f"{server_url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _build_token_ids(length: int, seed: int = 42) -> List[int]:
    """生成指定长度的随机 token_ids (范围 100-30000)。"""
    rng = random.Random(seed)
    return [rng.randint(100, 30000) for _ in range(length)]


def _send_generate(
    server_url: str,
    input_ids: List[int],
    max_tokens: int,
    request_id,
) -> Dict[str, Any]:
    """通过 /generate 端点发送请求 (支持 input_ids 精确控制 token)。"""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
        },
    }
    try:
        resp = _requests.post(
            f"{server_url}/generate", json=payload, timeout=300,
        )
        result = resp.json()
        meta = result.get("meta_info", {})
        print(
            f"  [req-{request_id}] prompt={meta.get('prompt_tokens', '?')}, "
            f"completion={meta.get('completion_tokens', '?')}, "
            f"cached={meta.get('cached_tokens', '?')}"
        )
        return result
    except Exception as e:
        print(f"  [req-{request_id}] 失败: {e}")
        return {"error": str(e)}


def _start_profile_http(
    server_url: str,
    output_dir: str,
    num_steps: int = 5,
    profile_by_stage: bool = True,
    with_stack: bool = True,
    record_shapes: bool = True,
    activities: Optional[List[str]] = None,
    profile_stages: Optional[List[str]] = None,
) -> bool:
    """通过 HTTP API 启动 profiling (阻塞式)。"""
    if activities is None:
        activities = ["CPU", "GPU"]

    payload = {
        "output_dir": output_dir,
        "num_steps": num_steps,
        "activities": activities,
        "profile_by_stage": profile_by_stage,
        "with_stack": with_stack,
        "record_shapes": record_shapes,
    }
    if profile_stages:
        payload["profile_stages"] = profile_stages

    print(f"[profiler] POST /start_profile:")
    for k, v in payload.items():
        print(f"  {k} = {v}")

    try:
        resp = _requests.post(
            f"{server_url}/start_profile", json=payload, timeout=600,
        )
        if resp.status_code == 200:
            print(f"[profiler] /start_profile 返回: {resp.text.strip()}")
            # profile_by_stage=true 时立即返回，需要等待 trace 生成
            if profile_by_stage:
                print("[profiler] profile_by_stage=true, profiling 将在后台随 forward step 进行")
            return True
        else:
            print(f"[profiler] /start_profile 失败: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[profiler] /start_profile 异常: {e}")
        return False


def _list_trace_files(output_dir: str) -> List[str]:
    patterns = ["*.trace.json.gz", "*.trace.json", "trace.json"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(output_dir, pat)))
        files.extend(glob.glob(os.path.join(output_dir, "**", pat), recursive=True))
    return sorted(set(files))


# ============================================================================
# Server 自动启动
# ============================================================================

def _launch_sglang_server(args, extra_sglang_args: List[str] = None) -> subprocess.Popen:
    """使用官方 python3 -m sglang.launch_server 启动服务。

    extra_sglang_args: 透传给 sglang 的额外参数列表 (来自 parse_known_args 的未识别参数)。
    会自动去重: 如果 extra_sglang_args 中已包含某参数, 则不再重复添加默认值。
    """
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    extra = extra_sglang_args or []

    def _has_arg(name: str) -> bool:
        """检查 extra_sglang_args 中是否已包含指定参数。"""
        return name in extra

    cmd = [sys.executable, "-m", "sglang.launch_server"]

    # --model-path: 优先使用显式参数, 否则从 extra 中获取
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    elif not _has_arg("--model-path"):
        print("错误: 需要通过 --model-path 指定模型路径")
        sys.exit(1)

    # --port: 确保 port 一致 (profiler 需要知道端口)
    if not _has_arg("--port"):
        cmd += ["--port", str(args.port)]

    # 关键: 通过 --json-model-override-args 减少层数 (profiler 专有功能)
    if args.num_layers:
        # 检查用户是否已在 extra 中手动指定了 --json-model-override-args
        if _has_arg("--json-model-override-args"):
            print("警告: --num-layers 与 extra args 中的 --json-model-override-args 冲突, 使用 --num-layers")
            # 移除 extra 中的 --json-model-override-args
            idx = extra.index("--json-model-override-args")
            extra = extra[:idx] + extra[idx + 2:]  # 移除 flag + value
        override = json.dumps({"num_hidden_layers": args.num_layers})
        cmd += ["--json-model-override-args", override]

    # 透传所有 extra sglang 参数
    cmd += extra

    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = output_dir
    # 绕过 flashinfer 版本检查 (常见环境问题)
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    cmd_str = " ".join(cmd)
    print(f"\n[launcher] 启动 SGLang server ...")
    print(f"  SGLANG_TORCH_PROFILER_DIR={output_dir}")
    print(f"  FLASHINFER_DISABLE_VERSION_CHECK={env.get('FLASHINFER_DISABLE_VERSION_CHECK', '0')}")
    print(f"  {cmd_str}")
    if args.num_layers:
        print(f"  [快速模式] num_hidden_layers 从原始值 -> {args.num_layers}")

    log_path = os.path.join(output_dir, "sglang_server.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"  PID={proc.pid}, 日志: {log_path}")
    return proc


def _wait_server_ready(server_url: str, timeout: float = 600, log_path: str = None):
    """等待 server 就绪。"""
    print(f"[launcher] 等待 server 就绪 (超时 {timeout}s) ...")
    start = time.time()
    last_log_check = 0
    while time.time() - start < timeout:
        if _health_check(server_url, timeout=3.0):
            elapsed = time.time() - start
            print(f"[launcher] Server 就绪! (耗时 {elapsed:.0f}s)")
            return True
        # 每 15 秒打印一次日志末尾
        if log_path and time.time() - last_log_check > 15:
            last_log_check = time.time()
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""
                    elapsed = time.time() - start
                    print(f"  [{elapsed:.0f}s] {last_line[:120]}")
            except Exception:
                pass
        time.sleep(2.0)
    print(f"[launcher] 超时! Server 未就绪")
    return False


def _kill_server(proc: subprocess.Popen):
    """停止 server 进程组。"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
        print("[launcher] Server 已停止")
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ============================================================================
# 主逻辑
# ============================================================================

def run_profiling(args, extra_sglang_args: List[str] = None) -> None:
    """Server 模式: 通过 HTTP API 进行 profiling。"""
    _check_requests()

    # 如果 extra_sglang_args 中指定了 --port, 提取出来同步给 profiler
    if extra_sglang_args and "--port" in extra_sglang_args:
        idx = extra_sglang_args.index("--port")
        if idx + 1 < len(extra_sglang_args):
            args.port = int(extra_sglang_args[idx + 1])

    server_url = f"http://localhost:{args.port}"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    server_proc = None

    print(f"\n{'=' * 70}")
    print(f"  SGLang Torch Profiler")
    print(f"{'=' * 70}")
    print(f"  server_url      = {server_url}")
    print(f"  output_dir      = {output_dir}")
    print(f"  input_len       = {args.input_len}")
    print(f"  prefix_len      = {args.prefix_len}")
    print(f"  output_len      = {args.output_len}")
    print(f"  num_requests    = {args.num_requests}")
    print(f"  num_steps       = {args.num_steps}")
    print(f"  profile_by_stage= {args.profile_by_stage}")
    if args.launch_server:
        print(f"  model_path      = {args.model_path}")
        print(f"  num_layers      = {args.num_layers or 'all (原始)'}")
        extra = getattr(args, '_extra_sglang_args', [])
        if extra:
            print(f"  sglang extra    = {' '.join(extra)}")

    try:
        # Step 1: 启动 server (如果需要)
        if args.launch_server:
            print(f"\n--- Step 1: 启动 SGLang Server ---")
            server_proc = _launch_sglang_server(args, extra_sglang_args)
            log_path = os.path.join(output_dir, "sglang_server.log")
            if not _wait_server_ready(server_url, timeout=args.launch_timeout, log_path=log_path):
                print("\n错误: Server 启动失败，检查日志:")
                print(f"  cat {log_path}")
                return
        else:
            print(f"\n--- Step 1: 检查 Server ---")
            if not _health_check(server_url):
                print(f"错误: Server 未就绪 ({server_url}/health)")
                print("\n请先启动 server, 例如:")
                print(f"  export SGLANG_TORCH_PROFILER_DIR={output_dir}")
                print(f"  python3 -m sglang.launch_server \\")
                print(f"    --model-path <MODEL_PATH> \\")
                print(f"    --tp-size 8 --ep-size 8 \\")
                print(f"    --json-model-override-args '{{\"num_hidden_layers\": 2}}' \\")
                print(f"    --trust-remote-code")
                return
            print("Server 就绪!")

        # Step 2: 预填充 prefix cache
        prefix_token_ids = None
        if args.prefix_len > 0:
            print(f"\n--- Step 2: 预填充 prefix cache (len={args.prefix_len}) ---")
            prefix_token_ids = _build_token_ids(args.prefix_len, seed=42)
            _send_generate(server_url, prefix_token_ids, max_tokens=1, request_id="prefix")
        else:
            print(f"\n--- Step 2: 无 prefix (跳过) ---")

        # Step 3: Warmup
        print(f"\n--- Step 3: Warmup ({args.warmup_runs} 次) ---")
        for w in range(args.warmup_runs):
            if prefix_token_ids:
                ids = prefix_token_ids + _build_token_ids(args.input_len, seed=100 + w)
            else:
                ids = _build_token_ids(args.input_len, seed=100 + w)
            _send_generate(server_url, ids, args.output_len, f"warmup-{w}")

        # Step 4: 记录已有 trace
        existing_traces = set(_list_trace_files(output_dir))

        # Step 5: 后台启动 profiling + 发送请求
        print(f"\n--- Step 4: Profiling ---")
        profile_stages = None
        if args.profile_stages:
            profile_stages = [s.strip() for s in args.profile_stages.split(",")]

        with ThreadPoolExecutor(max_workers=2) as executor:
            profile_future = executor.submit(
                _start_profile_http,
                server_url, output_dir,
                num_steps=args.num_steps,
                profile_by_stage=args.profile_by_stage,
                with_stack=args.with_stack,
                record_shapes=args.record_shapes,
                profile_stages=profile_stages,
            )

            time.sleep(1.0)

            print(f"\n发送 {args.num_requests} 个请求 (input_len={args.input_len}, output_len={args.output_len}) ...")
            request_futures = []
            for i in range(args.num_requests):
                if prefix_token_ids:
                    prompt_ids = prefix_token_ids + _build_token_ids(args.input_len, seed=200 + i)
                else:
                    prompt_ids = _build_token_ids(args.input_len, seed=200 + i)
                request_futures.append(
                    executor.submit(_send_generate, server_url, prompt_ids, args.output_len, i)
                )

            for f in as_completed(request_futures):
                f.result()

            # 持续发送请求直到 trace 文件出现
            max_extra_rounds = 20
            extra_round = 0
            while extra_round < max_extra_rounds:
                # 检查 trace 文件是否已生成
                new_traces_now = set(_list_trace_files(output_dir)) - existing_traces
                if new_traces_now:
                    print(f"\n[profiler] 检测到 {len(new_traces_now)} 个新 trace 文件!")
                    break

                extra_round += 1
                print(f"\n等待 trace 生成, 发送补充请求 (round {extra_round}/{max_extra_rounds}) ...")
                extra_futures = []
                for i in range(max(args.num_requests, 3)):
                    if prefix_token_ids:
                        prompt_ids = prefix_token_ids + _build_token_ids(
                            args.input_len, seed=1000 + extra_round * 100 + i)
                    else:
                        prompt_ids = _build_token_ids(
                            args.input_len, seed=1000 + extra_round * 100 + i)
                    extra_futures.append(
                        executor.submit(_send_generate, server_url, prompt_ids,
                                        args.output_len, f"extra-{extra_round}-{i}")
                    )
                for f in as_completed(extra_futures):
                    f.result()
                time.sleep(2.0)

            profile_ok = profile_future.result() if profile_future.done() else True

        # 等待 trace 文件写入完成 (给 scheduler 一些时间 flush)
        print("\n等待 trace 文件写入完成 ...")
        for wait_i in range(15):
            new_traces_check = set(_list_trace_files(output_dir)) - existing_traces
            if new_traces_check:
                # 再等 2 秒确保所有 rank 都写完
                time.sleep(2.0)
                break
            time.sleep(2.0)

        # Step 6: 展示结果
        print(f"\n{'=' * 70}")
        print(f"  Profiling {'完成' if profile_ok else '失败'}")
        print(f"{'=' * 70}")

        new_traces = sorted(set(_list_trace_files(output_dir)) - existing_traces)
        if new_traces:
            print(f"\n生成的 trace 文件 ({len(new_traces)} 个):")
            for tf in new_traces:
                size_mb = os.path.getsize(tf) / (1024 * 1024)
                basename = os.path.basename(tf)
                stage = "prefill" if "EXTEND" in basename else "decode" if "DECODE" in basename else "mixed"
                print(f"  [{stage:>7}] {tf}  ({size_mb:.1f} MB)")
            print(f"\n打开 https://ui.perfetto.dev 加载 trace 文件进行可视化分析")
        else:
            all_traces = _list_trace_files(output_dir)
            if all_traces:
                print(f"\noutput_dir 中已有 {len(all_traces)} 个 trace 文件:")
                for tf in all_traces[:10]:
                    print(f"  {tf}")
            else:
                print(f"\n警告: 未找到 trace 文件")
                print("  请确认启动 server 时设置了 SGLANG_TORCH_PROFILER_DIR 环境变量")

    finally:
        if server_proc and args.auto_shutdown:
            print(f"\n--- 停止 Server ---")
            _kill_server(server_proc)


# ============================================================================
# Engine 模式 (保留, 进阶用途)
# ============================================================================

def run_engine_profiling(args) -> None:
    """Engine 模式: monkey-patch Scheduler.run_batch 在 TP0 注入 profiler。"""
    try:
        from .config import DATA_DIR, SCHEDULE_JSONL_FILENAME
        from .nsys_profiler import (
            build_prompts,
            build_server_args_from_config,
            find_case_by_match,
            load_case_by_id,
        )
    except ImportError:
        print("错误: Engine 模式需要通过 python3 -m refactor_test_aic.torch_profiler 运行")
        sys.exit(1)

    if args.custom_prefix_len is not None and args.custom_extend_len is not None:
        prefix_len = args.custom_prefix_len
        extend_len = args.custom_extend_len
        tid = args.custom_token_id
        prefix_prompts = [[tid] * prefix_len] if prefix_len > 0 else [[tid]]
        full_prompts = [[tid] * prefix_len + [tid + 1] * extend_len]
        batch_type = "custom"
        request_infos = [{"input_length": extend_len, "past_kv_length": prefix_len}]
        print(f"[engine] 自定义: prefix_len={prefix_len}, extend_len={extend_len}")
    else:
        schedule_jsonl = args.schedule_jsonl or os.path.join(
            args.data_dir, SCHEDULE_JSONL_FILENAME
        )
        if args.case_id is not None:
            batch_type, request_infos = load_case_by_id(schedule_jsonl, args.case_id)
        elif args.batch_size is not None and args.seq_len is not None:
            case_id, rec = find_case_by_match(
                schedule_jsonl, batch_size=args.batch_size, seq_len=args.seq_len,
                forward_mode=args.forward_mode, match_index=args.match_index,
            )
            batch_type, request_infos = load_case_by_id(schedule_jsonl, case_id)
        else:
            print("错误: 需要 --custom-prefix-len + --custom-extend-len")
            sys.exit(1)
        if not request_infos:
            raise ValueError("Empty request_infos.")
        prefix_prompts, full_prompts = build_prompts(request_infos, token_id=args.token_id)

    for i, ri in enumerate(request_infos):
        print(f"  req[{i}] input_len={ri['input_length']} past_kv_len={ri['past_kv_length']}")

    if args.dry_run:
        return

    output_dir = os.path.abspath(args.output_dir)
    profiler_kwargs = {
        "record_shapes": args.record_shapes, "with_stack": args.with_stack,
        "with_flops": getattr(args, "with_flops", False),
        "with_modules": getattr(args, "with_modules", False),
        "profile_memory": True,
    }
    _install_profiling_hook(output_dir, profiler_kwargs)

    from sglang.srt.entrypoints.engine import Engine
    from dataclasses import asdict
    server_args = build_server_args_from_config(
        load_format=getattr(args, "load_format", "dummy"),
        override_model_path=getattr(args, "model", None),
    )
    llm = Engine(**asdict(server_args))
    sampling_params = {"temperature": 0, "top_p": 1, "max_new_tokens": getattr(args, "max_new_tokens", 2)}

    _ = llm.generate(input_ids=prefix_prompts,
                     sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 1})
    for w in range(getattr(args, "warmup_runs", 1)):
        _ = llm.generate(input_ids=full_prompts, sampling_params=sampling_params)

    expected_forwards = 1 + getattr(args, "max_new_tokens", 2)
    _start_profiling_signal(output_dir, expected_forwards)
    llm.generate(input_ids=full_prompts, sampling_params=sampling_params)
    _wait_profiling_done(output_dir, timeout=getattr(args, "profiling_timeout", 120.0))
    llm.shutdown()


# Engine 模式内部函数 (保留原有 monkey-patch 机制)

def _install_profiling_hook(output_dir: str, profiler_kwargs: dict):
    os.environ["_TORCH_PROFILE_DIR"] = os.path.abspath(output_dir)
    os.environ["_TORCH_PROFILE_KWARGS"] = json.dumps(profiler_kwargs)
    os.makedirs(output_dir, exist_ok=True)
    for fname in [".start_profiling", ".profiling_done"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    from sglang.srt.managers.scheduler import Scheduler
    _orig = Scheduler.run_batch

    def _patched(self, batch):
        tp_rank = getattr(self, "tp_rank", -1)
        pdir = os.environ.get("_TORCH_PROFILE_DIR", "")
        if tp_rank != 0 or not pdir:
            return _orig(self, batch)
        if not hasattr(self, "_tps"):
            self._tps = {"prof": None, "rem": 0, "active": False}
        st = self._tps
        sig = os.path.join(pdir, ".start_profiling")
        if not st["active"] and os.path.exists(sig):
            with open(sig) as f:
                n = int(f.read().strip())
            os.remove(sig)
            import torch
            from torch.profiler import ProfilerActivity, profile
            kw = json.loads(os.environ.get("_TORCH_PROFILE_KWARGS", "{}"))
            st["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **kw)
            st["prof"].start()
            st["rem"] = n
            st["active"] = True
        result = _orig(self, batch)
        if st["active"] and batch is not None:
            st["rem"] -= 1
            if st["rem"] <= 0:
                st["prof"].stop()
                st["prof"].export_chrome_trace(os.path.join(pdir, "trace.json"))
                tbl = st["prof"].key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=200)
                with open(os.path.join(pdir, "key_averages.txt"), "w") as f:
                    f.write(tbl)
                with open(os.path.join(pdir, ".profiling_done"), "w") as f:
                    f.write("done")
                st["active"] = False
                st["prof"] = None
        return result

    Scheduler.run_batch = _patched


def _start_profiling_signal(output_dir: str, n: int):
    with open(os.path.join(output_dir, ".start_profiling"), "w") as f:
        f.write(str(n))


def _wait_profiling_done(output_dir: str, timeout: float = 60.0) -> bool:
    done = os.path.join(output_dir, ".profiling_done")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(done):
            os.remove(done)
            return True
        time.sleep(0.1)
    return False


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="SGLang Torch Profiler — 基于 /start_profile API 的性能采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  # ===== 基础用法 (Server 已启动) =====

  # prefill profiling (input=128 tokens):
  python3 torch_profiler.py --input-len 128 --output-len 32

  # 带 prefix cache:
  python3 torch_profiler.py --input-len 64 --prefix-len 512 --output-len 32

  # 多请求:
  python3 torch_profiler.py --input-len 128 --output-len 64 --num-requests 4

  # 只 profile decode:
  python3 torch_profiler.py --input-len 32 --output-len 128 \\
    --profile-stages DECODE --num-steps 10

  # ===== 自动启动 Server (快速测试) =====

  # 2层快速 profile (所有 sglang 参数直接透传):
  python3 torch_profiler.py --launch-server \\
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \\
    --num-layers 2 \\
    --input-len 128 --output-len 64 \\
    --tp-size 8 --ep-size 8 --trust-remote-code \\
    --moe-a2a-backend deepep

  # piecewise cuda graph profile:
  python3 torch_profiler.py --launch-server \\
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \\
    --num-layers 2 \\
    --input-len 128 --output-len 64 \\
    --tp-size 8 --ep-size 8 --trust-remote-code \\
    --enable-piecewise-cuda-graph --disable-cuda-graph-padding \\
    --piecewise-cuda-graph-tokens 2 3 4 5 6 7 8

  # ===== Engine 模式 (进阶) =====

  python3 -m refactor_test_aic.torch_profiler --mode engine \\
    --custom-prefix-len 100 --custom-extend-len 10
""",
    )

    # 模式
    ap.add_argument("--mode", default="server", choices=["server", "engine"])

    # --- Profiling 参数 ---
    prof = ap.add_argument_group("Profiling 参数")
    prof.add_argument("--input-len", type=int, default=128,
                      help="输入 token 长度 (默认: 128)")
    prof.add_argument("--prefix-len", type=int, default=0,
                      help="prefix cache 长度 (默认: 0)")
    prof.add_argument("--output-len", type=int, default=64,
                      help="最大输出 token 数 (默认: 64)")
    prof.add_argument("--num-requests", type=int, default=1,
                      help="并发请求数 (默认: 1)")
    prof.add_argument("--num-steps", type=int, default=5,
                      help="每阶段采集 forward step 数 (默认: 5)")
    prof.add_argument("--profile-by-stage", action="store_true", default=True,
                      help="分别采集 prefill/decode trace (默认: True)")
    prof.add_argument("--no-profile-by-stage", dest="profile_by_stage",
                      action="store_false")
    prof.add_argument("--profile-stages", type=str, default=None,
                      help="只 profile 指定阶段, 逗号分隔 (如 EXTEND,DECODE)")
    prof.add_argument("--with-stack", action="store_true", default=True,
                      help="记录 Python 调用栈 (默认: True)")
    prof.add_argument("--no-with-stack", dest="with_stack",
                      action="store_false", help="禁用 Python 调用栈")
    prof.add_argument("--record-shapes", action="store_true", default=True)
    prof.add_argument("--warmup-runs", type=int, default=1)
    prof.add_argument("--output-dir", type=str, default="/tmp/sglang_profile",
                      help="trace 输出目录 (默认: /tmp/sglang_profile)")
    prof.add_argument("--port", type=int, default=30000)

    # --- Server 启动参数 (--launch-server 时使用) ---
    srv = ap.add_argument_group("Server 启动参数 (配合 --launch-server)")
    srv.add_argument("--launch-server", action="store_true",
                     help="自动启动 SGLang server")
    srv.add_argument("--model-path", type=str, default=None,
                     help="模型路径 (也可通过 extra sglang args 传入)")
    srv.add_argument("--num-layers", type=int, default=None,
                     help="覆盖 num_hidden_layers (减少层数加速 profiling)")
    srv.add_argument("--launch-timeout", type=float, default=600,
                     help="等待 server 启动超时 (秒, 默认: 600)")
    srv.add_argument("--auto-shutdown", action="store_true", default=True,
                     help="profiling 完成后自动停止 server (默认: True)")
    srv.add_argument("--no-auto-shutdown", dest="auto_shutdown",
                     action="store_false")

    # --- Engine 模式参数 ---
    eng = ap.add_argument_group("Engine 模式参数")
    eng.add_argument("--custom-prefix-len", type=int, default=None)
    eng.add_argument("--custom-extend-len", type=int, default=None)
    eng.add_argument("--custom-token-id", type=int, default=10000)
    eng.add_argument("--schedule-jsonl", type=str, default=None)
    eng.add_argument("--data-dir", type=str, default=None)
    eng.add_argument("--case-id", type=int, default=None)
    eng.add_argument("--batch-size", type=int, default=None)
    eng.add_argument("--seq-len", type=int, default=None)
    eng.add_argument("--forward-mode", type=int, default=1)
    eng.add_argument("--match-index", type=int, default=0)
    eng.add_argument("--model", type=str, default=None)
    eng.add_argument("--load-format", type=str, default="dummy")
    eng.add_argument("--token-id", type=int, default=325)
    eng.add_argument("--max-new-tokens", type=int, default=2)
    eng.add_argument("--with-flops", action="store_true")
    eng.add_argument("--with-modules", action="store_true")
    eng.add_argument("--profiling-timeout", type=float, default=120.0)

    # 通用
    ap.add_argument("--dry-run", action="store_true")

    args, extra_sglang_args = ap.parse_known_args()

    # 将 extra args 存储到 args 上, 便于打印
    args._extra_sglang_args = extra_sglang_args

    if args.mode == "server":
        if args.launch_server:
            has_model = args.model_path or "--model-path" in extra_sglang_args
            if not has_model:
                print("错误: --launch-server 需要 --model-path (可在 profiler 参数或 sglang extra args 中指定)")
                sys.exit(1)
        if extra_sglang_args and not args.launch_server:
            print(f"警告: 检测到额外 sglang 参数 {extra_sglang_args}, 但未指定 --launch-server, 这些参数将被忽略")
        run_profiling(args, extra_sglang_args if args.launch_server else None)
    else:
        if extra_sglang_args:
            print(f"警告: Engine 模式不支持额外 sglang 参数, 忽略: {extra_sglang_args}")
        run_engine_profiling(args)


if __name__ == "__main__":
    main()
