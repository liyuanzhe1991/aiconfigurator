#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect vLLM-realistic NCCL performance data for AIC.

Matches vLLM's actual pynccl call chains:
- AllGather:     grouped ncclAllGather via group_start/end (dispatch_router_logits)
- ReduceScatter: single ncclReduceScatter (combine)

Timing strategy:
- Grouped ops:  per-iteration torch.cuda.synchronize() + host timer
                (ncclGroupEnd doesn't sync back to user stream → CUDA events wrong)
- Single ops:   N-iteration CUDA events on user stream (ncclReduceScatter syncs
                properly → same approach as nccl-tests all_gather_perf)

Must run with torchrun:
    torchrun --nproc_per_node=8 collect_vllm_nccl.py [options]
"""

import os
import csv
import time
import argparse
from datetime import datetime

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
GLOO_GROUP = None
PYNCCL_COMM = None


def setup_distributed():
    global GLOO_GROUP, PYNCCL_COMM
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    GLOO_GROUP = dist.new_group(backend="gloo")
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        PYNCCL_COMM = PyNcclCommunicator(group=GLOO_GROUP, device=local_rank)
    except ImportError:
        PYNCCL_COMM = None
    return local_rank


# ---------------------------------------------------------------------------
# Timing: grouped ops (per-iteration sync)
# ---------------------------------------------------------------------------
def bench_grouped(fn, num_warmup=100, num_iter=500) -> float:
    """For grouped pynccl ops — per-iteration sync, median of 5 windows."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    latencies = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iter):
            fn()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) / num_iter * 1000)
    latencies.sort()
    return latencies[2]


# ---------------------------------------------------------------------------
# Timing: single pynccl ops — also needs per-iter sync
# ---------------------------------------------------------------------------
def bench_single(fn, num_warmup=100, num_iter=500) -> float:
    """Single pynccl ops also pipeline on NCCL internal streams.
    Must use per-iter sync, same as grouped ops."""
    return bench_grouped(fn, num_warmup, num_iter)


# ---------------------------------------------------------------------------
# Op 1: Grouped multi-tensor AllGather (vLLM dispatch_router_logits path)
# ---------------------------------------------------------------------------
def bench_grouped_allgather(tensor_specs, num_gpus):
    if PYNCCL_COMM is None:
        return -1.0
    sends = [torch.zeros(n, dtype=dt, device="cuda") for n, dt in tensor_specs]
    recvs = [torch.zeros(n * num_gpus, dtype=dt, device="cuda") for n, dt in tensor_specs]
    for t in sends:
        t.fill_(1)
    torch.cuda.synchronize()

    def fn():
        PYNCCL_COMM.group_start()
        for s, r in zip(sends, recvs):
            PYNCCL_COMM.all_gather(r, s)
        PYNCCL_COMM.group_end()

    return bench_grouped(fn)


# ---------------------------------------------------------------------------
# Op 2: Single ReduceScatter (vLLM combine path)
# ---------------------------------------------------------------------------
def bench_pynccl_reduce_scatter(total_numel, num_gpus):
    if PYNCCL_COMM is None:
        return -1.0
    per_rank = total_numel // num_gpus
    send = torch.zeros(total_numel, dtype=torch.bfloat16, device="cuda")
    recv = torch.zeros(per_rank, dtype=torch.bfloat16, device="cuda")
    send.fill_(1.0)
    torch.cuda.synchronize()

    def fn():
        PYNCCL_COMM.reduce_scatter(recv, send)

    return bench_single(fn)


# ---------------------------------------------------------------------------
# Op 3: Plain AllGather (nccl-tests equivalent, reference baseline)
# ---------------------------------------------------------------------------
def bench_plain_allgather(msg_bytes, dtype, num_gpus):
    if PYNCCL_COMM is None:
        return -1.0
    elem_size = torch.tensor([], dtype=dtype).element_size()
    per_rank = max(1, msg_bytes // elem_size)
    send = torch.zeros(per_rank, dtype=dtype, device="cuda")
    recv = torch.zeros(per_rank * num_gpus, dtype=dtype, device="cuda")
    send.fill_(1.0 if dtype != torch.uint8 else 1)
    torch.cuda.synchronize()

    def fn():
        PYNCCL_COMM.all_gather(recv, send)

    return bench_single(fn)


# ---------------------------------------------------------------------------
# Tensor specs
# ---------------------------------------------------------------------------
def make_tensor_specs(num_tokens, hidden_size, num_experts):
    """GLM5-style dispatch_router_logits tensors."""
    return [
        (num_tokens * (hidden_size // 2), torch.uint8),      # hidden_fp4
        (num_tokens * num_experts, torch.float32),            # router_logits
        (num_tokens * (hidden_size // 16), torch.uint8),      # blockscales
    ]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(rows, filename):
    if not rows:
        return
    fields = ["framework", "version", "device", "op_name", "kernel_source",
              "nccl_dtype", "num_gpus", "message_size", "latency"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect(args):
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    major, minor, patch = torch.cuda.nccl.version()
    nccl_ver = f"{major}.{minor}.{patch}"
    dev_name = torch.cuda.get_device_name()
    hidden_size = args.hidden_size
    num_experts = args.num_experts

    rows = []

    if rank == 0:
        print("=" * 80)
        print(f"vLLM NCCL Collector — {dev_name}, NCCL {nccl_ver}, {world_size} GPUs")
        print(f"  hidden_size={hidden_size}, num_experts={num_experts}")
        print(f"  pynccl: {'OK' if PYNCCL_COMM else 'MISSING'}")
        print(f"  Grouped ops: per-iter sync | Single ops: N-iter amortized")
        print("=" * 80)

    def add(op, ksrc, dname, elems, lat):
        rows.append({"framework": "vLLM", "version": nccl_ver, "device": dev_name,
                      "op_name": op, "kernel_source": ksrc, "nccl_dtype": dname,
                      "num_gpus": world_size, "message_size": elems,
                      "latency": f"{lat:.6f}"})

    # ── Part 1: Plain AllGather sweep (pynccl single, nccl-tests style) ──
    if rank == 0:
        print("\nPart 1: Plain AllGather sweep (pynccl single, N-iter amortized)")

    s = args.min_bytes
    while s <= args.max_bytes:
        lat = bench_plain_allgather(s, torch.float16, world_size)
        elems = max(1, s // 2)
        add("all_gather", "pynccl", "half", elems, lat)
        if rank == 0:
            print(f"  msg={s:>10}B  elems={elems:>10}  lat={lat*1000:>8.2f} us")
        s *= args.ratio

    # ── Part 2 & 3: Grouped AllGather + ReduceScatter ──
    phases = [("decode", [1, 2, 4, 8, 16, 32, 64]),
              ("prefill", [128, 256, 512, 1024, 2048, 4096, 8192])]

    for phase, token_counts in phases:
        if rank == 0:
            print(f"\n{'Decode' if phase == 'decode' else 'Prefill'}: "
                  f"Grouped AllGather (per-iter sync) + ReduceScatter (N-iter)")

        for nt in token_counts:
            specs = make_tensor_specs(nt, hidden_size, num_experts)
            # Grouped AllGather (per-iter sync — matches nsys)
            # message_size = nt * hidden_size * world_size (matches operations.py: volume * dp_size)
            lat_ag = bench_grouped_allgather(specs, world_size)
            add("vllm_dp_dispatch", "pynccl_grouped", "half", nt * hidden_size * world_size, lat_ag)

            # ReduceScatter via pynccl (per-iter sync)
            # message_size = nt * hidden_size * world_size (total elements, nccl-tests convention)
            rs_total = nt * hidden_size * world_size
            lat_rs = bench_pynccl_reduce_scatter(rs_total, world_size)
            add("vllm_dp_combine", "pynccl", "half", rs_total, lat_rs)

            if rank == 0:
                print(f"  [{phase:>7}] tokens={nt:>5}  ag={lat_ag*1000:>8.2f} us  "
                      f"rs={lat_rs*1000:>8.2f} us")

    # ── Save ──
    if rank == 0:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"vllm_nccl_perf_{ts}.csv")
        write_csv(rows, out_file)
        print(f"\nSaved {len(rows)} entries to {out_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-bytes", type=int, default=512)
    p.add_argument("--max-bytes", type=int, default=536870912)
    p.add_argument("--ratio", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--output-dir", default="/workspace/aic-alignment/experiments/allgather_bench/results")
    collect(p.parse_args())
