# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profile MLA prefill and decode for BF16/FP8 KV cache with NVTX annotations.

Mirrors collect_mla_1_1rc2.py (the actual collector for TRT-LLM >= 1.1.0)
so that the kernel code-path is identical to what runs in production.

Usage:
    nsys profile \\
        --trace=cuda,nvtx \\
        --capture-range=cudaProfilerApi \\
        --capture-range-end=stop \\
        -o mla_profile \\
        python profile_mla.py

Output:
    mla_profile.nsys-rep  -- open with Nsight Systems UI

NVTX markers in the report:
    PROFILE_context_bf16       -- Prefill, BF16 KV cache
    PROFILE_context_fp8        -- Prefill, FP8 KV cache
    PROFILE_generation_bf16    -- Decode,  BF16 KV cache
    PROFILE_generation_fp8     -- Decode,  FP8 KV cache
"""

import math
import os
import sys

# Allow running directly from collector/trtllm/ — add parent dir to path
# so that `from helper import ...` resolves to collector/helper.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

from helper import benchmark_with_power


# ── DeepSeek V3 MLA constants (matches collect_mla_1_1rc2.py Scenario) ──────
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
HIDDEN_SIZE = 7168
TOKENS_PER_BLOCK = 64

ROPE_CONFIG_DICT = {
    "hidden_size": HIDDEN_SIZE,
    "num_attention_heads": 128,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "max_position_embeddings": 163840,
    "rope_theta": 10000.0,
    "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
    "model_type": "deepseek_v3",
}


class _RopeConfig:
    """Minimal config object for RopeParams.from_config()."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def profile_mla(
    input_len, batch_size, output_len, kv_cache_dtype, num_heads,
    tp_size, is_context_phase, warming_up, profile_iters,
    label, device="cuda:0",
):
    """
    Profile a single MLA configuration.
    Uses the same API as collect_mla_1_1rc2.py for identical kernel behaviour.
    """
    device = torch.device(device)
    torch.cuda.set_device(device)

    dtype = torch.bfloat16
    qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

    assert num_heads % tp_size == 0
    num_heads = num_heads // tp_size
    num_kv_heads = num_heads

    context_sequence_lengths = [input_len] * batch_size
    num_generation_steps = 0 if is_context_phase else 1

    # ── Attention backend (same as collect_mla_1_1rc2.py) ───────────────
    attention_cls = get_attention_backend("TRTLLM")

    rope_config = _RopeConfig(ROPE_CONFIG_DICT)
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        predicted_tokens_per_seq=1,
    )

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    # quant_config: None for BF16, QuantConfig for FP8
    # (collect_mla_1_1rc2.py passes literal None for BF16)
    quant_config = None
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    if is_context_phase:
        attn_mla = attention_cls(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=qk_head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        )
    else:
        attn_mla = attention_cls(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            num_kv_heads=1,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        )

    # ── KV cache (same as collect_mla_1_1rc2.py) ───────────────────────
    world_size = tp_size
    mapping = Mapping(world_size=world_size, tp_size=tp_size, rank=0)

    max_context_sequence_length = max(context_sequence_lengths)
    max_num_contexts = len(context_sequence_lengths)

    max_tokens = (
        (
            max_context_sequence_length
            + (num_generation_steps + 1) * output_len
            + TOKENS_PER_BLOCK
            - 1
        )
        // TOKENS_PER_BLOCK
        * TOKENS_PER_BLOCK
        * max_num_contexts
    )

    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_context_sequence_length + (num_generation_steps + 1) * output_len,
        max_batch_size=max_num_contexts,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    # Use LlmRequest + add_sequence (same as collect_mla_1_1rc2.py)
    for req_id, ctx_len in enumerate(context_sequence_lengths):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=num_generation_steps + 1,
            input_tokens=[1] * ctx_len,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.paged_kv_block_ids = []
        kv_cache_manager.impl.add_sequence(req_id, ctx_len, 1, req)

    # ── Metadata (matches collect_mla_1_1rc2.py exactly) ────────────────
    attn_metadata = attention_cls.Metadata(
        seq_lens=torch.tensor(context_sequence_lengths, dtype=torch.int),
        request_ids=list(range(max_num_contexts)),
        max_num_requests=max_num_contexts,
        num_contexts=max_num_contexts,
        prompt_lens=context_sequence_lengths,
        max_num_tokens=max_context_sequence_length,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * max_num_contexts,
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()

    if not is_context_phase:
        for req_id in range(max_num_contexts):
            for _ in range(output_len):
                kv_cache_manager.impl.add_token(req_id)
        attn_metadata = attention_cls.Metadata(
            seq_lens=torch.tensor([output_len] * max_num_contexts, dtype=torch.int),
            request_ids=list(range(max_num_contexts)),
            max_num_requests=max_num_contexts,
            num_contexts=0,
            prompt_lens=context_sequence_lengths,
            max_num_tokens=max_context_sequence_length,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=list(context_sequence_lengths),
            ),
            mapping=mapping,
            enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
        )
        attn_metadata.prepare()

    # ── Input tensors (matches collect_mla_1_1rc2.py) ───────────────────
    if is_context_phase:
        ctx_len = input_len
        num_tokens = ctx_len * max_num_contexts

        compressed_kv = torch.randn([num_tokens, KV_LORA_RANK], dtype=dtype, device=device)
        k_pe = torch.randn([num_tokens, QK_ROPE_HEAD_DIM], dtype=dtype, device=device)

        q = torch.randn([num_tokens, num_heads * qk_head_dim], dtype=dtype, device=device)
        ctx_kv = torch.randn(
            [num_tokens, num_kv_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)],
            dtype=dtype, device=device,
        )
        ctx_k_nope, v = ctx_kv.split(
            [num_kv_heads * QK_NOPE_HEAD_DIM, num_kv_heads * V_HEAD_DIM], dim=-1
        )
        ctx_k_nope = ctx_k_nope.view(-1, num_kv_heads, QK_NOPE_HEAD_DIM)
        k = torch.cat(
            [ctx_k_nope, k_pe.view(-1, 1, QK_ROPE_HEAD_DIM).expand(-1, num_kv_heads, -1)],
            dim=-1,
        ).view(-1, num_kv_heads * qk_head_dim)

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

        # Dry run (same as collect_mla_1_1rc2.py L448-455)
        attn_mla.forward(
            q, k, v, attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
        )
    else:
        num_tokens = output_len * max_num_contexts

        compressed_kv = torch.randn([num_tokens, KV_LORA_RANK], dtype=dtype, device=device)
        k_pe = torch.randn([num_tokens, QK_ROPE_HEAD_DIM], dtype=dtype, device=device)

        fused_q = torch.randn(
            [num_tokens, num_heads * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)],
            dtype=dtype, device=device,
        )
        q_pe = torch.randn(
            [num_tokens, num_heads, QK_ROPE_HEAD_DIM],
            dtype=dtype, device=device,
        )

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

        # Generation setup (same as collect_mla_1_1rc2.py L487-542)
        num_seqs = max_num_contexts
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)

        has_fp8_kv_cache = attn_mla.has_fp8_kv_cache if hasattr(attn_mla, "has_fp8_kv_cache") else False
        if has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
            quant_q_buffer = torch.empty(
                num_tokens, num_heads * (KV_LORA_RANK + QK_ROPE_HEAD_DIM),
                dtype=torch.uint8, device=device,
            )
        else:
            mla_bmm1_scale = None
            mla_bmm2_scale = None
            quant_q_buffer = None

        # Dry run: mla_rope_generation + forward
        attn_mla.mla_rope_generation(
            fused_q, q_pe, latent_cache, attn_metadata,
            cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter,
            mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer,
        )
        attn_mla.forward(
            fused_q, None, None, attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
        )

    print(f"  [{label}] dry run succeeded")

    # ── Kernel function (matches collect_mla_1_1rc2.py kernel_func) ─────
    def kernel_func():
        if is_context_phase:
            attn_mla.forward(
                q, k, v, attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
            )
        else:
            attn_mla.mla_rope_generation(
                fused_q, q_pe, latent_cache, attn_metadata,
                cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter,
                mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer,
            )
            attn_mla.forward(
                fused_q, None, None, attn_metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                quant_q_buffer=quant_q_buffer,
            )

    # ── Profiled benchmark ──────────────────────────────────────────────
    torch.cuda.nvtx.range_push(f"PROFILE_{label}")

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=profile_iters,
        repeat_n=1,
        allow_graph_fail=True,
    ) as results:
        pass

    torch.cuda.nvtx.range_pop()

    latency = results["latency_ms"]
    used_graph = results.get("used_cuda_graph", "unknown")
    print(f"  [{label}] latency = {latency:.4f} ms  (cuda_graph={used_graph})")

    kv_cache_manager.shutdown()


def main():
    INPUT_LEN = 8192
    OUTPUT_LEN = 1      # context test: output_len=1 (matches collect_mla_1_1rc2.py)
    BATCH_SIZE = 1
    NUM_HEADS = 128
    TP_SIZE = 1
    DEVICE = "cuda:0"
    WARMUP_ITERS = 10
    PROFILE_ITERS = 6

    BF16 = tensorrt_llm.bindings.DataType.BF16
    FP8 = tensorrt_llm.bindings.DataType.FP8

    configs = [
        # (kv_cache_dtype, is_context_phase, output_len, label)
        (BF16, True,  1,   "context_bf16"),
        (FP8,  True,  1,   "context_fp8"),
        (BF16, False, 1,   "generation_bf16"),
        (FP8,  False, 1,   "generation_fp8"),
    ]

    torch.cuda.cudart().cudaProfilerStart()

    for kv_dtype, is_ctx, osl, label in configs:
        phase = "prefill" if is_ctx else "decode"
        dtype_name = "BF16" if kv_dtype == BF16 else "FP8"
        print(f"\n{'='*60}")
        print(f"[{label}] {phase} | KV={dtype_name} | "
              f"ISL={INPUT_LEN} OSL={osl} BS={BATCH_SIZE}")
        print(f"{'='*60}")

        try:
            profile_mla(
                input_len=INPUT_LEN, batch_size=BATCH_SIZE, output_len=osl,
                kv_cache_dtype=kv_dtype, num_heads=NUM_HEADS, tp_size=TP_SIZE,
                is_context_phase=is_ctx, warming_up=WARMUP_ITERS,
                profile_iters=PROFILE_ITERS, label=label, device=DEVICE,
            )
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            torch.cuda.empty_cache()

    torch.cuda.cudart().cudaProfilerStop()

    print(f"\n{'='*60}")
    print("All configurations profiled. Latencies printed above.")
    print("To generate nsys report, run:")
    print("  nsys profile --trace=cuda,nvtx "
          "--capture-range=cudaProfilerApi --capture-range-end=stop "
          "-o mla_profile python profile_mla.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
