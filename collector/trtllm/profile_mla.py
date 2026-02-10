# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profile MLA prefill and decode for BF16/FP8 KV cache with NVTX annotations.

Uses the same flow as collect_mla.py (create_attention + benchmark_with_power)
for accurate kernel-level profiling of DeepSeek MLA attention.

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
    PROFILE_generation_fp8     -- Decode,  FP8 KV cache (FP8 attention compute)
"""

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import AttentionInputType, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig, QuantAlgo

# ── DeepSeek V3 MLA constants ───────────────────────────────────────────────
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
TOKENS_PER_BLOCK = 64


def profile_mla(
    input_len, batch_size, output_len, kv_cache_dtype, num_heads,
    tp_size, is_context_phase, warming_up, profile_iters,
    label, device="cuda:0",
):
    """
    Profile a single MLA configuration.
    Uses CUDA Graph capture + replay with NVTX markers, matching collect_mla.py.
    """
    device = torch.device(device)
    torch.cuda.set_device(device)

    num_key_value_heads = num_heads
    assert num_key_value_heads % tp_size == 0
    num_key_value_heads = num_key_value_heads // tp_size
    assert num_heads % tp_size == 0
    num_heads = num_heads // tp_size

    world_size = tp_size
    has_fp8_kv = kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        embedder=None,
        rope=RopeParams(
            dim=56, theta=10000,
            scale_type=RotaryScalingType.yarn, scale=40,
            low_freq_factor=1.0, high_freq_factor=4.0,
            short_m_scale=1.0, long_m_scale=1.0,
            max_positions=163840, original_max_positions=4096,
            beta_fast=32, beta_slow=1,
            mscale=1.0, mscale_all_dim=1.0,
        ),
    )

    # quant_algo='FP8_BLOCK_SCALES' must only be set for FP8 KV cache;
    # setting it for BF16 causes illegal memory access on SM100 (GB200).
    quant_config = QuantConfig(
        quant_algo='FP8_BLOCK_SCALES' if has_fp8_kv else None,
        kv_cache_quant_algo=QuantAlgo.FP8 if has_fp8_kv else None,
        group_size=None, smoothquant_val=0.5, clamp_val=None,
        use_meta_recipe=False, has_zero_point=False,
        pre_quant_scale=False, exclude_modules=None,
    )

    attn = create_attention(
        backend_name="TRTLLM", layer_idx=0, num_heads=num_heads,
        head_dim=((QK_NOPE_HEAD_DIM if is_context_phase else KV_LORA_RANK)
                  + QK_ROPE_HEAD_DIM),
        num_kv_heads=num_key_value_heads if is_context_phase else 1,
        pos_embd_params=pos_embd_params, quant_config=quant_config,
        is_mla_enable=True,
        q_lora_rank=Q_LORA_RANK, kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM, qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM if is_context_phase else KV_LORA_RANK,
        predicted_tokens_per_seq=1,
    )

    # ── KV cache setup ───────────────────────────────────────────────────
    total_num_tokens = (input_len + output_len) * batch_size
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp_size)

    kv_cache_config = KvCacheConfig(
        max_tokens=(int((input_len + output_len - 1) / TOKENS_PER_BLOCK + 1)
                    * TOKENS_PER_BLOCK * batch_size * 2),
        enable_block_reuse=False,
    )

    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1, num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=input_len + output_len,
        max_batch_size=batch_size,
        mapping=mapping, dtype=kv_cache_dtype,
    )

    input_seq_lens = [input_len] * batch_size
    total_seq_lens = [input_len + output_len] * batch_size
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    # On SM100 (GB200), Flash MLA context path causes illegal memory access.
    sm_version = torch.cuda.get_device_capability()
    enable_flash_mla = sm_version == (9, 0)

    # Pre-allocate workspace to avoid dynamic resize crash on some platforms.
    workspace = torch.tensor([], device=device, dtype=torch.int8)
    if is_context_phase and input_len * batch_size > 1024:
        workspace_bytes = 2 * 1024 * 1024 * 1024  # 2 GB
        workspace = torch.empty(workspace_bytes, device=device, dtype=torch.int8)

    # ── Attention metadata ───────────────────────────────────────────────
    if is_context_phase:
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size, max_num_tokens=total_num_tokens,
            kv_cache_manager=kv_cache_manager, mapping=mapping,
            enable_flash_mla=enable_flash_mla,
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32),
            position_ids=None, num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0] * batch_size,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            cross=None, request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=workspace,
        )
    else:
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size, max_num_tokens=total_num_tokens,
            kv_cache_manager=kv_cache_manager, mapping=mapping,
            enable_flash_mla=enable_flash_mla,
            seq_lens=torch.tensor([1] * batch_size, dtype=torch.int32),
            position_ids=None, num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=input_seq_lens,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            cross=None, request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=workspace,
        )

    attn_metadata.prepare()

    # ── Input tensors ────────────────────────────────────────────────────
    num_tokens = input_len * batch_size if is_context_phase else batch_size

    compressed_kv = torch.randn([num_tokens, KV_LORA_RANK], dtype=torch.bfloat16, device=device)
    k_pe = torch.randn([num_tokens, QK_ROPE_HEAD_DIM], dtype=torch.bfloat16, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

    if is_context_phase:
        q = torch.randn(
            [num_tokens, num_heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)],
            dtype=torch.bfloat16, device=device,
        )
        k = torch.randn(
            [num_tokens, num_key_value_heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)],
            dtype=torch.bfloat16, device=device,
        )
        v = torch.randn(
            [num_tokens, num_key_value_heads * V_HEAD_DIM],
            dtype=torch.bfloat16, device=device,
        )
        q_pe = None
    else:
        fused_q = torch.randn(
            [num_tokens, num_heads * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)],
            dtype=torch.bfloat16, device=device,
        )
        q_pe = torch.randn(
            [num_tokens, num_heads, QK_ROPE_HEAD_DIM],
            dtype=torch.bfloat16, device=device,
        )

    # ── Generation parameters ────────────────────────────────────────────
    cu_q_seqlens = None
    cu_kv_seqlens = None
    fmha_scheduler_counter = None
    mla_bmm1_scale = None
    mla_bmm2_scale = None
    quant_q_buffer = None

    if not is_context_phase:
        cu_q_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)

        if has_fp8_kv:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
            quant_q_buffer = torch.empty(
                num_tokens, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                dtype=torch.uint8, device=device,
            )

        fused_q_3d = fused_q.view(num_tokens, num_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        attn.mla_rope_generation(
            fused_q_3d, q_pe, latent_cache, attn_metadata,
            cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter,
            mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer,
        )

    # ── Helper: single forward call ──────────────────────────────────────
    def _forward():
        if is_context_phase:
            attn.forward(
                q, k, v, attn_metadata,
                out_scale=None,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
            )
        else:
            attn.forward(
                fused_q, None, None, attn_metadata,
                out_scale=None,
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

    # ── Step 1: dry run ──────────────────────────────────────────────────
    _forward()
    torch.cuda.synchronize()
    print(f"  [{label}] dry run succeeded")

    # ── Step 2: CUDA Graph capture ───────────────────────────────────────
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _forward()

    # ── Step 3: warmup ───────────────────────────────────────────────────
    for _ in range(warming_up):
        g.replay()

    # ── Step 4: profiled replays with NVTX ───────────────────────────────
    torch.cuda.nvtx.range_push(f"PROFILE_{label}")
    for i in range(profile_iters):
        torch.cuda.nvtx.range_push(f"{label}/iter{i}")
        g.replay()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ── Timing ───────────────────────────────────────────────────────────
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(profile_iters):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event) / profile_iters
    print(f"  [{label}] latency = {latency:.4f} ms")


def main():
    INPUT_LEN = 8192
    OUTPUT_LEN = 1024
    BATCH_SIZE = 1
    NUM_HEADS = 128
    TP_SIZE = 1
    DEVICE = "cuda:0"
    WARMUP_ITERS = 10
    PROFILE_ITERS = 6

    BF16 = tensorrt_llm.bindings.DataType.BF16
    FP8 = tensorrt_llm.bindings.DataType.FP8

    configs = [
        # (kv_cache_dtype, is_context_phase, label)
        (BF16, True,  "context_bf16"),
        (FP8,  True,  "context_fp8"),
        (BF16, False, "generation_bf16"),
        (FP8,  False, "generation_fp8"),
    ]

    torch.cuda.cudart().cudaProfilerStart()

    for kv_dtype, is_ctx, label in configs:
        phase = "prefill" if is_ctx else "decode"
        dtype_name = "BF16" if kv_dtype == BF16 else "FP8"
        print(f"\n{'='*60}")
        print(f"[{label}] {phase} | KV={dtype_name} | "
              f"ISL={INPUT_LEN} OSL={OUTPUT_LEN} BS={BATCH_SIZE}")
        print(f"{'='*60}")

        try:
            profile_mla(
                input_len=INPUT_LEN, batch_size=BATCH_SIZE, output_len=OUTPUT_LEN,
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
