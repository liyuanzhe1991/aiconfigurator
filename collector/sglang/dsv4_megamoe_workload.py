# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload helpers for DeepSeek-V4 MegaMoE collection.

This module intentionally reuses AIC's existing MoE distribution helpers instead
of reimplementing power-law or balanced routing.  It adds only the source-rank
placement layer needed by real EP collection.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    from collector.helper import balanced_logits, power_law_logits_v3
except ImportError:
    from helper import balanced_logits, power_law_logits_v3


SUPPORTED_DISTRIBUTIONS = ("balanced", "power_law_1.01", "power_law_1.2")
SUPPORTED_SOURCE_POLICIES = ("random",)


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    kind: str
    alpha: float | None = None


@dataclass(frozen=True)
class RoutingPlan:
    distribution: str
    source_policy: str
    global_num_tokens: int
    tokens_per_rank: tuple[int, ...]
    routed_num_experts: int
    routed_topk: int
    ep_size: int
    rank: int
    local_topk_ids: torch.Tensor
    local_topk_weights: torch.Tensor
    routed_expert_counts: tuple[int, ...]
    dst_rank_loads: tuple[int, ...]
    src_dst_matrix: tuple[tuple[int, ...], ...]
    local_selection_ratio: float
    remote_selection_ratio: float
    bottleneck_rank: int
    routing_seed: int
    norm_topk_prob: bool

    def metadata(self) -> dict[str, object]:
        return {
            "distribution": self.distribution,
            "source_policy": self.source_policy,
            "global_num_tokens": self.global_num_tokens,
            "tokens_per_rank": json.dumps(list(self.tokens_per_rank), separators=(",", ":")),
            "routing_seed": self.routing_seed,
            "rank_loads": json.dumps(list(self.dst_rank_loads), separators=(",", ":")),
            "src_dst_matrix": json.dumps([list(row) for row in self.src_dst_matrix], separators=(",", ":")),
            "local_selection_ratio": f"{self.local_selection_ratio:.6f}",
            "remote_selection_ratio": f"{self.remote_selection_ratio:.6f}",
            "bottleneck_rank": self.bottleneck_rank,
            "norm_topk_prob": str(self.norm_topk_prob).lower(),
        }


def parse_distribution(distribution: str) -> DistributionSpec:
    if distribution == "balanced":
        return DistributionSpec(name=distribution, kind="balanced")
    if distribution.startswith("power_law_"):
        alpha = float(distribution.removeprefix("power_law_"))
        if alpha not in (1.01, 1.2):
            raise ValueError(f"unsupported power-law alpha: {alpha}")
        return DistributionSpec(name=distribution, kind="power_law", alpha=alpha)
    raise ValueError(f"unsupported distribution: {distribution}; expected one of {SUPPORTED_DISTRIBUTIONS}")


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        if not value:
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _validate_common(
    *,
    tokens_per_rank: Sequence[int],
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
    rank: int,
) -> tuple[int, ...]:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive")
    if rank < 0 or rank >= ep_size:
        raise ValueError(f"rank {rank} must be within [0, {ep_size})")
    if routed_topk <= 0:
        raise ValueError("routed_topk must be positive")
    if routed_num_experts <= 0:
        raise ValueError("routed_num_experts must be positive")
    if routed_num_experts % ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by ep_size")
    normalized = tuple(int(item) for item in tokens_per_rank)
    if len(normalized) != ep_size:
        raise ValueError("tokens_per_rank length must equal ep_size")
    if any(item <= 0 for item in normalized):
        raise ValueError("all tokens_per_rank entries must be positive")
    return normalized


def _logits_for_distribution(
    *,
    spec: DistributionSpec,
    global_num_tokens: int,
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
) -> torch.Tensor:
    if spec.kind == "balanced":
        return balanced_logits(global_num_tokens, routed_num_experts, routed_topk).to(dtype=torch.float32, device="cpu")
    if spec.kind == "power_law":
        if spec.alpha is None:
            raise ValueError("power-law distribution requires alpha")
        return power_law_logits_v3(
            global_num_tokens,
            routed_num_experts,
            routed_topk,
            ep_size,
            spec.alpha,
            use_eplb=False,
            return_rank0_info=False,
        ).to(dtype=torch.float32, device="cpu")
    raise ValueError(f"unsupported distribution kind: {spec.kind}")


def _route_matrix(
    topk_ids_by_rank: Sequence[torch.Tensor],
    *,
    routed_num_experts: int,
    ep_size: int,
) -> list[list[int]]:
    experts_per_rank = routed_num_experts // ep_size
    matrix = [[0 for _ in range(ep_size)] for _ in range(ep_size)]
    for src_rank, topk_ids in enumerate(topk_ids_by_rank):
        owner = torch.div(topk_ids.to(dtype=torch.int64), experts_per_rank, rounding_mode="floor")
        if torch.any(owner < 0) or torch.any(owner >= ep_size):
            raise ValueError("topk_ids contain expert ids outside routed expert range")
        counts = torch.bincount(owner.reshape(-1), minlength=ep_size).tolist()
        matrix[src_rank] = [int(value) for value in counts[:ep_size]]
    return matrix


def _validate_plan(
    *,
    topk_ids_by_rank: Sequence[torch.Tensor],
    topk_weights_by_rank: Sequence[torch.Tensor],
    routed_num_experts: int,
    routed_topk: int,
    tokens_per_rank: Sequence[int],
    expected_expert_counts: torch.Tensor,
) -> None:
    if len(topk_ids_by_rank) != len(tokens_per_rank):
        raise ValueError("rank count mismatch")
    for rank, (topk_ids, topk_weights, tokens) in enumerate(
        zip(topk_ids_by_rank, topk_weights_by_rank, tokens_per_rank, strict=True)
    ):
        if tuple(topk_ids.shape) != (tokens, routed_topk):
            raise ValueError(f"rank {rank} topk_ids shape mismatch: {tuple(topk_ids.shape)}")
        if tuple(topk_weights.shape) != (tokens, routed_topk):
            raise ValueError(f"rank {rank} topk_weights shape mismatch: {tuple(topk_weights.shape)}")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_ids must be integer")
        if torch.any(topk_ids < 0) or torch.any(topk_ids >= routed_num_experts):
            raise ValueError("topk_ids contain invalid routed expert ids")
        sorted_ids = torch.sort(topk_ids.to(dtype=torch.int64), dim=1).values
        if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("a token row contains duplicate expert ids")

    merged = torch.cat([item.reshape(-1).to(dtype=torch.int64) for item in topk_ids_by_rank])
    actual_counts = torch.bincount(merged, minlength=routed_num_experts).to(dtype=torch.int64)
    if not torch.equal(actual_counts[:routed_num_experts], expected_expert_counts.to(dtype=torch.int64)):
        raise ValueError("expert counts changed while assigning source ranks")


def build_routing_plan(
    *,
    distribution: str,
    tokens_per_rank: Sequence[int],
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
    rank: int,
    source_policy: str = "random",
    routing_seed: int = 0,
    norm_topk_prob: bool = True,
) -> RoutingPlan:
    """Build a local routing plan from an AIC distribution helper.

    ``distribution`` controls destination expert load.  ``source_policy`` controls
    source-rank placement of the generated token rows.

    ``random`` shuffles whole token rows before assigning them to source ranks,
    preserving both expert counts and per-token top-k structure.

    """
    tokens_per_rank = _validate_common(
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=rank,
    )
    if source_policy not in SUPPORTED_SOURCE_POLICIES:
        raise ValueError(
            f"unsupported source_policy: {source_policy}; expected one of {SUPPORTED_SOURCE_POLICIES}"
        )

    spec = parse_distribution(distribution)
    global_num_tokens = int(sum(tokens_per_rank))
    logits = _logits_for_distribution(
        spec=spec,
        global_num_tokens=global_num_tokens,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
    )
    topk_weights, topk_ids = torch.topk(logits, k=routed_topk, dim=-1, largest=True, sorted=False)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    topk_ids = topk_ids.to(dtype=torch.int32, device="cpu").contiguous()
    topk_weights = topk_weights.to(dtype=torch.float32, device="cpu").contiguous()
    expected_expert_counts = torch.bincount(
        topk_ids.reshape(-1).to(dtype=torch.int64),
        minlength=routed_num_experts,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(routing_seed))
    permutation = torch.randperm(global_num_tokens, generator=generator)
    topk_ids = topk_ids[permutation].contiguous()
    topk_weights = topk_weights[permutation].contiguous()

    topk_ids_by_rank = []
    topk_weights_by_rank = []
    offset = 0
    for tokens in tokens_per_rank:
        end = offset + tokens
        topk_ids_by_rank.append(topk_ids[offset:end].contiguous())
        topk_weights_by_rank.append(topk_weights[offset:end].contiguous())
        offset = end

    _validate_plan(
        topk_ids_by_rank=topk_ids_by_rank,
        topk_weights_by_rank=topk_weights_by_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        tokens_per_rank=tokens_per_rank,
        expected_expert_counts=expected_expert_counts[:routed_num_experts],
    )

    matrix = _route_matrix(topk_ids_by_rank, routed_num_experts=routed_num_experts, ep_size=ep_size)
    experts_per_rank = routed_num_experts // ep_size
    dst_rank_loads = tuple(
        int(expected_expert_counts[dst * experts_per_rank : (dst + 1) * experts_per_rank].sum().item())
        for dst in range(ep_size)
    )
    local_selections = sum(matrix[src][src] for src in range(ep_size))
    total_selections = global_num_tokens * routed_topk
    local_ratio = local_selections / total_selections if total_selections else 0.0

    return RoutingPlan(
        distribution=distribution,
        source_policy=source_policy,
        global_num_tokens=global_num_tokens,
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=int(routed_num_experts),
        routed_topk=int(routed_topk),
        ep_size=int(ep_size),
        rank=int(rank),
        local_topk_ids=topk_ids_by_rank[rank],
        local_topk_weights=topk_weights_by_rank[rank],
        routed_expert_counts=tuple(int(value) for value in expected_expert_counts[:routed_num_experts].tolist()),
        dst_rank_loads=dst_rank_loads,
        src_dst_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        local_selection_ratio=float(local_ratio),
        remote_selection_ratio=float(1.0 - local_ratio),
        bottleneck_rank=int(max(range(ep_size), key=lambda idx: dst_rank_loads[idx])),
        routing_seed=int(routing_seed),
        norm_topk_prob=bool(norm_topk_prob),
    )


def append_fused_shared_experts(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    routed_num_experts: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append SGLang-style fused shared expert slots to a routed top-k plan."""
    if num_fused_shared_experts == 0:
        return topk_ids.contiguous(), topk_weights.contiguous()
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")
    if routed_scaling_factor <= 0:
        raise ValueError("routed_scaling_factor must be positive")

    num_tokens = topk_ids.shape[0]
    shared_ids = torch.arange(
        routed_num_experts,
        routed_num_experts + num_fused_shared_experts,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    ).expand(num_tokens, num_fused_shared_experts)
    routed_sum = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    normalized_routed_weights = topk_weights / routed_sum
    shared_weights = torch.full(
        (num_tokens, num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    return (
        torch.cat([topk_ids, shared_ids], dim=1).contiguous(),
        torch.cat([normalized_routed_weights, shared_weights], dim=1).contiguous(),
    )
