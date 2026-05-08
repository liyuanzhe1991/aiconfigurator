# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

_saved_mock = None
if isinstance(sys.modules.get("torch"), MagicMock):
    _saved_mock = sys.modules.pop("torch")

try:
    import torch
except ImportError:
    if _saved_mock is not None:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required for tensor operations", allow_module_level=True)

from collector.sglang.dsv4_megamoe_workload import build_routing_plan


@pytest.mark.unit
def test_random_routing_plan_is_deterministic_and_preserves_counts():
    ep_size = 4
    tokens_per_rank = [16] * ep_size
    routed_num_experts = 16
    routed_topk = 2

    plan = build_routing_plan(
        distribution="balanced",
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=0,
        source_policy="random",
        routing_seed=123,
    )
    plan_again = build_routing_plan(
        distribution="balanced",
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=0,
        source_policy="random",
        routing_seed=123,
    )

    assert plan.source_policy == "random"
    assert tuple(plan.local_topk_ids.shape) == (tokens_per_rank[0], routed_topk)
    assert tuple(plan.local_topk_weights.shape) == (tokens_per_rank[0], routed_topk)
    assert torch.equal(plan.local_topk_ids, plan_again.local_topk_ids)
    assert torch.equal(plan.local_topk_weights, plan_again.local_topk_weights)
    assert sum(plan.routed_expert_counts) == sum(tokens_per_rank) * routed_topk
    assert sum(plan.dst_rank_loads) == sum(tokens_per_rank) * routed_topk
    for src_rank, row in enumerate(plan.src_dst_matrix):
        assert sum(row) == tokens_per_rank[src_rank] * routed_topk
    assert plan.local_selection_ratio + plan.remote_selection_ratio == pytest.approx(1.0)


@pytest.mark.unit
def test_unknown_source_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported source_policy"):
        build_routing_plan(
            distribution="power_law_1.01",
            tokens_per_rank=[16] * 4,
            routed_num_experts=16,
            routed_topk=2,
            ep_size=4,
            rank=0,
            source_policy="unsupported",
        )
