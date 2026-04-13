from kunlun_commons.hardwares.accelerator import AcceleratorInfo
from kunlun_commons.model_info import ModelInfo

from insight_benchmark.simulation.schedule_emulator.types import (
    PlatformConfig,
    FakeRequest,
    SchedulerConfig,
)
from insight_benchmark.simulation.schedule_emulator.base import GlobalValues
from insight_benchmark.simulation.schedule_emulator.schedule_policy import (
    SchedulePolicy,
)
from insight_benchmark.simulation.schedule_emulator.prefix_cache import HiRadixCache

from insight_benchmark.simulation.infer_time_predictor import (
    InferTimePredictor,
)


class SimpleInferTimePredictor(InferTimePredictor):
    def __init__(self, model, hw, config):
        super().__init__(model, hw, config)

    def predict_infer_time(self, batch):
        # Each token requires 3 time unit to compute.
        return batch.num_context_tokens * 3


def test_insight_policy_demo():
    hw = AcceleratorInfo.find_by_hw_name("H20")
    model = ModelInfo.find_by_model_name("Qwen3-8B")
    scheduler_config = SchedulerConfig(model)

    time_predictor = SimpleInferTimePredictor(model, hw, scheduler_config)

    platform_config = PlatformConfig(
        hw,
        disk_read_bandwidth_gb=1e-9,  # Each token requires 1 time unit to retrieve.
        memory_read_bandwidth_gb=float("+inf"),
    )

    waiting_queue = [
        FakeRequest(0, 10, 8, disk_cache_hit_length=1),
        FakeRequest(1, 10, 8, disk_cache_hit_length=2),
        FakeRequest(2, 10, 8, disk_cache_hit_length=2),
        FakeRequest(3, 10, 8, disk_cache_hit_length=2),
        FakeRequest(4, 10, 8, disk_cache_hit_length=0),
        FakeRequest(5, 10, 8, disk_cache_hit_length=2),
    ]
    tree_cache = HiRadixCache(platform_config, 1, 1, GlobalValues())
    for req in waiting_queue:
        tree_cache.add_to_prefetch_queue(req)

    policy = SchedulePolicy("plg", tree_cache, time_predictor)
    policy.calc_priority(waiting_queue)
    priority_req_ids = [req.id for req in waiting_queue]
    assert priority_req_ids == [3, 4, 5, 0, 1, 2]

    # recover original order
    waiting_queue.sort(key=lambda req: req.id)
    tree_cache.reset()
    for req in waiting_queue:
        tree_cache.add_to_prefetch_queue(req)
    policy = SchedulePolicy("mcr", tree_cache, time_predictor)
    policy.calc_priority(waiting_queue)
    priority_req_ids = [req.id for req in waiting_queue]
    assert priority_req_ids == [4, 0, 1, 2, 3, 5]


if __name__ == "__main__":
    test_insight_policy_demo()
