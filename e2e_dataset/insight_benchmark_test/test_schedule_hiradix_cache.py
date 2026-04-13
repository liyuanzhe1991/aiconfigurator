import numpy
import random
import os

from insight_benchmark.simulation.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
)
from insight_benchmark.simulation.schedule_emulator.run import (
    BenchmarkRunner,
)


random.seed(0)
numpy.random.seed(0)


def test_scheduler_hiradix_cache():
    benchmark_config = BenchmarkConfig(
        num_prompts=8,
        dataset_path=os.path.join(
            os.path.dirname(__file__), "assets/dataset/multiturn_requests.jsonl"
        ),
    )

    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        chunked_prefill_size=1000,
        hicache_storage_backend="hf3fs",
        schedule_policy="fcfs",
        enable_stats=True,
        max_running_requests=10,
    )
    platform_config = PlatformConfig(
        # Memory bandwidth is shared through a common PCIe switch across multiple devices,
        # whereas disk read bandwidth is shared via the same network interface by all devices.
        device="H20",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )

    benchmark_config.num_prompts = 8
    scheduler_config.schedule_policy = "fcfs"
    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
        # radix cache
        use_real_token=True,
    )

    metrics = runner.run_benchmark_emulation()
    assert len(metrics) != 0


if __name__ == "__main__":
    test_scheduler_hiradix_cache()
