import time
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
    DisaggBenchmarkRunner,
)


random.seed(0)
numpy.random.seed(0)


def test_benchmark_emulation():
    start = time.time()

    benchmark_config = BenchmarkConfig(
        num_prompts=100,
        min_input_length=900,
        max_input_length=1100,
        min_output_length=250,
        max_output_length=350,
        max_concurrency=20,
        min_prefix_disk_hit_rate=0.3,
        max_prefix_disk_hit_rate=0.5,
        min_prefix_host_hit_rate=0.2,
        max_prefix_host_hit_rate=0.3,
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

    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
    )

    # metrics = insight_benchmark.simulation.schedule_emulator.run.run_benchmark_emulation(...)
    for _ in range(2):
        # Run multiple benchmarks with a single runner to keep the radix cache
        metrics = runner.run_benchmark_emulation(benchmark_config)
        iter_stats = runner.get_iteration_stats()
        request_cache_fetch_stats = runner.get_request_cache_fetch_stats()
        iteration_cache_fetch_stats = runner.get_iteration_cache_fetch_stats()
        response_results = runner.get_response_results()

        assert len(metrics) != 0
        assert len(iter_stats) != 0
        assert len(request_cache_fetch_stats) != 0
        assert len(iteration_cache_fetch_stats) != 0
        assert all(
            [
                len(stats.request_stats) <= scheduler_config.max_running_requests
                for stats in iter_stats
            ]
        )
        assert len(response_results) == benchmark_config.num_prompts
        end = time.time()

        print(metrics)
        print(f"Emulator Running Time: {end - start}s")

        # test with dataset from file
        benchmark_config.dataset_path = os.path.join(
            os.path.dirname(__file__), "assets/dataset/prefix_cache_requests.jsonl"
        )
        benchmark_config.num_prompts = 2
        scheduler_config.schedule_policy = "fcfs"
        runner = BenchmarkRunner(
            benchmark_config=benchmark_config,
            scheduler_config=scheduler_config,
            platform_config=platform_config,
        )

    # metrics = insight_benchmark.simulation.schedule_emulator.run.run_benchmark_emulation(...)
    metrics = runner.run_benchmark_emulation(benchmark_config)
    assert len(metrics) != 0


def test_disagg_emulation():
    benchmark_config = BenchmarkConfig(
        num_prompts=100,
        min_input_length=900,
        max_input_length=1100,
        min_output_length=250,
        max_output_length=350,
        max_concurrency=20,
    )

    p_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B", enable_stats=True, scenario="disagg_prefill"
    )
    d_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B", enable_stats=True, scenario="disagg_decode"
    )
    p_platform_config = PlatformConfig(
        device="A100-SXM4-80GB",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )
    d_platform_config = PlatformConfig(
        device="H20",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )

    runner = DisaggBenchmarkRunner(
        benchmark_config=benchmark_config,
        p_scheduler_config=p_scheduler_config,
        d_scheduler_config=d_scheduler_config,
        p_platform_config=p_platform_config,
        d_platform_config=d_platform_config,
        num_p_instance=2,
        num_d_instance=5,
    )

    metrics = runner.run_benchmark_emulation()
    assert len(metrics) != 0

    reps = runner.get_response_results()
    assert len(reps) != 0

    stats = runner.get_request_cache_fetch_stats()
    assert len(stats) != 0 and len(stats[0]) != 0


if __name__ == "__main__":
    test_benchmark_emulation()
    test_disagg_emulation()
