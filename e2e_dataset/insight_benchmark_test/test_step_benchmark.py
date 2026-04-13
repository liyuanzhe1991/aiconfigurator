from insight_benchmark.step_benchmark import (
    StepBenchmarkRuntimeArgs,
    StepBenchmarkConfigs,
)
from insight_benchmark.step_benchmark.step_benchmark import StepBenchmark
from insight_benchmark.analyze import (
    StepBenchmarkThroughputAnalyzer,
    StepBenchmarkPrefixCacheAnalyzer,
    StepBenchmarkSlaAnalyzer,
    SLAConfig,
)
import pytest

from env import MODEL_PATH, check_framework


@pytest.mark.skipif(not check_framework("sglang"), reason="sglang is not installed")
@pytest.mark.skipif(True, reason="Hook is invalid in pytest environment.")
def test_step_benchmark_sglang():
    if not check_framework("sglang"):
        return

    runtime_args = StepBenchmarkRuntimeArgs(
        strategy="prefix_cache",
        dataset_name="random",
        batch_sizes=[2, 4],
        input_lengths=[128, 256],
        output_lengths=[8],
        prefix_hit_rate=0.8,
        num_replay=5,
    )

    bench = StepBenchmark(
        StepBenchmarkConfigs(
            backend="sglang",
            backend_args={
                "model_path": MODEL_PATH,
                "disable_overlap_schedule": True,
                "disable_cuda_graph": True,
                "load_format": "dummy",
                "mem_fraction_static": 0.5,
            },
            runtime_args=runtime_args,
        )
    )

    runner_config = bench.get_runner_configs()
    assert runner_config is not None

    metrics = bench.benchmark()
    assert len(metrics) != 0
    total_reused_tokens = sum([item.mean_reused_tokens for item in metrics])
    assert total_reused_tokens > 0

    analyzer = StepBenchmarkThroughputAnalyzer(
        runtime_args=runtime_args, benchmark_metrics=metrics
    )
    result = analyzer.run()
    assert "message" in result

    analyzer = StepBenchmarkPrefixCacheAnalyzer(
        runtime_args=runtime_args,
        benchmark_metrics=metrics,
    )
    result = analyzer.run()
    assert "message" in result

    analyzer = StepBenchmarkSlaAnalyzer(
        runtime_args=runtime_args,
        benchmark_metrics=metrics,
        sla_config=SLAConfig(
            max_ttft_ms=1000,
            max_tpot_ms=50,
            mean_input_length=1024,
        ),
    )
    result = analyzer.run()
    assert "message" in result


@pytest.mark.skipif(
    not check_framework("tensorrt_llm"), reason="tensorrt_llm is not installed"
)
@pytest.mark.skipif(True, reason="Hook is invalid in pytest environment.")
def test_step_benchmark_trtllm():
    if not check_framework("tensorrt_llm"):
        return

    runtime_args = StepBenchmarkRuntimeArgs(
        strategy="normal",
        dataset_name="random",
        batch_sizes=[4, 8],
        input_lengths=[8, 16],
        output_lengths=[8, 16],
    )

    bench = StepBenchmark(
        StepBenchmarkConfigs(
            backend="trtllm",
            backend_args={
                "model": MODEL_PATH,
                "backend": "pytorch",
            },
            runtime_args=runtime_args,
        )
    )

    metrics = bench.benchmark()
    assert len(metrics) != 0

    analyzer = StepBenchmarkThroughputAnalyzer(
        runtime_args=runtime_args, benchmark_metrics=metrics
    )
    result = analyzer.run()
    assert "message" in result


if __name__ == "__main__":
    test_step_benchmark_sglang()
    test_step_benchmark_trtllm()
