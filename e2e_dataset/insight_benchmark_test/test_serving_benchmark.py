from insight_benchmark.serving_benchmark import (
    ServingBenchmark,
    ServingBenchmarkRuntimeArgs,
    ServingBenchmarkConfigs,
)
from insight_benchmark.analyze import ServingBenchmarkSlaAnalyzer, SLAConfig
from insight_benchmark.dataset import DatasetArgs
import subprocess
import pytest

from env import MODEL_PATH, check_framework


@pytest.mark.skipif(not check_framework("sglang"), reason="sglang is not installed.")
def test_serving_benchmark_sglang():
    sglang_server_proc = subprocess.Popen(
        f"python3 -m sglang.launch_server --port 8000 --model-path {MODEL_PATH} --mem-fraction-static=0.5",
        shell=True,
    )

    bench_runtime_args = ServingBenchmarkRuntimeArgs(
        strategy="sla",
        sla_max_ttft_ms=180,
        sla_max_tpot_ms=50,
        max_concurrency=8,
    )

    dataset_args = DatasetArgs(
        name="random",
        num_prompts=100,
        min_input_len=16,
        max_input_len=32,
        min_output_len=8,
        max_output_len=16,
    )

    benchmark_configs = ServingBenchmarkConfigs(
        backend="sglang-oai-chat",
        host="127.0.0.1",
        port=8000,
        model_path=MODEL_PATH,
        request_rate=float("+inf"),
        disable_tqdm=False,
        dataset_args=dataset_args,
        runtime_args=bench_runtime_args,
    )

    bench = ServingBenchmark(benchmark_configs)

    metrics = bench.benchmark()
    assert len(metrics) > 0

    analyzer = ServingBenchmarkSlaAnalyzer(
        sla_config=SLAConfig(
            max_ttft_ms=bench_runtime_args.sla_max_ttft_ms,
            max_tpot_ms=bench_runtime_args.sla_max_tpot_ms,
            mean_input_length=(dataset_args.max_input_len + dataset_args.min_input_len)
            // 2,
            mean_output_length=(
                dataset_args.max_output_len + dataset_args.min_output_len
            )
            // 2,
        ),
        benchmark_metrics=metrics,
    )

    result = analyzer.run()
    assert "message" in result

    sglang_server_proc.kill()
    subprocess.run("pkill -9 -f sglang.launch_server", shell=True)


if __name__ == "__main__":
    test_serving_benchmark_sglang()
