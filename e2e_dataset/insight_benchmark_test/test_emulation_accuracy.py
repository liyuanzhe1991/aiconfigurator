import os
import json
import pandas as pd
import pytest

from insight_benchmark.simulation.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
)
from insight_benchmark.simulation.infer_time_predictor import (
    StepBenchmarkTimePredictor,
)
from insight_benchmark.simulation.schedule_emulator.run import run_benchmark_emulation
from kunlun_commons.hardwares.accelerator import AcceleratorInfo
from kunlun_commons.model_info import ModelInfo


@pytest.mark.skipif(False, reason="Accuracy validation is time costly")
def test_emulation_accuracy():
    real_benchmark_data = []
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "assets/qwen3_32b/sglang_bench_serving_4090_4tp.jsonl",
        )
    ) as f:
        real_benchmark_data.extend([json.loads(line) for line in f.readlines()])

    real_benchmark_data = list(
        filter(lambda x: x["max_concurrency"] < 10, real_benchmark_data)
    )

    model = ModelInfo.find_by_model_name("Qwen3-32B")
    device = AcceleratorInfo.find_by_hw_name("RTX-4090")
    scheduler_config = SchedulerConfig(model, tp_size=4)
    platform_config = PlatformConfig(device=device)

    metrics_names = [
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
    ]

    mpe_accuracy = []

    for real_metrics in real_benchmark_data:
        max_inlen = real_metrics["random_input_len"]
        min_inlen = (
            real_metrics["random_input_len"] * real_metrics["random_range_ratio"]
        )
        max_outlen = real_metrics["random_output_len"]
        min_outlen = (
            real_metrics["random_output_len"] * real_metrics["random_range_ratio"]
        )

        benchmark_config = BenchmarkConfig(
            max_concurrency=real_metrics["max_concurrency"],
            num_prompts=real_metrics["completed"],
            min_input_length=min_inlen,
            max_input_length=max_inlen,
            min_output_length=min_outlen,
            max_output_length=max_outlen,
            request_rate=real_metrics["request_rate"],
        )

        emulated_metrics = run_benchmark_emulation(
            benchmark_config=benchmark_config,
            scheduler_config=scheduler_config,
            platform_config=platform_config,
            infer_time_predictor=StepBenchmarkTimePredictor(
                model=model,
                hw=device,
                config=scheduler_config,
                database_path=os.path.join(
                    os.path.dirname(__file__),
                    "assets/qwen3_32b/sglang_step_benchmark_4090_4tp.csv",
                ),
            ),
        )

        mpe = {
            "concurrency": real_metrics["max_concurrency"],
            "mean_inlen": (min_inlen + max_inlen) // 2,
            "mean_outlen": (min_outlen + max_outlen) // 2,
        }

        for m in metrics_names:
            mpe[m] = (emulated_metrics[m] - real_metrics[m]) / real_metrics[m]

        mpe_accuracy.append(mpe)

    df = pd.DataFrame(mpe_accuracy).round(4)
    df.sort_values(by=["concurrency", "mean_inlen", "mean_outlen"], inplace=True)
    # df.to_csv("mpe_accuracy.csv", index=False)
    print(df)


if __name__ == "__main__":
    test_emulation_accuracy()
