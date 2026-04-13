import os

os.environ["KUNLUN_INSIGHT_BENCHMARK_LOG_LEVEL"] = "DEBUG"

import json
from pathlib import Path
import pandas as pd
from insight_benchmark.simulation.internal_hisim.sglang.runner import (
    SGLangBenchmarkRunner,
)
from insight_benchmark.simulation.infer_time_predictor import (
    ScheduleReplayTimePredictor,
)
from insight_benchmark.simulation.schedule_emulator.types import (
    BenchmarkConfig,
)
from insight_benchmark.dataset import DatasetArgs
from sglang.srt.server_args import ServerArgs

cur_dir = Path(__file__).parent


def validate_hicache_replay(
    model_path: str, data_dir: Path, case_list=["no_cache", "l1"]
) -> list[dict]:
    data_path_list = []
    for case_name in case_list:
        data_path_list.append(f"{str(data_dir / f'{case_name}.schedule_batch.jsonl')}")

    mock_config = {
        "platform": {
            "accelerator": {"name": "H20"},
            "disk_read_bandwidth_gb": 0.5,
            "disk_write_bandwidth_gb": 0.5,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "schedule_replay",
            "database_path": ",".join(data_path_list),
        },
        "scheduler": {
            "tp_size": 1,
            "data_type": "FP16",
            "kv_cache_data_type": "FP16",
            "backend_version": "0.5.6.post2",
        },
    }

    config_path = "/tmp/mock.config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)

    os.environ["HISIM_CONFIG_PATH"] = str(config_path)
    with open(data_dir / "no_cache.metrics.json") as f:
        real_metrics = json.load(f)
        max_total_num_tokens = real_metrics["server_info"]["max_total_num_tokens"]
        page_size = real_metrics["server_info"]["page_size"]
        enable_hicache = real_metrics["server_info"]["enable_hierarchical_cache"]
        disable_overlap_schedule = real_metrics["server_info"][
            "disable_overlap_schedule"
        ]

    server_args = ServerArgs(
        model_path=model_path,
        enable_hierarchical_cache=enable_hicache,
        disable_cuda_graph=True,
        disable_overlap_schedule=disable_overlap_schedule,
        page_size=page_size,
        decode_attention_backend="fa3",
        max_total_tokens=max_total_num_tokens,
        load_format="dummy",
    )

    runner = SGLangBenchmarkRunner(server_args)

    result = []

    for case_name in case_list:
        with open(data_dir / f"{case_name}.metrics.json") as f:
            real_metrics = json.load(f)

        sim_metrics = runner.benchmark(
            BenchmarkConfig(with_queue_start=True),
            dataset_args=DatasetArgs(
                name="insight_hook",
                filepath=str(data_dir / f"{case_name}.requests.jsonl"),
            ),
        )

        sim_iteration_stats = runner.get_iteration_stats()
        sim_iteration_stats = [item["requests"] for item in sim_iteration_stats]

        predictor = ScheduleReplayTimePredictor(
            None,
            None,
            None,
            database_path=str(data_dir / f"{case_name}.schedule_batch.jsonl"),
        )
        real_iteration_stats = [item[0].request_info() for item in predictor.data]

        if sim_iteration_stats != real_iteration_stats:
            success = False
        else:
            success = True

        out_dir = cur_dir / f"tmp/replay/{data_dir.name}"
        out_dir.mkdir(exist_ok=True, parents=True)
        with open(out_dir / f"sim_{case_name}_iteration.json", "w") as f:
            for item in sim_iteration_stats:
                f.write(json.dumps(item) + "\n")

        with open(out_dir / f"real_{case_name}_iteration.json", "w") as f:
            for item in real_iteration_stats:
                f.write(json.dumps(item) + "\n")

        with open(out_dir / "sim_request.jsonl", "w") as f:
            for item in runner.get_request_stats():
                f.write(json.dumps(item) + "\n")

        with open(out_dir / "real_request.jsonl", "w") as f:
            data = []
            for line in open(data_dir / f"{case_name}.requests.jsonl").readlines():
                data.append(json.loads(line))
            sorted(data, key=lambda x: x["created_time"])
            start_ts = data[0]["created_time"]
            for item in data:
                item["created_time"] -= start_ts
                if "client_created_time" in item:
                    item["client_created_time"] -= start_ts
                item["server_created_time"] -= start_ts
                item["queue_start"] -= start_ts
                item["queue_end"] -= start_ts
                f.write(json.dumps(item) + "\n")

        result.append(
            {
                "case": case_name,
                "success": success,
                "dir": int(data_dir.name),
                "sim_mean_ttft_ms": sim_metrics["mean_ttft_ms"],
                "real_mean_ttft_ms": real_metrics["mean_ttft_ms"],
                "mape_mean_ttft_ms": (
                    sim_metrics["mean_ttft_ms"] - real_metrics["mean_ttft_ms"]
                )
                / real_metrics["mean_ttft_ms"],
            }
        )

    return result


def main():
    result = []
    root_dir = cur_dir / "data/GLM-5/serving/L1/GB300-GLM-5-FP8-GSP-No-Overlap/data"
    for item in root_dir.iterdir():
        result.extend(
            validate_hicache_replay(
                model_path="zai-org/GLM-5-FP8",
                # model_path="Qwen/Qwen3-8B",
                data_dir=item,
                case_list=["no_cache"],
            )
        )
    df = pd.DataFrame(result).sort_values(by="dir")
    df.round(4).to_csv(
        cur_dir / "tmp" / (root_dir.parent.name + ".hicache.replay.result.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
