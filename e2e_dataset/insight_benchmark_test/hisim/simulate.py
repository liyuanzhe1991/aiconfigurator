import os
from pathlib import Path
import argparse
import json
import time
import config
import metrics_handle

from insight_benchmark.simulation.schedule_emulator.types import BenchmarkConfig
from insight_benchmark.dataset import DatasetArgs
# from sgl_serving_runner import SGLangServingRunner
from sgl_simulator_runner import SGLangSimulatorRunner


os.environ["HISIM_LOG_LEVEL"] = "INFO"

cur_dir = Path(__file__).parent
DEFAULT_OUTPUT_DIR = cur_dir / "output"


MODEL_ID_MAP = {
    "Qwen3-30B-A3B-FP8": "Qwen/Qwen3-30B-A3B-FP8",
    "Qwen3-32B-FP8": "Qwen/Qwen3-32B-FP8",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    # "Qwen3-8B": "/home/linsiyuan.lsy/models/Qwen3-8B/",
    "GLM-5-FP8": "zai-org/GLM-5-FP8",
}


def run_simulation(
    model_bench_dir: Path,
    output_dir: Path,
    predictor_db_path: str,
    predictor_name: str = "aiconfigurator",
    predictor_extra_config: dict = {},
    case_list=["idle"],
    runner_name: str = "serving",
    device_name: str = "h20_sxm",
    backend_version: str | None = None,
    model_path: str | None = None,
):

    if model_path is None:
        for k, v in MODEL_ID_MAP.items():
            if k in str(model_bench_dir):
                model_path = v
                break
    if model_path is None:
        raise "Fail to parse model name from directory."

    # Get server info
    with open(
        model_bench_dir
        / "data"
        / os.listdir(model_bench_dir / "data")[0]
        / f"{case_list[0]}.metrics.json"
    ) as f:
        real_metrics = json.load(f)
        max_total_num_tokens = real_metrics["server_info"]["max_total_num_tokens"]
        page_size = real_metrics["server_info"]["page_size"]
        enable_hicache = real_metrics["server_info"]["enable_hierarchical_cache"]
        hicache_storage_backend = real_metrics["server_info"]["hicache_storage_backend"]
        hicache_storage_prefetch_policy = real_metrics["server_info"][
            "hicache_storage_prefetch_policy"
        ]
        disable_overlap_schedule = real_metrics["server_info"][
            "disable_overlap_schedule"
        ]
        chunked_prefill_size = real_metrics["server_info"]["chunked_prefill_size"]

        # Parse backend version if it's not set.
        if backend_version is None:
            backend_version = real_metrics["server_info"]["version"]

    server_config = {
        "model_path": model_path,
        "max_total_tokens": max_total_num_tokens,
        "page_size": page_size,
        "chunked_prefill_size": chunked_prefill_size,
        "enable_hierarchical_cache": enable_hicache,
        "disable_overlap_schedule": disable_overlap_schedule,
        "hicache_storage_backend": hicache_storage_backend,
        "hicache_storage_prefetch_policy": hicache_storage_prefetch_policy,
        "disable_cuda_graph": True,
        "load_format": "dummy",
        "decode_attention_backend": "fa3",
    }

    schedule_config = config.get_schedule_config_from_sglang_server_info(
        real_metrics["server_info"], backend_version=backend_version
    )

    platform_config = config.get_platform_config(device_name=device_name)

    predictor_config = config.get_prefictor_config(
        predictor_name=predictor_name,
        db_path=predictor_db_path,
        extra_config=predictor_extra_config,
        device_name=device_name,
    )

    hisim_config = {
        "platform": platform_config,
        "predictor": predictor_config,
        "scheduler": schedule_config,
    }

    config_path = output_dir / "hisim_config.json"
    with open(config_path, "w") as f:
        json.dump(hisim_config, f, indent=4)

    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(config_path)

    if runner_name == "serving":
        runner = SGLangServingRunner(server_config)
    elif runner_name == "offline":
        from sglang.srt.server_args import ServerArgs

        runner = SGLangSimulatorRunner(ServerArgs(**server_config))
    else:
        raise ValueError(f"Unknown runner: {runner_name}")

    for request_rate_dir in model_bench_dir.joinpath("data").iterdir():
        print("Running: ", request_rate_dir)
        # hang up???
        # runner.flush_cache()
        for case_name in case_list:
            f_path = request_rate_dir / f"{case_name}.requests.jsonl"

            if case_name in ["l3", "L3", "disk", "Disk"]:
                runner.flush_cache()
                print("sleep 1s......")
                time.sleep(1)

            metrics = runner.benchmark(
                BenchmarkConfig(request_rate=float(request_rate_dir.name)),
                dataset_args=DatasetArgs(name="hisim-collection", filepath=f_path),
            )

            (output_dir / request_rate_dir.name).mkdir(exist_ok=True)

            with open(
                output_dir
                / request_rate_dir.name
                / f"{predictor_name}.{case_name}.metrics.json",
                "w",
            ) as f:
                json.dump(metrics, f)

            if hasattr(runner, "get_iteration_stats"):
                with open(
                    output_dir
                    / request_rate_dir.name
                    / f"{predictor_name}.{case_name}.iteration.jsonl",
                    "w",
                ) as f:
                    data = runner.get_iteration_stats()
                    for item in data:
                        f.write(json.dumps(item) + "\n")

            if hasattr(runner, "get_request_stats"):
                with open(
                    output_dir
                    / request_rate_dir.name
                    / f"{predictor_name}.{case_name}.request.jsonl",
                    "w",
                ) as f:
                    data = runner.get_request_stats()
                    for item in data:
                        f.write(json.dumps(item) + "\n")

            # internal_metrics = metrics_handle.calc_real_metrics_from_requests_stats(
            #     f_path
            # )
            # with open(
            #     output_dir
            #     / request_rate_dir.name
            #     / f"internal.{case_name}.metrics.json",
            #     "w",
            # ) as f:
            #     json.dump(internal_metrics, f)

            if case_name in ["l3", "L3", "disk", "Disk"]:
                runner.flush_cache()
                runner.reset_storage_cache()
                print("sleep 1s......")
                time.sleep(1)

        runner.flush_cache()

    runner.shutdown()


def main(
    data_dir: str | Path,
    runner: str = "offline",
    predictors: list[str] = ["aiconfigurator"],
    case_list: list[str] = ["no_cache", "l1"],
    output_dir: str | None = None,
    predictor_db_path: str | None = None,
    predictor_extra_config: dict = {},
    device_name: str = "h100_sxm",
    backend_version: str | None = None,
    model_path: str | None = None,
):

    if output_dir is None:
        version, model = data_dir.parts[-2], data_dir.parts[-1]
        output_dir = DEFAULT_OUTPUT_DIR / version / model
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for predictor_name in predictors:
        if predictor_db_path is None:
            print("Warning: predictor database directory is not set.")

        run_simulation(
            model_bench_dir=data_dir,
            output_dir=output_dir,
            predictor_name=predictor_name,
            predictor_db_path=predictor_db_path,
            predictor_extra_config=predictor_extra_config,
            case_list=case_list,
            runner_name=runner,
            device_name=device_name,
            backend_version=backend_version,
            model_path=model_path,
        )

    df = metrics_handle.merge_metrics(
        real_data_dir=data_dir / "data",
        sim_data_dir=output_dir,
        case_list=case_list,
        predictors=predictors,
    )
    df.round(4).to_csv(output_dir / "result.csv", index=False)

    metrics_handle.show_metrics(
        output_dir / "result.csv",
        title=data_dir.name,
        cases=case_list,
        output_file=output_dir / "metrics.png",
    )
    metrics_handle.cal_metrics_mpe(output_dir / "result.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The common serving benchmark data can be found in `33.254.38.20:/apsarapangu/disk2/projects/kunlun_insight/data/serving`",
    )
    parser.add_argument(
        "--predictor-db-path",
        type=str,
        help="The common predictor data can be found in `33.254.38.20:/apsarapangu/disk2/projects/kunlun_insight/data/`",
    )
    parser.add_argument(
        "--predictor-extra-config",
        type=json.loads,
        default={},
        help="Extra config for predictor in json format, e.g. '{\"prefill_scale_factor\": 1.2}'",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="sgl_simulator_offline",
        choices=["hisim_offline", "hisim_serving", "sgl_simulator_offline"],
    )
    parser.add_argument("--cases", type=str, nargs="+", default=["no_cache", "l1"])
    parser.add_argument("--predictors", type=str, nargs="+", default=["aiconfigurator"])
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()

    main(
        data_dir=Path(args.data_dir),
        runner=args.runner,
        predictors=args.predictors,
        case_list=args.cases,
        output_dir=args.output_dir,
        predictor_db_path=args.predictor_db_path,
    )
