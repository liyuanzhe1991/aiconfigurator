import json
from typing import Optional
from sglang_simulator.simulation.benchmark import (
    BenchmarkConfig as SGLangSimulatorBenchmarkConfig,
)
from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner as SGLangSimulatorBenchmarkRunner,
)
from sglang_simulator.dataset import SimpleDataset, GenericRequest

from sglang.srt.server_args import ServerArgs
from hisim.simulation.types import BenchmarkConfig
from hisim.dataset import BaseDataset, DatasetArgs


class SGLangSimulatorRunner:
    def __init__(self, server_args: ServerArgs):
        self._runner: SGLangSimulatorBenchmarkRunner = SGLangSimulatorBenchmarkRunner(
            server_args
        )

    def benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Optional[BaseDataset] = None,
        dataset_args: Optional[DatasetArgs] = None,
    ):
        assert dataset_args is not None, (
            "Current simulation support `dataset_args` only."
        )

        sim_benchmark_config = SGLangSimulatorBenchmarkConfig(
            request_rate=benchmark_config.request_rate,
            max_concurrency=benchmark_config.max_concurrency,
            ignore_request_timestamp=benchmark_config.ignore_request_timestamp,
        )
        dataset = self._load_hisim_collection(dataset_args.filepath)
        return self._runner.benchmark(sim_benchmark_config, dataset)

    def flush_cache(self):
        self._runner.flush_cache()

    def reset_storage_cache(self):
        self._runner.clear_hicache_storage()

    def shutdown(self):
        self._runner.shutdown()

    def _load_hisim_collection(self, filepath: str) -> SimpleDataset:

        dataset = SimpleDataset()

        with open(filepath) as f:
            line = f.readline()
            min_ts = float("inf")
            while line:
                req = json.loads(line)
                ts = req.get("timestamp") or req.get("created_time")
                min_ts = min(min_ts, ts)
                dataset.add_request(
                    GenericRequest(
                        token_ids=req["input_ids"],
                        input_length=req["input_length"],
                        output_length=req["output_length"],
                        custom_params={"created_time": ts},
                    )
                )
                line = f.readline()
            for req in dataset.data:
                req.custom_params["created_time"] -= min_ts

        return dataset

    def get_iteration_stats(self):
        return self._runner.get_iteration_stats()

    def get_request_stats(self):
        return self._runner.get_request_stats()
