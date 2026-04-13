import os
import signal
import subprocess
import sys
import requests
import json
import time
from typing import Optional

from hisim.simulation.base.runner import BaseBenchmarkRunner
from hisim.simulation.types import BenchmarkConfig
from hisim.dataset import BaseDataset, DatasetArgs


class SGLangServingRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: dict):
        # If the process terminates unexpectedly, the serving runner will remain running.
        subprocess.run(
            args=["pkill", "-9", "-f", "insight_benchmark.simulation.internal_hisim"]
        )

        cmd = [
            sys.executable,
            "-m",
            "insight_benchmark.simulation.internal_hisim.sglang.launch_server",
        ]

        for k, v in server_args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            elif v is False:
                pass
            else:
                cmd.extend([flag, str(v)])

        self.server_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        dur = 0
        while dur < 120:
            try:
                r = requests.get(url="http://localhost:30000")
                if r.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(1)
            dur += 1
        raise RuntimeError("Fail to start llm server.")

    def benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Optional[BaseDataset] = None,
        dataset_args: Optional[DatasetArgs] = None,
    ):
        start = time.time()
        output_file = "/tmp/hisim_serving_benchmark.json"
        cmd = [
            sys.executable,
            "-m",
            "hisim.simulation.bench_serving",
            f"--request-rate={benchmark_config.request_rate}",
            "--warmup-request=0",
            "--bench-mode=simulation",
            "--backend=sglang",
            f"--dataset-name={dataset_args.name}",
            f"--dataset-path={dataset_args.filepath}",
            f"--output-file={output_file}",
        ]
        subprocess.run(cmd)

        with open(output_file) as f:
            metrics = json.load(f)

        with open(output_file, "w") as f:
            # clear data
            pass

        metrics["time_cost"] = time.time() - start

        return metrics

    def flush_cache(self):
        requests.get(url="http://localhost:30000/flush_cache")

    def reset_storage_cache(self):
        print("1 reset_storage_cache")
        requests.get(url="http://localhost:30000/clear_hicache_storage_backend")

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None
