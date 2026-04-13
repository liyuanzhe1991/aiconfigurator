import os
import json
import subprocess
import pytest

from env import check_framework, MODEL_PATH

cur_dir = os.path.dirname(__file__)


@pytest.mark.skipif(not check_framework("sglang"), reason="sglang is not installed.")
def test_serving_benchmark_sglang():
    custom_env = os.environ.copy()
    custom_env.update(
        {"HISIM_CONFIG_PATH": os.path.dirname(__file__) + "/assets/mock/config.json"}
    )

    port = 8000
    sglang_server_proc = subprocess.Popen(
        f"python3 -m insight_benchmark.simulation.internal_hisim.sglang.launch_server --port {port} --model-path {MODEL_PATH} --mem-fraction-static=0.5 --skip-server-warmup",
        shell=True,
        env=custom_env,
    )

    from insight_benchmark.serving_benchmark.utils import check_service

    if not check_service(api_url=f"http://127.0.0.1:{port}"):
        raise RuntimeError("SGLang service is not available")

    output_path = "/tmp/insight_benchmark/hisim_serving_test.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # clear data
        pass

    num_prompts = 5
    _ = subprocess.run(
        "python3 -m hisim.simulation.bench_serving "
        f"--port {port} "
        "--warmup-requests 0 "
        "--bench-mode simulation "
        "--dataset-name hisim-collection "
        f"--dataset-path {cur_dir}/assets/dataset/hook_raw_request.jsonl "
        f"--num-prompts {num_prompts} "
        f"--output-file {output_path}",
        shell=True,
    )

    try:
        # dump internal metrics
        with open(output_path) as f:
            data = json.loads(f.readline())
            assert data.get("completed") == num_prompts
    except Exception as e:
        raise e
    finally:
        sglang_server_proc.kill()
        subprocess.run("pkill -9 -f sglang.launch_server", shell=True)


if __name__ == "__main__":
    test_serving_benchmark_sglang()
