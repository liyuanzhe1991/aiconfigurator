import subprocess
import json


def test_estmate():
    result_path = "/tmp/insight_benchmark_estimate_result.json"

    cmd = [
        "python3",
        "-m",
        "insight_benchmark.commands.estimate",
        "--device-names",
        "H20",
        "H800",
        "--tp",
        "1",
        "2",
        "--data-types",
        "FP16",
        "--model-name=Qwen2.5-7B-Instruct",
        "--sla-max-ttft-ms=1000",
        "--sla-max-tpot-ms=50",
        "--sla-mean-input-length=1024",
        "--sla-mean-output-length=256",
        f"--result-path={result_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Command failed with error:\n{result.stderr}"

    result = json.load(open(result_path))

    analyzers = [item["analyzer"] for item in result]
    for name in ["throughput", "sla", "device_capability", "pd_disagg"]:
        assert name in analyzers


if __name__ == "__main__":
    test_estmate()
