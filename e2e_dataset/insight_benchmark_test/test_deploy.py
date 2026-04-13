import time
import pytest
from insight_benchmark.deploy import (
    LlmServiceConfig,
    ResourceConfig,
    PodServiceConfig,
    DeployConfig,
    DeployStatus,
    init_deployer,
)
from insight_benchmark.dataset import DatasetArgs
from insight_benchmark.serving_benchmark import (
    ServingBenchmark,
    ServingBenchmarkRuntimeArgs,
    ServingBenchmarkConfigs,
)


@pytest.mark.skipif(True, reason="Testing in tair kvcache cluster")
def test_tair_kvcace_deloyer():
    config = DeployConfig(
        deploy_tool="tair_kvcache",
        name="insight-benchmark-test",
        llm_service=LlmServiceConfig(
            model_path="/data/Qwen2.5-0.5B-Instruct", framework="sglang", tp_size=1
        ),
        resource=ResourceConfig(xpu_type="NVIDIA-GeForce-RTX-4090"),
        pod_service=PodServiceConfig(svc_type="ClusterIP", port=8000, target_port=8000),
    )

    deployer = init_deployer(config)

    deployer.deploy()
    auth_info = deployer.get_auth_info()

    while deployer.status() != DeployStatus.READY:
        time.sleep(1)

    base_url = deployer.get_svc_base_url()

    bench = ServingBenchmark(
        ServingBenchmarkConfigs(
            backend="sglang-oai-chat",
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_args=DatasetArgs(
                name="random",
                num_prompts=100,
                min_input_len=128,
                max_input_len=256,
                min_output_len=16,
                max_output_len=16,
            ),
            runtime_args=ServingBenchmarkRuntimeArgs(max_concurrency=10),
            base_url=base_url,
            auth_info=auth_info,
        )
    )

    metrics = bench.benchmark()
    assert len(metrics) > 0


if __name__ == "__main__":
    test_tair_kvcace_deloyer()
