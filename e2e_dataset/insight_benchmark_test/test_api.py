from kunlun_commons.model_info import ModelInfo
from kunlun_commons.data_type import DataType
from kunlun_commons.hardwares.accelerator import AcceleratorInfo

from insight_benchmark.api.model import list_supported_models, query_model_info
from insight_benchmark.api.estimate import (
    estimate_performance_sla,
    estimate_optimization_options,
    estimate_performance_sla_single_config,
    estimate_max_throughput,
    estimate_min_device_number,
    recommend_inference_config,
    estimate_max_num_tokens,
)
from insight_benchmark.api.topo import (
    analyze_performance_topology_serving,
    analyze_performance_topology_static,
    analyze_model_topology,
)
from insight_benchmark.api.schedule_stats import estimate_scheduler_stats
from insight_benchmark.api.filter import filter_machine
from insight_benchmark.api.static_performence import (
    estimate_static_performence,
    estimate_static_performance_with_real_latency,
    estimate_static_roofline,
    estimate_max_batch_size_by_latency,
)
from insight_benchmark.api.types import (
    IterRange,
    InferenceConfig,
)

from insight_benchmark.simulation.schedule_emulator.types import BenchmarkConfig

from insight_benchmark.analyze.types import (
    SLAConfig,
    PayloadConfig,
    InferenceConfigArgs,
)


def test_list_supported_models():
    models = list_supported_models()
    assert len(models) > 0


def test_query_model_info():
    model = ModelInfo.find_by_model_name("Qwen2.5-7B")

    result = query_model_info(model, InferenceConfig(tp=1))
    assert result["communication_bytes"] == 0

    result = query_model_info(model, InferenceConfig(tp=2))
    assert result["communication_bytes"] > 0


def test_filter_machine():
    result = filter_machine(
        ModelInfo.find_by_model_name("Qwen2.5-7B"), framework_name="sglang"
    )
    assert len(result) != 0
    result = filter_machine(
        ModelInfo.find_by_model_name("Qwen3-235B-A22B"), framework_name="sglang"
    )
    assert len(result) != 0


def test_estimate_performance_sla():
    result = estimate_performance_sla(
        model_name="Deepseek-V3",
        device_names=["A100-SXM4-80GB", "H20"],
        sla_config=SLAConfig(max_ttft_ms=1000, max_tpot_ms=100),
        payload_config=PayloadConfig(
            min_input_length=256,
            max_input_length=512,
            min_output_length=256,
            max_output_length=512,
        ),
        inference_config_args=InferenceConfigArgs(frameworks=["sglang"]),
    )
    assert len(result) != 0


def test_estimate_performance_sla_single_config():
    result = estimate_performance_sla_single_config(
        model_name="Qwen2.5-7B",
        device_name="A30",
        sla_config=SLAConfig(max_ttft_ms=1000, max_tpot_ms=100),
        payload_config=PayloadConfig(
            min_input_length=256,
            max_input_length=512,
            min_output_length=256,
            max_output_length=512,
        ),
        inference_config_args=InferenceConfigArgs(
            frameworks=["sglang"], data_types=["FP16"]
        ),
    )
    assert len(result) != 0


def test_estimate_optimization_options():
    result = estimate_optimization_options(
        model_name="Qwen3-235B-A22B",
        device_name="A100-SXM4-80GB",
        sla_config=SLAConfig(max_ttft_ms=1000, max_tpot_ms=100),
        payload_config=PayloadConfig(
            min_input_length=256,
            max_input_length=512,
            min_output_length=256,
            max_output_length=512,
        ),
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1], frameworks=["sglang"], data_types=["FP16"]
        ),
    )
    assert len(result) != 0


def test_estimate_scheduler_stats():
    result = estimate_scheduler_stats(
        model=ModelInfo.find_by_model_name("Qwen2.5-7B"),
        device=AcceleratorInfo.find_by_hw_name("A30"),
        inference_config=InferenceConfig(),
        emulate_benchmark_config=BenchmarkConfig(1000, 16, 32, 8, 16, 100),
    )
    assert "prefill" in result["run_batch"]
    assert "decode" in result["run_batch"]


def test_analyze_performance_topology_serving():
    result = analyze_performance_topology_serving(
        model_name="Qwen2.5-7B",
        device_name="GB200",
        sla_config=SLAConfig(max_ttft_ms=1000, max_tpot_ms=100),
        payload_config=PayloadConfig(
            min_input_length=256,
            max_input_length=512,
            min_output_length=256,
            max_output_length=512,
        ),
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1], frameworks=["sglang"], data_types=["FP16"]
        ),
    )
    assert len(result["devices"]) != 0


def test_estimate_static_roofline():
    for dev in ["GB200", "A100-SXM4-80GB"]:
        result = estimate_static_roofline(
            model=ModelInfo.find_by_model_name("Deepseek-R1"),
            device=AcceleratorInfo.find_by_hw_name(dev),
            inference_config=InferenceConfig(
                tp=16,
                framework="sglang",
                dt="FP16",
                # The parameter to enable DeepEP:
                dp=16,
                enable_deep_ep=True,
                enable_dp_attention=True,
            ),
            input_length=1024,
            output_length=8,
            batch_size=1,
            mode="decode",
            with_operator=True,
        )

        assert len(result["roofline"]) > 0
        assert len(result["operator_roofline"]) > 0

        op_names = [item["name"] for item in result["operator_roofline"]]

        if dev == "GB200":
            assert "mlp.token_dispatch" in op_names
        elif dev == "A100-SXM4-80GB":  # A100 dose not support DeepEP
            assert "mlp.token_dispatch" not in op_names


def test_analyze_performeance_topology_static():
    for dev in ["GB200", "A100-SXM4-80GB"]:
        result = analyze_performance_topology_static(
            model=ModelInfo.find_by_model_name("Deepseek-R1"),
            device=AcceleratorInfo.find_by_hw_name(dev),
            infer_config=InferenceConfig(tp=16, framework="sglang", dt="FP8"),
            input_length=1024,
            output_length=8,
            batch_size=1,
            mode="decode",
        )
        for n in result["devices"]:
            if n["status"] == "busy":
                assert 0 <= n.get("utilization", {}).get("estimated") <= 1


def test_estimate_static_performence():
    result = estimate_static_performence(
        model=ModelInfo.find_by_model_name("Deepseek-R1"),
        device=AcceleratorInfo(
            name="GB200",
            vendor="",
            device_family="",
            vendor_id=0,
            device_id=0,
            max_power_watt=0,
            hbm_bandwidth_gb=80,
            tflops={"FP8": 10000},
            inter_node_bandwidth_gb=1800,
            intra_node_bandwidth_gb=4800,
            hbm_capacity_gb=18,
        ),
        input_range=IterRange(1, 128, 16),
        batch_size_range=IterRange(2, 8, 2),
        mode="prefill",
    )
    assert len(result["data"]) != 0
    assert result["inference_config"]["parallelism"]["tp_size"] == 256

    result = estimate_static_performence(
        model=ModelInfo.find_by_model_name("Qwen3-235B-A22B"),
        device=AcceleratorInfo(
            name="A100-SXM4-80GB",
            vendor="",
            device_family="",
            vendor_id=0,
            device_id=0,
            max_power_watt=0,
            hbm_bandwidth_gb=80,
            inter_node_bandwidth_gb=1800,
            intra_node_bandwidth_gb=4800,
            hbm_capacity_gb=203.9,
        ),
        input_range=IterRange(1, 128, 16),
        batch_size_range=IterRange(2, 8, 2),
        mode="decode",
    )
    assert result["inference_config"]["parallelism"]["tp_size"] == 4
    assert len(result["data"]) != 0
    for item in result["data"]:
        for k, v in item["performance"].items():
            if k.endswith("_eff"):
                assert 0 <= v <= 1


def test_estimate_static_performance_with_real_latency():
    result = estimate_static_performance_with_real_latency(
        model=ModelInfo.find_by_model_name("Qwen3-235B-A22B"),
        device=AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB"),
        input_length=1024,
        batch_size=2,
        mode="prefill",
        real_latency_s=3,
        infer_config=InferenceConfig(tp=8),
    )
    assert 0 < result["data"]["performance"]["computation_eff"] <= 1


def test_estimate_min_device_number():
    result = estimate_min_device_number(
        model=ModelInfo.find_by_model_name("Deepseek-R1"),
        device=AcceleratorInfo.find_by_hw_name("NVIDIA GB200"),
    )
    assert result == 8

    result = estimate_min_device_number(
        model=ModelInfo.find_by_model_name("Qwen3-235B-A22B"),
        device=AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB"),
    )
    assert result == 8


def test_recommend_inference_config():
    result = recommend_inference_config(
        model=ModelInfo.find_by_model_name("Deepseek-R1"),
        device=AcceleratorInfo.find_by_hw_name("NVIDIA GB200"),
        dtype=DataType.FP8,
        mode="prefill",
    )
    assert len(result) > 0
    best_config = result[0]
    assert best_config.tp == 8

    result = recommend_inference_config(
        model=ModelInfo.find_by_model_name("Deepseek-R1"),
        device=AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB"),
        mode="decode",
    )
    assert len(result) > 0
    best_config = result[0]
    # A100 dose not support FP8
    assert best_config.dt.value == "FP16"
    assert best_config.tp == 32


def test_estimate_max_throughput():
    model = ModelInfo.find_by_model_name("Qwen2.5-7B")
    device = AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB")
    infer_config = recommend_inference_config(model, device)[0]

    result = estimate_max_throughput(
        model=model, device=device, infer_config=infer_config
    )
    assert result["prefill_throughput_tokens"] == 2304

    model = ModelInfo.find_by_model_name("Deepseek-R1")
    device = AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB")
    infer_config = recommend_inference_config(model, device)[0]

    result = estimate_max_throughput(
        model=model, device=device, infer_config=infer_config
    )
    assert result["prefill_throughput_tokens"] == 2560


def test_model_topology():
    result = analyze_model_topology(
        ModelInfo.find_by_model_name("Qwen2.5-7B"),
        infer_config=InferenceConfig(tp=1, pp=2),
    )
    assert len(result) != 0


def test_estimate_max_batch_size_by_ttft():
    model = ModelInfo.find_by_model_name("Qwen2.5-7B")
    device = AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB")
    infer_config = recommend_inference_config(model, device)[0]

    for m in ["prefill", "decode"]:
        result = estimate_max_batch_size_by_latency(
            model=model,
            device=device,
            inference_config=infer_config,
            latency_limitation_s=2,
            input_length=1024,
            mode=m,
        )
        pass

    assert result > 0


def test_estimate_max_num_tokens():
    model = ModelInfo.find_by_model_name("Qwen2.5-7B")
    device = AcceleratorInfo.find_by_hw_name("NVIDIA A100-SXM4-80GB")
    infer_config = recommend_inference_config(model, device)[0]
    num_tokens = estimate_max_num_tokens(model, device, infer_config)
    assert num_tokens > 0


if __name__ == "__main__":
    test_query_model_info()
    test_list_supported_models()
    test_filter_machine()
    test_estimate_performance_sla()
    test_estimate_performance_sla_single_config()
    test_estimate_optimization_options()
    test_estimate_scheduler_stats()
    test_analyze_performance_topology_serving()
    test_analyze_performeance_topology_static()
    test_estimate_static_performence()
    test_estimate_min_device_number()
    test_recommend_inference_config()
    test_estimate_max_throughput()
    test_model_topology()
    test_estimate_static_roofline()
    test_estimate_max_batch_size_by_ttft()
    test_estimate_static_performance_with_real_latency()
    test_estimate_max_num_tokens()
