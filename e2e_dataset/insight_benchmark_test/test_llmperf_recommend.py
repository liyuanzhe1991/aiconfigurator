from insight_benchmark.analyze import (
    LlmperfThroughputAnalyzer,
    LlmperfSlaAnalyzer,
    LlmperfDeviceCapabilityAnalyzer,
    LlmperfPDDisaggregationAnalyzer,
    LlmperfPrefixCacheAnalyzer,
    PDDisaggAnalyzer,
    SLAConfig,
    InferenceConfigArgs,
    InferenceConfig,
    PayloadConfig,
)
from kunlun_commons.model_info import ModelInfo
from kunlun_commons.hardwares.accelerator import AcceleratorInfo


def test_llmperf_theoretical_analyzer():
    # without inference configuration
    analyzer = LlmperfThroughputAnalyzer(
        devices=["NVIDIA H20"],
        model="Qwen2.5-7B-Instruct",
        inference_config_args=InferenceConfigArgs(),
    )

    result = analyzer.run()
    assert "message" in result and len(result["data"]) > 1

    # with specific inference configuration
    analyzer = LlmperfThroughputAnalyzer(
        devices=["NVIDIA H20"],
        model="Qwen2.5-7B-Instruct",
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1],
            ep_sizes=[1],
            frameworks=["sglang"],
            data_types=["FP16"],
        ),
    )
    result = analyzer.run()
    assert "message" in result and len(result["data"]) == 1


def test_llmperf_sla_analyzer():
    analyzer = LlmperfSlaAnalyzer(
        devices=["NVIDIA H20", "Tesla T4"],
        model="Qwen2.5-7B-Instruct",
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1],
            ep_sizes=[1],
            frameworks=["sglang"],
            data_types=["FP16"],
        ),
        sla_config=SLAConfig(
            max_ttft_ms=1000,
            max_tpot_ms=50,
            mean_input_length=1024,
            mean_output_length=256,
        ),
    )

    result = analyzer.run()
    assert len(result["data"]) == 2


def test_llmperf_device_cap_analyzer():
    analyzer = LlmperfDeviceCapabilityAnalyzer(
        model="Qwen2.5-7B-Instruct",
        devices=["H20", "L20"],
    )
    result = analyzer.run()
    assert len(result["data"]) > 0


def test_llmperf_pd_disagg_analyzer():
    analyzer = LlmperfPDDisaggregationAnalyzer(
        devices=["NVIDIA H20", "Tesla T4"],
        model="Qwen2.5-7B-Instruct",
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1],
            ep_sizes=[1],
            frameworks=["sglang"],
            data_types=["FP16"],
        ),
        sla_config=SLAConfig(
            max_ttft_ms=1000,
            max_tpot_ms=50,
            mean_input_length=1024,
            mean_output_length=256,
        ),
    )
    result = analyzer.run()
    assert len(result["table"]) == 2


def test_llmperf_prefix_cache_analyzer():
    analyzer = LlmperfPrefixCacheAnalyzer(
        devices=["NVIDIA H20", "Tesla T4"],
        model="Qwen2.5-7B-Instruct",
        inference_config_args=InferenceConfigArgs(
            tp_sizes=[1],
            ep_sizes=[1],
            frameworks=["sglang"],
            data_types=["FP16"],
        ),
        payload_config=PayloadConfig(prefix_cache_hit_rate=0.75),
    )

    result = analyzer.run()

    assert len(result["table"]) > 0


def test_pd_disagg_v2():
    analyzer = PDDisaggAnalyzer(
        model=ModelInfo.find_by_model_name("Qwen3-235B-A22B"),
        p_device=AcceleratorInfo.find_by_hw_name(
            "H800-SXM5-80GB"
        ),  # Prefill Instance: high perfermance
        d_device=AcceleratorInfo.find_by_hw_name(
            "H20-SXM5-141GB"
        ),  # Decode Instance: high bandwidth and high hbm capacity
        p_infer_config=InferenceConfig(tp=8),
        d_infer_config=InferenceConfig(tp=8),
        payload_config=PayloadConfig(
            min_input_length=1024,
            max_input_length=1024,
            min_output_length=1024,
            max_output_length=1024,
        ),
        sla_config=SLAConfig(max_ttft_ms=3000, max_tpot_ms=100),
    )

    result = analyzer.run()

    assert result["num_pd_ratio"] > 0


if __name__ == "__main__":
    test_llmperf_theoretical_analyzer()
    test_llmperf_sla_analyzer()
    test_llmperf_device_cap_analyzer()
    test_llmperf_pd_disagg_analyzer()
    test_llmperf_prefix_cache_analyzer()
    test_pd_disagg_v2()
