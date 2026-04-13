import insight_benchmark.hook as insight_hook
from insight_benchmark.step_benchmark import sglang_hook, trtllm_hook, transformer_hook
from insight_benchmark.step_benchmark.stats_manager import get_stats_manager
from env import MODEL_PATH, check_framework
import os
import pytest

insight_hook.install_class_hooks(
    [
        # sglang_hook.C_SglangModelRunnerHook,
        transformer_hook.C_TransformersAutoConfig,
        sglang_hook.C_SglangSchedulerHook,
        trtllm_hook.C_TrtllmPyExecutorHook,
        trtllm_hook.C_TrtllmPyTorchModelEngineHook,
    ]
)
insight_hook.install_module_hooks(
    [
        trtllm_hook.M_TrtllmGetRequestHook,
    ]
)


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# FIXME: Hook is invalid in pytest environment. -> force to disable testing
@pytest.mark.skipif(True, reason="Hook is invalid in pytest environment.")
@pytest.mark.skipif(not check_framework("sglang"), reason="sglang is not installed")
def test_hook_llm_sglang():
    if not check_framework("sglang"):
        return

    # init stats manager befor starting engine
    stats_manager = get_stats_manager()

    import sglang as sgl

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        disable_overlap_schedule=True,
        tp_size=1,
        mem_fraction_static=0.5,
        chunked_prefill_size=8192,  # The chunked prefill size will be adjusted based on the memory size specified in ServerArgs.
    )

    prompts = [
        "Hello, my name is "
        * 32,  # In order to reuse the block, the prompt's length should be larger than the block size.
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": 8,
    }

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    data = stats_manager.gets()
    assert len(data) != 0
    total_reused_tokens = sum([item.num_reused_tokens for item in data])
    assert total_reused_tokens == 0

    # Rerun with the same prompts to check prefix caching.
    outputs = llm.generate(prompts, sampling_params)
    data = get_stats_manager().gets()
    assert len(data) != 0
    total_reused_tokens = sum([item.num_reused_tokens for item in data])
    assert total_reused_tokens > 0

    llm.shutdown()
    stats_manager.clean()


@pytest.mark.skipif(True, reason="Hook is invalid in pytest environment.")
@pytest.mark.skipif(
    check_framework("tensorrt_llm"), reason="tensorrt_llm is not installed"
)
def test_hook_llm_trtllm():
    if not check_framework("tensorrt_llm"):
        return

    # init stats manager befor starting engine
    stats_manager = get_stats_manager()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

    prompts = [
        "Hello, my name is "
        * 32,  # In order to reuse the block, the prompt's length should be larger than the block size.
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=8)

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        backend="pytorch",
        enable_block_reuse=True,  # enable block reuse
        pytorch_backend_config=PyTorchConfig(use_cuda_graph=False),
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    data = stats_manager.gets()
    assert len(data) > 0
    total_reused_tokens = sum([item.num_reused_tokens for item in data])
    assert total_reused_tokens == 0

    # Rerun with the same prompts to check block reusing.
    outputs = llm.generate(prompts, sampling_params)
    data = get_stats_manager().gets()
    assert len(data) > 0
    total_reused_tokens = sum([item.num_reused_tokens for item in data])
    assert total_reused_tokens > 0

    llm.shutdown()
    stats_manager.clean()


if __name__ == "__main__":
    test_hook_llm_sglang()
    test_hook_llm_trtllm()
