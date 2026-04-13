from pathlib import Path
from kunlun_commons.hardwares.accelerator import AcceleratorInfo
from kunlun_commons.model_info import ModelInfo

from insight_benchmark.simulation.infer_time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from insight_benchmark.simulation.schedule_emulator.types import (
    SchedulerConfig,
)


cur_dir = Path(__file__).parent


def test_aic_xgb_predictor():
    model = ModelInfo.find_by_model_name("Qwen3-8B")
    hw = AcceleratorInfo.find_by_hw_name("H100-SXM5-80G")
    hw.name = "h100_sxm"  # AIConfigurator internal device name
    config = SchedulerConfig(
        model=model, backend_name="sglang", backend_version="0.5.6.post2"
    )

    xgb_model_path = cur_dir / "assets/aic_xgb_models/"
    predictor = AIConfiguratorTimePredictor(
        model, hw, config, xgb_model_path=xgb_model_path
    )

    # Prefill
    reqs = [
        ScheduleRequest(512, 512),
        ScheduleRequest(1024, 0),
        ScheduleRequest(512, 0),
    ]

    latency = predictor.predict_infer_time(ScheduleBatch(reqs))
    assert latency > 0

    # Decode
    reqs = [
        ScheduleRequest(1, 1024),
        ScheduleRequest(1, 1024),
        ScheduleRequest(1, 1024),
    ]

    latency = predictor.predict_infer_time(ScheduleBatch(reqs))
    assert latency > 0


if __name__ == "__main__":
    test_aic_xgb_predictor()
