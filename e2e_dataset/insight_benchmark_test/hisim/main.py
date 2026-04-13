from pathlib import Path
import simulate


def run_qwen():
    # data_root = Path("/nfs/disk2/projects/kunlun_insight/data/serving/sglang/")
    data_root = Path(
        "/host/aiconfigurator/e2e_dataset/tmp/"
    )

    for version_dir in data_root.iterdir():
        if version_dir.name != "L1":
            continue
        for model_dir in version_dir.iterdir():
            if model_dir.name not in [
                # "260414-L1-H20-GLM-5-FP8-LAYER48-TP8-EP8-EPLB-FAKE-No-Overlap",
                # "260408-L1-H20-Qwen3-8B-GSP-No-Overlap",
                # "260408-L1-H20-Qwen3-32B-FP8-GSP-No-Overlap",
                # "260408-L2-H20-Qwen3-8B-GSP-No-Overlap",
                # "260408-L2-H20-Qwen3-32B-FP8-GSP-No-Overlap"
                # "H20-3e-GLM5-FP8-TP8-EP8-BALANCED-GSP",
                # "H20-3e-GLM-5-FP8-TP8-EP8-BALANCED-RANDOM",
                "H20-Qwen3-235B-FP8",
                # "H20-Qwen3-235B-FP8-BALANCE",
            ]:
                continue

            if "L1" in model_dir.name:
                case_list = ["no_cache", "l1"]
            elif "L2" in model_dir.name:
                case_list = ["no_cache", "l1", "evict_l1", "l2"]
            else:
                case_list = ["no_cache"]
            # case_list = ["no_cache", "l1"]

            if "H20-3e" in model_dir.name:
                device_name = "h20_3e_sxm"
            elif "H20" in model_dir.name:
                device_name = "h20_sxm"
            elif "A100" in model_dir.name:
                device_name = "a100_sxm"
            else:
                raise f"Unexpected device: {model_dir.name}"

            if "Qwen3-8B" in model_dir.name:
                extra_config = {
                    "prefill_scale_factor": 1.045,
                    "decode_scale_factor": 1.0,
                }
            elif "Qwen3-32B" in model_dir.name:
                extra_config = {
                    "prefill_scale_factor": 1.045,
                    "decode_scale_factor": 1.011,
                }
            elif "Qwen3-235B" in model_dir.name:
                extra_config = {
                    "prefill_scale_factor": 1.011,
                    "decode_scale_factor": 1.041,
                }
            else:
                extra_config = {}

            print(f"Processing model: {model_dir.name}")
            if "BALANCE" not in model_dir.name:
                extra_config["workload_distribution"] = "power_law_1.2"
                print(f"Use power_law_1.2 for {model_dir.name}")

            # import time
            # time.sleep(100)

            simulate.main(
                data_dir=model_dir,
                runner="offline",
                predictors=["aiconfigurator"],
                case_list=case_list,
                predictor_db_path="/host/aiconfigurator/src/aiconfigurator/systems/",
                predictor_extra_config=extra_config,
                device_name=device_name,
                backend_version="0.5.9",
                model_path="/models/Qwen3-235B-A22B-Instruct-2507-FP8",
            )


if __name__ == "__main__":
    run_qwen()
