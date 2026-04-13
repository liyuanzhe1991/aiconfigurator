from pathlib import Path
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from insight_benchmark.simulation.schedule_emulator.types import (
    FakeRequest,
    RequestStage,
)
from insight_benchmark.simulation.schedule_emulator.utils import calc_metrics


def calc_real_prefix_hit_rate(f_path: str) -> float:
    total_input_len = 0
    total_device_hit_len = 0

    for line in open(f_path).readlines():
        item = json.loads(line)

        total_input_len += item["input_length"]
        if "final_prefix_cache_len" in item:
            total_device_hit_len += item.get("final_prefix_cache_len")
        else:
            total_device_hit_len += item.get("final_device_hit_len")

    return total_device_hit_len / total_input_len


def calc_real_prefetch_rate(f_path: str) -> float:
    total_input_len = 0
    total_disk_hit_len = 0

    for idx, line in enumerate(open(f_path).readlines()):
        # if idx == 0:
        #     # warmup
        #     continue

        item = json.loads(line)

        total_input_len += item["input_length"]
        total_disk_hit_len += item.get("final_disk_hit_len", 0)
        # print(f"[calc_real_prefetch_rate] {item["input_length"]=}, {item["final_disk_hit_len"]=}")

    return total_disk_hit_len / total_input_len


def calc_real_metrics_from_requests_stats(real_requests_path: str) -> dict:
    requests = []
    start_ts = float("inf")
    for line in open(real_requests_path).readlines():
        item = json.loads(line)
        requests.append(
            FakeRequest(
                id=-1,
                input_token_length=item["input_length"],
                output_token_length=item["output_length"],
                gen_token_latencies=item["gen_token_latencies"],
                final_reused_tokens=item.get(
                    "final_prefix_cache_len", item.get("final_device_hit_len", 0)
                ),
                last_event_time=item["last_event_time"],
                stage=RequestStage.COMPLETE,
            )
        )
        start_ts = min(start_ts, item["created_time"])

    for req in requests:
        req.last_event_time -= start_ts

    return calc_metrics(requests)


def merge_metrics(
    real_data_dir: Path,
    sim_data_dir: Path,
    case_list=["no_cache", "l1"],
    predictors: list[str] = ["aiconfigurator"],
) -> pd.DataFrame:
    result = []

    print("Running: ", real_data_dir)

    for rate_dir in os.listdir(real_data_dir):
        for run_case in case_list:
            real_metrics_path = real_data_dir / rate_dir / f"{run_case}.metrics.json"
            if not real_metrics_path.exists():
                continue

            real_requests_path = real_data_dir / rate_dir / f"{run_case}.requests.jsonl"

            real_metrics = json.load(open(real_metrics_path))

            metadata = {
                "total_input_tokens": real_metrics["total_input_tokens"],
                "total_output_tokens": real_metrics["total_output_tokens"],
                "random_range_ratio": real_metrics["random_range_ratio"],
                "request_rate": real_metrics["request_rate"],
                "max_concurrency": real_metrics["max_concurrency"],
            }

            common_keys = [
                "duration",
                "mean_ttft_ms",
                "median_ttft_ms",
                "mean_tpot_ms",
                "median_tpot_ms",
                "mean_itl_ms",
                "median_itl_ms",
                "mean_e2e_latency_ms",
                "input_throughput",
                "output_throughput",
            ]

            real_data = {k: real_metrics[k] for k in common_keys}
            real_data.update(
                {
                    "case": run_case,
                    "mode": "real",
                    "predictor": "benchmark",
                    "prefix_hit_ratio": calc_real_prefix_hit_rate(real_requests_path),
                    "disk_prefetch_ratio": calc_real_prefetch_rate(real_requests_path),
                    "time_cost": real_metrics["duration"],
                }
            )
            real_data.update(metadata)
            result.append(real_data)

            for predictor_name in predictors:
                metrics_file = (
                    sim_data_dir
                    / rate_dir
                    / f"{predictor_name}.{run_case}.metrics.json"
                )
                if not metrics_file.exists():
                    continue

                sim_metrics = json.load(metrics_file.open())

                sim_data = {k: sim_metrics[k] for k in common_keys}
                sim_data.update(
                    {
                        "case": run_case,
                        "mode": "simulation",
                        "predictor": predictor_name,
                        "prefix_hit_ratio": sim_metrics["prefix_cache_reused_ratio"],
                        "disk_prefetch_ratio": sim_metrics["disk_prefetch_ratio"],
                        "time_cost": sim_metrics["time_cost"],
                    }
                )
                sim_data.update(metadata)

                result.append(sim_data)

    return pd.DataFrame(result)


def show_metrics(
    data_or_path: Path | pd.DataFrame,
    title: str,
    output_file: Path,
    cases: list = ["no_cache"],
    hue="mode",
):
    if isinstance(data_or_path, Path):
        df = pd.read_csv(data_or_path)
    else:
        df = data_or_path

    # 设置学术风样式
    sns.set_theme(style="whitegrid", font_scale=1.0)

    metrics_list = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "mean_e2e_latency_ms",
        "input_throughput",
        "prefix_hit_ratio",
        "disk_prefetch_ratio",
    ]

    fig, axes = plt.subplots(
        len(cases),
        len(metrics_list),
        figsize=(4 * len(metrics_list), 4 * len(cases)),
        dpi=300,
    )

    legend_handles = legend_labels = None

    for i, case in enumerate(cases):
        case_df = df[df["case"] == case]
        for j, metrics_name in enumerate(metrics_list):
            ax = axes[i, j] if len(cases) > 1 else axes[j]

            sns.lineplot(
                data=case_df,
                x="request_rate",
                y=metrics_name,
                hue=hue,
                style="mode",
                ax=ax,
                legend="full",  # 关键：生成完整 legend（含 style）
            )

            if (
                metrics_name == "prefix_hit_ratio"
                or metrics_name == "disk_prefetch_ratio"
            ):
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0)

            # 只从第一个子图抓一次 legend 信息
            if legend_handles is None:
                leg = ax.get_legend()
                legend_handles = leg.legend_handles
                legend_labels = [t.get_text() for t in leg.get_texts()]

            # 每个子图都不要自己的 legend
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    plt.suptitle(title)

    # 共享图例（包含 hue + style）
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, -0.02 / len(cases)),
    )

    fig.tight_layout(rect=[0.03, 0.1 / len(cases), 0.96, 0.97])
    for i, case in enumerate(cases):
        ax0 = axes[i, 0] if len(cases) > 1 else axes[0]
        pos = ax0.get_position()
        y = (pos.y0 + pos.y1) / 2
        x = pos.x0 - 0.04
        fig.text(
            x, y, f"{case}", ha="right", va="center", rotation=90, fontweight="bold"
        )

    plt.savefig(output_file)


def cal_metrics_mpe(csv_path: Path):
    df = pd.read_csv(csv_path)

    metrics = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "input_throughput",
        "output_throughput",
        "duration",
        "prefix_hit_ratio",
    ]

    # 1) 聚合：每个 request_rate 下，real 一行；mock_offline 则每个 predictor 一行
    grp_keys = ["request_rate", "case", "mode", "predictor"]

    agg = df.groupby(grp_keys, as_index=False)[metrics].mean(numeric_only=True)

    # 2) 用聚合后的表算 MPE（real 作为 true）
    real = agg[agg["mode"] == "real"][["case", "request_rate"] + metrics].rename(
        columns={m: f"real_{m}" for m in metrics}
    )

    mock = agg[agg["mode"] == "simulation"].copy()

    merged = mock.merge(real, on=["case", "request_rate"], how="left")

    for m in metrics:
        diff = merged[m] - merged[f"real_{m}"]
        real_val = merged[f"real_{m}"]
        merged[f"mpe_{m}"] = np.where(
            (diff == 0) & diff.notna() & real_val.notna(), 0.0, diff / real_val
        )

    mpe_cols = ["case", "request_rate", "predictor"] + [f"mpe_{m}" for m in metrics]
    mpe = merged[mpe_cols].copy()
    mpe["mode"] = "simulation"

    # 3) 合并回 agg：只会给 mock_offline 行补上 MPE；real 行保持 NaN
    out = agg.merge(mpe, on=["case", "mode", "request_rate", "predictor"], how="left")

    out_path = csv_path.parent / f"mpe.{csv_path.name}"
    out.to_csv(out_path, index=False)
    print(out)

    mpe_columns = [f"mpe_{item}" for item in metrics]

    df = out
    mock_df = df[df["mode"] == "simulation"]

    result_df = mock_df.groupby(["predictor", "case"])[mpe_columns].apply(
        lambda x: x.abs().mean()
    )
    result_df = result_df.reset_index()

    result_df.round(3).to_csv(csv_path.parent / "avg_abs_mpe_summary.csv", index=False)
    # essential metrics
    essential_metrics = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "input_throughput",
        "duration",
        "prefix_hit_ratio",
    ]
    essential_mpe_cols = [f"mpe_{m}" for m in essential_metrics]
    # 只对 essential 的 MPE 列计算平均绝对误差
    essential_result_df = (
        mock_df.groupby(["predictor", "case"])[essential_mpe_cols]
        .apply(lambda x: (x.abs().mean() * 100).round(2).astype(str) + "%")
        .reset_index()
    )

    essential_result_df.to_csv(
        csv_path.parent / "essential_mpe_summary.csv", index=False
    )

    return out
