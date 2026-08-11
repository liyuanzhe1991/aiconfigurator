# Level 0 closure: query FPMForwardOp at every collected grid point and
# verify the prediction reproduces the measured row (exact-lookup path).
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

MODEL_PATH = "MiniMaxAI/MiniMax-M2.7"
PARQUET = "fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"

db = get_database("h200_sxm", "vllm", "0.25.1")
df = pd.read_parquet(PARQUET)


def make_op(phase: str, tp: int, ep: int) -> FPMForwardOp:
    cfg = sdk_config.ModelConfig(
        tp_size=tp,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=ep,
        cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    if phase == "prefill":
        sol = lambda b, tp_, tk: 0.001
    else:
        sol = lambda b, tk: 0.001
    return FPMForwardOp(phase, cfg, MODEL_PATH, sol_fn=sol, weight_bytes=1e9)


overall_worst = 0.0
for (tp, ep), topo_df in df.groupby(["tp", "moe_ep"]):
    for phase, phase_df in topo_df.groupby("workload_kind"):
        op = make_op(phase, int(tp), int(ep))
        n = exact = 0
        worst = 0.0
        skipped = 0
        for row in phase_df.itertuples():
            b = int(row.batch_size)
            tot_p = int(row.total_prefill_tokens)
            tot_kv = int(row.total_kv_read_tokens)
            if phase == "prefill":
                if tot_p % b or tot_kv % b:
                    skipped += 1  # coords not expressible as (b, s, prefix)
                    continue
                res = op.query(db, batch_size=b, s=tot_p // b, prefix=tot_kv // b)
            else:
                if tot_kv % b:
                    skipped += 1
                    continue
                res = op.query(db, batch_size=b, s=tot_kv // b)
            n += 1
            pred = float(res)
            meas = float(row.latency_ms)
            rel = abs(pred - meas) / meas
            worst = max(worst, rel)
            if rel < 1e-9:
                exact += 1
        overall_worst = max(overall_worst, worst)
        print(
            f"TEP{tp:<2d} {phase:8s}: {n:5d} points, exact={exact:5d}, "
            f"worst rel err={worst:.3e}, skipped(indivisible)={skipped}"
        )

print(f"\noverall worst relative error: {overall_worst:.3e}")
print("closure PASSED" if overall_worst < 1e-6 else "closure FAILED")
