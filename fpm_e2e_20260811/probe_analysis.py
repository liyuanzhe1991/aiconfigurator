# Channel-B probe analysis: 5-repeat medians per explicit point vs the formal
# parquet. Decode points are config-matched (decode-cell params); prefill
# points ran WITHOUT the prefill-cell graph config and are reported separately
# as config-effect measurements.
import json

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

db = get_database("h200_sxm", "vllm", "0.25.1")
FPMForwardOp.clear_cache()
pq = pd.read_parquet("fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")


def make_op(phase, tp):
    cfg = sdk_config.ModelConfig(
        tp_size=tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=tp, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block, moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16, comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8)
    sol = (lambda b, tp_, tk: 0.001) if phase == "prefill" else (lambda b, tk: 0.001)
    return FPMForwardOp(phase, cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=sol, weight_bytes=1e9)


for tp in (4, 8):
    print(f"\n================ TEP{tp} ================")
    reps = []
    for r in range(1, 6):
        d = json.load(open(f"fpm_e2e_20260811/probes/tep{tp}/probe_r{r}.json"))
        for g in d["iteration_groups"]:
            p = g["point"]
            reps.append({
                "rep": r, "kind": p["point_type"], "b": p["batch_size"],
                "tokens": p["total_prefill_tokens"], "kv": p["total_kv_read_tokens"],
                "ms": g["wall_time"] * 1000, "complete": g["complete"],
            })
    df = pd.DataFrame(reps)
    df = df[df.complete]
    agg = df.groupby(["kind", "b", "tokens", "kv"]).ms.agg(
        med="median", n="size", spread=lambda s: (s.max() - s.min()) / s.median()).reset_index()

    sub = pq[(pq.tp == tp)]
    grid_dec = set(zip(sub[sub.workload_kind == "decode"].batch_size, sub[sub.workload_kind == "decode"].total_kv_read_tokens))
    grid_pre = set(zip(sub[sub.workload_kind == "prefill"].batch_size, sub[sub.workload_kind == "prefill"].total_prefill_tokens, sub[sub.workload_kind == "prefill"].total_kv_read_tokens))
    row_dec = {(r.batch_size, r.total_kv_read_tokens): r.latency_ms for r in sub[sub.workload_kind == "decode"].itertuples()}
    row_pre = {(r.batch_size, r.total_prefill_tokens, r.total_kv_read_tokens): r.latency_ms for r in sub[sub.workload_kind == "prefill"].itertuples()}
    op_d = make_op("decode", tp)

    print("--- DECODE points (config-matched) ---")
    print(f"{'grp':4s} {'b':>5s} {'kv_total':>9s} {'probe med':>9s} {'5-rep spread':>12s} {'parquet/model':>13s} {'delta':>7s}")
    a_deltas, a_spreads, b_deltas = [], [], []
    for r in agg[agg.kind == "decode"].itertuples():
        key = (r.b, r.kv)
        if key in grid_dec:
            ref, grp = row_dec[key], "A"
            a_deltas.append((r.med - ref) / ref); a_spreads.append(r.spread)
        else:
            ref = float(op_d.query(db, batch_size=int(r.b), s=max(1, round(r.kv / r.b))))
            grp = "B"
            b_deltas.append((r.med - ref) / ref)
        print(f"{grp:4s} {int(r.b):5d} {int(r.kv):9d} {r.med:9.2f} {r.spread*100:11.1f}% {ref:13.2f} {(r.med-ref)/ref*100:+6.1f}%")
    if a_deltas:
        print(f"A-group: drift median {np.median(a_deltas)*100:+.2f}% | noise floor (5-rep spread median) {np.median(a_spreads)*100:.2f}%")
    if b_deltas:
        print(f"B-group: interpolation delta median {np.median(np.abs(b_deltas))*100:.2f}% (signed median {np.median(b_deltas)*100:+.2f}%)")

    print("--- PREFILL points (decode-config; NOT parity-comparable, config effect) ---")
    print(f"{'b':>2s} {'tokens':>7s} {'probe med':>9s} {'parquet(prefill-cfg)':>20s} {'ratio':>6s}")
    for r in agg[agg.kind == "prefill"].itertuples():
        key = (r.b, r.tokens, r.kv)
        ref = row_pre.get(key)
        if ref:
            print(f"{int(r.b):2d} {int(r.tokens):7d} {r.med:9.2f} {ref:20.2f} {r.med/ref:6.2f}x")
