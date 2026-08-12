# Strict mixed-step validation of the LANDED #1461 formula (task A).
# Ground truth: the v2 cap2048-aligned sweep's raw FPM stream, per-step.
# Model side replicates _get_fpm_mix_step_latency exactly, but addressed at
# each real step's ACTUAL scheduled coordinates from the stream:
#   pre  = query_totals(bp, tp + bd, kvp)          (regime by scheduled total)
#   dec  = query(bd, kvd/bd) - query_pass_baseline(bd)   (marginal, floor 0)
#   mixed = pre + max(0, dec)
# Old-formula values recomputed side by side for the delta-vs-delta table.
import json

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

STREAM = "fpm_e2e_20260811/serve_results/stream_stack1_v3.jsonl"
WINDOWS = "fpm_e2e_20260811/serve_results/mixed_windows_v2.tsv"

db = get_database("h200_sxm", "vllm", "0.25.1")
FPMForwardOp.clear_cache()
cfg = sdk_config.ModelConfig(
    tp_size=8, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=8, cp_size=1,
    gemm_quant_mode=common.GEMMQuantMode.fp8_block, moe_quant_mode=common.MoEQuantMode.fp8_block,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16, comm_quant_mode=common.CommQuantMode.half,
    kvcache_quant_mode=common.KVCacheQuantMode.fp8)
op_p = FPMForwardOp("prefill", cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=lambda b, tp, tk: 0.001, weight_bytes=1e9)
op_d = FPMForwardOp("decode", cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=lambda b, tk: 0.001, weight_bytes=1e9)

lines = open(STREAM).readlines()
rows = []
for w in open(WINDOWS):
    name, bd_nom, chunk, s, e = w.split()
    for line in lines[int(s):int(e)]:
        d = json.loads(line)
        sr = d["scheduled_requests"]
        bp, tp_, kvp = sr["num_prefill_requests"], sr["sum_prefill_tokens"], sr["sum_prefill_kv_tokens"]
        bd, kvd = sr["num_decode_requests"], sr["sum_decode_kv_tokens"]
        if bp == 0 or bd == 0:
            continue  # strict: mixed steps only
        meas = d["wall_time"] * 1000
        if meas <= 0:
            continue
        try:
            pre_new = float(op_p.query_totals(db, batch_size=bp, total_prefill_tokens=tp_ + bd,
                                              total_kv_read_tokens=kvp))
            pre_old = float(op_p.query(db, batch_size=bp, s=max(1, round(tp_ / bp)), prefix=round(kvp / bp)))
            dec = float(op_d.query(db, batch_size=bd, s=max(1, round(kvd / bd))))
            base = float(op_d.query_pass_baseline(db, batch_size=bd))
            marginal = max(0.0, dec - base)
            rows.append({"window": name, "bd_nom": int(bd_nom), "chunk": int(chunk),
                         "bp": bp, "tp": tp_, "kvp": kvp, "bd": bd, "kvd": kvd,
                         "meas": meas, "new": pre_new + marginal, "old": pre_old + marginal})
        except Exception as exc:
            rows.append({"window": name, "bd_nom": int(bd_nom), "chunk": int(chunk),
                         "bp": bp, "tp": tp_, "kvp": kvp, "bd": bd, "kvd": kvd,
                         "meas": meas, "new": None, "old": None, "err": type(exc).__name__})

df = pd.DataFrame(rows)
df.to_csv("fpm_e2e_20260811/mixed_validation_v3_fixed_formula.csv", index=False)
ok = df[df.new.notna()].copy()
ok["d_new"] = (ok.new - ok.meas) / ok.meas
ok["d_old"] = (ok.old - ok.meas) / ok.meas
print(f"mixed steps scored: {len(ok)}/{len(df)} (unanswerable: {len(df) - len(ok)})")
print(f"\n{'window':12s} {'n':>4s} {'meas med':>9s} {'old med':>8s} {'old δmed':>8s} {'new med':>8s} {'new δmed':>8s} {'new MAPE':>8s}")
for (bdn, ch), g in sorted(ok.groupby(["bd_nom", "chunk"])):
    print(f"bd{bdn}_c{ch:<6d} {len(g):4d} {g.meas.median():9.2f} {g.old.median():8.2f} "
          f"{g.d_old.median()*100:+7.1f}% {g.new.median():8.2f} {g.d_new.median()*100:+7.1f}% "
          f"{np.mean(np.abs(g.d_new))*100:7.2f}%")
print(f"\nGRID: old median|δ| {np.median(np.abs(ok.d_old))*100:.2f}%  ->  "
      f"new median|δ| {np.median(np.abs(ok.d_new))*100:.2f}%   "
      f"(new MAPE {np.mean(np.abs(ok.d_new))*100:.2f}%, p90|δ| {np.percentile(np.abs(ok.d_new), 90)*100:.2f}%)")
bound = ok[ok.chunk == 2048]
print(f"BOUNDARY rows (chunk=2048): old δmed {bound.d_old.median()*100:+.1f}%  ->  "
      f"new δmed {bound.d_new.median()*100:+.1f}%  (gate ±15%)")
