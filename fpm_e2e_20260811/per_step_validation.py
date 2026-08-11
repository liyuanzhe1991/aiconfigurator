# Per-step validation: EVERY real-traffic engine step recorded by the FPM
# stream vs the model value composed from the formal parquet at that step's
# exact workload coordinates. No client/end-to-end metrics involved.
import json

import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

STREAM = "fpm_e2e_20260811/serve_results/fpm_stream_full.jsonl"
OUT = "fpm_e2e_20260811/per_step_validation.csv"

db = get_database("h200_sxm", "vllm", "0.25.1")
FPMForwardOp.clear_cache()
cfg = sdk_config.ModelConfig(
    tp_size=8, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=8, cp_size=1,
    gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    moe_quant_mode=common.MoEQuantMode.fp8_block,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    comm_quant_mode=common.CommQuantMode.half,
    kvcache_quant_mode=common.KVCacheQuantMode.fp8,
)
op_p = FPMForwardOp("prefill", cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=lambda b, tp, tk: 0.001, weight_bytes=1e9)
op_d = FPMForwardOp("decode", cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=lambda b, tk: 0.001, weight_bytes=1e9)

rows = []
for line in open(STREAM):
    d = json.loads(line)
    s = d["scheduled_requests"]
    bp, tp_, kvp = s["num_prefill_requests"], s["sum_prefill_tokens"], s["sum_prefill_kv_tokens"]
    bd, kvd = s["num_decode_requests"], s["sum_decode_kv_tokens"]
    if bp == 0 and bd == 0:
        continue  # heartbeat
    meas = d["wall_time"] * 1000
    if meas <= 0:
        continue
    kind = "decode" if bp == 0 else ("prefill" if bd == 0 else "mixed")
    model_ms = None
    note = ""
    try:
        if kind == "decode":
            model_ms = float(op_d.query(db, batch_size=bd, s=max(1, round(kvd / bd))))
        elif kind == "prefill":
            model_ms = float(op_p.query(db, batch_size=bp, s=max(1, round(tp_ / bp)), prefix=round(kvp / bp)))
        else:
            pre = float(op_p.query(db, batch_size=bp, s=max(1, round(tp_ / bp)), prefix=round(kvp / bp)))
            dec = float(op_d.query(db, batch_size=bd, s=max(1, round(kvd / bd))))
            base = float(op_d.query_pass_baseline(db, batch_size=bd))
            model_ms = pre + max(0.0, dec - base)
    except Exception as exc:  # out-of-domain etc.
        note = type(exc).__name__
    rows.append({
        "counter": d["counter_id"], "kind": kind,
        "n_prefill": bp, "prefill_tokens": tp_, "prefill_kv": kvp,
        "n_decode": bd, "decode_kv": kvd,
        "measured_ms": round(meas, 3),
        "model_ms": round(model_ms, 3) if model_ms is not None else None,
        "delta_pct": round((model_ms - meas) / meas * 100, 2) if model_ms else None,
        "note": note,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
ok = df[df.model_ms.notna()].copy()
print(f"steps total={len(df)}  scored={len(ok)}  unanswerable={len(df) - len(ok)}")
print(f"artifact: {OUT}\n")

print(f"{'class':22s} {'n':>6s} {'meas med':>9s} {'model med':>9s} {'delta med':>9s} {'delta p10':>9s} {'delta p90':>9s}")
def bucket(r):
    if r.kind == "decode":
        return f"decode B={int(r.n_decode)}"
    if r.kind == "mixed":
        return f"mixed  Bd={int(r.n_decode)}" if r.n_decode >= 32 else "mixed  Bd<32"
    return "prefill-only"
ok["bucket"] = ok.apply(bucket, axis=1)
for b, g in sorted(ok.groupby("bucket"), key=lambda kv: -len(kv[1])):
    if len(g) < 5:
        continue
    print(f"{b:22s} {len(g):6d} {g.measured_ms.median():9.2f} {g.model_ms.median():9.2f} "
          f"{g.delta_pct.median():+8.1f}% {g.delta_pct.quantile(.1):+8.1f}% {g.delta_pct.quantile(.9):+8.1f}%")
EOF_MARKER = None
