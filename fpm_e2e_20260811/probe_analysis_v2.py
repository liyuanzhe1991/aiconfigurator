# v2 probe analysis against the RANDTOK parquet.
# Both manifests are config-matched to their parquet regime:
#   decode probes  ran under the decode-cell engine config,
#   prefill probes ran under the prefill-cell engine config (sync sched,
#   graphs<=2048, prefix caching ON) — so prefill points ARE parity-comparable.
# A-group = on-grid anchors (drift/noise); B-group = off-grid (interpolation).
# Prints EVERY point; writes probes_v2_scores.csv.
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

db = get_database("h200_sxm", "vllm", "0.25.1")
FPMForwardOp.clear_cache()
pq = pd.read_parquet("fpm_formal_database_randtok/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")


def make_op(phase, tp):
    cfg = sdk_config.ModelConfig(
        tp_size=tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=tp, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block, moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16, comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8)
    sol = (lambda b, tp_, tk: 0.001) if phase == "prefill" else (lambda b, tk: 0.001)
    return FPMForwardOp(phase, cfg, "MiniMaxAI/MiniMax-M2.7", sol_fn=sol, weight_bytes=1e9)


out_rows = []
for tp in (4, 8):
    sub = pq[pq.tp == tp]
    dec_rows = {(r.batch_size, r.total_kv_read_tokens): r.latency_ms
                for r in sub[sub.workload_kind == "decode"].itertuples()}
    pre_rows = {(r.batch_size, r.total_prefill_tokens, r.total_kv_read_tokens): r.latency_ms
                for r in sub[sub.workload_kind == "prefill"].itertuples()}
    op_d, op_p = make_op("decode", tp), make_op("prefill", tp)

    for cfg_name in ("decode", "prefill"):
        files = sorted(glob.glob(f"fpm_e2e_20260811/probes_v2/tep{tp}/{cfg_name}_r*.json"))
        if not files:
            continue
        reps = []
        for f in files:
            r = int(re.search(r"_r(\d+)\.json", f).group(1))
            d = json.load(open(f))
            for g in d["iteration_groups"]:
                p = g["point"]
                reps.append({"rep": r, "kind": p["point_type"], "b": p["batch_size"],
                             "tokens": p["total_prefill_tokens"], "kv": p["total_kv_read_tokens"],
                             "ms": g["wall_time"] * 1000, "complete": g["complete"]})
        df = pd.DataFrame(reps)
        n_bad = int((~df.complete).sum())
        df = df[df.complete]
        agg = df.groupby(["kind", "b", "tokens", "kv"]).ms.agg(
            med="median", n="size",
            spread=lambda s: (s.max() - s.min()) / s.median() if len(s) > 1 else 0.0).reset_index()

        print(f"\n========== TEP{tp} / {cfg_name}-config probes "
              f"({len(files)} repeats, {len(agg)} points, incomplete dropped: {n_bad}) ==========")
        hdr = f"{'grp':3s} {'kind':7s} {'b':>4s} {'tokens':>7s} {'kv_total':>9s} {'probe med':>10s} {'spread':>7s} {'reference':>10s} {'delta':>7s}"
        print(hdr)
        a_d, b_d, spreads = [], [], []
        for r in agg.itertuples():
            if r.kind == "decode":
                key = (r.b, r.kv)
                on = key in dec_rows
                ref = dec_rows[key] if on else float(op_d.query(db, batch_size=int(r.b), s=max(1, round(r.kv / r.b))))
            else:
                key = (r.b, r.tokens, r.kv)
                on = key in pre_rows
                ref = pre_rows[key] if on else float(op_p.query(
                    db, batch_size=int(r.b), s=max(1, round(r.tokens / r.b)), prefix=round(r.kv / r.b)))
            grp = "A" if on else "B"
            delta = (r.med - ref) / ref
            (a_d if on else b_d).append(delta)
            spreads.append(r.spread)
            print(f"{grp:3s} {r.kind:7s} {int(r.b):4d} {int(r.tokens):7d} {int(r.kv):9d} "
                  f"{r.med:10.3f} {r.spread*100:6.1f}% {ref:10.3f} {delta*100:+6.1f}%")
            out_rows.append({"tp": tp, "config": cfg_name, "group": grp, "kind": r.kind,
                             "b": int(r.b), "tokens": int(r.tokens), "kv": int(r.kv),
                             "probe_med_ms": r.med, "n_reps": int(r.n), "spread": r.spread,
                             "reference_ms": ref, "delta": delta})
        for name, arr in (("A(anchor drift)", a_d), ("B(interpolation)", b_d)):
            if arr:
                arr = np.array(arr)
                print(f"{name}: n={len(arr)}  MAPE={np.mean(np.abs(arr))*100:.2f}%  "
                      f"median|d|={np.median(np.abs(arr))*100:.2f}%  p90|d|={np.percentile(np.abs(arr),90)*100:.2f}%  "
                      f"max|d|={np.max(np.abs(arr))*100:.2f}%  (signed median {np.median(arr)*100:+.2f}%)")
        if spreads:
            print(f"repeat noise floor: median spread {np.median(spreads)*100:.2f}%")

os.makedirs("fpm_e2e_20260811", exist_ok=True)
pd.DataFrame(out_rows).to_csv("fpm_e2e_20260811/probes_v2_scores.csv", index=False)
print(f"\nwrote fpm_e2e_20260811/probes_v2_scores.csv ({len(out_rows)} rows)")
