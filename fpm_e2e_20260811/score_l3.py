# L3 offline scorer — per-step strata classification + the fixed metric set:
# per phase x stratum: MAPE / P95 APE / MAX APE (with coordinates), plus the
# C-stratum error-vs-extrapolation-multiple curve. Ground truth = FPM stream;
# model = the branch's fixed FPMForwardOp (query_totals path incl. batch clamp).
#
# Usage: score_l3.py --dir <harvest_dir> --parquet <...> --tp 4 [--model-path M]
import argparse
import json
import sys

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--parquet", required=True)
ap.add_argument("--tp", type=int, required=True)
ap.add_argument("--model-path", default="MiniMaxAI/MiniMax-M2.7")
args = ap.parse_args()
D = args.dir

db = get_database("h200_sxm", "vllm", "0.25.1")
FPMForwardOp.clear_cache()
cfg = sdk_config.ModelConfig(
    tp_size=args.tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=args.tp, cp_size=1,
    gemm_quant_mode=common.GEMMQuantMode.fp8_block, moe_quant_mode=common.MoEQuantMode.fp8_block,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16, comm_quant_mode=common.CommQuantMode.half,
    kvcache_quant_mode=common.KVCacheQuantMode.fp8)
op_p = FPMForwardOp("prefill", cfg, args.model_path, sol_fn=lambda b, t, k: 0.001, weight_bytes=1e9)
op_d = FPMForwardOp("decode", cfg, args.model_path, sol_fn=lambda b, k: 0.001, weight_bytes=1e9)

pq = pd.read_parquet(args.parquet)
pre = pq[(pq.workload_kind == "prefill") & (pq.tp == args.tp)]
dec = pq[(pq.workload_kind == "decode") & (pq.tp == args.tp)]
pre_grid = set(zip(pre.batch_size, pre.total_prefill_tokens, pre.total_kv_read_tokens))
dec_grid = set(zip(dec.batch_size, dec.total_kv_read_tokens))
B_c = int(pre.batch_size.max())

stream = []
for line in open(f"{D}/fpm_stream.jsonl"):
    try:
        d = json.loads(line)
        stream.append(d["scheduled_requests"] | {"wall": d["wall_time"] * 1000})
    except Exception:
        stream.append(None)

def q_pre(bp, tot, kv):
    try:
        return float(op_p.query_totals(db, batch_size=bp, total_prefill_tokens=tot, total_kv_read_tokens=kv))
    except Exception:
        return None

def q_dec(b, kvtot):
    try:
        return float(op_d.query_totals(db, batch_size=b, total_kv_read_tokens=kvtot))
    except Exception:
        return None

rows = []

# ---------------- PREFILL (burst windows) ----------------
bw = pd.read_csv(f"{D}/burst_windows.tsv", sep="\t",
                 names=["grp", "bp", "n", "kv", "rep", "s0", "s1"])
plan = pd.read_csv(f"{D}/prefill_plan.csv")
BUDGET = 8192
for (grp, bp, n, kv), g in bw.groupby(["grp", "bp", "n", "kv"]):
    single = n <= BUDGET
    meas = []
    for r in g.itertuples():
        for s in stream[int(r.s0):int(r.s1)]:
            if not s or s["num_decode_requests"] > 0:
                continue
            if single and (s["num_prefill_requests"] == bp
                           and s["sum_prefill_tokens"] == bp * n
                           and s["sum_prefill_kv_tokens"] == bp * kv):
                meas.append(s["wall"])
            elif not single and s["num_prefill_requests"] == bp and s["sum_prefill_tokens"] <= BUDGET * bp:
                # 长请求:逐 chunk 步按各自坐标计入(独立评分单元)
                mm = q_pre(bp, s["sum_prefill_tokens"], s["sum_prefill_kv_tokens"])
                if mm:
                    st = "A" if (bp, s["sum_prefill_tokens"], s["sum_prefill_kv_tokens"]) in pre_grid else "B"
                    rows.append(dict(phase="prefill", stratum=st, bp=bp, n=s["sum_prefill_tokens"],
                                     kv=s["sum_prefill_kv_tokens"], Bd=0, meas=s["wall"], model=mm,
                                     mult=1.0, src="长请求步"))
    if single and meas:
        med = float(np.median(meas))
        mm = q_pre(bp, bp * n, bp * kv)
        stratum = grp if grp in ("C", "Cpfx") else ("A" if (bp, bp * n, bp * kv) in pre_grid else "B")
        rows.append(dict(phase="prefill", stratum=stratum, bp=bp, n=n, kv=kv, Bd=0,
                         meas=med, model=mm, mult=bp / B_c if stratum.startswith("C") else 1.0,
                         src=f"burst n={len(meas)}"))

# ---------------- DECODE (window sweeps) ----------------
try:
    dw = pd.read_csv(f"{D}/decode_windows.tsv", sep="\t",
                     names=["grp", "C", "isl", "osl", "kind", "s0", "s1"])
except FileNotFoundError:
    dw = pd.DataFrame()
for r in dw.itertuples():
    seen = set()
    for s in stream[int(r.s0):int(r.s1)]:
        if not s or s["num_prefill_requests"] > 0 or s["num_decode_requests"] != r.C:
            continue
        kvt = s["sum_decode_kv_tokens"]
        if kvt in seen:
            continue
        seen.add(kvt)
        mm = q_dec(r.C, kvt)
        if mm:
            st = "A" if (r.C, kvt) in dec_grid else "B"
            rows.append(dict(phase="decode", stratum=st, bp=0, n=0, kv=kvt, Bd=r.C,
                             meas=s["wall"], model=mm, mult=1.0, src=r.kind))

# ---------------- MIXED (pool windows) ----------------
try:
    mw = pd.read_csv(f"{D}/mixed_windows.tsv", sep="\t",
                     names=["grp", "bp", "Bd", "chunk", "pfl", "s0", "s1"])
except FileNotFoundError:
    mw = pd.DataFrame()
for r in mw.itertuples():
    meas = []
    kvds = []
    for s in stream[int(r.s0):int(r.s1)]:
        if not s:
            continue
        if (s["num_prefill_requests"] == max(1, r.bp) and s["num_decode_requests"] == r.Bd
                and s["sum_prefill_tokens"] == r.chunk
                and s["sum_prefill_kv_tokens"] == max(1, r.bp) * r.pfl):
            meas.append(s["wall"]); kvds.append(s["sum_decode_kv_tokens"])
    if len(meas) >= 5:
        med = float(np.median(meas)); kvd = int(np.median(kvds))
        bp_eff = max(1, r.bp)
        pre_v = q_pre(bp_eff, r.chunk + r.Bd, bp_eff * r.pfl)
        dec_v = q_dec(r.Bd, kvd)
        try:
            base = float(op_d.query_pass_baseline(db, batch_size=r.Bd))
        except Exception:
            base = None
        if pre_v and dec_v and base is not None:
            mm = pre_v + max(0.0, dec_v - base)
            comp_on = (bp_eff, r.chunk + r.Bd, bp_eff * r.pfl) in pre_grid
            st = ("C" if r.grp.startswith("C") and r.bp > B_c else
                  "C锚" if r.grp.startswith("C") else ("A" if comp_on else "B"))
            rows.append(dict(phase="mixed", stratum=st, bp=bp_eff, n=r.chunk, kv=bp_eff * r.pfl,
                             Bd=r.Bd, meas=med, model=mm,
                             mult=bp_eff / B_c if st == "C" else 1.0, src=f"稳态n={len(meas)}"))

df = pd.DataFrame(rows)
df["ape"] = (df.model - df.meas).abs() / df.meas
df.to_csv(f"{D}/l3_scores.csv", index=False)

print(f"===== L3 评分 — {len(df)} 个评分单元(tp={args.tp})=====")
for (ph, st), g in df.groupby(["phase", "stratum"]):
    p95 = g.ape.quantile(0.95); mx = g.ape.max()
    r95 = g.iloc[(g.ape - p95).abs().argmin()]; rmx = g.loc[g.ape.idxmax()]
    print(f"{ph:8s} {st:4s} n={len(g):5d}  MAPE={g.ape.mean()*100:6.2f}%  "
          f"P95={p95*100:6.2f}% @(bp{int(r95.bp)},n{int(r95.n)},kv{int(r95.kv)},Bd{int(r95.Bd)})  "
          f"MAX={mx*100:6.2f}% @(bp{int(rmx.bp)},n{int(rmx.n)},kv{int(rmx.kv)},Bd{int(rmx.Bd)})")
c = df[(df.phase == "prefill") & (df.stratum.isin(["C", "Cpfx"]))]
if len(c):
    print("\n----- C 组:误差 vs 外插倍数 -----")
    for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 16.01)):
        g = c[(c.mult > lo) & (c.mult <= hi)]
        if len(g):
            print(f"  {lo}x-{hi:.0f}x: n={len(g):3d}  MAPE={g.ape.mean()*100:6.2f}%  "
                  f"P95={g.ape.quantile(0.95)*100:6.2f}%  MAX={g.ape.max()*100:6.2f}%")
print(f"\nscores -> {D}/l3_scores.csv")
