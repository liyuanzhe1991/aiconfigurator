# L3 offline scorer — dep4 (dp=4) variant. Ground truth = per-rank FPM stream.
#
# Core constraint: the database's coordinate system is per-DP-rank totals under
# balanced_v1 — the collector measured every coordinate with ALL FOUR ranks
# concurrently running the same shape, so a coordinate's latency includes the
# full EP all-to-all load of four balanced ranks. A real-traffic step where
# only some ranks are active spreads its MoE tokens over all four ranks
# (per-rank expert load up to 4x lighter, with a non-linear latency-bound
# floor), so its wall is NOT comparable to the balanced coordinate. Empirical:
# bp=1 windows measure ~2x faster than the balanced grid value at n>=4096.
#
# Therefore prefill/mixed windows are scored ONLY when the stream shows a
# balanced step: a cluster of 4 lines (one per dp_rank, adjacent in arrival
# order) with the identical consistent shape; the cluster wall is the max over
# ranks (collector semantics). Windows that cannot form balanced steps
# (bp not divisible by 4, single-rank long-request chunking, single-rank mixed
# probes) are reported as structurally incommensurable, not as model error.
#
# Decode windows are balanced by construction (phase script pre-multiplied the
# pool by DP so every rank holds C decodes) — verified per-rank kv spread
# ~0.05% — and are scored per line like the tep scorer.
#
# Stratum + extrapolation multiple are classified against the OBSERVED
# per-rank batch vs B_c (per-rank parquet max), not the plan's grp label.
#
# Usage: score_l3_dep4.py --dir <harvest> --plans <bundle_dir> --root <analysis_systems_root>
import argparse
import json
import os

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--plans", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--model-path", default="MiniMaxAI/MiniMax-M2.7")
args = ap.parse_args()
D = args.dir

db = get_database("h200_sxm", "vllm", "0.25.1", systems_paths=[os.path.abspath(args.root)])
FPMForwardOp.clear_cache()
cfg = sdk_config.ModelConfig(
    tp_size=1, pp_size=1, attention_dp_size=4, moe_tp_size=1, moe_ep_size=4, cp_size=1,
    gemm_quant_mode=common.GEMMQuantMode.fp8_block, moe_quant_mode=common.MoEQuantMode.fp8_block,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16, comm_quant_mode=common.CommQuantMode.half,
    kvcache_quant_mode=common.KVCacheQuantMode.fp8)
op_p = FPMForwardOp("prefill", cfg, args.model_path, sol_fn=lambda b, t, k: 0.001, weight_bytes=1e9)
op_d = FPMForwardOp("decode", cfg, args.model_path, sol_fn=lambda b, k: 0.001, weight_bytes=1e9)

pq = pd.read_parquet(f"{args.root}/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")
pre = pq[pq.workload_kind == "prefill"]
dec = pq[pq.workload_kind == "decode"]
pre_grid = set(zip(pre.batch_size, pre.total_prefill_tokens, pre.total_kv_read_tokens))
dec_grid = set(zip(dec.batch_size, dec.total_kv_read_tokens))
B_c = int(pre.batch_size.max())

stream = []
for line in open(f"{D}/fpm_stream.jsonl"):
    try:
        d = json.loads(line)
        s = d["scheduled_requests"]
        s["wall"] = d["wall_time"] * 1000
        s["rank"] = d.get("dp_rank", 0)
        stream.append(s)
    except Exception:
        stream.append(None)

DP = 4

def balanced_clusters(lines, shape_of, tol=1.03):
    """Group active lines into balanced clean steps: DP adjacent-in-arrival
    lines, one per rank, identical shape, walls mutually within tol (lockstep
    drag by a concurrent heavier step shows up as a 2-3x wall outlier — e.g.
    the burst blocker inflates concurrent measure steps from 88 to 258ms — and
    disqualifies the cluster). Returns [(shape, max_wall), ...]."""
    out = []
    cur_shape, cur = None, {}
    for s in lines:
        shp = shape_of(s)
        if shp is None:
            continue
        if shp != cur_shape or s["rank"] in cur:
            cur_shape, cur = shp, {}
        cur[s["rank"]] = s["wall"]
        if len(cur) == DP:
            walls = list(cur.values())
            if max(walls) / min(walls) <= tol:
                out.append((cur_shape, max(walls)))
            cur_shape, cur = None, {}
    return out

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

def pre_stratum(npf, tot, kv):
    if (npf, tot, kv) in pre_grid:
        return "A", 1.0
    if npf > B_c:
        return ("Cpfx" if kv > 0 else "C"), npf / B_c
    return "B", 1.0

rows = []
dropped = []

# ---------------- PREFILL (burst windows) ----------------
bw = pd.read_csv(f"{D}/burst_windows.tsv", sep="\t",
                 names=["grp", "bp", "n", "kv", "rep", "s0", "s1"])
BUDGET = 8192
incommensurable = []
for (grp, bp, n, kv), g in bw.groupby(["grp", "bp", "n", "kv"]):
    single = n <= BUDGET
    if not single or bp % DP != 0:
        # single-rank chunking / bp not divisible by DP: no balanced step exists
        incommensurable.append(f"prefill ({grp},bp{bp},n{n},kv{kv})")
        continue
    def shape_of(s, n=n, kv=kv):
        if s["wall"] <= 0 or s["num_decode_requests"] > 0:
            return None
        npf = s["num_prefill_requests"]
        if npf > 0 and s["sum_prefill_tokens"] == npf * n and s["sum_prefill_kv_tokens"] == npf * kv:
            return (npf, npf * n, npf * kv)
        return None
    target = bp // DP
    tshape = (target, target * n, target * kv)
    by_shape = {}
    loners = []
    for r in g.itertuples():
        lines = [s for s in stream[int(r.s0):int(r.s1)] if s]
        for shp, wall in balanced_clusters(lines, shape_of):
            by_shape.setdefault(shp, []).append(wall)
        # blocker windows (bp>=6): the blocker's own rank runs its measure step
        # AFTER the blocker finishes, alone (other ranks already done+idle) —
        # drag-free. With per-rank totals <=2048 the EP-dilution effect sits in
        # the grouped-GEMM latency floor (validated -3.9% at (4,1024,0)), so
        # the loner is a clean estimate for these small-total windows.
        blocker_ranks = {s["rank"] for s in lines
                         if s["wall"] > 0 and s["num_decode_requests"] == 0
                         and s["num_prefill_requests"] == 1
                         and s["sum_prefill_tokens"] == BUDGET and (1, BUDGET, s["sum_prefill_kv_tokens"]) != tshape}
        if blocker_ranks and target * n <= 2048:
            loners.extend(s["wall"] for s in lines
                          if s["rank"] in blocker_ranks and shape_of(s) == tshape)
    hit, src = by_shape.get(tshape), "均衡簇"
    if not hit and loners:
        hit, src = loners, "blocker后独跑"
    if not hit:
        dropped.append(f"prefill ({grp},bp{bp},n{n},kv{kv}): 无均衡簇/独跑 (簇形状: {sorted(by_shape)})")
        continue
    npf, tot, kvt = tshape
    mm = q_pre(npf, tot, kvt)
    if mm:
        st, mult = pre_stratum(npf, tot, kvt)
        rows.append(dict(phase="prefill", stratum=st, bp=npf, n=n, kv=kv, Bd=0,
                         meas=float(np.median(hit)), model=mm, mult=mult,
                         src=f"burst plan_bp{bp} {src}n={len(hit)}"))

# ---------------- DECODE (window sweeps; C is per-rank target) ----------------
try:
    dw = pd.read_csv(f"{D}/decode_windows.tsv", sep="\t",
                     names=["grp", "C", "isl", "osl", "kind", "s0", "s1"])
except FileNotFoundError:
    dw = pd.DataFrame()
for r in dw.itertuples():
    seq = []
    seen = set()
    for s in stream[int(r.s0):int(r.s1)]:
        if not s or s["wall"] <= 0 or s["num_prefill_requests"] > 0 or s["num_decode_requests"] != r.C:
            continue
        kvt = s["sum_decode_kv_tokens"]
        if kvt in seen:
            continue
        seen.add(kvt)
        seq.append((kvt, s["wall"]))
    walls = np.array([w for _, w in seq])
    for i, (kvt, _) in enumerate(seq):
        lo = max(0, i - 20); hi = min(len(seq), i + 21)
        wall = float(np.median(walls[lo:hi]))
        mm = q_dec(r.C, kvt)
        if mm:
            st = "A" if (r.C, kvt) in dec_grid else "B"
            rows.append(dict(phase="decode", stratum=st, bp=0, n=0, kv=kvt, Bd=r.C,
                             meas=wall, model=mm, mult=1.0, src=r.kind))

# ---------------- MIXED (pool windows; Bd is per-rank target) ----------------
try:
    mw = pd.read_csv(f"{D}/mixed_windows.tsv", sep="\t",
                     names=["grp", "bp", "Bd", "chunk", "pfl", "s0", "s1"])
except FileNotFoundError:
    mw = pd.DataFrame()
for r in mw.itertuples():
    bp_plan = max(1, r.bp)
    if bp_plan % DP != 0:
        # probe occupies 1-2 ranks; the other ranks run pure decode — the mixed
        # step's MoE load is diluted vs the balanced coordinate
        incommensurable.append(f"mixed ({r.grp},bp{r.bp},Bd{r.Bd},chunk{r.chunk},pfl{r.pfl})")
        continue
    req_len = r.chunk // bp_plan
    target = bp_plan // DP
    clusters = []
    cur = {}
    for s in stream[int(r.s0):int(r.s1)]:
        if not s or s["wall"] <= 0:
            continue
        npf = s["num_prefill_requests"]
        ok = (npf == target and abs(s["num_decode_requests"] - r.Bd) <= 1
              and s["sum_prefill_tokens"] == npf * req_len
              and s["sum_prefill_kv_tokens"] == npf * r.pfl)
        if not ok:
            if s["num_decode_requests"] > 0 or npf > 0:
                cur = {}
            continue
        if s["rank"] in cur:
            cur = {}
        cur[s["rank"]] = (s["wall"], s["num_decode_requests"], s["sum_decode_kv_tokens"])
        if len(cur) == DP:
            walls = [v[0] for v in cur.values()]
            if max(walls) / min(walls) <= 1.03:
                clusters.append((max(walls),
                                 int(np.median([v[1] for v in cur.values()])),
                                 int(np.median([v[2] for v in cur.values()]))))
            cur = {}
    if len(clusters) < 5:
        dropped.append(f"mixed ({r.grp},bp{r.bp},Bd{r.Bd},chunk{r.chunk},pfl{r.pfl}): {len(clusters)} 均衡簇")
        continue
    med = float(np.median([w for w, _, _ in clusters]))
    ndec = int(np.median([nd for _, nd, _ in clusters]))
    kvd = int(np.median([dk for _, _, dk in clusters]))
    npf = target
    tok = npf * req_len
    kvp = npf * r.pfl
    pre_v = q_pre(npf, tok + ndec, kvp)
    dec_v = q_dec(ndec, kvd)
    try:
        base = float(op_d.query_pass_baseline(db, batch_size=ndec))
    except Exception:
        base = None
    if pre_v and dec_v and base is not None:
        mm = pre_v + max(0.0, dec_v - base)
        if npf > B_c:
            st, mult = "C", npf / B_c
        elif str(r.grp).startswith("C"):
            st, mult = "C锚", 1.0
        else:
            st = "A" if (npf, tok + ndec, kvp) in pre_grid else "B"
            mult = 1.0
        rows.append(dict(phase="mixed", stratum=st, bp=npf, n=tok, kv=kvp, Bd=ndec,
                         meas=med, model=mm, mult=mult, src=f"稳态均衡簇n={len(clusters)} plan_bp{r.bp}"))

df = pd.DataFrame(rows)
df["ape"] = (df.model - df.meas).abs() / df.meas
df.to_csv(f"{D}/l3_scores.csv", index=False)

print(f"===== L3 评分 — dep4 (dp=4, tp=1, moe_ep=4) — {len(df)} 个评分单元 =====")
print(f"B_c (per-rank) = {B_c}")
for (ph, st), g in df.groupby(["phase", "stratum"]):
    p95 = g.ape.quantile(0.95); mx = g.ape.max()
    r95 = g.iloc[(g.ape - p95).abs().argmin()]; rmx = g.loc[g.ape.idxmax()]
    print(f"{ph:8s} {st:4s} n={len(g):5d}  MAPE={g.ape.mean()*100:6.2f}%  "
          f"P95={p95*100:6.2f}% @(bp{int(r95.bp)},n{int(r95.n)},kv{int(r95.kv)},Bd{int(r95.Bd)})  "
          f"MAX={mx*100:6.2f}% @(bp{int(rmx.bp)},n{int(rmx.n)},kv{int(rmx.kv)},Bd{int(rmx.Bd)})")
c = df[(df.phase == "prefill") & (df.stratum.isin(["C", "Cpfx"]))]
if len(c):
    print("\n----- C 组:误差 vs 外插倍数(per-rank)-----")
    for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 16.01)):
        g = c[(c.mult > lo) & (c.mult <= hi)]
        if len(g):
            print(f"  {lo}x-{hi:.0f}x: n={len(g):3d}  MAPE={g.ape.mean()*100:6.2f}%  "
                  f"P95={g.ape.quantile(0.95)*100:6.2f}%  MAX={g.ape.max()*100:6.2f}%")
if incommensurable:
    print(f"\n----- 结构性不可通约窗口(非均衡步 vs balanced_v1 坐标,不计入精度)({len(incommensurable)})-----")
    from collections import Counter
    kinds = Counter(w.split(" ")[0] for w in incommensurable)
    print("  按相:", dict(kinds))
if dropped:
    print(f"\n----- 无有效均衡样本被弃的窗口({len(dropped)})-----")
    for d_ in dropped:
        print(" ", d_)
print(f"\nscores -> {D}/l3_scores.csv")
