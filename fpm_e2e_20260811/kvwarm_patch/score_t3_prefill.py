# T3 prefill 层三方打分:T2 burst 真值 vs 新采 vs 旧采。
# 口径与 score_l3/score_l3_dep4 的 prefill 部分一致:burst 窗内取纯 prefill 步,
# 按实际形成的 (bp, total_prefill, kv_read) 入账(score-what-forms),
# 同坐标跨 rep 取中位;A/B 分层按各自 DB 网格;C 组(bp 超网格)单列。
import argparse
import json

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp

ap = argparse.ArgumentParser()
ap.add_argument("--topo", choices=["tep4", "dep4"], required=True)
ap.add_argument("--stream", required=True)
ap.add_argument("--windows", required=True, help="burst_windows.tsv")
ap.add_argument("--new-root", required=True)
ap.add_argument("--old-root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

def make_query(root):
    perf_database.set_systems_paths([root])
    db = perf_database.get_database("h200_sxm", "vllm", "0.25.1")
    FPMForwardOp.clear_cache()
    cfg = sdk_config.ModelConfig(
        tp_size=4, pp_size=1, attention_dp_size=1,
        moe_tp_size=1, moe_ep_size=4, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8)
    op = FPMForwardOp("prefill", cfg, "MiniMaxAI/MiniMax-M2.7",
                      sol_fn=lambda b, t, k: 0.001, weight_bytes=1e9)
    pq = pd.read_parquet(
        f"{root}/data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")
    pre = pq[pq.workload_kind == "prefill"]
    grid = set(zip(pre.batch_size.astype(int),
                   pre.total_prefill_tokens.astype(int),
                   pre.total_kv_read_tokens.astype(int)))
    bmax = int(pre.batch_size.max())
    def q(bp, tot, kv):
        try:
            return float(op.query_totals(
                db, batch_size=bp, total_prefill_tokens=tot,
                total_kv_read_tokens=kv))
        except Exception:
            return None
    return q, grid, bmax

steps = []
for line in open(args.stream):
    try:
        d = json.loads(line); s = d["scheduled_requests"]
        steps.append((s["num_prefill_requests"], s["sum_prefill_tokens"],
                      s["sum_prefill_kv_tokens"], s["num_decode_requests"],
                      d["wall_time"] * 1000, d.get("dp_rank", 0)))
    except Exception:
        steps.append(None)

# 真值:burst 窗(grp,bp,n,kv,rep,s0,s1);dep=均衡簇/blocker后独跑
# (判据移植自 score_l3_dep4.py:77-171),tep=形状直读跨 rep 中位。
DP = 4 if args.topo == "dep4" else 1
BUDGET = 8192
bw = pd.read_csv(args.windows, sep="\t",
                 names=["grp", "bp", "n", "kv", "rep", "s0", "s1"])
truth = {}
skipped = 0
for (grp, bp, n, kv), g in bw.groupby(["grp", "bp", "n", "kv"]):
    if DP > 1 and (n > BUDGET or bp % DP != 0):
        skipped += 1  # 单 rank 分块/不可整除:无均衡步存在(波语义不可通约)
        continue
    target = bp // DP
    tshape = (target, target * n, target * kv)
    def shape_of(s, n=n, kv=kv):
        if s[4] <= 0 or s[3] > 0 or s[0] == 0:
            return None
        if s[1] == s[0] * n and s[2] == s[0] * kv:
            return (s[0], s[1], s[2])
        return None
    hits, loners = [], []
    for r in g.itertuples():
        lines = [s for s in steps[int(r.s0):int(r.s1)] if s]
        if DP == 1:
            hits.extend(s[4] for s in lines if shape_of(s) == tshape)
            continue
        cur_shape, cur = None, {}
        for s in lines:
            shp = shape_of(s)
            if shp is None:
                continue
            if shp != cur_shape or s[5] in cur:
                cur_shape, cur = shp, {}
            cur[s[5]] = s[4]
            if len(cur) == DP:
                walls = list(cur.values())
                if max(walls) / min(walls) <= 1.03 and cur_shape == tshape:
                    hits.append(max(walls))
                cur_shape, cur = None, {}
        blocker_ranks = {s[5] for s in lines
                         if s[4] > 0 and s[3] == 0 and s[0] == 1
                         and s[1] == BUDGET and (1, BUDGET, s[2]) != tshape}
        if blocker_ranks and target * n <= 2048:
            loners.extend(s[4] for s in lines
                          if s[5] in blocker_ranks and shape_of(s) == tshape)
    vals = hits or loners
    if vals:
        truth.setdefault(tshape, []).extend(vals)
truth = {k: float(np.median(v)) for k, v in truth.items()}
print(f"prefill 真值坐标 {len(truth)} 个(不可通约窗跳过 {skipped})")

rows = []
for tag, root in (("new", args.new_root), ("old", args.old_root)):
    q, grid, bmax = make_query(root)
    for (bp, tot, kv), tv in truth.items():
        mv = q(bp, tot, kv)
        if mv is None:
            continue
        st = ("C" if bp > bmax else
              "A" if (bp, tot, kv) in grid else "B")
        rows.append(dict(side=tag, bp=bp, tot=tot, kv=kv, truth=tv, model=mv,
                         ape=abs(mv - tv) / tv, stratum=st))
assert rows
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"===== {args.topo} prefill 层 =====")
for side in ("old", "new"):
    g = df[df.side == side]
    for st in sorted(g.stratum.unique()):
        gg = g[g.stratum == st]
        print(f"  {side:>3} {st} n={len(gg):5d}  MAPE={gg.ape.mean()*100:6.2f}%  "
              f"P95={gg.ape.quantile(.95)*100:6.2f}%  MAX={gg.ape.max()*100:6.2f}%")
