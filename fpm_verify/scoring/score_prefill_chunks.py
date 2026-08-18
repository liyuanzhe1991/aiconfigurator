# r13 prefill 层三方打分:真值 = L3 收割流中的全部纯 prefill 步(ShareGPT
# 真实内容灌注,含 chunked 续段),score-what-forms 按实际形成的
# (bp, total_prefill_tokens, kv_read) 入账,同坐标跨步中位。
# dep4 口径移植 score_t3_prefill(均衡簇:连续同形 × DP rank 齐、wall 差 ≤3%,
# 一簇一票取最大 wall);tep4 逐步直读。配对弃行与 decode 打分器同源。
# 分层:A=坐标在网格;B=网格内插;C=bp 超网格上限(单列)。
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
ap.add_argument("--stream", required=True, nargs="+")
ap.add_argument("--new-root", required=True)
ap.add_argument("--old-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model-id", default="MiniMaxAI/MiniMax-M2.7")
ap.add_argument("--system", default="h200_sxm")
ap.add_argument("--backend", default="vllm")
ap.add_argument("--backend-version", default="0.25.1")
args = ap.parse_args()

DP = 4 if args.topo == "dep4" else 1

def make_query(root):
    perf_database.set_systems_paths([root])
    db = perf_database.get_database(args.system, args.backend, args.backend_version)
    FPMForwardOp.clear_cache()
    cfg = sdk_config.ModelConfig(
        tp_size=4, pp_size=1, attention_dp_size=1,
        moe_tp_size=1, moe_ep_size=4, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8)
    op = FPMForwardOp("prefill", cfg, args.model_id,
                      sol_fn=lambda b, t, k: 0.001, weight_bytes=1e9)
    pq = pd.read_parquet(
        f"{root}/data/{args.system}/{args.backend}/{args.backend_version}/fpm_forward_perf.parquet")
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

truth = {}
clusters = loners_skipped = 0
for spath in args.stream:
    recs = []
    for line in open(spath):
        try:
            d = json.loads(line); s = d["scheduled_requests"]
            if (s["num_prefill_requests"] >= 1 and s["num_decode_requests"] == 0
                    and d["wall_time"] > 0):
                recs.append((s["num_prefill_requests"], s["sum_prefill_tokens"],
                             s["sum_prefill_kv_tokens"], d["wall_time"] * 1000,
                             d.get("dp_rank", 0)))
        except Exception:
            continue
    if DP == 1:
        for bp, tot, kv, w, _ in recs:
            truth.setdefault((bp, tot, kv), []).append(w)
        continue
    # dep:连续同形 × DP rank 齐 + wall 近等(≤3%)才构成均衡簇,一簇一票。
    # 形不齐/wall 散的步是跨 rank 波形错位,单 rank wall 含同步等待,弃
    cur_shape, cur = None, {}
    for bp, tot, kv, w, rank in recs:
        shp = (bp, tot, kv)
        if shp != cur_shape or rank in cur:
            if cur_shape is not None and len(cur) < DP:
                loners_skipped += 1
            cur_shape, cur = shp, {}
        cur[rank] = w
        if len(cur) == DP:
            walls = list(cur.values())
            if max(walls) / min(walls) <= 1.03:
                truth.setdefault(cur_shape, []).append(max(walls))
                clusters += 1
            cur_shape, cur = None, {}
truth = {k: float(np.median(v)) for k, v in truth.items()}
print(f"prefill 真值坐标 {len(truth)} 个"
      + (f"(均衡簇 {clusters},弃散簇 {loners_skipped})" if DP > 1 else ""))

sides, grids, bmaxes = {}, {}, {}
for tag, root in (("new", args.new_root), ("old", args.old_root)):
    q, grid, bmax = make_query(root)
    sides[tag] = {k: q(*k) for k in truth}
    grids[tag], bmaxes[tag] = grid, bmax

rows, dropped = [], 0
for k, tv in truth.items():
    if sides["new"][k] is None or sides["old"][k] is None:
        dropped += 1
        continue
    for tag in ("new", "old"):
        mv = sides[tag][k]
        st = ("C" if k[0] > bmaxes[tag]
              else "A" if k in grids[tag] else "B")
        rows.append(dict(side=tag, bp=k[0], tot=k[1], kv=k[2], truth=tv,
                         model=mv, ape=abs(mv - tv) / tv, stratum=st))
print(f"配对弃行(任一侧 None): {dropped} 坐标")
assert rows, "两侧库均零行"
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"\n===== {args.topo} prefill 层:同真值坐标三方对比 =====")
for side in ("old", "new"):
    g = df[df.side == side]
    for st in sorted(g.stratum.unique()):
        gg = g[g.stratum == st]
        print(f"  {side:>3} {st} n={len(gg):5d}  MAPE={gg.ape.mean()*100:6.2f}%  "
              f"P95={gg.ape.quantile(.95)*100:6.2f}%  MAX={gg.ape.max()*100:6.2f}%")
    print(f"  {side:>3} 全 n={len(g):5d}  MAPE={g.ape.mean()*100:6.2f}%  "
          f"P95={g.ape.quantile(.95)*100:6.2f}%")
