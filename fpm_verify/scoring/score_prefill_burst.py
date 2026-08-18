# r13 prefill burst 三方打分:真值 = parity stock 栈下重收的 burst 窗。
# tep(dp=1):窗内精确命中目标形 (bp, bp*n, bp*kv) 的纯 prefill 步,跨 rep 中位。
# dep(dp>1):多 rank 遥测下的配速者口径——逐 rank 各取一条组成同拍组,
#   步 wall=组内最大(锁步同步语义),坐标记最大负载 rank 的实际形,
#   一组一票,同坐标跨组中位。分层 A/B/C(C: bp 超网格上限)。
# 配对弃行:任一侧查询 None 即两侧同弃,保同坐标配对性。
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
ap.add_argument("--windows", required=True)
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

steps = []
for line in open(args.stream):
    try:
        d = json.loads(line); s = d["scheduled_requests"]
        steps.append((s["num_prefill_requests"], s["sum_prefill_tokens"],
                      s["sum_prefill_kv_tokens"], s["num_decode_requests"],
                      d["wall_time"] * 1000, d.get("dp_rank", 0)))
    except Exception:
        steps.append(None)

truth = {}
groups = 0
for ln in open(args.windows):
    f = ln.rstrip("\n").split("\t")
    grp, bp, n, kv = f[0], int(f[1]), int(f[2]), int(f[3])
    s0, s1 = int(f[5]), int(f[6])
    recs = [s for s in steps[s0:s1]
            if s and s[0] >= 1 and s[3] == 0 and s[4] > 0]
    if DP == 1:
        tshape = (bp, bp * n, bp * kv)
        for s in recs:
            if (s[0], s[1], s[2]) == tshape:
                truth.setdefault(tshape, []).append(s[4])
        continue
    # dep 均衡簇(与驱动器 dp_balanced 同语义):窗内逐 rank 取首条
    # "每 rank 恰 bp/DP 份"同形纯 prefill 记录,DP 个 rank 齐且
    # wall 差 ≤3% 记一票(取最大 wall=锁步语义),一窗一票;
    # 坐标 = 每 rank 形(与网格同轴,可通约)。
    # 不要求记录连续:rank 交错/夹杂他步不构成否决(驱动器验收即此语义)
    tb = bp // DP
    tgt = (tb, tb * n, tb * kv)
    walls = {}
    for s in recs:
        if (s[0], s[1], s[2]) == tgt and s[5] not in walls:
            walls[s[5]] = s[4]
            if len(walls) >= DP:
                break
    if len(walls) >= DP:
        v = list(walls.values())
        if max(v) <= min(v) * 1.03:
            truth.setdefault(tgt, []).append(max(v))
            groups += 1
truth = {k: float(np.median(v)) for k, v in truth.items()}
print(f"prefill 真值坐标 {len(truth)} 个" + (f"(同拍组 {groups})" if DP > 1 else ""))

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
print(f"\n===== {args.topo} prefill burst 层:同真值坐标三方对比 =====")
for side in ("old", "new"):
    g = df[df.side == side]
    for st in sorted(g.stratum.unique()):
        gg = g[g.stratum == st]
        print(f"  {side:>3} {st} n={len(gg):5d}  MAPE={gg.ape.mean()*100:6.2f}%  "
              f"P95={gg.ape.quantile(.95)*100:6.2f}%  MAX={gg.ape.max()*100:6.2f}%")
    print(f"  {side:>3} 全 n={len(g):5d}  MAPE={g.ape.mean()*100:6.2f}%  "
          f"P95={g.ape.quantile(.95)*100:6.2f}%")
