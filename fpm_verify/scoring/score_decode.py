# T3 decode 层三方打分:T2 真值(ShareGPT serving)vs 新采(kvwarm)vs 旧采。
# 口径:score-what-forms——每条 FPM 记录按实际形成的 (batch, kv_total) 入账。
#  - tep 单调度器 = 池即批,逐记录入账;
#  - dep 多 rank 流走配速者口径:DP 锁步下步 wall 由最慢 rank 决定,各 rank
#    第 k 个纯 decode 步同拍(锁步核验窗口精确;isl>1 窗口进场错位仅数步、
#    kv 慢变,配对误差可忽略),一步一票,坐标取最大负载 rank——治"小 shape
#    配大 wall"的稀释性高估。rank0-only 旧流自动退回逐记录口径。
# 同坐标滚动中位(±20 邻域),一坐标一票;A/B 分层按各自 DB 网格成员判定;
# F 层 = 低于网格地板挡位(per-req<2)的钳位坐标,rung-2 表示,单列不入 A/B。
# 多段真值:--stream/--windows 成对可重复(如 r11 v2 + r13 v3 合并入账)。
import argparse
import json

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp

ap = argparse.ArgumentParser()
ap.add_argument("--topo", choices=["tep4", "dep4", "tp4"], required=True)
ap.add_argument("--stream", required=True, nargs="+")
ap.add_argument("--windows", required=True, nargs="+")
ap.add_argument("--new-root", required=True)
ap.add_argument("--old-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model-id", default="MiniMaxAI/MiniMax-M2.7")
ap.add_argument("--system", default="h200_sxm")
ap.add_argument("--backend", default="vllm")
ap.add_argument("--backend-version", default="0.25.1")
args = ap.parse_args()
assert len(args.stream) == len(args.windows), "stream/windows 必须成对给"

DP = 4 if args.topo == "dep4" else 1
MOE_TP = 4 if args.topo == "tp4" else 1
MOE_EP = 1 if args.topo == "tp4" else 4

def make_query(root):
    perf_database.set_systems_paths([root])
    db = perf_database.get_database(args.system, args.backend, args.backend_version)
    FPMForwardOp.clear_cache()
    cfg = sdk_config.ModelConfig(
        # dep 库的既定查询口径 = tp4 形身份(先例:score_l3_dp.py)
        tp_size=4, pp_size=1, attention_dp_size=1,
        moe_tp_size=MOE_TP, moe_ep_size=MOE_EP, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8)
    op = FPMForwardOp("decode", cfg, args.model_id,
                      sol_fn=lambda b, k: 0.001, weight_bytes=1e9)
    pq = pd.read_parquet(
        f"{root}/data/{args.system}/{args.backend}/{args.backend_version}/fpm_forward_perf.parquet")
    dec = pq[pq.workload_kind == "decode"]
    grid = set(zip(dec.batch_size.astype(int), dec.total_kv_read_tokens.astype(int)))
    def q(b, kvt):
        try:
            return float(op.query_totals(db, batch_size=b, total_kv_read_tokens=kvt))
        except Exception:
            return None
    return q, grid

def load_steps(path):
    steps = []
    for line in open(path):
        try:
            d = json.loads(line); s = d["scheduled_requests"]
            steps.append((s["num_prefill_requests"], s["num_decode_requests"],
                          s["sum_decode_kv_tokens"], d["wall_time"] * 1000,
                          d.get("dp_rank", 0)))
        except Exception:
            steps.append(None)
    return steps

def _pace_key(b):
    """定拍键:cudagraph 捕获桶(stock 默认 1,2,4 + 8 的倍数至 512;>512 eager)。
    同桶内 forward 被 pad 到同尺寸、wall 近等,裸 (b,kv) 字典序会在桶边界
    选错定拍 rank(实测桶效应可达 +26%);跨 eager 界更是断崖(+146%)。"""
    if b > 512:
        return (1, b)
    if b <= 2:
        return (0, b)
    if b <= 4:
        return (0, 4)
    return (0, (b + 7) // 8 * 8)

# 真值坐标表:窗口内 score-what-forms,滚动中位
truth = {}
skew_windows = 0        # rank 步数差超限被弃的窗口
wall_spread_steps = 0   # 组内 wall max/min>1.2 的步(近等假设违例计数)
for spath, wpath in zip(args.stream, args.windows):
    steps = load_steps(spath)
    for ln in open(wpath):
        parts = ln.rstrip("\n").split("\t")
        # 窗口三代格式,前 7 列一致(tag,C,isl,osl,rep,s0,s1):
        # v1 7 列;v2 +ok(8 列);v3 +mark(9 列,lockstep/ragged/na)
        s0, s1 = int(parts[5]), int(parts[6])
        lockstep = len(parts) >= 9 and parts[8] == "lockstep" and int(parts[2]) == 1
        recs = [s for s in steps[s0:s1] if s and s[0] == 0 and s[1] >= 1]
        ranks = sorted({r[4] for r in recs})
        if len(ranks) > 1:
            # 配速者合并:各 rank 第 k 步同拍,wall 取组内中位(同步后近等,
            # 中位挡单 rank 计时毛刺),坐标取定拍 rank,一步一票
            per = {r: [x for x in recs if x[4] == r] for r in ranks}
            n, N = min(len(v) for v in per.values()), max(len(v) for v in per.values())
            if N > n * 1.25 + 2:
                skew_windows += 1
                continue  # rank 步数差过大,第 k 步同拍假设不可信,整窗弃
            seq_steps = []
            for k in range(n):
                group = [per[r][k] for r in ranks]
                walls_g = [g[3] for g in group]
                if max(walls_g) > min(walls_g) * 1.2:
                    wall_spread_steps += 1
                wall = float(np.median(walls_g))
                g = max(group, key=lambda x: (_pace_key(x[1]), x[2], x[1]))
                seq_steps.append((g[1], g[2], wall))
        else:
            seq_steps = [(x[1], x[2], x[3]) for x in recs]
        by_batch = {}
        for i, (b_, kv_, w_) in enumerate(seq_steps):
            if not lockstep and i < 4:
                continue  # 参差进场碎批,剔除;锁步核验窗口保留全程(kv=1..4 入账)
            by_batch.setdefault(b_, {}).setdefault(kv_, []).append(w_)
        for b, kvmap in by_batch.items():
            if sum(len(v) for v in kvmap.values()) < 50:
                continue  # 爬坡/收尾碎批,不构成稳定池
            seq = sorted((k, float(np.median(w))) for k, w in kvmap.items())
            if lockstep:
                # 锁步浅池:逐坐标直录不平滑——极浅端 wall 随 kv 有真实斜率,
                # ±20 邻域中位会把 kv=1 的真值抬向 kv≈11;噪声靠跨窗多票中位压
                for kvt, v in seq:
                    truth.setdefault((b, kvt), []).append(v)
                continue
            walls = np.array([w for _, w in seq])
            for i, (kvt, _) in enumerate(seq):
                lo, hi = max(0, i - 20), min(len(seq), i + 21)
                if hi - lo < 9:
                    continue  # 邻域太薄的碎批不入账
                truth.setdefault((b, kvt), []).append(float(np.median(walls[lo:hi])))
truth = {k: float(np.median(v)) for k, v in truth.items()}
print(f"真值坐标 {len(truth)} 个(score-what-forms);"
      f"弃窗(rank步数差)={skew_windows} wall近等违例步={wall_spread_steps}")

# 两个 root 不能共存缓存:逐侧构建、即时批量查询、暂存后配对
sides, grids = {}, {}
for tag, root in (("new", args.new_root), ("old", args.old_root)):
    q, grid = make_query(root)
    vals = {}
    for (b, kvt) in truth:
        if kvt < 2 * b:
            # 低于网格地板挡位(per-req<2):既定钳位语义,用 rung-2 行表示,
            # 单列 F 层如实报——有界诚实近似,不并入 A/B 精度口径
            vals[(b, kvt)] = (q(b, 2 * b), "F")
        else:
            vals[(b, kvt)] = (q(b, kvt), None)
    sides[tag], grids[tag] = vals, grid

rows, dropped = [], 0
for (b, kvt), tv in truth.items():
    (mv_new, st), (mv_old, _) = sides["new"][(b, kvt)], sides["old"][(b, kvt)]
    if mv_new is None or mv_old is None:
        dropped += 1  # 任一侧查不到即两侧同弃:保同坐标配对性,禁止单侧吃难点
        continue
    for tag, mv in (("new", mv_new), ("old", mv_old)):
        rows.append(dict(side=tag, b=b, kv=kvt, truth=tv, model=mv,
                         ape=abs(mv - tv) / tv,
                         stratum=st or ("A" if (b, kvt) in grids[tag] else "B")))
print(f"配对弃行(任一侧 None): {dropped} 坐标")
assert rows, "两侧库均零行——检查库载入"
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"\n===== {args.topo} decode 层:同真值坐标三方对比 =====")
for side in ("old", "new"):
    g = df[df.side == side]
    for st in sorted(g.stratum.unique()):
        gg = g[g.stratum == st]
        print(f"  {side:>3} {st} n={len(gg):5d}  MAPE={gg.ape.mean()*100:6.2f}%  "
              f"P95={gg.ape.quantile(.95)*100:6.2f}%  MAX={gg.ape.max()*100:6.2f}%")
    print(f"  {side:>3} 全 n={len(g):5d}  MAPE={g.ape.mean()*100:6.2f}%  "
          f"P95={g.ape.quantile(.95)*100:6.2f}%")
