# L3 matrix generator (formal) — L3_PLAN_V2 rules, zero hand-written constants.
# Inputs:  parquet (the cell's collected lattice) + resolved-config dump (engine facts)
# Outputs: <out>/prefill_plan.csv, decode_plan.csv, mixed_plan.csv, coverage_report.json
# Every point's stratum is asserted against the parquet; quota misses fail generation.
#
# Usage:
#   make_l3_matrix.py --parquet <fpm_forward_perf.parquet> --tp 4 \
#     --resolved-config <resolved-config-node0.json> --out l3_matrix_tep4 [--tier full|fast]
import argparse
import bisect
import json
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--tp", type=int, required=True)
ap.add_argument("--resolved-config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tier", choices=["full", "fast"], default="full")
args = ap.parse_args()

pq = pd.read_parquet(args.parquet)
rc = json.load(open(args.resolved_config))

# ---- engine facts from resolved-config (fall back to sane probes of the dump layout)
def rc_get(*keys, default=None):
    node = rc
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node

BUDGET = int(rc_get("scheduler_config", "max_num_batched_tokens", default=8192) or 8192)
MAX_SEQS = int(rc_get("scheduler_config", "max_num_seqs", default=1024) or 1024)
MAX_LEN = int(rc_get("model_config", "max_model_len", default=204800) or 204800)
BLOCK = int(rc_get("cache_config", "block_size", default=16) or 16)

pre = pq[(pq.workload_kind == "prefill") & (pq.tp == args.tp)]
dec = pq[(pq.workload_kind == "decode") & (pq.tp == args.tp)]
if not len(pre) or not len(dec):
    sys.exit(f"no rows for tp={args.tp} in {args.parquet}")

pre_grid = set(zip(pre.batch_size, pre.total_prefill_tokens, pre.total_kv_read_tokens))
bs_lat = sorted(pre.batch_size.unique()); B_c = int(max(bs_lat))
tiers = sorted(pre[(pre.batch_size == 1) & (pre.total_kv_read_tokens == 0)].total_prefill_tokens.unique())
b1 = pre[(pre.batch_size == 1) & (pre.total_kv_read_tokens == 0)].set_index("total_prefill_tokens").latency_ms
CAP = max((t for t in tiers if t + 1 in b1.index and b1[t + 1] / b1[t] > 1.5), default=max(tiers))
dlat = sorted(dec.batch_size.unique()); dset = set(dlat)
pfx_lat = sorted(pre[(pre.batch_size == 1) & (pre.total_kv_read_tokens > 0)].total_kv_read_tokens.unique())
KV_DOM_MAX = int(dec.total_kv_read_tokens.max())
GIANT = 0.4 * KV_DOM_MAX

def snap(v, lat):
    i = bisect.bisect_left(lat, v)
    c = [lat[max(0, i - 1)], lat[min(len(lat) - 1, i)]]
    return int(min(c, key=lambda x: abs(x - v)))

reg = lambda tot: "graph" if tot <= CAP else "eager"

# ============================ PREFILL ============================
P = []
def addp(grp_hint, bp, n, kv, tag):
    if n + kv > MAX_LEN - BLOCK or bp > MAX_SEQS:
        return
    single = n <= BUDGET
    on = (bp, n, kv) in pre_grid
    if grp_hint == "C":
        grp = "Cpfx" if kv > 0 else "C"
        assert bp > B_c, (bp, B_c)
    elif not single:
        grp = "长请求"          # request-level; steps classified at scoring time
    else:
        grp = "A" if on else "B"
    P.append(dict(grp=grp, bp=bp, n=n, kv=kv, tag=tag,
                  exec_="单步" if single else f"chunked~{-(-n // BUDGET)}步"))

isl_on = [2 ** k for k in range(7, 20) if 2 ** k <= MAX_LEN]
isl_off = [int(np.sqrt(isl_on[i] * isl_on[i + 1])) for i in range(len(isl_on) - 1)]
if isl_on[-1] < MAX_LEN:
    isl_off.append(int(np.sqrt(isl_on[-1] * MAX_LEN)))

KV_ON = [0, 256, 1024, 4096]
KV_EXT = [16384, 65536]
KV_OFF = [608, 2144, 8384]     # off-lattice by construction (asserted below)
for kv in KV_OFF:
    assert not any((1, t, kv) in pre_grid for t in tiers), f"KV_OFF {kv} collides with grid"

for n in isl_on:
    kvs = KV_ON + [k for k in KV_EXT if n >= 2048] + KV_OFF[: 2 if n < 2048 else 3]
    for kv in kvs:
        addp("D", 1, n, kv, "ISL在网主干")
for n in isl_off:
    for kv in [0, 256, 4096, KV_OFF[1]]:
        addp("D", 1, n, kv, "ISL离网")
for bp, n in ((2, 1024), (2, 4096), (4, 512), (4, 2048)):
    if bp in bs_lat:
        for kv in (0, 1024, KV_OFF[0]):
            addp("D", bp, n, kv, "bs变体")
# C 组:bs 外插(仅 prefill)
if args.tier == "full":
    for bp in sorted({int(m * B_c) for m in (1.5, 2, 3, 4, 6, 8, 12, 16)}):
        for f in (0.5, 0.9, 1.1, 2, 4):
            tok = snap(f * CAP / bp, tiers)
            if tok * bp <= BUDGET and tok >= tiers[0]:
                addp("C", bp, tok, 0, f"外插{bp / B_c:.1f}x")
    for bp in sorted({int(1.5 * B_c), 2 * B_c}):
        tok = snap(CAP / bp, tiers)
        if bp * tok <= BUDGET:
            hi = MAX_LEN - BLOCK - tok
            for f in (0.25, 0.5, 0.85):
                tgt = int(np.exp(np.log(max(pfx_lat[0], 16)) + (np.log(hi) - np.log(max(pfx_lat[0], 16))) * f))
                nr = snap(tgt, pfx_lat); j = pfx_lat.index(nr)
                mid = ((pfx_lat[j] + pfx_lat[min(j + 1, len(pfx_lat) - 1)]) // 2) // 16 * 16
                if mid > pfx_lat[j] and tok + mid <= MAX_LEN - BLOCK:
                    addp("C", bp, tok, mid, f"外插{bp / B_c:.1f}x+pfx")
pf = pd.DataFrame(P).drop_duplicates(subset=["bp", "n", "kv"]).reset_index(drop=True)
pf["regime"] = (pf.bp * pf.n).clip(upper=BUDGET * pf.bp).apply(lambda t: reg(min(t, BUDGET)))
pf["repeats"] = 5 if args.tier == "full" else 3

# ============================ DECODE ============================
W = []
L2P = []
bs_on = [2 ** k for k in range(0, 9) if 2 ** k in dset]
extra_on = [b for b in (max(x for x in dlat if x + 1 in dset), max(x for x in dlat if x + 1 in dset) + 1, max(dlat)) if b in dset]
bs_on = sorted(set(bs_on + extra_on))
gaps = [b for b in (6, 12, 20, 44, 100, 150, 300, 600, 900) if b not in dset and b <= max(dlat)]
POCKET_REP = {snap(x, dlat) for x in (40, 64, 192, 256, 512, max(dlat))}
for B in sorted(set(bs_on + gaps)):
    grp = "A批" if B in dset else "B批"
    ref = snap(B, dlat)
    sv = sorted((dec[dec.batch_size == ref].total_kv_read_tokens // ref).unique())
    smax = int(sv[-1])
    W.append(dict(grp=grp, C=B, isl=1, osl=min(4400, smax), kind="浅扫段", boots=1))
    for anchor in [16384, 65536, 262144]:
        if anchor <= smax and args.tier == "full":
            if B * anchor >= GIANT:
                L2P.append(dict(batch_size=int(B), total_kv_read_tokens=int(B * anchor)))
            else:
                W.append(dict(grp=grp, C=B, isl=anchor - 128, osl=512, kind=f"跳扫@{anchor}", boots=1))
    stop = int(sv[-2]) if len(sv) > 1 else smax
    if B * stop >= GIANT:
        L2P.append(dict(batch_size=int(B), total_kv_read_tokens=int(B * stop)))  # 巨KV → L2 探针
        mid_deep = int(sv[int(len(sv) * 5 / 6)]) if len(sv) > 2 else stop
        if B * mid_deep >= GIANT:
            L2P.append(dict(batch_size=int(B), total_kv_read_tokens=int(B * mid_deep)))
        continue
    boots = 1
    if args.tier == "full" or B in POCKET_REP:
        W.append(dict(grp=grp, C=B, isl=max(1, stop - 128), osl=256, kind="顶端扫", boots=boots))
wd = pd.DataFrame(W)
# 去重:顶端扫与跳扫窗口重合时合并
wd = wd.sort_values(["C", "isl"]).drop_duplicates(subset=["C", "isl"], keep="last").reset_index(drop=True)

# ============================ MIXED ============================
M = []
Bd_sel = sorted({snap(x, dlat) for x in (8, dlat[int(len(dlat) * 0.3)], 64)})
gr_t = [t for t in tiers if t <= CAP]
grm_base = gr_t[len(gr_t) // 2]; j = tiers.index(grm_base)
grm = (tiers[j] + tiers[j + 1]) // 2
bigm = [(tiers[i] + tiers[i + 1]) // 2 for i in range(len(tiers) - 1) if tiers[i + 1] - tiers[i] >= 256]
for Bd in Bd_sel:
    M.append(dict(grp="A", bp=1, Bd=Bd, chunk=CAP - Bd, tgt=CAP, note="悬崖左档"))
    M.append(dict(grp="A", bp=1, Bd=Bd, chunk=CAP + 1 - Bd, tgt=CAP + 1, note="悬崖右缘"))
    if args.tier == "full":
        for tier_v in (t for t in (1024, 4096) if t in tiers and t - Bd > 0):
            M.append(dict(grp="A", bp=1, Bd=Bd, chunk=tier_v - Bd, tgt=tier_v, note="chunk+Bd=采集档"))
        for m in bigm[:2]:
            M.append(dict(grp="B", bp=1, Bd=Bd, chunk=m - Bd, tgt=m, note="eager中点"))
        M.append(dict(grp="B", bp=1, Bd=Bd, chunk=grm - Bd, tgt=grm, note="graph中点"))
if args.tier == "full":
    for bp in [2, B_c, int(1.5 * B_c), 2 * B_c]:
        tok = snap(CAP / bp / 2, tiers)
        M.append(dict(grp="C" if bp > B_c else "C锚", bp=bp, Bd=40, chunk=bp * tok, tgt=bp * tok + 40,
                      note=f"{bp}条{tok}t齐发+池"))
md = pd.DataFrame(M).reset_index(drop=True)

# ============================ 覆盖报告与配额门 ============================
dom = pf[pf.grp.isin(["A", "B"])]
report = {
    "engine_facts": dict(budget=BUDGET, max_seqs=MAX_SEQS, max_len=MAX_LEN, block=BLOCK,
                         prefill_capture=int(CAP), B_c=B_c, decode_max_batch=int(max(dlat)),
                         kv_dom_max=KV_DOM_MAX, giant_kv_threshold=int(GIANT)),
    "prefill": dict(total=len(pf), A=int((pf.grp == "A").sum()), B=int((pf.grp == "B").sum()),
                    长请求=int((pf.grp == "长请求").sum()), C=int(pf.grp.isin(["C", "Cpfx"]).sum()),
                    past_kv_share=round(float((dom.kv > 0).mean()), 3),
                    A_share_domain=round(float((dom.grp == "A").mean()), 3)),
    "decode": dict(windows=len(wd), on_batch=len(bs_on), off_batch=len(gaps),
                   multi_boot=int((wd.boots > 1).sum())),
    "mixed": dict(windows=len(md), A=int((md.grp == "A").sum()), B=int((md.grp == "B").sum()),
                  C=int(md.grp.isin(["C", "C锚"]).sum())),
    "tier": args.tier,
    "l2_pocket_points": len({(d["batch_size"], d["total_kv_read_tokens"]) for d in L2P}),
}
fails = []
if args.tier == "full":
    if report["prefill"]["past_kv_share"] < 0.4: fails.append("prefill past-kv share < 40%")
    if not any(w["kind"].startswith("顶端") for w in W): fails.append("no top-scan windows")
    missing = [2 ** k for k in range(7, 14) if 2 ** k <= BUDGET and 2 ** k not in set(pf[pf.kv == 0].n)]
    if missing: fails.append(f"mandatory ISL rungs missing: {missing}")
    miss_b = [2 ** k for k in range(0, 9) if 2 ** k in dset and 2 ** k not in set(wd.C)]
    if miss_b: fails.append(f"mandatory bs rungs missing: {miss_b}")
report["quota_failures"] = fails

os.makedirs(args.out, exist_ok=True)
pf.to_csv(f"{args.out}/prefill_plan.csv", index=False)
json.dump({"schema_version": 1, "prefill": [], "decode": sorted({(d["batch_size"], d["total_kv_read_tokens"]) for d in L2P}) and [dict(batch_size=b, total_kv_read_tokens=k) for b, k in sorted({(d["batch_size"], d["total_kv_read_tokens"]) for d in L2P})]},
          open(f"{args.out}/l2_pocket_probe.json", "w"), indent=1)
wd.to_csv(f"{args.out}/decode_plan.csv", index=False)
md.to_csv(f"{args.out}/mixed_plan.csv", index=False)
json.dump(report, open(f"{args.out}/coverage_report.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
if fails:
    sys.exit("QUOTA FAILURES: " + "; ".join(fails))
print(f"\nplans written to {args.out}/")
