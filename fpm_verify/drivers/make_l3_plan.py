# L3 v3 补采计划生成器:逐 B 档计算"最终打分口径下保留的真值覆盖"在低 kv 区
# 的空洞并精确填补。零 hardcode:B 档从 v2 计划导出,kv 上限从采集点单导出,
# 池容量从引擎 resolved-config 导出(经 --pool-tokens 传入)。
# 覆盖模型(与打分口径严格一致):
#   计入覆盖的 = v2 计划中 tag 匹配 --retain-tags 的行(打分会保留其 r11 窗口)
#              + 本计划自产的 shallow / rerun 行。
#   不匹配 --retain-tags 且不在 --rerun-bands 的 v2 行(如 dep4 的 band1/
#   longsweep,rank0-only 稀释时代产物)不算覆盖——它们的窗口会被打分丢弃。
# 覆盖片段:isl=x, osl=y 的格形成 per-req kv [x, x+y)(x=1 时从 1 起)。
# 输出 CSV: tag,C,isl,osl,reps
import argparse
import csv
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--v2-plan", required=True, help="v2 计划 CSV(定义 B 档与既有覆盖)")
ap.add_argument("--manifest", required=True, help="采集点单(定义各 B 的 kv 上限)")
ap.add_argument("--dp", type=int, required=True)
ap.add_argument("--pool-tokens", type=int, required=True, help="每 rank KV 池容量")
ap.add_argument("--fill-ceiling", type=int, required=True,
                help="填洞上界(per-req kv);上界内'保留覆盖∪自产'必须无洞或如实记档")
ap.add_argument("--retain-tags", default=".*",
                help="正则:v2 计划中打分保留的 tag(须与打分链的窗口过滤一致)")
ap.add_argument("--shallow-osl", type=int, default=96)
ap.add_argument("--shallow-reps", type=int, default=6)
ap.add_argument("--fill-reps", type=int, default=2)
ap.add_argument("--deep-fill-reps", type=int, default=1,
                help="osl>1024 的重洞用低重复(耗时红线)")
ap.add_argument("--min-osl", type=int, default=60,
                help="填洞格最短扫段:格实际步数=osl-1,非锁步窗再剔前 4 步且稳定池"
                     "需 ≥50 样本(osl≥55 才可能入账),留参差余量取 60")
ap.add_argument("--rerun-bands", default="",
                help="逗号分隔 isl 列表:重收这些 v2 带(多 rank 遥测口径)")
ap.add_argument("--rerun-reps", type=int, default=2)
ap.add_argument("--out", required=True)
args = ap.parse_args()

MARGIN = 0.92  # 池可行性护栏(与 v2 生成器同源)

v2 = list(csv.DictReader(open(args.v2_plan)))
retain = re.compile(args.retain_tags)
rerun_isls = [int(x) for x in args.rerun_bands.split(",") if x]
m = json.load(open(args.manifest))
kv_max = {}
for p in m["decode"]:
    b = p["batch_size"]
    kv_max[b] = max(kv_max.get(b, 0), p["total_kv_read_tokens"] // b)

rows, report = [], []
c_levels = sorted({int(r["C"]) for r in v2})
for pool_c in c_levels:
    b = pool_c // args.dp
    ceiling = min(args.fill_ceiling, kv_max.get(b, args.fill_ceiling))
    budget_len = int(args.pool_tokens * args.dp * MARGIN / pool_c)

    # ① 本档自产行:浅池(锁步直录)+ 旧带重收
    my_rows = [("shallow", pool_c, 1, args.shallow_osl, args.shallow_reps)]
    for isl in rerun_isls:
        r0 = [r for r in v2 if int(r["C"]) == pool_c and r["tag"] == f"band{isl}"]
        if r0 and isl + int(r0[0]["osl"]) <= budget_len:
            my_rows.append((f"rerun{isl}", pool_c, isl, int(r0[0]["osl"]),
                            args.rerun_reps))

    # ② 覆盖 = 打分保留的 v2 行 + 自产行。
    # 格 (isl=x, osl=y) 实际只产生 y-1 个纯 decode 步(首 token 由 prefill 步
    # 产出),真实覆盖 [x, x+y-1)。rerun 带的覆盖只来自"实际生成的 rerun 行"
    # (在 my_rows 里)——被池护栏拒绝的带按洞处理,走 DROP/CLIP 记档
    covered = []
    for r in v2:
        if int(r["C"]) != pool_c or r["tag"] == "shallow":
            continue
        if not retain.fullmatch(r["tag"]):
            continue  # 打分口径下该窗口会被丢弃,不算覆盖
        x, y = int(r["isl"]), int(r["osl"])
        covered.append((max(x, 1), x + y - 1))
    for tag, _, x, y, _reps in my_rows:
        covered.append((max(x, 1), x + y - 1))
    covered.sort()

    # ③ 扫描 [1, ceiling) 找洞并填补
    holes, cur = [], 1
    for a, e in covered:
        if a > cur:
            holes.append((cur, min(a, ceiling)))
        cur = max(cur, e)
        if cur >= ceiling:
            break
    if cur < ceiling:
        holes.append((cur, ceiling))
    rows.extend(my_rows)
    for a, e in holes:
        a = max(a, 1)
        # 覆盖 [a,e) 需要 osl = e-a+1(实际步数 osl-1);
        # 池放不下整洞时按可行长度截断:部分覆盖优于空白,截断如实记档
        osl = min(e - a + 1, budget_len - a)
        if osl < args.min_osl:
            report.append(f"DROP C={pool_c} hole[{a},{e}) 可行扫段 {max(osl,0)} "
                          f"< 打分最短 {args.min_osl},不可填")
            continue
        if osl < e - a + 1:
            report.append(f"CLIP C={pool_c} hole[{a},{e})→[{a},{a + osl - 1})")
        reps = args.fill_reps if osl <= 1024 else args.deep_fill_reps
        rows.append((f"fill{a}", pool_c, a, osl, reps))

rows.sort(key=lambda r: (r[1], r[2]))
with open(args.out, "w") as f:
    f.write("tag,C,isl,osl,reps\n")
    for tag, c, isl, osl, reps in rows:
        f.write(f"{tag},{c},{isl},{osl},{reps}\n")

steps = sum(r[3] * r[4] for r in rows)
inj = sum(r[1] * r[2] * r[4] for r in rows)
entries = sum(r[4] for r in rows)
print(f"{args.out}: {len(c_levels)} 档 {len(rows)} 行 | decode 步 {steps} | "
      f"灌注 {inj/1e6:.1f}M tok | 进场 {entries} 次")
for line in report:
    print(" ", line)
