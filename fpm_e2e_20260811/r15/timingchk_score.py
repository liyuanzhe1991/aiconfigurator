# timing 补丁验证判分:A=基线 B/C=+timing
# 判据:
#  1) 逐点配对 wall(物理坐标键):B、C 相对 A 的中位漂移与逐点漂移在 boot 噪声内
#     (Guaranteed pod 带内历史极差 3.3%;带外 <1%);
#  2) benchmark_elapsed_seconds:B/C vs A ±5%(含 kvwarm/boot 自然波动);
#  3) B/C 的 timing.phases 存在、键齐全、数值自洽:
#     sum(可加相位) + measured + kvwarm_stage 秒数 ≤ elapsed;
#  4) A 的输出不含 phases(基线未打补丁),其余结构与 B/C 一致。
import json
import sys

A, B, C = (json.load(open(p)) for p in sys.argv[1:4])

def points(d):
    out = {}
    for g in d["iteration_groups"]:
        p = g["point"]
        key = (p["point_type"], p["batch_size"],
               p.get("total_prefill_tokens", 0), p["total_kv_read_tokens"])
        out[key] = float(g["wall_time"])
    return out

pa, pb, pc = points(A), points(B), points(C)
assert set(pa) == set(pb) == set(pc), "坐标集不一致"
print(f"{'coord':<38}{'A_ms':>9}{'B_ms':>9}{'C_ms':>9}{'B/A':>8}{'C/A':>8}")
worst = 0.0
for k in sorted(pa):
    ra, rb, rc = pa[k], pb[k], pc[k]
    db, dc = rb / ra - 1, rc / ra - 1
    worst = max(worst, abs(db), abs(dc))
    print(f"{str(k):<38}{ra*1e3:>9.2f}{rb*1e3:>9.2f}{rc*1e3:>9.2f}"
          f"{db*100:>+7.2f}%{dc*100:>+7.2f}%")
ea = A["timing"]["benchmark_elapsed_seconds"]
eb = B["timing"]["benchmark_elapsed_seconds"]
ec = C["timing"]["benchmark_elapsed_seconds"]
print(f"\nelapsed A={ea:.1f}s B={eb:.1f}s ({eb/ea-1:+.2%}) C={ec:.1f}s ({ec/ea-1:+.2%})")
print(f"measured A={A['timing']['measured_iteration_seconds']:.3f}s "
      f"B={B['timing']['measured_iteration_seconds']:.3f}s "
      f"C={C['timing']['measured_iteration_seconds']:.3f}s")

assert "phases" not in A["timing"], "A(基线)不应有 phases"
ok = True
for tag, d in (("B", B), ("C", C)):
    ph = d["timing"].get("phases")
    assert ph, f"{tag} 缺 phases"
    need = ["import_to_first_bench_activity_s", "dataset_load_s", "tokenizer_load_s",
            "kvwarm_plan_s", "content_tokenize_s", "content_gen_s", "seed_total_s"]
    missing = [k for k in need if k not in ph]
    if missing:
        print(f"{tag} phases 缺键: {missing}")
        ok = False
    kvw = sum(s["build_seconds"] for s in d.get("kvwarm", {}).get("stages", []))
    # 相位嵌套树(勿平铺求和):
    #   content_gen_s ⊃ 首调用池构建 ⊃ dataset_load + tokenizer_load(首次)
    #   kvwarm.stages[].build_seconds ⊃ content_tokenize_s(链分词在 stage 窗口内)
    #   顶层互斥相位 = content_gen + kvwarm_stages + seed + measured + kvwarm_plan
    top = (ph.get("content_gen_s", 0) + kvw + ph.get("seed_total_s", 0)
           + ph.get("kvwarm_plan_s", 0))
    meas = d["timing"]["measured_iteration_seconds"]
    el = d["timing"]["benchmark_elapsed_seconds"]
    print(f"{tag}: 顶层互斥相位={top:.2f}s (content_gen={ph.get('content_gen_s',0):.2f} "
          f"kvwarm_stages={kvw:.2f} seed={ph.get('seed_total_s',0):.3f}) "
          f"measured={meas:.2f}s elapsed={el:.2f}s "
          f"守恒余量(other)={el - top - meas:.2f}s")
    if top + meas > el * 1.02:
        print(f"{tag} 守恒违例:顶层相位和超过 elapsed")
        ok = False
    print(f"{tag} phases: {json.dumps(ph, ensure_ascii=False)}")

print(f"\n逐点最坏漂移 {worst:.2%}")
verdict = ok and worst < 0.05 and abs(eb/ea-1) < 0.05 and abs(ec/ea-1) < 0.05
print("VERDICT:", "PASS" if verdict else "FAIL")
sys.exit(0 if verdict else 1)
