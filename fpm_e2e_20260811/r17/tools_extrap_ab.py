# 外插 A/B 验证脚本(C1 终态提案配套;用法见 PROPOSAL_C1_FINAL_FORM.md)
# 输入:同机真值打分 CSV 对(裸/滤)+ 库 parquet;输出:顶格带三方 MAPE。
import sys
import pandas as pd, numpy as np

LIB, RAW, FIL, TOPO = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lib = pd.read_parquet(LIB)
sel = {"tp4": (lib.tp==4)&(lib.moe_tp==4), "tep4": (lib.tp==4)&(lib.moe_ep==4)&(lib.dp==1),
       "tp2": (lib.tp==2)&(lib.moe_tp==2), "tep2": (lib.tp==2)&(lib.moe_ep==2)&(lib.dp==1),
       "dep4": (lib.dp==4), "dep2": (lib.dp==2)}[TOPO]
d = lib[sel & (lib.workload_kind=="decode")]
real = d[d.kv_seed_regime=="real_kv"]

def extrap(lad, kv):
    lad = lad.sort_values("total_kv_read_tokens")
    if len(lad) < 2: return None
    r1, r2 = lad.iloc[-2], lad.iloc[-1]
    if kv <= r2.total_kv_read_tokens: return None
    s = (r2.latency_ms - r1.latency_ms)/(r2.total_kv_read_tokens - r1.total_kv_read_tokens)
    return r2.latency_ms + (kv - r2.total_kv_read_tokens)*s

raw = pd.read_csv(RAW); raw = raw[raw.side=="new"]
fil = pd.read_csv(FIL); fil = fil[fil.side=="new"]
fk = set(zip(fil.b, fil.kv))
band = raw[[ (b,k) not in fk for b,k in zip(raw.b, raw.kv) ]]
band = band[band.b <= 512]
bs = np.array(sorted(real.batch_size.unique()))
rows = []
for _, v in band.iterrows():
    lo = bs[bs<=v.b].max() if (bs<=v.b).any() else None
    hi = bs[bs>=v.b].min() if (bs>=v.b).any() else None
    if lo is None or hi is None: continue
    e1 = extrap(real[real.batch_size==lo], v.kv)
    e2 = extrap(real[real.batch_size==hi], v.kv) if hi!=lo else e1
    if e1 is None or e2 is None: continue
    w = 0 if hi==lo else (v.b-lo)/(hi-lo)
    rows.append((v.truth, v.model, e1*(1-w)+e2*w))
r = pd.DataFrame(rows, columns=["truth","fake_pred","extrap_pred"])
for c in ["fake_pred","extrap_pred"]:
    e = abs(r[c]-r.truth)/r.truth
    print(f"{TOPO} {c}: MAPE {e.mean()*100:.2f}% P95 {e.quantile(.95)*100:.2f}% n={len(r)}")
