#!/usr/bin/env python3
"""AIC op 级建模对同一 eval set 坐标出预测 (三方对比: op vs FPM vs 真值).

口径:
- op 库 = 树内 h200_sxm vllm 0.24.0 (0.25.1 无 op 数据);
- 坐标换算 = 等分假设: decode 每请求 KV S=round(kv/b); prefill 每请求
  chunk=round(tot/bp), past-kv=round(kv/bp);
- decode: run_static_latency_only(mode=static_gen, osl=2) 单步;
- prefill: run_static_latency_only(mode=static_ctx) 单 chunk。
"""
import sys

import pandas as pd

sys.path.insert(0, "/Users/yuanzhe/Desktop/codex_workspace/fpm/worktrees/aic-collector-generator-decoupling/aic-core/src")

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk import models, perf_database
from aiconfigurator_core.sdk.backends.factory import get_backend

EVAL = "/Users/yuanzhe/Desktop/codex_workspace/fpm/artifacts/agx_p1_tep4"
ROOT = "/Users/yuanzhe/Desktop/codex_workspace/fpm/worktrees/aic-collector-generator-decoupling/aic-core/src/aiconfigurator_core/systems"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = sys.argv[2] if len(sys.argv) > 2 else "op_pred.csv"

perf_database.set_systems_paths([ROOT])
db = perf_database.get_database("h200_sxm", "vllm", "0.24.0")

cfg = sdk_config.ModelConfig(
    tp_size=4, pp_size=1, attention_dp_size=1,
    moe_tp_size=1, moe_ep_size=4, cp_size=1,
    gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    moe_quant_mode=common.MoEQuantMode.fp8_block,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    comm_quant_mode=common.CommQuantMode.half,
)
model = models.get_model("MiniMaxAI/MiniMax-M2.7", cfg, "vllm")
backend = get_backend("vllm")
print("model/backend ready, forward_model =", cfg.forward_model, flush=True)


def rt(**kw):
    return sdk_config.RuntimeConfig(beam_width=1, engine_step_backend="python", **kw)


def dec_pred(b, kv):
    s = max(2, round(kv / b))
    return backend.run_static_latency_only(
        model, db, rt(batch_size=int(b), isl=s - 1, osl=2), mode="static_gen")


def pre_pred(bp, tot, kv):
    chunk = max(1, round(tot / bp))
    px = max(0, round(kv / bp))
    return backend.run_static_latency_only(
        model, db, rt(batch_size=int(bp), isl=chunk + px, osl=2, prefix=px), mode="static_ctx")


d = pd.read_csv(f"{EVAL}/scores_decode.csv")
p = pd.read_csv(f"{EVAL}/scores_prefill.csv")
dn = d[d.side == "new"][["b", "kv", "truth"]].reset_index(drop=True)
pp = p[p.side == "new"][["bp", "tot", "kv", "truth"]].reset_index(drop=True)
if LIMIT:
    dn, pp = dn.head(LIMIT), pp.head(max(LIMIT // 10, 20))

import time
rows = []
t0 = time.time()
for i, r in dn.iterrows():
    try:
        v = dec_pred(r.b, r.kv)
    except Exception as e:
        v = float("nan")
        if i < 5:
            print("dec err:", type(e).__name__, e, flush=True)
    rows.append(("decode", r.b, r.kv, 0, r.truth, v))
    if i % 5000 == 0:
        print(f"decode {i}/{len(dn)}  {time.time()-t0:.0f}s", flush=True)
for i, r in pp.iterrows():
    try:
        v = pre_pred(r.bp, r.tot, r.kv)
    except Exception as e:
        v = float("nan")
        if i < 5:
            print("pre err:", type(e).__name__, e, flush=True)
    rows.append(("prefill", r.bp, r.kv, r.tot, r.truth, v))
    if i % 1000 == 0:
        print(f"prefill {i}/{len(pp)}  {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows, columns=["kind", "b", "kv", "tot", "truth", "op_model"])
out.to_csv(OUT, index=False)
ok = out.dropna(subset=["op_model"])
print(f"\ndone {time.time()-t0:.0f}s  ok={len(ok)}/{len(out)}")
for kind, g in ok.groupby("kind"):
    s = (g.op_model - g.truth) / g.truth * 100
    print(f"{kind}: n={len(g)} med={s.median():.1f}% MAPE={s.abs().mean():.1f}% P95={s.abs().quantile(.95):.1f}%")
