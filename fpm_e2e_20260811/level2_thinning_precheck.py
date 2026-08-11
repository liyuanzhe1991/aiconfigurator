# L2 pre-check: thinned-grid holdout, fully local.
# Hold out ~20% interior points per curve, rebuild a compliant parquet+sidecar
# in a scratch systems root, then interpolate the held-out coordinates and
# compare against their (hidden) measured values.
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.operations.fpm_forward import FPMForwardOp
from aiconfigurator_core.sdk.perf_database import get_database

RNG = np.random.default_rng(20260811)
MODEL_PATH = "MiniMaxAI/MiniMax-M2.7"
SRC_DIR = "fpm_formal_database/h200_sxm/vllm/0.25.1"
SCRATCH = os.path.abspath("fpm_e2e_20260811/thinned_systems_root")
HOLDOUT_FRAC = 0.2

df = pd.read_parquet(f"{SRC_DIR}/fpm_forward_perf.parquet")
df = df.reset_index(drop=True)

holdout_idx = []
# Decode: per (tp, batch) KV curve — interior, addressable, ≤20% of curve.
for (tp, b), g in df[df.workload_kind == "decode"].groupby(["tp", "batch_size"]):
    g = g.sort_values("total_kv_read_tokens")
    inner = g.iloc[1:-1]
    inner = inner[inner.total_kv_read_tokens % inner.batch_size == 0]
    if len(inner) < 3:
        continue
    k = max(1, int(len(inner) * HOLDOUT_FRAC))
    holdout_idx.extend(RNG.choice(inner.index.to_numpy(), size=k, replace=False))
# Prefill: per (tp, batch, prefix-kv) new-token curve — same rules.
for (tp, b, kv), g in df[df.workload_kind == "prefill"].groupby(["tp", "batch_size", "total_kv_read_tokens"]):
    g = g.sort_values("total_prefill_tokens")
    inner = g.iloc[1:-1]
    inner = inner[(inner.total_prefill_tokens % inner.batch_size == 0) & (kv % b == 0)]
    if len(inner) < 3:
        continue
    k = max(1, int(len(inner) * HOLDOUT_FRAC))
    holdout_idx.extend(RNG.choice(inner.index.to_numpy(), size=k, replace=False))

holdout_idx = sorted(set(int(i) for i in holdout_idx))
held = df.loc[holdout_idx]
kept = df.drop(index=holdout_idx)
print(f"total={len(df)} held-out={len(held)} kept={len(kept)}")

# Build the scratch systems root: yaml + thinned pair + compliant sidecar.
data_dir = os.path.join(SCRATCH, "data", "h200_sxm", "vllm", "0.25.1")
os.makedirs(data_dir, exist_ok=True)
shutil.copy("aic-core/src/aiconfigurator_core/systems/h200_sxm.yaml", SCRATCH)
pq_path = os.path.join(data_dir, "fpm_forward_perf.parquet")
kept.to_parquet(pq_path, index=False)
meta = json.load(open(f"{SRC_DIR}/fpm_forward_perf.metadata.json"))
meta["parquet_sha256"] = hashlib.sha256(open(pq_path, "rb").read()).hexdigest()
meta["row_count"] = len(kept)
with open(os.path.join(data_dir, "fpm_forward_perf.metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2, sort_keys=True)

db = get_database("h200_sxm", "vllm", "0.25.1", systems_paths=[SCRATCH])
FPMForwardOp.clear_cache()


def make_op(phase, tp, ep):
    cfg = sdk_config.ModelConfig(
        tp_size=tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=ep, cp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    sol = (lambda b, tp_, tk: 0.001) if phase == "prefill" else (lambda b, tk: 0.001)
    return FPMForwardOp(phase, cfg, MODEL_PATH, sol_fn=sol, weight_bytes=1e9)


rows_out = []
for (tp, phase), g in held.groupby(["tp", "workload_kind"]):
    op = make_op(phase, int(tp), int(tp))
    apes = []
    for row in g.itertuples():
        b = int(row.batch_size)
        try:
            if phase == "prefill":
                res = op.query(db, batch_size=b, s=int(row.total_prefill_tokens) // b,
                               prefix=int(row.total_kv_read_tokens) // b)
            else:
                res = op.query(db, batch_size=b, s=int(row.total_kv_read_tokens) // b)
        except Exception as e:
            rows_out.append((tp, phase, row.batch_size, "REFUSED", str(e)[:60]))
            continue
        ape = abs(float(res) - float(row.latency_ms)) / float(row.latency_ms)
        apes.append(ape)
    if apes:
        a = np.array(apes)
        print(f"TEP{tp} {phase:8s}: n={len(a):4d} refused={len(g)-len(a):3d} "
              f"median APE={np.median(a)*100:5.2f}%  p95={np.percentile(a,95)*100:5.2f}%  max={a.max()*100:5.2f}%")

refused = [r for r in rows_out if r[3] == "REFUSED"]
if refused:
    print(f"\nrefused total: {len(refused)} (first 3): {refused[:3]}")
