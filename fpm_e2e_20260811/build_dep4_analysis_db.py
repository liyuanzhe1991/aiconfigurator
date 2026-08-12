# Build an ANALYSIS-ONLY systems root holding the two dep4 cells from the
# stranded plan 4df9b0f2 artifacts, so L3 scoring can query the model at dep4
# coordinates. This is NOT publication: rows are aggregated by the collector's
# own aggregate_cell (full native-artifact validation, max-rank latency), but
# the output lives in a scratch root consumed only via
# get_database(..., systems_paths=[ROOT]).
import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from collector.fpm_forward.database import _ROW_KEY, aggregate_cell  # noqa: E402
from collector.fpm_forward.planner import BackendPolicy, FPMCell  # noqa: E402
from collector.fpm_forward.types import ParallelTopology  # noqa: E402

ART = Path("fpm_forward_artifacts/4df9b0f29115c63d")
ROOT = Path("fpm_e2e_20260811/dep4_analysis_root")
CELLS = ["fpm-947eff015e9514d1", "fpm-c19369ef6ddd8012"]  # dep4 prefill / decode

plan_json = json.load(open(ART / "collection-plan.json"))
cap = plan_json["capability"]
plan = SimpleNamespace(
    sha256=plan_json["sha256"],
    model_path=plan_json["model_path"],
    system=plan_json["system"],
    backend=plan_json["backend"],
    options=SimpleNamespace(warmup_iterations=plan_json["options"]["global_warmup_iterations"]),
    capability=SimpleNamespace(
        support_level=cap["support_level"],
        template_id=cap["template_id"],
        template_version=cap["template_version"],
        aic_database_version=cap["aic_database_version"],
    ),
)

rows = []
for cid in CELLS:
    cdir = ART / "cells" / cid
    cj = json.load(open(cdir / "cell.json"))
    rd = cj["resolved_dtypes"]
    bp = cj["backend_policy"]
    cell = FPMCell(
        cell_id=cj["cell_id"],
        workload_kind=cj["workload_kind"],
        topology=ParallelTopology(**cj["topology"]),
        weight_quantization=cj["weight_quantization"],
        kv_cache_dtype=cj["kv_cache_dtype"],
        backend_policy=BackendPolicy(
            policy_id=bp["policy_id"],
            generator_overrides=bp["generator_overrides"],
            expected_markers=bp["expected_markers"],
            aic_fields=bp["aic_fields"],
            admission_reason=bp["admission_reason"],
        ),
        gemm_quant_mode=rd["gemm_quant_mode"],
        parallel_strategy=cj["parallel_strategy"],
        moe_quant_mode=rd["moe_quant_mode"],
        fmha_quant_mode=rd["fmha_quant_mode"],
        comm_quant_mode=rd["comm_quant_mode"],
        fmha_resolution=rd["fmha_resolution"],
    )
    prov_files = glob.glob(str(cdir / "raw" / "*" / "collector-provenance.json"))
    assert len(prov_files) == 1, prov_files
    attempt_id = json.load(open(prov_files[0]))["attempt_id"]
    cell_rows = aggregate_cell(plan, cell, cdir, expected_attempt_id=attempt_id)
    print(f"{cid}: {cj['workload_kind']} -> {len(cell_rows)} rows")
    rows.extend(cell_rows)

df = pd.DataFrame(rows).sort_values(list(_ROW_KEY)).reset_index(drop=True)
data_dir = ROOT / "data" / "h200_sxm" / "vllm" / "0.25.1"
data_dir.mkdir(parents=True, exist_ok=True)
shutil.copy("aic-core/src/aiconfigurator_core/systems/h200_sxm.yaml", ROOT)
pq_path = data_dir / "fpm_forward_perf.parquet"
df.to_parquet(pq_path, index=False, compression="zstd")

sidecar = {
    "schema_name": "aic_fpm_forward_perf",
    "schema_version": 6,
    "coordinate_system": "iteration_totals_balanced_v1",
    "measurement_policy": "dynamo_native_single_sample_v1",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.25.1",
    "parquet_sha256": hashlib.sha256(pq_path.read_bytes()).hexdigest(),
    "row_count": len(df),
    "note": "ANALYSIS-ONLY scratch build from plan 4df9b0f2 dep4 cells; not a formal publication",
    "source_plan_sha256": sorted(df.source_plan_sha256.unique().tolist()),
    "collector_attempt_ids": sorted(df.collector_attempt_id.unique().tolist()),
}
(data_dir / "fpm_forward_perf.metadata.json").write_text(json.dumps(sidecar, indent=1))
print(f"wrote {pq_path} rows={len(df)}")
print(df.groupby(["workload_kind"]).agg(n=("latency_ms", "size"), bmax=("batch_size", "max")).to_string())
