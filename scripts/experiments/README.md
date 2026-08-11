# FPM Collection + Modeling Experiments — Step-by-Step Guide

Goal: **run FPM self-benchmark collection on H100 / H200 / GB200 / B200 with one command**, then run FPM modeling (`forward_model="fpm"`) against the collected data — on one branch that merges all four FPM PRs (collection + modeling).

---

## Part 1 — How this branch was assembled (merge order)

The PRs form a stack and **must be merged in this exact order** (each depends on the previous one):

| Order | PR | Branch | What it adds |
|---|---|---|---|
| 1 | #1473 | `fpm-pr0-contract` | The FPM contract table (file names, env vars, result-path rules) shared by both sides |
| 2 | #1474 | `fpm-prg-generator` | The generator's FPM render target (produces `k8s_deploy.yaml` + `fpm_env.sh` + thin `run.sh`), the CUDA-PATH fix, the nvcc fail-fast guard |
| 3 | #1475 | `fpm-prc-collector` | The collector's FPM workflow (planner, runner, in-pod runtime, follower watchdog) — the **consumer** of what #1474 renders |
| 4 | #1461 | `fpm-modeling-rust` | FPM modeling on both engines (Python + Rust `Op::FpmForward`), sweep integration, schema-v6 loaders — the **consumer** of what #1475 collects. Contains all of #1384 (`feature/fpm-modeling`) as a superset; do **not** merge #1384 separately |

Why this order: the contract defines the interface, the generator produces the artifacts, the collector consumes them and produces perf data, the modeling stack consumes the perf data.

This branch was built exactly like this (reproducible):

```bash
git fetch upstream main
git checkout -b fpm-all-20260811 upstream/main
git merge --no-ff fpm-pr0-contract    -m "merge: PR #1473 fpm contract"
git merge --no-ff fpm-prg-generator   -m "merge: PR #1474 generator FPM render surface"
git merge --no-ff fpm-prc-collector   -m "merge: PR #1475 collector FPM workflow"
git merge --no-ff fpm-modeling-rust   -m "merge: PR #1461 fpm modeling on the Rust engine"
# + restore .agents/skills/aic-codeowners/SKILL.md (dropped by a stray merge resolution on #1461)
# + one extra commit adding scripts/experiments/ (experiment-only, never for upstream)
```

All four merges are conflict-free against `upstream/main` @ `648aebcc`; the `--no-ff` merge commits only record the order.

**Modeling caveat:** #1461 adds new Rust operators, so the native extension
(`_aiconfigurator_core.abi3.so`) must be built **from this branch's Rust source** —
an `.so` copied from an older build imports fine but lacks `Op::FpmForward` and the
Rust engine will fail at plan time. See Step 2.

---

## Part 2 — Running an experiment, step by step

### Step 1: Teleport login (every ~7 hours)

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh:443
for C in dynamo-aws-dev-02 dynamo-aws-dev-01 dynamo-nebius-2 dynamo-nscale-dev-cluster; do tsh kube login $C; done
tsh status | grep 'Valid until'
```

Expect: browser SSO completes; validity ~7h.
Warning: `tsh logout` (or expiry) wipes **all** kube contexts — rerun the `for` loop after every login.

### Step 2: point the script at your Python env (once per terminal)

```bash
export FPM_PYTHON=/path/to/venv/bin/python   # a venv with the collector dependencies
PYTHONPATH=$PWD/src:$PWD/aic-core/src $FPM_PYTHON collector/collect.py --help | head -2
```

Expect: the usage text, with `fpm_forward` in the ops list. (The collector shells out
to kubectl — there is no Python kubernetes-client dependency.)

**Required: the compiled native core.** Real runs import the SDK, which loads the Rust
extension `aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so` (a gitignored
build artifact — a fresh clone does not have it). Provide it once per worktree, either by
building (`cd aic-core/rust/aiconfigurator-core && maturin develop`) or by copying the file
from any existing build/wheel of the same revision. Verify with:

```bash
PYTHONPATH=$PWD/src:$PWD/aic-core/src $FPM_PYTHON -c "import aiconfigurator.sdk.common; print('sdk OK')"
```

### Step 3: verify cluster prerequisites (once per cluster, ever)

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nebius-2        # see Appendix A for all contexts
kubectl --context=$CTX get secret nvcr-push-secret -n yuanli-aic
kubectl --context=$CTX get pvc -n yuanli-aic
```

Expect: the image-pull secret exists, and the model-cache PVC is present
(`shared-model-cache` on h100/b200, `model-cache` on h200/gb200).
If the model itself was never downloaded, run a one-off download Job
(`python:3.12-slim` + `pip install hf_transfer`, `HF_HUB_CACHE=/cache` mounted on the PVC).

### Step 4: run

Always preview first:

```bash
scripts/experiments/fpm_collect.sh h200 8 m27 --dry-run
```

Then run for real (drop `--dry-run`):

```bash
scripts/experiments/fpm_collect.sh h200 8 m27
```

Expect: a header line `== 集群 h200 | 8卡 | m27 | --smoke ==`, render logs, pods starting, engine boot; 20–40 min total (first multinode runs take longer).

All validated combinations, copy-paste ready:

```bash
scripts/experiments/fpm_collect.sh h100  4  m27
scripts/experiments/fpm_collect.sh h100  8  m27
scripts/experiments/fpm_collect.sh h100  16 m27                      # 2 nodes
scripts/experiments/fpm_collect.sh h200  4  m27
scripts/experiments/fpm_collect.sh h200  8  m27
scripts/experiments/fpm_collect.sh h200  16 m27                      # 2 nodes
scripts/experiments/fpm_collect.sh gb200 8  m27 --imex-workaround    # ARM; 2 nodes
scripts/experiments/fpm_collect.sh gb200 16 m27 --imex-workaround    # ARM; 4 nodes
scripts/experiments/fpm_collect.sh b200  8  glm-nvfp4
scripts/experiments/fpm_collect.sh b200  16 glm-nvfp4                # 2 nodes
```

Smoke vs formal: the default is **smoke** (`--smoke --limit 1`: one cell, 4 benchmark points, no formal database writes). Add `--formal` for a **full collection** (full sampling grid, parquet output). Always smoke first.

### Step 5: watch progress (optional, second terminal)

```bash
watch -n 30 "kubectl --context=$CTX get pods -n yuanli-aic | grep fpm-"
```

Expect: pods go `Pending` → `Running` (one leader + N-1 followers for multinode), then disappear on completion.

### Step 6: read the result

```bash
grep '"status"' .collector_checkpoint/fpm_forward_smoke.json   # smoke runs
grep '"status"' .collector_checkpoint/fpm_forward.json         # formal runs
```

`"status": "passed"` = success. (Smoke and formal keep separate checkpoints.)

| What | Where |
|---|---|
| Engine logs on failure | `fpm_forward_artifacts/<run>/smoke/cells/<cell>/raw/<pod>/engine.std{out,err}.log` |
| Rendered manifests / launch script | same dir: `k8s_deploy.yaml`, `run.sh` |
| Collection summary (error taxonomy) | `all_<timestamp>/collection_summary_vllm.json` |
| **Formal parquet** (only with `--formal`) | `fpm_formal_database/<system>/vllm/<version>/fpm_forward_perf.parquet` + `.metadata.json` |

Verify a formal run's output (schema v6):

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_parquet("fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")
print(len(df), "rows")
print(df[["moe_backend", "attention_backend", "enable_wideep", "enable_eplb"]].drop_duplicates())
EOF
```

Expect: string "auto" for unpinned backends, real booleans `False` for
wide-EP/EPLB, positive `latency_ms` across the grid.

### Step 7: verify cleanup (hard rule: zero residue)

A normal run prints `零残留 ✓` (zero residue) at the end. After an abnormal stop (Ctrl-C, network loss, timeout), clean up manually:

```bash
kubectl --context=$CTX get pods,podcliquesets,computedomains -n yuanli-aic
kubectl --context=$CTX delete podcliqueset <name> -n yuanli-aic
kubectl --context=$CTX delete computedomain <name> -n yuanli-aic     # GB200 only
```

If you used `--imex-workaround`, restore the in-place-edited facts file:

```bash
git checkout -- src/aiconfigurator/generator/facts/hardware.yaml
```

---

## Part 3 — How to change each dimension

The driver script `fpm_collect.sh` takes `<cluster> <gpus> <model>` positionally. Here is exactly what each knob does and how to go beyond the presets.

### 3.1 Cluster

`h100 | h200 | gb200 | b200` selects a block in the script's **cluster registry** (the `case "$CLUSTER"` block) which sets, per cluster:

- kubectl context (`CTX`)
- GPU profile passed to the collector (`--gpu h100_sxm / h200_sxm / gb200 / b200_sxm`)
- transport (`--transport efa / ib / nvlink`, or none for B200's LWS default)
- orchestrator (`--fpm-orchestrator grove` everywhere except B200)
- container image (x86 clusters use the steady image; GB200 **must** use the ARM image `gc-steady-arm64-schedonly-20260810`)
- KAI queue-compliance labels and known-bad-node blacklists

**To add a cluster**: copy the closest `case` block, change context/PVC/image/transport, and keep the KAI labels if the cluster enforces queue scheduling (pods that bypass the queue get reclaimed with exit 137).

### 3.2 Model

`m27 | glm-nvfp4 | qwen32b` selects a block in the **model registry** (the `case "$MODEL_KEY"` block) which sets the HF path and the snapshot directory inside the model-cache PVC.

**To add a model**:

1. Download it to the target cluster's PVC (one-off Job, see Step 3).
2. Add a `case` entry with `MODEL_PATH` (HF id) and `SNAPSHOT` (the `models--ORG--NAME/snapshots/<rev>` path inside the PVC).
3. Mind hardware constraints — the script encodes one already: NVFP4 models are sm100-only, and the GB200 ARM image lacks FP4 kernels, so `glm-nvfp4` is restricted to B200.

### 3.3 GPU count

`4 | 8 | 16` maps to `--fpm-max-gpus N --fpm-tp-sizes N`.
Node math: a run needs `N / gpus-per-node` **whole** nodes — 8 GPUs/node on h100/h200/b200, only 4 GPUs/node on GB200 (so gb200×16 = 4 nodes). Anything above one node goes through the multinode path (leader + headless followers).

### 3.4 Parallelism mode (the part you edit by hand)

The script pins the campaign-validated shape: **TEP** (`--fpm-parallel-presets tep --fpm-tp-sizes N`), and for 16 GPUs it additionally pins `--fpm-dp-sizes 1` so the planner selects TEP16/DP1 instead of a mixed shape.

The collector exposes these knobs (pass them by editing the `CMD=(...)` array in the script, or by invoking `collector/collect.py` directly):

| Knob | Values | Meaning | Constraints |
|---|---|---|---|
| `--fpm-parallel-presets` | `tep`, `dep` (MoE); `tp` (dense only) | which parallel family the planner enumerates | MoE models only accept tep/dep |
| `--fpm-tp-sizes` | e.g. `8`, `16` | tensor/expert-parallel width | **tep only** |
| `--fpm-dp-sizes` | e.g. `1`, `2` | data-parallel replicas | with `dep`, this is the sharding axis; with tep×16 keep `1` |
| `--fpm-max-gpus` | total GPUs | upper bound for the plan | must equal tp×dp for a single pinned shape |

Concrete recipes:

```bash
# TEP16, one DP replica (the validated 16-GPU shape) — what the script does:
--fpm-parallel-presets tep --fpm-tp-sizes 16 --fpm-dp-sizes 1 --fpm-max-gpus 16

# DEP16 (expert-parallel, DP-sharded across all 16 GPUs):
--fpm-parallel-presets dep --fpm-dp-sizes 16 --fpm-max-gpus 16

# Mixed shape, 2 DP replicas x 8 GPUs each — use the dep family; the CLI
# rejects --fpm-dp-sizes together with tep (dp filters only apply to dep):
--fpm-parallel-presets dep --fpm-dp-sizes 2 --fpm-max-gpus 16

# Dense model with plain TP (e.g. qwen32b):
--fpm-parallel-presets tp --fpm-tp-sizes 8 --fpm-max-gpus 8
```

### 3.5 Backend identity knobs (schema v6)

Four columns identify which engine backends a row was measured under.
Defaults: the string backends record `"auto"` (engine decides); the two
booleans default to `false`. Pinning a value makes the collector deliver it
to the engine and demand resolved-config evidence — combinations without
verified plumbing are rejected up front:

| Knob | Values | Plumbing (vllm) |
|---|---|---|
| `--fpm-moe-backend` | auto / kernel name | `--kernel-config {"moe_backend": ...}` + marker |
| `--fpm-attention-backend` | auto only for now | rejected if pinned (no verified plumbing yet) |
| `--fpm-enable-wideep` | false only on vllm | wide-EP is SGLang-only; `true` is rejected |
| `--fpm-enable-eplb` | true / false | `--enable-eplb` + marker when `true` |

Pass them through the driver by editing the `CMD` array, or invoke
`collector/collect.py` directly. Note: pinned values need working
resolved-config dumps in the image (see the image-vintage caveat in the
troubleshooting table).

### 3.6 Smoke vs formal, cell count

- `--formal` in the driver removes `--smoke` → full sampling grid + database/parquet writes.
- `--limit N` caps how many cells the plan executes (default 1).

---

## Part 4 — End-to-end: from collected parquet to FPM modeling

Collection (Parts 2–3) produces the data; this part closes the loop by running
`forward_model="fpm"` predictions against it. Everything here is CPU-only —
no cluster needed.

### 4.1 One-time environment (this worktree)

```bash
uv sync --extra dev
```

This creates `.venv` AND compiles the native Rust extension
(`aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so`) from **this
branch's** Rust source via maturin. That matters: #1461 adds `Op::FpmForward`
to the Rust engine (the default engine since #1454); an `.so` from any older
source imports fine but fails at plan time. The rustup toolchain lives at
`/opt/homebrew/opt/rustup/bin` (not on the default PATH). Sanity check —
the parity suite has a dedicated `forward_model="fpm"` section:

```bash
.venv/bin/python -m pytest aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py -q
```

### 4.2 Stage the collected pair into the systems tree

The modeling loader resolves exactly one path per (system, backend, version):

```
<systems_root>/data/<system>/<backend>/<version>/fpm_forward_perf.parquet   (+ .metadata.json)
```

With the editable install, the systems root is the in-repo tree, so staging is
one copy (example: an h200 formal run on vllm 0.25.1):

```bash
mkdir -p aic-core/src/aiconfigurator_core/systems/data/h200_sxm/vllm/0.25.1
cp fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet \
   fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.metadata.json \
   aic-core/src/aiconfigurator_core/systems/data/h200_sxm/vllm/0.25.1/
```

Rules the loader enforces (all fail loudly, by design):

- **Copy the pair, never one file** — the sidecar carries the parquet's sha256;
  an unmatched pair is rejected.
- **Never edit the parquet** — any byte change breaks the digest gate.
- **Never relocate across systems** — the sidecar pins system/backend/version;
  an h200 pair copied into a b200 tree is rejected.
- Do **not** commit staged pairs; they are experiment state, not curated data.

### 4.3 Run FPM modeling

```bash
.venv/bin/aiconfigurator cli default \
  --model-path MiniMaxAI/MiniMax-M2.7 \
  --system h200_sxm --backend vllm --backend-version 0.25.1 \
  --total-gpus 16 \
  --forward-model fpm
```

`--forward-model fpm` is accepted by `default`, `exp`, `generate`, and
`estimate` (not `afd`). Three exact-match rules decide whether a query is
answerable:

1. **`--model-path` must equal the collected `model_path` exactly** — cell
   selection is exact-string (decision D1); no family/architecture borrowing.
2. **Backend identity must match** the v6 identity columns the run was
   collected under (`moe_backend`/`attention_backend` strings, wide-EP/EPLB
   booleans; unpinned collections record `"auto"`/`false`).
3. **No extrapolation, ever**: points outside the collected
   (batch, prefill-tokens, kv-read-tokens) hull are *unanswerable* — sweeps
   skip them rather than guessing. If the search comes back empty, your grid
   doesn't cover the workload; collect more, don't loosen the query.

---

## Part 5 — Accuracy validation methodology

Method distilled from previous silicon-alignment campaigns: climb a ladder of
comparisons, each rung isolating one error source, and **never attribute a
residual across more than one rung**. Keep a running ledger
(`prediction | measured | delta | attribution`) per campaign; record retracted
claims with the reason.

### Level 0 — Closure (CPU-only, run first, costs nothing)

Predict at exactly the collected grid points; FPM must reproduce its own
input rows (error ≈ 0, exact-lookup path). This validates loader identity,
cell matching, and phase bookkeeping — plumbing, not physics. Also run the
Rust/Python parity suite against the *real* staged parquet (not just the
synthetic fixture): both engines must agree before either is compared to
silicon.

### Level 1 — Noise floor (same silicon, re-measure)

Re-collect a small probe subset twice on the same cluster (different pods,
non-blacklisted nodes; medians of ≥3 benchmark repeats with the first run
discarded — JIT/autotune warm-up skews percentiles, and the vLLM autotune
cache is warm across pods, so a retry is not an independent sample).
Define the noise floor δ = median |run-to-run delta|. **A prediction error
below δ is unfalsifiable** — acceptance gates below are clamped to ≥ 2δ.

### Level 2 — Interpolation holdout (the core FPM claim)

FPM's promise is accurate *interpolation inside* the collected domain. Test
exactly that:

1. Run a formal collection (the full sampling grid) → the staged parquet.
2. Collect a **probe set** of interior off-grid points — batch/prefill/kv
   combinations strictly inside the hull but *not* on the lattice (drive
   `collector/collect.py` directly with pinned shapes to hit them).
3. Compare FPM predictions (from the formal parquet) against probe medians.

Metrics per (system, GPUs, parallel shape, phase): median APE and p95 APE,
prefill and decode reported separately. Gates from prior campaigns:
**median ≤ 5%, p95 ≤ 10%**, clamped to ≥ 2δ. Investigate any point > 15%
individually before touching the aggregate verdict.

### Level 3 — Deployment-faithful end-to-end

Compare the FPM-backed *serving* prediction (TTFT/TPOT/throughput at a target
concurrency) against a real deployment measured with a load generator:

- Deploy with the generator's own artifacts, on the same cluster, with the
  **same pinned image the data was collected under** (Appendix B) — image
  vintage changes kernels, and kernels are what FPM measured.
- Match the measurement to what the model models: CUDA-graph capture sizes
  must equal the deployment's; a standalone decode benchmark is polluted by
  chunked-prefill mixing; EP/TP ranks run in lockstep so account per-step
  across all ranks; discard first-run numbers.
- Before blaming kernel speed, do the consistency arithmetic:
  `throughput = concurrency × per-user speed`. Derive *effective* concurrency
  from the measurement; a gap vs configured concurrency localizes the error
  to admission/KV-memory, not forward-pass latency — that is a different rung.
- Cross-check the memory side with the engine's own ledger lines (weights GiB,
  KV tokens, max concurrency) against AIC's per-component memory dict —
  component by component, never totals (component errors cancel in totals).

### MoE-specific caveat (m27 is FP8 MoE)

The collector benchmarks with synthetic inputs; near-identical hidden states
collapse expert routing (few unique experts → several-fold less weight
traffic), biasing measured forward latency **fast** relative to real text.
Real weights (from the PVC snapshot) don't fix this — routing is
data-dependent. So: treat Level-2 numbers as internally consistent
(prediction and probe share the bias), but at Level 3 run the load generator
with **realistic prompts** and expect a systematic gap on MoE models; measure
it, attribute it to routing entropy explicitly in the ledger, and do not
"fix" it by tuning unrelated knobs.

### Campaign deliverables

- The ledger (one row per comparison, with attribution).
- Pinned provenance: branch SHA, image tags (Appendix B), parquet sha256s
  (from the sidecars), cluster + node set, probe-point definitions.
- The residuals deliberately left unexplained, stated as such.

---

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `timed out waiting for N FPM pods` | no capacity (16-GPU needs 2–4 whole idle nodes) | pure queueing — retry in ~30 min |
| Pods stuck `Pending` | same (gang scheduling can't assemble whole nodes) | same |
| `NCCL error: unhandled cuda error` (GB200) | cluster NVLink/IMEX fabric fault | use `--imex-workaround`; if it persists you drew a bad node — retry |
| Engine crashes minutes in with `!cubin.empty()` assert | the old CUDA-PATH bug | make sure you are on this branch (it carries the fix) |
| Pods vanish / exit code 137 | bypassed the KAI queue and got reclaimed | use this script (labels included); don't strip scheduling params |
| `unrecognized arguments: --prefill-...` | image's dynamo build lacks the FPM CLI | keep the image tags pinned in the script (they are validated) |
| `no <family>/vllm/<ver> directory with measured data exists` | formal publish refuses to invent curated trees | the driver passes `--fpm-database-root` automatically; keep it |
| pinned backend value rejected at evidence check | the image's resolved-config dump is broken (holdout-signature vintage bug) | collect with auto, or fix the image's config dump first |

---

## Appendix A — quick reference

| Cluster | kubectl context | Arch | PVC | 16 GPUs = nodes |
|---|---|---|---|---|
| h100 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02` | x86 | shared-model-cache | 2 |
| h200 | `nv-prd-dgxc.teleport.sh-dynamo-nebius-2` | x86 | model-cache | 2 |
| gb200 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01` | **ARM** | model-cache | 4 (4 GPUs/node) |
| b200 | `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster` | x86 | shared-model-cache | 2 |

| Model key | Actual model | Notes |
|---|---|---|
| m27 | MiniMaxAI/MiniMax-M2.7 (FP8 MoE, 222 GB) | main multinode workhorse |
| glm-nvfp4 | nvidia/GLM-5.2-NVFP4 | sm100-only → B200 |
| qwen32b | Qwen/Qwen3-32B (dense) | download to the target cluster first |

Built into the script so you don't have to think about it: per-cluster image selection, KAI queue-compliance labels, known-bad-node blacklists, TEP16/DP1 pinning at 16 GPUs, and post-run residue verification.

---

## Appendix B — pinned container images

All images live in the NGC private registry
`nvcr.io/0980761089281446/dynamo-fpm-frozen` (**NGC, not Docker Hub** — pods
need the `nvcr-push-secret` image-pull secret). They are crane-appended
variants of the frozen dynamo build; the driver pins them per cluster:

| Cluster | Image tag | Why this one |
|---|---|---|
| h100, h200 | `gc-steady-16xfix-20260809` | x86 steady build + the 16-GPU multinode fix |
| gb200 | `gc-steady-arm64-schedonly-20260810` | ARM64 build; carries the dual-signature scheduler fix; **lacks FP4 kernels** (why glm-nvfp4 is barred from GB200) |
| b200 | `d719cca-gc-steady-20260729` | the original frozen baseline (dynamo `d719cca`) |

Rules:

- **Do not bump a tag casually.** Image vintage changes kernels and the
  engine's CLI surface; the tags above are exactly what the collected data
  and the troubleshooting table were validated against (e.g. older vintages
  lack the FPM CLI args; one vintage ships a broken resolved-config dump that
  breaks pinned-backend evidence checks).
- **Level-3 accuracy validation must deploy the same tag the data was
  collected under** (see Part 5) — comparing predictions from one kernel
  vintage against measurements from another invalidates the comparison.
- If a new image is unavoidable, re-run a smoke collection first and treat
  every prior parquet as suspect until a Level-1 re-measure confirms the
  kernels didn't move.
