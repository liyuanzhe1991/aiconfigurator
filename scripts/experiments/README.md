# FPM Collection Experiments — Step-by-Step Guide

Goal: **run FPM self-benchmark collection on H100 / H200 / GB200 / B200 with one command**, on a branch that merges the three FPM PRs.

---

## Part 1 — How this branch was assembled (merge order)

The three PRs form a stack and **must be merged in this exact order** (each depends on the previous one):

| Order | PR | Branch | What it adds |
|---|---|---|---|
| 1 | #1473 | `fpm-pr0-contract` | The FPM contract table (file names, env vars, result-path rules) shared by both sides |
| 2 | #1474 | `fpm-prg-generator` | The generator's FPM render target (produces `k8s_deploy.yaml` + `fpm_env.sh` + thin `run.sh`), the CUDA-PATH fix, the nvcc fail-fast guard |
| 3 | #1475 | `fpm-prc-collector` | The collector's FPM workflow (planner, runner, in-pod runtime, follower watchdog) — the **consumer** of what #1474 renders |

Why this order: the contract defines the interface, the generator produces the artifacts, the collector consumes them. Reversing the order leaves the collector without anything to render with.

This branch was built exactly like this (reproducible):

```bash
git fetch upstream main
git checkout -b fpm-decoupled-merged-20260810 upstream/main
git merge --no-ff fpm-pr0-contract    -m "merge: PR #1473 fpm contract"
git merge --no-ff fpm-prg-generator   -m "merge: PR #1474 generator FPM render surface"
git merge --no-ff fpm-prc-collector   -m "merge: PR #1475 collector FPM workflow"
# plus one extra commit adding scripts/experiments/ (experiment-only, never for upstream)
```

The resulting tree is byte-identical to the top of the PR stack; the `--no-ff` merge commits only record the order.

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

### Step 2: Python env (once per terminal)

All commands below run from the repo root and use `$PY`:

```bash
PY=/path/to/venv/bin/python                    # a venv with the collector dependencies
export PYTHONPATH=$PWD/src:$PWD/aic-core/src
$PY collector/collect.py --help | head -2
```

Expect: the usage text, with `fpm_forward` in the ops list. (The collector shells out
to kubectl — there is no Python kubernetes-client dependency.)

**Required: the compiled native core.** Real runs import the SDK, which loads the Rust
extension `aic-core/src/aiconfigurator_core/_aiconfigurator_core.abi3.so` (a gitignored
build artifact — a fresh clone does not have it). Provide it once per worktree, either by
building (`cd aic-core/rust/aiconfigurator-core && maturin develop`) or by copying the file
from any existing build/wheel of the same revision. Verify with:

```bash
$PY -c "import aiconfigurator.sdk.common; print('sdk OK')"
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

There is no wrapper script: every command below is the complete, literal
`collect.py` invocation (run from the repo root, after Step 2's `$PY` and
`PYTHONPATH`). What you see is exactly what runs.

#### What every flag does

| Flag | What it does | When you change it |
|---|---|---|
| `FPM_KUBECTL="kubectl --context=..."` | env var: which cluster every kubectl call targets (contexts in Appendix A) | per cluster |
| `--backend vllm` | inference backend being measured | never (this campaign) |
| `--ops fpm_forward` | selects whole-model FPM collection (must be the only op) | never |
| `--model-path` | HF model id — plan identity and the engine's `--model` | per model |
| `--gpu` | AIC system profile (`h100_sxm`/`h200_sxm`/`gb200`/`b200_sxm`): nodeSelector, VRAM, GPUs-per-node facts | per cluster |
| `--fpm-max-gpus N` | total GPUs in the plan | per scale |
| `--fpm-parallel-presets` | parallel family: `tep`/`dep` (MoE), `tp` (dense) — see 3.4 | per model family |
| `--fpm-tp-sizes N` | pins TP/EP width (= total GPUs → exactly one shape) | per scale |
| `--fpm-dp-sizes 1` | **16 GPUs only**: pins TEP16/DP1 so the planner doesn't pick a mixed shape | 16-GPU runs |
| `--namespace` | K8s namespace (quota + resources live here) | per environment |
| `--model-cache PVC:MOUNT:SUBPATH` | model cache: PVC name / mount point in pod / snapshot dir. **SUBPATH must reach `models--ORG--NAME/snapshots/<rev>`** (the dir holding `config.json`) — the cache root is not a valid model dir. Look it up: see 3.2 | per model & cluster |
| `--image-pull-secret nvcr-push-secret` | NGC private-registry pull credential | never |
| `--generator-set K8sConfig.k8s_image=...` | engine container image — pinned per cluster, do not bump casually (Appendix B) | per cluster |
| `--generator-set 'K8sConfig.extra_env=[...]'` | content source for prefill points (`DYN_BENCH_PREFILL_CONTENT=sharegpt`): the engine draws from the sharegpt **even** pool (collection pool; the odd pool is reserved for verification truth). Only meaningful on the x86 timing image; ARM (GB200) lacks the content dispatcher — omit there | x86 clusters |
| `--generator-set K8sConfig.fpm_resource_labels=...` | extra pod labels — KAI queue compliance (missing ⇒ pods reclaimed, exit 137) | KAI clusters |
| `--generator-set K8sConfig.worker_extra_pod_spec=...` | pod-spec patch: `runAsUser:0` (FlashInfer cubin write access), `schedulerName: kai-scheduler`, bad-node blacklist affinity, GFD label overrides | per cluster |
| `--fpm-orchestrator grove` | multinode orchestrator (Grove PodCliqueSet); omitted on B200 where LWS works | per cluster |
| `--transport efa/ib/nvlink` | multinode interconnect | per cluster |
| `--smoke --limit 1` | smoke tier: 1 cell, minimal axes, **no database writes**; `--limit` is smoke-only | smoke runs |
| `--fpm-database-root DIR` | formal runs: where the parquet is published (publication refuses to invent curated in-repo trees, so this is required) | formal runs |

Smoke vs formal: **always smoke first** for a new scenario. A formal command
becomes its smoke variant by removing `--fpm-database-root ...` and appending
`--smoke --limit 1`.

#### h200 (nebius-2) · MiniMax-M2.7 · 8 GPUs · formal

```bash
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-nebius-2"
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --model-path MiniMaxAI/MiniMax-M2.7 --gpu h200_sxm \
  --fpm-max-gpus 8 --fpm-parallel-presets tep --fpm-tp-sizes 8 \
  --namespace yuanli-aic \
  --model-cache model-cache:/workspace/model_cache:models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b \
  --image-pull-secret nvcr-push-secret \
  --generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818 \
  --generator-set 'K8sConfig.extra_env=[{"name":"DYN_BENCH_PREFILL_CONTENT","value":"sharegpt"}]' \
  --generator-set 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}' \
  --generator-set 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0}}' \
  --fpm-orchestrator grove --transport ib \
  --fpm-database-root "$PWD/fpm_formal_database"
```

16-GPU variant (2 nodes): change `--fpm-max-gpus 16 --fpm-tp-sizes 16` and append `--fpm-dp-sizes 1`.

#### h100 (aws-dev-02) · MiniMax-M2.7 · 8 GPUs · formal

aws-dev-02 has non-default GFD labels (hence the explicit nodeSelector) and no enforced KAI queue:

```bash
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02"
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --model-path MiniMaxAI/MiniMax-M2.7 --gpu h100_sxm \
  --fpm-max-gpus 8 --fpm-parallel-presets tep --fpm-tp-sizes 8 \
  --namespace yuanli-aic \
  --model-cache shared-model-cache:/workspace/model_cache:models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b \
  --image-pull-secret nvcr-push-secret \
  --generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818 \
  --generator-set 'K8sConfig.extra_env=[{"name":"DYN_BENCH_PREFILL_CONTENT","value":"sharegpt"}]' \
  --generator-set 'K8sConfig.worker_extra_pod_spec={"nodeSelector":{"nvidia.com/gpu.product":"NVIDIA-H100-80GB-HBM3"},"securityContext":{"runAsUser":0,"runAsGroup":0}}' \
  --fpm-orchestrator grove --transport efa \
  --fpm-database-root "$PWD/fpm_formal_database"
```

#### gb200 (aws-dev-01, ARM) · MiniMax-M2.7 · 16 GPUs (4 nodes) · formal

ARM image is mandatory; the affinity blacklist skips the 2026-08-10 IMEX-incident nodes (drop it once the cluster is fixed). For the manual NVLS/cumem workaround during fabric incidents see the end of this step.

```bash
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01"
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --model-path MiniMaxAI/MiniMax-M2.7 --gpu gb200 \
  --fpm-max-gpus 16 --fpm-parallel-presets tep --fpm-tp-sizes 16 --fpm-dp-sizes 1 \
  --namespace yuanli-aic \
  --model-cache model-cache:/workspace/model_cache:models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b \
  --image-pull-secret nvcr-push-secret \
  --generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-steady-arm64-schedonly-20260810 \
  --generator-set 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"default-queue"}' \
  --generator-set 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0},"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"NotIn","values":["ip-100-64-148-63.ec2.internal","ip-100-64-173-248.ec2.internal","ip-100-64-174-195.ec2.internal","ip-100-64-226-152.ec2.internal"]}]}]}}}}' \
  --fpm-orchestrator grove --transport nvlink \
  --fpm-database-root "$PWD/fpm_formal_database"
```

#### b200 (nscale) · GLM-5.2-NVFP4 · 8 GPUs · formal

NVFP4 is sm100-only and the GB200 ARM image lacks FP4 kernels — GLM-5.2-NVFP4 runs on B200 only. The affinity blacklist skips the known dirty-GPU nodes (`xmhbj`, `7wrxm`). B200's LWS works, so no `--fpm-orchestrator`/`--transport`:

```bash
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster"
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --model-path nvidia/GLM-5.2-NVFP4 --gpu b200_sxm \
  --fpm-max-gpus 8 --fpm-parallel-presets tep --fpm-tp-sizes 8 \
  --namespace yuanli-aic \
  --model-cache shared-model-cache:/workspace/model_cache:models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa \
  --image-pull-secret nvcr-push-secret \
  --generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818 \
  --generator-set 'K8sConfig.extra_env=[{"name":"DYN_BENCH_PREFILL_CONTENT","value":"sharegpt"}]' \
  --generator-set 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}' \
  --generator-set 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0},"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"NotIn","values":["cluster-0967a26d-pool-14bee067-prctr-xmhbj","cluster-0967a26d-pool-14bee067-prctr-7wrxm"]}]}]}}}}' \
  --fpm-database-root "$PWD/fpm_formal_database"
```

Dense-model variant (Qwen3-32B on b200, 4 GPUs): same command with
`--model-path Qwen/Qwen3-32B`,
`--model-cache shared-model-cache:/workspace/model_cache:models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`,
`--fpm-max-gpus 4 --fpm-parallel-presets tp --fpm-tp-sizes 4`.

To run any of the above on a different model, swap the three model-bound
values (`--model-path`, the `--model-cache` SUBPATH, the preset family) —
lookup procedure in 3.2.

Expect: render logs, pods starting, engine boot; 20–40 min total for smoke
(first multinode runs take longer).

#### GB200 IMEX incident workaround (manual, only during fabric faults)

This edits a **tracked** facts file in place — restore it after the run and
never commit it:

```bash
sed -i.bak -e 's/NCCL_NVLS_ENABLE: "1"/NCCL_NVLS_ENABLE: "0"/' \
           -e 's/NCCL_CUMEM_ENABLE: "1"/NCCL_CUMEM_ENABLE: "0"/' \
           -e 's/VLLM_USE_NCCL_SYMM_MEM: "1"/VLLM_USE_NCCL_SYMM_MEM: "0"/' \
           src/aiconfigurator/generator/facts/hardware.yaml
# ... run the gb200 command ...
git checkout -- src/aiconfigurator/generator/facts/hardware.yaml && rm -f src/aiconfigurator/generator/facts/hardware.yaml.bak
```

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

Run this after **every** collection, normal or not:

```bash
kubectl --context=$CTX get pods -n yuanli-aic | grep fpm- || echo 'zero residue OK'
```

After an abnormal stop (Ctrl-C, network loss, timeout), clean up manually:

```bash
kubectl --context=$CTX get pods,podcliquesets,computedomains -n yuanli-aic
kubectl --context=$CTX delete podcliqueset <name> -n yuanli-aic
kubectl --context=$CTX delete computedomain <name> -n yuanli-aic     # GB200 only
```

If you applied the IMEX workaround, restore the facts file (see end of Step 4).

---

## Part 3 — How to change each dimension

The Step 4 command blocks pin the campaign-validated values per dimension. Here is what each dimension means and how to go beyond the presets.

### 3.1 Cluster

Each cluster's command block in Step 4 differs in exactly these values:

- kubectl context (`CTX`)
- GPU profile passed to the collector (`--gpu h100_sxm / h200_sxm / gb200 / b200_sxm`)
- transport (`--transport efa / ib / nvlink`, or none for B200's LWS default)
- orchestrator (`--fpm-orchestrator grove` everywhere except B200)
- container image (x86 clusters use the timing image `gc-timing-20260818`; GB200 **must** use the ARM image `gc-steady-arm64-schedonly-20260810`)
- KAI queue-compliance labels and known-bad-node blacklists

**To add a cluster**: copy the closest Step 4 command block, change context/PVC/image/transport, and keep the KAI labels if the cluster enforces queue scheduling (pods that bypass the queue get reclaimed with exit 137).

### 3.2 Model

A model contributes three values to a command: `--model-path` (HF id), the `--model-cache` SUBPATH (`models--ORG--NAME/snapshots/<rev>` inside the PVC), and the preset family (`tep`/`dep` for MoE, `tp` for dense).

Look up the snapshot revision on the target cluster (it changes when the model is re-downloaded — never trust a stale copy):

```bash
kubectl --context=$CTX run pvc-peek --rm -i --restart=Never -n yuanli-aic --image=busybox:1.36 \
  --overrides='{"spec":{"containers":[{"name":"pvc-peek","image":"busybox:1.36","command":["sh","-c","ls /cache/models--MiniMaxAI--MiniMax-M2.7/snapshots"],"volumeMounts":[{"name":"cache","mountPath":"/cache"}]}],"volumes":[{"name":"cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}'
```

**To add a model**:

1. Download it to the target cluster's PVC (one-off Job, see Step 3).
2. Look up its snapshot revision (command above) and substitute the three model-bound values into a Step 4 command.
3. Mind hardware constraints — e.g. NVFP4 models are sm100-only, and the GB200 ARM image lacks FP4 kernels, so GLM-5.2-NVFP4 is restricted to B200.

### 3.3 GPU count

`4 | 8 | 16` maps to `--fpm-max-gpus N --fpm-tp-sizes N`.
Node math: a run needs `N / gpus-per-node` **whole** nodes — 8 GPUs/node on h100/h200/b200, only 4 GPUs/node on GB200 (so gb200×16 = 4 nodes). Anything above one node goes through the multinode path (leader + headless followers).

### 3.4 Parallelism mode (the part you edit by hand)

The Step 4 commands pin the campaign-validated shape: **TEP** (`--fpm-parallel-presets tep --fpm-tp-sizes N`), and for 16 GPUs additionally `--fpm-dp-sizes 1` so the planner selects TEP16/DP1 instead of a mixed shape.

The collector exposes these knobs (edit them directly in the command):

| Knob | Values | Meaning | Constraints |
|---|---|---|---|
| `--fpm-parallel-presets` | `tep`, `dep`, `pure_tp` (MoE); `tp` (dense) | which parallel families the planner enumerates — a space-separated **list** yields one plan covering all of them | `pure_tp` = tensor-parallel experts (`moe_tp` axis, experts sliced instead of distributed); it is capability-gated — models whose runtime capability does not declare it fail enumeration loudly. GLM-5.2 and MiniMax-M2.7 declare it |
| `--fpm-tp-sizes` | e.g. `8`, `16` | tensor/expert-parallel width | **tep only** |
| `--fpm-dp-sizes` | e.g. `1`, `2` | data-parallel replicas | with `dep`, this is the sharding axis; with tep×16 keep `1` |
| `--fpm-max-gpus` | total GPUs | upper bound for the plan | must equal tp×dp for a single pinned shape |

Concrete recipes:

```bash
# TEP16, one DP replica (the validated 16-GPU shape) — what Step 4 uses:
--fpm-parallel-presets tep --fpm-tp-sizes 16 --fpm-dp-sizes 1 --fpm-max-gpus 16

# DEP16 (expert-parallel, DP-sharded across all 16 GPUs):
--fpm-parallel-presets dep --fpm-dp-sizes 16 --fpm-max-gpus 16

# Mixed shape, 2 DP replicas x 8 GPUs each — use the dep family; the CLI
# rejects --fpm-dp-sizes together with tep (dp filters only apply to dep):
--fpm-parallel-presets dep --fpm-dp-sizes 2 --fpm-max-gpus 16

# Dense model with plain TP (e.g. qwen32b):
--fpm-parallel-presets tp --fpm-tp-sizes 8 --fpm-max-gpus 8

# Multi-shape sweep in ONE run: presets take a list, and unpinned sizes
# enumerate every width within the budget. Verified with --plan-only: this
# plans 4 topologies / 8 cells (tep4, tep8, dep4, dep8), executed
# sequentially under one plan sha / one checkpoint:
--fpm-parallel-presets tep dep --fpm-max-gpus 8

# pure_tp on a MoE model (capability-gated; verified on GLM-5.2: plans
# tp4/moe_tp4 + tp8/moe_tp8):
--fpm-parallel-presets pure_tp --fpm-max-gpus 8
```

Notes on multi-shape runs: `--fpm-max-gpus` is an upper bound — without
`--fpm-tp-sizes` the planner enumerates **all** valid widths under it (hence
tep4 *and* tep8 above), and every shape costs 2 cells (prefill + decode).
The campaign commands in Step 4 pin sizes precisely to avoid this fan-out;
use multi-shape runs to fill a topology matrix on one cluster in one sitting.
Repeated runs (any mix of shapes) merge into the same formal parquet —
row-key uniqueness and run identity are checked at publication.

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

Append them to any Step 4 command. Note: pinned values need working
resolved-config dumps in the image (see the image-vintage caveat in the
troubleshooting table).

### 3.6 Smoke vs formal, cell count

- Formal = full sampling grid + parquet writes; its commands carry `--fpm-database-root` and no `--smoke`.
- `--smoke --limit 1` = one cell, minimal axes, no database writes; `--limit` is smoke-only (formal must run its full plan).

---

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `timed out waiting for N FPM pods` | no capacity (16-GPU needs 2–4 whole idle nodes) | pure queueing — retry in ~30 min |
| Pods stuck `Pending` | same (gang scheduling can't assemble whole nodes) | same |
| `NCCL error: unhandled cuda error` (GB200) | cluster NVLink/IMEX fabric fault | apply the manual IMEX workaround (end of Step 4); if it persists you drew a bad node — retry |
| Engine crashes minutes in with `!cubin.empty()` assert | the old CUDA-PATH bug | make sure you are on this branch (it carries the fix) |
| Pods vanish / exit code 137 | bypassed the KAI queue and got reclaimed | use the Step 4 commands verbatim (KAI labels included); don't strip scheduling params |
| `unrecognized arguments: --prefill-...` | image's dynamo build lacks the FPM CLI | keep the image tags pinned in Appendix B (they are validated) |
| `no <family>/vllm/<ver> directory with measured data exists` | formal publish refuses to invent curated trees | keep `--fpm-database-root` in every formal command |
| pinned backend value rejected at evidence check | the image's resolved-config dump is broken (holdout-signature vintage bug) | collect with auto, or fix the image's config dump first |

---

## Appendix A — cluster / image / model reference (single source)

**This table is the complete registry the retired `fpm_collect.sh` used to
encode.** Every value below is baked into a Step 4 command block; if a
command and this table ever disagree, one of them is stale — fix both in the
same commit.

### Per-cluster configuration

| Cluster | kubectl context | Arch | `--gpu` | PVC | Image tag (details: Appendix B) | Orchestrator / transport | KAI queue label | Known-bad nodes (affinity blacklist) | 16 GPUs = nodes |
|---|---|---|---|---|---|---|---|---|---|
| h100 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02` | x86 | `h100_sxm` | `shared-model-cache` | `gc-timing-20260818` | grove / efa | — (not enforced; needs explicit GFD nodeSelector `NVIDIA-H100-80GB-HBM3`) | — | 2 |
| h200 | `nv-prd-dgxc.teleport.sh-dynamo-nebius-2` | x86 | `h200_sxm` | `model-cache` | `gc-timing-20260818` | grove / ib | `dynamo` | — | 2 |
| gb200 | `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01` | **ARM** | `gb200` | `model-cache` | `gc-steady-arm64-schedonly-20260810` | grove / nvlink | `default-queue` | `ip-100-64-148-63/-173-248/-174-195/-226-152.ec2.internal` (2026-08-10 IMEX incident; drop when cluster fixed) | 4 (4 GPUs/node) |
| b200 | `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster` | x86 | `b200_sxm` | `shared-model-cache` | `gc-timing-20260818` | LWS default / — | `dynamo` | `…-prctr-xmhbj`, `…-prctr-7wrxm` (dirty GPUs) | 2 |

### Per-model values

Snapshot revisions are **as observed 2026-08-11** — they change whenever the
model is re-downloaded; always re-verify with the 3.2 lookup before a
campaign.

| Model (`--model-path`) | `--model-cache` SUBPATH (2026-08-11) | Preset | Notes |
|---|---|---|---|
| `MiniMaxAI/MiniMax-M2.7` (FP8 MoE, 222 GB) | `models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b` | tep | main multinode workhorse |
| `nvidia/GLM-5.2-NVFP4` | `models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` | tep | sm100-only → B200 (ARM image lacks FP4 kernels) |
| `Qwen/Qwen3-32B` (dense) | `models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137` | tp | download to the target cluster first |

Also baked into the Step 4 commands: `runAsUser:0` (FlashInfer cubin write
access), `schedulerName: kai-scheduler` on KAI clusters, and TEP16/DP1
pinning at 16 GPUs. Residue verification is Step 7.

---

## Appendix B — pinned container images

All images live in the NGC private registry
`nvcr.io/0980761089281446/dynamo-fpm-frozen` (**NGC, not Docker Hub** — pods
need the `nvcr-push-secret` image-pull secret). They are crane-appended
variants of the frozen dynamo build; the Step 4 commands pin them per cluster:

| Cluster | Image tag | Why this one |
|---|---|---|
| h100, h200, b200 | `gc-timing-20260818` | x86 build: real-content pools (sharegpt, even/odd split) + engine phase timing (`timing.phases`); carries the 16-GPU multinode fix; zero measurement overhead verified (5-boot Guaranteed-pod A/B, in-envelope). Digest `sha256:adcd28a943c7811e71931a4c154d31cb95d126469bc9b4fe2a8981f5327e8c48` |
| gb200 | `gc-steady-arm64-schedonly-20260810` | ARM64 build; carries the dual-signature scheduler fix; **lacks FP4 kernels** (why glm-nvfp4 is barred from GB200) and has no timing/content-pool rev yet |

Rules:

- **Do not bump a tag casually.** Image vintage changes kernels and the
  engine's CLI surface; the tags above are exactly what the collected data
  and the troubleshooting table were validated against (e.g. older vintages
  lack the FPM CLI args; one vintage ships a broken resolved-config dump that
  breaks pinned-backend evidence checks).
- **Accuracy validation must deploy the same tag the data was collected
  under** — comparing predictions from one kernel vintage against
  measurements from another invalidates the comparison.
- If a new image is unavoidable, re-run a smoke collection first and treat
  every prior parquet as suspect until a same-silicon re-measure confirms
  the kernels didn't move.
