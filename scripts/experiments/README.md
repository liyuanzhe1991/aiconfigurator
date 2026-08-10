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

### Step 2: point the script at your Python env (once per terminal)

```bash
export FPM_PYTHON=/path/to/venv/bin/python   # a venv with the collector dependencies
$FPM_PYTHON -c "import kubernetes, yaml; print('venv OK')"
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
grep '"status"' .collector_checkpoint/fpm_forward_smoke.json
```

`"status": "passed"` = success.

| What | Where |
|---|---|
| Engine logs on failure | `fpm_forward_artifacts/<run>/smoke/cells/<cell>/raw/<pod>/engine.std{out,err}.log` |
| Rendered manifests / launch script | same dir: `k8s_deploy.yaml`, `run.sh` |
| Collection summary (error taxonomy) | `all_<timestamp>/collection_summary_vllm.json` |

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

# DEP16 (expert-parallel with DP sharding) — edit the CMD array to:
--fpm-parallel-presets dep --fpm-dp-sizes 16 --fpm-max-gpus 16

# TEP8 x DP2 on 16 GPUs (mixed shape):
--fpm-parallel-presets tep --fpm-tp-sizes 8 --fpm-dp-sizes 2 --fpm-max-gpus 16

# Dense model with plain TP (e.g. qwen32b):
--fpm-parallel-presets tp --fpm-tp-sizes 8 --fpm-max-gpus 8
```

### 3.5 Smoke vs formal, cell count

- `--formal` in the driver removes `--smoke` → full sampling grid + database/parquet writes.
- `--limit N` caps how many cells the plan executes (default 1).

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
