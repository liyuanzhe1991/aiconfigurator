# FPM E2E Campaign Ledger — 2026-08-11 (h200, m27, 4/8 GPU, 8k/1k)

## Provenance (pinned)

| What | Value |
|---|---|
| Branch | `fpm-all-20260811` @ `f55ff21d` (upstream/main 648aebcc + #1473/#1474/#1475/#1461) |
| Image (h200 collection) | `nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-steady-16xfix-20260809` |
| Cluster | nebius-2 (`nv-prd-dgxc.teleport.sh-dynamo-nebius-2`), namespace `yuanli-aic` |
| Model | `MiniMaxAI/MiniMax-M2.7` (FP8-block MoE), snapshot `d494266a` |
| Parquet | `h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet`, sha256 `3c688b034b9f3228…`, 22,217 rows, schema v6 |
| Cells | TEP4/TEP8 × prefill/decode (4 cells; backend identity auto/auto/false/false) |
| Native .so | built from this branch via `uv sync` (contains Op::FpmForward); parity suite 335 passed |

## Runs

| Run | Wall time | Result |
|---|---|---|
| h100 4-GPU smoke (13:28) | 4 min | FAILED — dirty GPUs (4.99/79.18 GiB free at startup; neighbor tenant under-reporting suspected). Recorded, zero residue. h100 postponed. |
| h200 4-GPU formal (13:57–14:44) | 47 min | PASSED — 0 errors, 11,062 rows, zero residue |
| h200 8-GPU formal (14:47–15:16) | 29 min | PASSED — 0 errors, 11,155 rows, zero residue |

## Predictions (8k/1k: --isl 8192 --osl 1024, forward_model=fpm)

| Config | tokens/s/gpu | tokens/s/user | TTFT (ms) | bs | Source |
|---|---|---|---|---|---|
| 4 GPU, agg TEP4 | 336.99 | 34.20 | 448.13 | 40 | modeling_4gpu/ |
| 8 GPU, 2×TEP4 replicas (rank 1) | 336.99 | 34.20 | 448.13 | 40×2 | modeling_8gpu/ |
| 8 GPU, 1×TEP8 (rank 2) | 308.27 | 39.11 | 393.72 | 64 | modeling_8gpu/ |

Disagg at 4/8 GPUs: correctly UNANSWERABLE (would need TEP1/TEP2 cells; FPM
never extrapolates — sweep skipped, no fabricated rows). ISL 8192 sits on the
prefill-domain boundary (grid max 8192) — in-domain, answerable.

## Accuracy validation

| Level | prediction | measured | delta | attribution |
|---|---|---|---|---|
| L0a closure (13,281 addressable grid pts, all 4 cells) | op.query at exact coords | parquet latency_ms | **0.000e+00 worst rel err** (bit-exact) | exact-lookup path healthy: loader, identity match, phase bookkeeping all correct |
| L0b engine parity (8-GPU sweep, real data) | Rust engine top-2 rows | Python engine top-2 rows | **identical to printed precision** | Op::FpmForward Rust port consistent with Python reference |
| L1 noise floor | 5-repeat probe spreads | — | TEP4 1.74% / TEP8 3.80% median | DONE (see "L2 campaign results") |
| L2 interpolation holdout | model at off-grid explicit points | 5-rep medians | TEP4 2.51% / TEP8 1.31% median | DONE — gate passed (see "L2 campaign results") |
| L3 deployment-faithful, per-step via FPM stream | parquet/model per coordinate | real-traffic steps (3 classes, full sweeps) | decode −5±4%; prefill −33%..+16% (structured); mixed −75%..+19% (config break) | DONE (see Channel A + Mixed sections) |

Notes:
- 8,936 rows are not addressable through the per-request query interface
  (total tokens not divisible by batch — ragged interpolation-support points);
  they participate in interpolation but cannot be individually closure-checked
  via query(). Not a defect; noted for completeness.
- Loader printed a "legacy perf-data layout" migration warning for the staged
  path (h200_sxm/vllm/0.25.1 without a family level) — cosmetic today, tracked
  for Collector V3.

## Channel A — real-traffic ground truth (2026-08-11 17:00, TEP8, C=64, 8k/1k)

Setup: pod `fpm-serve-groundtruth` (same image/engine args, benchmark OFF),
etcd+nats+frontend+worker, passive FPM stream (DYN_FORWARDPASS_METRIC_PORT),
`vllm bench serve` 256 reqs, ISL 8192 / OSL 1024, C=64, ignore-eos.
117.7 s, 0 failed. Replicates the historical arm-A/arm-D design.

| prediction | measured | delta | attribution |
|---|---|---|---|
| parquet@(64, 531k..582k) = 15.43..15.72 ms | real steady pure-decode steps (3,836 @ B=64): 16.94..17.22 ms, IQR 0.23 ms | **self-benchmark data systematically FAST by 8.7–9.3%** | uniform across KV → systematic, not noise. Leading suspect: MoE routing entropy (benchmark-seeded requests collapse experts; bench uses random-token prompts). UNRESOLVED — needs a discriminating experiment |
| client ITL median 17.23 ms | FPM-stream steady median 17.12 ms | +0.6% | instruments agree (no common-mode error; echoes historical 4.82 vs 4.83) |
| FPM e2e TPOT 25.57 ms (39.11 tok/s/user) | client TPOT median 27.36 ms | −6.5% | consistent with the −9% data-layer bias partially diluted by mixed-step composition |
| FPM e2e TTFT 393.7 ms | client TTFT median 783.9 ms | not comparable as-is | client TTFT includes queueing (256 burst arrivals @ C=64) and chunked-prefill interleaving; needs a controlled-admission rerun to compare |

Prefill side not isolated (chunked prefill mixes phases; only 1 prefill-only
step in stream). Channel B (explicit-point probes, 5 repeats) running in
parallel; prefix-cache probe points dropped (engine-flag conflict with the
single-agg-launch config — prefill cells collect prefix points under
prefix-caching ON, decode cells under OFF).

## L2 campaign results (2026-08-11 evening)

**Channel B — explicit-point probes (5 repeats × 32 pts × TEP4/TEP8, additional-config
injection route, decode-cell engine params):**

| prediction | measured | delta | attribution |
|---|---|---|---|
| model interpolation at off-grid points | 5-rep medians | **TEP4 median 2.51%, TEP8 1.31%** | interpolation gate PASSED (≈ historical 2.33%); noise floor 1.7%/3.8% |
| parquet at on-grid anchors | re-measured | high-KV anchors ±0.3-1.8%; regime-boundary pts (B=9, off-site 12/52, low-KV mid-B) +5~13% | regime transitions re-measure unstably; big anchors reproduce |
| parquet prefill rows (prefill-cfg) | probe prefill (decode-cfg) | ≤2048: 1.4-3.1× slower; ≥2049: 0.97-1.01× | graph-config effect isolated; eager regime perfectly reproducible |

**Channel A — real-traffic ground truth (same image; stack① = decode-cell params,
stack② = prefill-cell params):**

| finding | evidence |
|---|---|
| decode data optimistic, routing-entropy signature | bias flat along KV (B=8: −3~−5% from 0.6k to 200k/req), bell-shaped along B (B=1: −0.9% → B=16-64: −7~−11% → B=512: −1.3%); temperature-only experiment moved real steady steps 17.12→16.37ms (−4.4%) |
| prefill small-step host floor | real serving carries ~6ms/step constant overhead absent from benchmark rows (128t: 12.4→18.4ms; scales down in relative terms with step size) |
| prefill eager-regime rows pessimistic | real steps 13-16% FASTER than parquet at 2049-8192 tokens (echoes June golden launch-bound finding); opposite sign to decode bias — partial cancellation explains e2e TPOT −6.5% |
| prefix-axis interpolation healthy | new_tokens ≥2048 with prefix 2k..131k: ±3-8% |
| coverage holes | b1024/kv≥64k high-C windows (steady pool never assembled; OSL too short — rerunnable); mixed bd=8 tail chunks (pool drained) |

Artifacts: per_step_validation.csv (12,647 steps), decode_validation_stack1.csv,
prefill_validation_stack2.csv, probes/tep{4,8}/probe_r{1..5}.json, streams in
serve_results/.

## Mixed-step grid (chunk × Bd, dual-stream method, 30 windows)

Delta%% (model composition vs real mixed steps), decode-config deployment:

|chunk\Bd|8|16|32|40|64|
|---|---|---|---|---|---|
|256|-20.6|-19.1|-15.1|-13.9|-12.5|
|512|-75.0|-73.2|-70.5|-69.1|—|
|1024|-64.2|-63.3|-60.2|-57.0|-50.7|
|2048|-46.5|-44.5|-39.9|-40.5|-35.3|
|4096|—|+12.5|+15.5|+16.2|+18.5|
|6144|—|+18.3|+18.9|+19.1|+18.9|

**Root cause: config-contract break.** The deployment engine (decode-cell
config, no prefill graph capture) pays the eager launch floor (~78-91 ms) for
chunks 512-4096, while the composition formula splices in parquet prefill rows
measured under the prefill-cell config (graphs to 2048, ~20-50 ms). The
-50~-75% zone is exactly the graphs-vs-eager disagreement region — the bread
and butter of chunked-prefill serving. DECISION NEEDED (owner): pin ONE engine
config across collection phases and deployment, or collect both phases under
the deployment config. Until then mixed-step prediction is structurally broken
in that zone; no formula tuning can fix a data-identity mismatch.

## Mixed v2 discriminator + fix validation (2026-08-11 late evening)

**v2 sweep (capture-2048-aligned engine, same 30-window grid):** the -75%
zone collapsed exactly where configs became commensurable (chunk 512: real
81.0→23.8 ms, delta -75→-15.1%; chunk 1024: -64→-5.5%), while chunk 2048
stayed broken (-44%) because chunk+Bd straddles the NEW boundary — proving the
regime is decided by total scheduled tokens, wherever the boundary sits.
Cold-start cost of the capture-2048 config: ~70 min (~100 large graphs) —
a real input to the contract decision.

**Offline formula validation against the v2 ground truth (60 windows):**

| formula | mixed |delta| median | notes |
|---|---|---|---|
| current `prefill(chunk)` | 16.8% | -44% at the boundary row |
| + total-token regime `prefill(chunk+Bd)` | 13.7% | boundary row healed; eager rows inherit +15-20% |
| + eager correction (÷1.145) + per-step scoring | **5.4%** | grid ±10% except one bimodal cell (29.2% @ 2048/Bd=64); residual small-chunk pattern = host-floor signature |

Recommended fix stack (by owner): ① collection reverts prefill cells to the
deployment-default engine config (serving-parity doctrine) + record the
engine's actual capture list in the sidecar; ② composition queries the
prefill component at chunk+Bd (Python + Rust); ③ fix the eager launch-bound
measurement in dynamo rather than shipping the 1.145 constant; host floor as
an explicit per-step overhead term. Constants 1.145 / ~6 ms are single-model,
single-HW — cross-validate (qwen32b, h100) before any lands in the model.

## Residuals deliberately left unexplained

- Decode bias: routing-entropy attribution is now triple-evidenced
  (temperature experiment, KV-flatness, B-bell-shape) but the final
  discriminator (dense-model control, qwen32b) has not run; the [-5%,-9%]
  entropy band is recorded, not folded into any constant.
- Prefill small-step ~6ms host floor: measured, unattributed to a specific
  code path (API/detokenize/scheduler serialization candidates).
- L0 exactness ≠ interpolation ≠ serving fidelity — different rungs, keep
  them separate when quoting this campaign.
