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

---

# Round 2 — randtok2 re-collection & re-validation (2026-08-12)

## Provenance delta from round 1

| What | Value |
|---|---|
| Image | `nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-steady-randtok2-20260812` (sha256:3067293d…) — salt-paired randomized benchmark inputs; first attempt `randtok-20260812` (940e6bd4…) is BROKEN (independent RNG → prefix hash mismatch → 8,914 points dead) and must not be used |
| Parquet | `fpm_formal_database_randtok/h200_sxm/vllm/0.25.1/` — 22,217 rows, schema v6, staged into aic-core systems tree replacing the round-1 (routing-collapse-biased) data |
| Collection | 4/4 cells passed, 0 errors (TEP8 prefill needed `--resume-retry-failed` after a Teleport stream break — infra, not workload) |
| L0 closure | 13,281 addressable points, worst rel err 0.000e+00 (bit-exact) |
| 8k/1k prediction | TEP4 346.15 tok/s/gpu; TEP8 327.13 tok/s/gpu, 41.48 tok/s/user, TTFT 352.84 ms |

## Headline: the -6.5% TPOT "accuracy" of round 1 was error cancellation

Round-1 prefill rows were biased slow at large M (+15% @8192) and fast at
small M; decode rows fast by -9%. In TPOT composition these partially
cancelled → -6.5%. With prefill FIXED by randtok2 (aligned ≤±1.6% to real
traffic across 128-8192), the honest decode gap is exposed: predicted TPOT
24.11 ms vs real 27.36 ms = **-11.9%**, dominated by the unresolved decode
routing-distribution band. Better data made the number worse — that is the
point of this campaign.

## Per-step rescore (12,647 real TEP8 steps vs randtok2 parquet)

Full table: `per_step_validation.csv` (old-parquet copy kept as
`per_step_validation_oldparquet.csv`).

- decode main buckets: B=40 → -9.2% median, B=64 → -8.9% (routing band, unchanged by randtok2 — real-text routing is skewed, benchmark's uniform-random is not);
- decode pad-up substructure: B∈{34..39,41} → -14~-17.4% (engine pads batch up to next capture size {40,48}, model interpolates linearly);
- mixed: +8~10% median at Bd 32-61 (formula unfixed — expected; catastrophic tails -92~-98% at cross-boundary steps remain until MIXED_FORMULA_SPEC lands);
- small-batch decode B≤5: -2.6~-4.9% (band shrinks with batch).

## v2 expanded probes (TEP4, config-matched, randtok2 engine+data)

Artifacts: `probes_v2/tep4/` (decode r1-r3 × 53 pts, prefill r1-r2 × 31 pts,
decomp r4 × 7 pts), scores in `probes_v2_scores.csv`.

**prefill (prefill-cell config: sync sched, graphs≤2048, prefix ON):**
anchors MAPE 1.45% (median 0.65%), off-grid MAPE 2.57% (median 1.09%),
2-rep noise floor 0.26%. Prefix axis (kv 128-98k) within ±4.3%. The one
structural residual: the eager gap right after the 2048-cliff — real curve
DIPS (2049→99.9 ms but 2304→93.5, flat to 3328, rises to 4096→108.2) so
linear interp {2049→4096} overestimates by 9-11% at 2304-3328 (equivalently
b=2/3328 -9.1%, b=4/2816 -8.7%). Fix owner: COLLECTOR lattice — densify
eager tokens (add ~2304/2560/3072/3584); not a formula defect.

**decode (decode-cell config):** 3-rep noise floor 1.21%. Anchors MAPE 3.85%,
off-grid MAPE 7.14% — but the structure decomposes exactly:

| mechanism | evidence (live vs model) | owner |
|---|---|---|
| batch-axis cliff bridging | b=600 +97%, b=768 +51%, b=550 +87%, b=900 +12% — interpolation skips the b=513 eager row and bridges b=512(graph)→b=1024; eager plateau is FLAT (513/550/900 → 87.9/86.3/88.8 ms), correct eager-segment interp would give -0.4% at b=900 | modeling PR (MIXED_FORMULA_SPEC §5) |
| pad-up staircase | real(36)=31.19 ≈ real(40)=30.68 (Δ1.7%) vs linear model(36)=25.79 (+20.9%); off-lattice batches 12/20/28/44 read +7~10% | modeling PR (§5, step semantics) |
| regime-transition rows contaminated | parquet b=513 row 98.70 vs live 87.90 (+12%); parquet (256,4096) row 24.72 vs live 20.88 stable across 3 reps (+18%) | collector (warmup at regime-transition coordinates) |
| routing-distribution band | everything else lands -4~-9% signed (b=512 -6.9%, anchors -1~-5%), consistent with per-step stream | dense control (qwen32b) pending |

## Cluster hygiene

TEP4 probe pods deleted after harvest; namespace `yuanli-aic` verified empty.
TEP8 probes pending full-node capacity (6/8 H200 nodes held by a neighbor
tenant's 8-GPU jobs; capacity watch armed).

## v2 probes — TEP8 + decomposition rounds (final, 2026-08-12)

Final probe stats (all config-matched, randtok2 engine + parquet):

| set | anchors MAPE | off-grid MAPE (median) | noise floor |
|---|---|---|---|
| TEP4 decode ×3 | 3.85% | 7.14% (3.31%) | 1.21% |
| TEP4 prefill ×2 | 1.45% | 2.57% (1.09%) | 0.26% |
| TEP8 decode ×2 | 2.70% | 14.28% (4.17%) | 1.03% |
| TEP8 prefill ×2 | 2.03% | 3.61% (2.82%) | 0.94% |

Decode tails are fully attributed (below); prefill is single-digit everywhere.

### Data-quality scan + decomposition verdicts

Whole-parquet monotonicity scan: 14 decode + 50 prefill non-monotonic row
pairs. Decomposition probes split them into TWO distinct classes:

1. **Corrupt rows (collection failures, must be caught by QA)** — physically
   impossible values, live truth measured on fresh engines:
   - tp8 (256, 6,557,530): row 11.22 ms → live **57.46** (-80% wrong)
   - tp8 (481, 2,097,152): row 12.35 ms → live **39.77**
   - tp8 (496, 1,048,576): row 17.29 ms → live **30.97**
   - tp4 (256, 4,096): row 24.72 → live 20.88 (3-rep, +18% wrong)
   - tp4/tp8 (513, 65,536) eager-transition rows read +11~12% vs live plateau
   All are giant-KV or regime-transition coordinates → **collector fix:
   monotonicity/sanity gate + re-run policy for flagged cells**.
2. **Real jagged engine behavior (rows faithful, smoothing is the error)** —
   the tp8 b=1 kv=131072 prefix curve oscillates between two kernel-plan
   levels (segments flip at tok 385/416/480/496/576/640/705/833/960/1280/1409);
   fresh-engine probes reproduce 5/6 rows within ±1.6% including the inverted
   cliff (832→94.9 slow / 833→75.5 fast). Same phenomenon found on the decode
   KV axis: (64, 3.15M) live 41.3 is a slow pocket while the surrounding curve
   (2.1M→25.7, 3.67M→33.5, 4.19M→35.8) is healthy. Exact-hit queries are
   correct; off-lattice interpolation inside a pocket carries ~±25% local
   error. Not fixable by data; documented as an interpolation floor in
   oscillation zones.

### Cliff/pad decomposition (TEP8 confirms TEP4)

- eager plateau flat at ~95-101 ms (513→101.1, 900→95.2 @ kv≈131k); batch-axis
  bridging remains the dominant off-grid error (+95~164% at b=600/768).
- pad-up staircase has counterexamples: live(12)=11.32 EXCEEDS live(16)=10.12
  (tp8) — padded ragged batches can cost more than the pad target; ceil-step
  semantics is an approximation, not an upper bound.

### Round-2 close-out state

- All four cells probed, TEP4+TEP8 pods deleted, namespace zero-residue.
- Handoffs: MIXED_FORMULA_SPEC §5 (modeling PR: regime-aware batch axis);
  collector QA gate + eager-lattice densification (2304/2560/3072/3584) +
  giant-KV re-run policy (collector PR); dense-model routing control (qwen32b)
  still the open discriminator for the -5~-9% band.

## Acceptance replay — #1461 tasks A/B landed (2026-08-12)

PR head c72bc9fa (mixed totals pricing 74de0ff6 + decode regime partition
6dada527 + review follow-ups) merged into this branch (c2a243ff, one doc-wording
conflict resolved ours). Hard constraint verified: `git diff` over the whole PR
range touches NO perf_interp path (Python engine.py / Rust perf_interp.rs).

probe_analysis_v2.py replay against the randtok2 parquet:

| gate | requirement | result |
|---|---|---|
| cliff points | \|δ\| ≤ 12% | tp4 b600 **-10.6%** (was +97.2), tp4 b768 **-5.7%** (was +50.6), tp8 b600 **-2.6%** (was +163.2), tp8 b768 **+0.7%** (was +95.3) — PASS |
| TEP4 decode off-grid MAPE | ≤ 6% | 7.14% → **3.93%** — PASS |
| TEP8 decode off-grid MAPE (excl. bad-row-poisoned (256,5375744)) | ≤ 6% | 14.28% → **5.64%** — PASS |
| on-grid / graph-side regression | zero drift | anchor stats bitwise identical pre/post — PASS |

Remaining >12% tails (6 pts, all in documented classes, no action for #1461):
pad-up staircase (b=12 +19.2%, b=20 +17.2% — optional snap semantics skipped by
design), giant-KV/oscillation pockets (b=36@1.68M +28.4%, b=64@3.15M +21.0%,
b=40@5.38M +15.8%, b=64@5.38M +14.9% — real jagged engine behavior per decomp
probes; collector QA + lattice work).

## Strict three-phase validation of the landed #1461 formula (2026-08-12)

Method notes (all trips found and closed): steady-state design points only
(tp==chunk AND bd==bd_nom; ramp/drain full-prefill steps excluded); ground
truth restricted to the cap2048 engine section (engine restart at stream line
141390 — the cumulative stream mixes configs before that); sweep wall-time
semantics validated against sync probes via pure-prefill parity
(2048: 46.76 vs 46.25 = +1.1%; 8192: 153.2 vs 150.2 = +2.1% — no async
discount, graphs active).

| phase | strict check | verdict |
|---|---|---|
| prefill | 62 probe points: post-#1461 references bitwise identical to pre-fix (max drift 0.00e+00); all off-grid single-digit | **PASS** |
| decode | 102 graph-side points bitwise identical; 4 eager-side points changed by design (+97/+51/+164/+94% → -10.6/-5.7/-2.6/+0.7%); MAPE gates met (3.93% / 5.64% excl. poisoned) | **PASS** |
| mixed | 375 steady steps, 24 windows (`mixed_validation_v3_steady.csv`): regime catastrophe healed (c2048 rows -39~-46% → +16~+25%, right side of the cliff) but gates FAIL: grid median\|δ\| 13.22% (gate ≤8%), boundary rows +16~+25% (gate ±15%) | **REGIME FIX VERIFIED; GATES FAIL — matches SPEC's predicted "coordinate-fix-only" stage (13.7%)** |

Mixed residual fully decomposed (component split per window):

1. **Graph-side windows (c256-c1024): overshoot ≈ the marginal term exactly**
   (over-marginal residue ±1-2ms ≈ 0). Real graph-side mixed steps cost ≈ the
   pure-prefill row at (chunk+Bd) ALONE — the decode KV-read marginal is
   absorbed into the padded graph replay. The `+ max(0, dec-base)` term
   double-counts here (+8~15% relative).
2. **Eager-side windows (c2048-c6144): the prefill component itself overprices
   by +6~15ms** (real mixed eager ≈ 0.86-0.88 × pure-row at same totals).
   Interacts with the eager-lattice dip (collector densification pending) plus
   a genuine mixed-vs-pure eager gap (mixed avoids the ragged post-cliff
   attention shape).

Follow-up owners: formula — graph-side marginal suppression (regime-aware:
marginal only when the step is eager? needs a discriminating experiment);
collector — eager densification (already assigned); the 0.86-0.88 eager
factor must NOT ship as a constant (single-model/single-HW; same doctrine as
the rejected 1.145).

## Giant-KV pocket root-cause (7-boot discriminator, 2026-08-12)

Setup: 17-pt pocket manifest (flip coords × micro-offsets ±4-8k tokens +
stable controls), one TEP8 decode-config pod, independent engine boots.
Artifacts: `probes_v2/pocket/pocket_b{1..7}.json`, manifests
`probe_pocket_tep8.json` / `probe_pocket_eager.json`.

Exclusion chain (each step measured, not argued):

1. **Measurement noise — excluded.** Controls repeat to 0.1-1.1% across all
   7 boots ((64, 393k): 14.80-15.03 ms). Levels are discrete, not scatter.
2. **flashinfer autotune — exonerated.** TP>1 disables its persistent cache
   (kernel_warmup.py:150, every boot re-tunes per rank) — plausible suspect,
   but boots 4-5 with `--kernel-config '{"enable_flashinfer_autotune":false}'`
   flip among themselves (29-40% spans) and share the same level set as the
   AT-on group. Neither necessary nor sufficient. (Attention backend is
   FLASH_ATTN; autotune covers MoE/allreduce here.)
3. **CUDA-graph capture state — prime suspect (not sole).** Eager giant-KV
   points (b=600, no graph) hold to ≤4.4% across boots while graph-region
   points span up to 40%; but (768, 5.38M) moved +12.7% across the same pair,
   so a secondary runtime sensitivity on the eager side cannot be ruled out
   at n=2. Verdict: capture-time state is the dominant amplifier of the
   lottery; sole-cause claim NOT made.

Phenomenology (binding for data consumers): at kv ≳ 3M and b ∈ {40..256},
step latency lands on 2-3 discrete levels 15-40% apart; selection flips both
across boots at identical shape and within a boot across ±4k-token
micro-shapes; each boot+shape is internally stable.

Resolution (owners):
- collector: flagged giant-KV coordinates collected as multi-boot medians
  (kills the single-boot lottery in the data; residual ±10-20% floor);
- upstream vLLM: determinism report with this 7-boot table (FULL-graph
  capture path first, FA split scheduling second);
- modeling/docs: intrinsic floor documented; interpolation layer NOT at
  fault in this region.
