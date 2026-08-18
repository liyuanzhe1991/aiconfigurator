# MiniMax M2.7 x H200 FPM Performance-Model Verification -- r15 Report

**Scope**: tep4 (tp4+ep4) and dep4 (dp4+ep4); decode + prefill; full loop = collection (real-content injection, in-band 3-boot median) -> DB build -> independent serving ground truth -> paired scoring. Image gc-steady-randtok2-20260812, stock-parity engine config, ShareGPT even pool for collection / odd pool for truth (train-test split).

## 1. Collection commands (reproducible, pod side)

Point manifests (r14 grid, boundary-capped at max_model_len=204800 from config.json):
```bash
python fpm_e2e_20260811/r14/make_points_r14.py \
  --manifest fpm_e2e_20260811/r11/points_{tep4,dep4}_r11.json \
  --max-model-len 204800 --out fpm_e2e_20260811/r14/points_{tep4,dep4}_r14.json
# prefill-only variants + in-band (512<tot<4096) supplement manifests:
python fpm_e2e_20260811/r15/make_r15_supp_points.py --manifest ... --out ...
```
Staging (all files via exec-cat + sha256 verification; kubectl cp is banned
after repeated silent truncation) and launch:
```bash
bash fpm_e2e_20260811/r15/stage_r15.sh          # 15 files, per-file sha check
kubectl exec ... -- nohup bash /tmp/fpm-kvwarm/r15_sequencer.sh &
# per topology: full prefill pass (real content) + 2 in-band supplement boots
# (content pool re-drawn per boot via DYN_BENCH_POOL_TAG)
```
Per-boot engine preparation (r15_run.sh): restore pristine scheduler ->
kvwarm_patch (decode KV warm-up, 11 surgeries) -> prefill_content_patch v4
(real-content injection: flat 2.4M-token ShareGPT pool, per-request
deterministic window; both the measured-request and the kv-prefix-seeding
generation sites are patched so the fake-prefix-cache invariant holds) ->
resolved-config dump fix. decode rows are reused from the r14 collection
(content fix does not affect decode; saves ~80 min/topology).

## 2. Modeling (DB build) commands (local)

```bash
# per topology (dep4: x4 rank files):
python fpm_e2e_20260811/r15/merge_r15_boots.py   --full full.json \
  --supp supp2.json supp3.json --out merged.json     # in-band 3-boot median
python fpm_e2e_20260811/r15/combine_prefill_decode.py --prefill merged.json \
  --decode-src r14_rank.json --out combined.json     # r15 prefill + r14 decode
python fpm_e2e_20260811/kvwarm_patch/build_t3_db.py --ranks combined*.json \
  --template <old parquet> --require "tp=4,moe_ep=4" \
  --out-root db_<topo> --cell-id fpm-kvwarm-<topo>-r15
```

## 3. Verification methodology

**Ground truth** (same image, same parity engine config, stock capture,
ShareGPT odd pool):
- decode: pristine serve stack + per-rank telemetry; single-session harvest of
  "uniform v2 grid + low-kv hole-filling + lockstep shallow pool"; lockstep
  windows validated by entry-clean accounting.
- prefill: burst driver (DP roadblock same-tick admission); in dep4 a mirror
  gate verifies all 4 ranks show the identical pure-prefill step shape
  (per-rank request count / token sum / kv sum equal) with walls within 3%.

**Scoring** (score-what-forms, paired by coordinate):
- every FPM record scored at the coordinate it actually formed; dep pacing
  merge (step wall = group max, cudagraph-bucket-aware pace key);
  one step one vote, one coordinate one vote, median across windows;
- paired drop rule: if either DB fails a lookup, both sides drop the row;
- strata: A = exact grid hit, B = interpolation, C = extrapolation beyond grid.

```bash
python fpm_e2e_20260811/kvwarm_patch/score_t3_decode.py  --topo <t> ...
python fpm_e2e_20260811/r13/score_r13_burst.py --topo <t> \
  --stream burst_stream.jsonl --windows burst_windows.tsv \
  --new-root db_<topo> --old-root <old systems root> --out scores.csv
```

## 4. Results

### tep4 / decode

*DB: r14 (kvwarm real-KV collection); truth: r14-native single-session harvest.*

| DB | n | MAPE | P95 | MAX | >5% | >10% |
|---|---|---|---|---|---|---|
| old DB (random-content) | 103,710 | 7.66% | 15.61% | 34.0% | 61.8% | 34.1% |
| new DB (this campaign) | 103,710 | 1.70% | 4.63% | 19.2% | 4.4% | 1.5% |

### dep4 / decode

*DB: r14 (kvwarm real-KV collection); truth: r14-native single-session harvest.*

| DB | n | MAPE | P95 | MAX | >5% | >10% |
|---|---|---|---|---|---|---|
| old DB (random-content) | 73,781 | 7.39% | 14.78% | 32.3% | 68.2% | 25.1% |
| new DB (this campaign) | 73,781 | 1.60% | 3.71% | 24.4% | 0.1% | 0.0% |

### tep4 / prefill

*DB: r15 (real-content + in-band 3-boot median); truth: r14-native single-session harvest.*

| DB | n | MAPE | P95 | MAX | >5% | >10% |
|---|---|---|---|---|---|---|
| old DB (random-content) | 121 | 6.89% | 14.29% | 66.0% | 58.7% | 16.5% |
| new DB (this campaign) | 121 | 4.76% | 13.03% | 64.7% | 30.6% | 9.1% |

Band decomposition (new DB, signed median = systematic bias):

| band | n | signed median | MAPE |
|---|---|---|---|
| <=512 (cudagraph) | 26 | -3.15% | 4.21% |
| 700-3000 (mem-bound) | 58 | +4.33% | 6.13% |
| >=4096 (token-linear) | 37 | +2.76% | 3.01% |

### dep4 / prefill

**SCOPE DECISION (2026-08-18): dep4 prefill is out of the delivery scope.** Real deployments run prefill on TP/EP workers; attention-DP is a decode-side technique, so this cell has no production consumer. Numbers below are kept for the record; the small-step gap is fully attributed (execution-regime mismatch, see Findings) and archived.

*DB: r15 (real-content + in-band 3-boot median); truth: r14-native single-session harvest.*

| DB | n | MAPE | P95 | MAX | >5% | >10% |
|---|---|---|---|---|---|---|
| old DB (random-content) | 38 | 11.22% | 30.66% | 40.8% | 60.5% | 36.8% |
| new DB (this campaign) | 38 | 10.94% | 29.69% | 38.8% | 57.9% | 42.1% |

Band decomposition (new DB, signed median = systematic bias):

| band | n | signed median | MAPE |
|---|---|---|---|
| <=512 (cudagraph) | 17 | -19.23% | 20.12% |
| 700-3000 (mem-bound) | 19 | +0.70% | 3.82% |
| >=4096 (token-linear) | 2 | -0.03% | 0.55% |

## 5. Findings established in this campaign (each data- or source-verified)

- **Content bias (fixed in r15)**: random-token prefill collection is
  systematically slower than real text in the mem-bound band (+5..12%,
  same-boot A/B). Real-content injection drives the in-band signed bias from
  +5.3% to +0.6% (tep4) / +2.3% (dep4). Random-vs-random scoring hides this
  bias entirely (both sides wrong together) -- the pre-campaign "4-5% gap" was
  this cancellation artifact.
- **Boot-state dispersion (protocol, not fixable in code)**: in the mem-bound
  band (step total 700-3000 tokens) identical hardware + config + content
  disperses +/-8..18% across engine boots (5-boot same-machine corridor;
  boot-internal reps stable +/-1-2%; GPU clocks/temps identical by sampler).
  Countermeasure: in-band 3-boot median in collection; residual is the truth
  side's own single-boot draw.
- **Fleet heterogeneity (operational, RESOLVED for this DB)**: two specimens --
  a mixed-board H200 node (3x 692- + 1x 965- SKU), and a node with one GPU
  clock-locked at 1590 MHz (others 1980; memory clocks all full). The locked
  GPU inflates token-linear-regime prefill by 2-4% via TP lockstep.
  **Verified in vivo**: re-collection on a clock-guard-verified clean node
  moved (b2,tot8192) from 192.9 -> 184.4 ms against a serving truth of 182.6
  (+1.0%), and the >=4096 band from +7.9% signed to +2.8% (MAPE 3.0%).
  Collector protocol now includes a per-GPU boost-clock assertion
  (light-load peak >= 1900 MHz per GPU) before any collection; the v1 guard
  (full-power burn) was itself a power-virus and false-flagged healthy GPUs
  -- the shipped guard uses light-load boost detection.
- **dep4 small-step execution regime (model-semantics decision pending)**:
  raw dep4 serving telemetry for small prefill steps is bimodal -- fast mode
  == DB value (+/-1%, cudagraph) vs slow mode = 1.78x (eager; ratio matches
  the known graph/eager signature). vLLM DP source
  (dp_utils._synchronize_dp_ranks) confirms per-step group mode sync by MIN
  across ranks. Crucially, **mirror-gated balanced lockstep groups (the
  gold-standard truth protocol, and arguably realistic synchronized serving)
  land almost exclusively in the eager regime** (only 2% of gated groups are
  fast); the fast mode lives in ragged/partial ticks. So the captured-mode DB
  genuinely under-predicts dep4 small-step synchronized serving by ~1.5-1.7x.
  Regime-split scoring (DB vs fast-anchored cluster) yields 8.77% MAPE (clean-node DB) but
  cannot bridge the semantic gap. Two candidate fixes, decision pending:
  (a) collect the dep4 small-tot band in eager mode (one enforce-eager boot,
  ~10 min) so DB rows match the dominant serving regime; (b) keep captured
  DB and add a modeling-side regime multiplier (~1.78x) for dep4 small steps.
  The per-step cudagraph-mode telemetry field remains queued to nail the
  local trigger.
- **Excluded by experiment**: machine attribution of the historical 29% gap
  (same-machine cross-boot 85->103ms killed it), content-draw luck (5 distinct
  draws equally slow), MoE routing imbalance (EP-rank LLN at these token
  counts), generator construction (pool == chain, three-arm same-node A/B),
  capture-boundary padding effects, blocker adjacency, prefix-cache hits
  (kv accounting proves zero cache hits in scored steps).

## 6. Known open items

1. tep4 prefill >=4096: re-collect on a clean node (queued; expected to bring
   the band from +7.9% signed to ~+1..2%).
2. dep4 prefill: OUT OF SCOPE by product decision (no production
   consumer; prefill runs on TP/EP workers). Mechanism dossier archived:
   eager-dominant regime under group mode-vote, per-step mode telemetry
   field remains a debt if the cell is ever revived.
3. Giant-kv extrapolation corners (stratum C, 2 points up to 81% APE) --
   grid/modeling item, pre-existing.
4. Mechanism debts (one-line telemetry each, ride along any future boot):
   per-step cudagraph mode per rank; per-step expert routing histogram.
5. dep4 DP truth floor: only 38 scoreable coordinates survive the mirror gate;
   structural (64/162 burst cells infeasible for kv>0 due to single-rank
   prefix residency).
