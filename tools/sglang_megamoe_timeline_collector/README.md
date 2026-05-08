# SGLang MegaMoE Timeline Collector

This project renders and optionally runs the native SGLang PD-disaggregation
MegaMoE nsys timeline collection flow used for the GB200 DeepSeek-V4-Pro
validation.

The rendered artifact is self-contained: it includes the K8s YAML, cleanup,
profile gate, result listing, chunked download, and NVTX verification scripts.
Rendering does not create cluster resources unless `--run` is passed.

## Quick Start

From the `aiconfigurator` repo root:

```bash
python3 tools/sglang_megamoe_timeline_collector/render.py \
  --config tools/sglang_megamoe_timeline_collector/configs/dsv4_gb200_dp8_ep8_decodebs16.yaml
```

The command prints the rendered artifact directory. To launch immediately:

```bash
python3 tools/sglang_megamoe_timeline_collector/render.py \
  --config tools/sglang_megamoe_timeline_collector/configs/dsv4_gb200_dp8_ep8_decodebs16.yaml \
  --run
```

After a run finishes, pull timelines back through the generated copy pod:

```bash
<artifact_dir>/download_results.sh
```

Verify scheduler NVTX labels in the local timelines:

```bash
<artifact_dir>/verify_nvtx.py <artifact_dir> --strict
```

Manual cleanup for a rendered artifact:

```bash
<artifact_dir>/cleanup.sh
```

## What It Collects

- Native SGLang PD-disaggregation, not Dynamo.
- Router launched with `sglang_router.launch_router --pd-disaggregation`.
- Prefill and decode SGLang servers wrapped by `nsys profile`.
- MegaMoE enforced with `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`,
  `--moe-runner-backend deep_gemm`, and `--moe-a2a-backend deepep`.
- Scheduler NVTX patch enabled with `SGLANG_SCHEDULER_NVTX=1`.
- Profile gates watch P/D logs and call `/start_profile` only after target
  batches are observed:
  - prefill: `Prefill batch`, `#new-seq == profile.prefill_local_bs`,
    `#new-token == profile.isl`
  - decode: all decode DP ranks have `Decode batch`,
    `#running-req == target local bs`

The generated gate does not call `/stop_profile`; it relies on SGLang
`num_steps` to stop profiling automatically.

## Config Notes

The baseline config is `configs/dsv4_gb200_dp8_ep8_decodebs16.yaml`.

Common edits:

- `profile.decode_local_bs_list`: target local decode batch sizes.
- `profile.e2e_global_concurrency_override`: global load concurrency.
- `decode.max_running_requests`: must be at least
  `max(profile.decode_local_bs_list) * decode.dp`.
- `decode.cuda_graph_max_bs`: must be at least
  `max(profile.decode_local_bs_list)`.
- `decode.cuda_graph_bs`: include the target batch sizes.
- `nsys.capture_range_end`: for multiple target batch sizes, set repeat high
  enough for the number of profile windows per SGLang process.
- `deepgemm.prefill_cache_dir` and `deepgemm.decode_cache_dir`: existing
  DeepGEMM JIT cache roots.

`profile.prefill_profile_steps` and `profile.decode_profile_steps` are sent to
SGLang `/start_profile` as `num_steps`. They do not directly set AIPerf
`--num-requests`; the generated AIPerf request count is
`concurrency * profile.e2e_request_multiplier`.
