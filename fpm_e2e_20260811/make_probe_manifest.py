# Build the L2 probe manifests (per topology): A-group on-grid anchors and
# B-group off-grid interior probes, both verified against the actual lattice.
import json

import pandas as pd

df = pd.read_parquet("fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet")

for tp in (4, 8):
    pre = df[(df.workload_kind == "prefill") & (df.tp == tp)]
    dec = df[(df.workload_kind == "decode") & (df.tp == tp)]
    pre_grid = set(zip(pre.batch_size, pre.total_prefill_tokens, pre.total_kv_read_tokens))
    dec_grid = set(zip(dec.batch_size, dec.total_kv_read_tokens))

    prefill_pts, decode_pts, rows = [], [], []

    def add_pre(b, tokens, kv, group, note):
        b, tokens, kv = int(b), int(tokens), int(kv)
        on_grid = (b, tokens, kv) in pre_grid
        assert on_grid == (group == "A"), f"grid check failed: {b},{tokens},{kv} group={group} on_grid={on_grid}"
        prefill_pts.append({"batch_size": b, "total_prefill_tokens": tokens, "total_kv_read_tokens": kv})
        rows.append(("prefill", group, b, tokens, kv, note))

    def add_dec(b, kv, group, note):
        b, kv = int(b), int(kv)
        on_grid = (b, kv) in dec_grid
        assert on_grid == (group == "A"), f"grid check failed: {b},{kv} group={group} on_grid={on_grid}"
        decode_pts.append({"batch_size": b, "total_kv_read_tokens": kv})
        rows.append(("decode", group, b, kv, "", note))

    # ---- A-group: exact grid rows (noise floor + drift + cliff stability) ----
    for t in (1024, 2048, 2049, 4096, 8192):
        add_pre(1, t, 0, "A", "cliff bracket + eager supports" if t in (2048, 2049, 4096) else "fast/boundary anchor")
    # NOTE: prefix-cache (kv>0) prefill probes are excluded — they require
    # prefix caching ON, but the single-agg-launch probe reuses the decode
    # cell's engine flags (--no-enable-prefix-caching, parity with formal
    # decode). Prefix-region interpolation is left unprobed; see LEDGER.
    # decode anchors: per batch stratum pick the median collected KV of that curve
    for b in (8, 9, 40, 64, 256, 1024):
        curve = sorted(dec[dec.batch_size == b].total_kv_read_tokens)
        if not curve:
            continue
        add_dec(b, int(curve[len(curve) // 2]), "A", "graph-pair anchor" if b in (8, 9) else "8k/1k op point" if b in (40, 64) else "stratum anchor")

    # ---- B-group: off-grid interior probes ----
    # (1) the sparse eager segment — the flagged risk region
    for t in (2560, 3072, 5120, 6144, 7168):
        add_pre(1, t, 0, "B", "eager-segment probe (sparse support)")
    # (2) fast-segment midpoints, batch 1 and 4: computed from the actual
    # lattice so they are guaranteed off-grid and divisible
    for b in (1, 4):
        curve = sorted(pre[(pre.batch_size == b) & (pre.total_kv_read_tokens == 0)].total_prefill_tokens.unique())
        fast = [t for t in curve if 512 <= t <= 2048]
        added = 0
        for i in range(len(fast) - 1):
            mid = ((fast[i] + fast[i + 1]) // 2) // b * b
            if mid > fast[i] and (b, mid, 0) not in pre_grid:
                add_pre(b, mid, 0, "B", "fast-segment midpoint")
                added += 1
                if added >= 2:
                    break
    # (3) prefix-cache off-grid probes: excluded, same reason as above.
    # (4) decode: KV midpoints on collected batch curves (log-mid, divisible)
    for b in (40, 64, 256):
        curve = sorted(dec[dec.batch_size == b].total_kv_read_tokens)
        m = len(curve) // 2
        for i in (m - 1, m):
            mid = ((curve[i] + curve[i + 1]) // 2) // b * b
            if (b, mid) not in dec_grid and mid > curve[i]:
                add_dec(b, mid, "B", "KV-curve midpoint")
    # (4b) the actual 8k/1k decode operating region: kv ≈ batch × (8192+512)
    for b in (40, 64):
        target = b * 8704
        curve = sorted(dec[dec.batch_size == b].total_kv_read_tokens)
        nearest = min(curve, key=lambda v: abs(v - target))
        add_dec(b, int(nearest), "A", "8k/1k operating-region anchor")
        above = [v for v in curve if v > nearest]
        if above:
            mid = ((nearest + above[0]) // 2) // b * b
            if (b, mid) not in dec_grid and mid > nearest:
                add_dec(b, mid, "B", "8k/1k operating-region midpoint")
    # (5) decode off-site batches (site transfer): batch not collected at all
    for b in (12, 48 + 4):  # 12 between sites 9/16; 52 between 49/56
        curve = sorted(dec[dec.batch_size == 40].total_kv_read_tokens)
        kv = (curve[len(curve) // 2]) // b * b
        if (b, kv) not in dec_grid:
            add_dec(b, kv, "B", "off-site batch (site transfer)")

    manifest = {"schema_version": 1, "prefill": prefill_pts, "decode": decode_pts}
    path = f"fpm_e2e_20260811/probe_points_tep{tp}.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"=== TEP{tp}: {len(prefill_pts)} prefill + {len(decode_pts)} decode -> {path} ===")
    print(f"{'phase':8s} {'grp':3s} {'bs':>4s} {'tokens/kv':>9s} {'prefix-kv':>9s}  note")
    for ph, grp, b, x, kv, note in rows:
        print(f"{ph:8s} {grp:3s} {b:4d} {x:9d} {str(kv):>9s}  {note}")
