# Expanded L2 probe manifests (v2) against the RANDTOK parquet.
# Two manifests per topology, mirroring the collection's two-cell configs:
#   - decode manifest  (run under the decode-cell engine config)
#   - prefill manifest (run under the prefill-cell engine config: sync sched,
#     graphs<=2048, prefix caching ON — same regime as the parquet rows)
# Every point is programmatically asserted OFF-grid (B-group) unless tagged A.
import json
import sys

import pandas as pd

PARQUET = sys.argv[1] if len(sys.argv) > 1 else "fpm_formal_database_randtok/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
df = pd.read_parquet(PARQUET)

for tp in (4, 8):
    pre = df[(df.workload_kind == "prefill") & (df.tp == tp)]
    dec = df[(df.workload_kind == "decode") & (df.tp == tp)]
    pre_grid = set(zip(pre.batch_size, pre.total_prefill_tokens, pre.total_kv_read_tokens))
    dec_grid = set(zip(dec.batch_size, dec.total_kv_read_tokens))
    dec_batches = set(dec.batch_size)

    # ---------------- decode manifest ----------------
    dpoints, rows = [], []

    def add_dec(b, kv, note, expect_offgrid=True):
        b, kv = int(b), int(kv)
        if kv < b: return
        on = (b, kv) in dec_grid
        assert on != expect_offgrid, f"grid membership mismatch: ({b},{kv}) on={on}"
        dpoints.append({"batch_size": b, "total_kv_read_tokens": kv})
        rows.append(("decode", "B" if expect_offgrid else "A", b, kv, note))

    # anchors (on-grid, drift/noise reference)
    for b in (8, 40, 64, 256):
        curve = sorted(dec[dec.batch_size == b].total_kv_read_tokens)
        for idx in (len(curve)//4, len(curve)//2, 3*len(curve)//4):
            add_dec(b, curve[idx], "anchor", expect_offgrid=False)
    # off-lattice batches spread across the axis (KV chosen divisible, mid-range)
    for b in (6, 12, 20, 28, 36, 44, 52, 60, 100, 150, 200, 300, 400, 600, 768):
        if b in dec_batches: continue
        ref = sorted(dec[dec.batch_size == min(dec_batches, key=lambda x: abs(x-b))].total_kv_read_tokens)
        kv = (ref[len(ref)//2]) // b * b
        if (b, kv) not in dec_grid and kv >= b:
            add_dec(b, kv, "off-site batch")
    # KV log-midpoints on collected curves, low->very-high
    for b in (8, 40, 64, 256):
        curve = sorted(dec[dec.batch_size == b].total_kv_read_tokens)
        idxs = [max(1, len(curve)//6), len(curve)//3, len(curve)//2, 2*len(curve)//3, 5*len(curve)//6, len(curve)-2]
        for i in sorted(set(idxs)):
            mid = ((curve[i] + curve[i+1]) // 2) // b * b
            if mid > curve[i] and (b, mid) not in dec_grid:
                add_dec(b, mid, "KV midpoint")
    # both-axes-off
    for b, frac in ((12, 0.7), (36, 0.4), (100, 0.6), (300, 0.3)):
        if b in dec_batches: continue
        ref = sorted(dec[dec.batch_size == min(dec_batches, key=lambda x: abs(x-b))].total_kv_read_tokens)
        kv = (int(ref[-2] * frac)) // b * b
        if kv >= b and (b, kv) not in dec_grid:
            add_dec(b, kv, "both-axes off")

    json.dump({"schema_version": 1, "prefill": [], "decode": dpoints},
              open(f"fpm_e2e_20260811/probe_v2_decode_tep{tp}.json", "w"), indent=1)

    # ---------------- prefill manifest (prefill-cell config) ----------------
    ppoints = []

    def add_pre(b, tok, kv, note, expect_offgrid=True):
        b, tok, kv = int(b), int(tok), int(kv)
        on = (b, tok, kv) in pre_grid
        assert on != expect_offgrid, f"grid membership mismatch: ({b},{tok},{kv}) on={on}"
        ppoints.append({"batch_size": b, "total_prefill_tokens": tok, "total_kv_read_tokens": kv})
        rows.append(("prefill", "B" if expect_offgrid else "A", b, (tok, kv), note))

    tok_lattice = set(pre[pre.total_kv_read_tokens == 0].total_prefill_tokens)
    # anchors incl. cliff pair and budget point
    for t in (1024, 2048, 2049, 4096, 8192):
        if (1, t, 0) in pre_grid: add_pre(1, t, 0, "anchor", expect_offgrid=False)
    # graph-region midpoints (NEW: same regime as parquet under prefill config)
    for t in (192, 320, 448, 640, 896, 1152, 1408, 1664, 1920):
        if t not in tok_lattice: add_pre(1, t, 0, "graph-region midpoint")
    # eager densification
    for t in (2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888, 6400, 6912, 7424, 7936):
        if t not in tok_lattice: add_pre(1, t, 0, "eager midpoint")
    # batch 2-4 off totals
    for b, t in ((2, 1664), (2, 3328), (3, 1920), (3, 5376), (4, 2816), (4, 6400)):
        if (b, t, 0) not in pre_grid: add_pre(b, t, 0, "batch off-total")
    # prefix axis: anchors + midpoints (prefix caching ON in this config)
    kv_lattice = sorted(pre[(pre.batch_size == 1) & (pre.total_kv_read_tokens > 0)].total_kv_read_tokens.unique())
    if kv_lattice:
        # on-grid prefix anchors
        g = pre[(pre.batch_size == 1) & (pre.total_kv_read_tokens > 0)]
        for idx in (len(g)//4, len(g)//2, 3*len(g)//4):
            r = g.sort_values(["total_kv_read_tokens", "total_prefill_tokens"]).iloc[idx]
            add_pre(1, r.total_prefill_tokens, r.total_kv_read_tokens, "prefix anchor", expect_offgrid=False)
        # off-grid prefix midpoints at fixed tokens
        # feasibility: per-request context (kv + new tokens) must fit max_model_len,
        # minus one 16-token block of margin — infeasible points kill the engine.
        # The lattice is dense near max_model_len, so quartiles are taken over the
        # FEASIBLE sub-lattice for this tok; midpoints are 16-block aligned.
        MAX_LEN = 204800
        for tok in (2048, 4096):
            feas = [kv for kv in kv_lattice if tok + kv <= MAX_LEN - 16]
            for i in (len(feas)//4, len(feas)//2, 3*len(feas)//4):
                if i + 1 >= len(feas):
                    continue
                mid = ((feas[i] + feas[i+1]) // 2) // 16 * 16
                if mid <= feas[i] or tok + mid > MAX_LEN - 16 or (1, tok, mid) in pre_grid:
                    continue
                add_pre(1, tok, mid, "prefix midpoint")

    json.dump({"schema_version": 1, "prefill": ppoints, "decode": []},
              open(f"fpm_e2e_20260811/probe_v2_prefill_tep{tp}.json", "w"), indent=1)

    nb = sum(1 for r in rows if r[1] == "B")
    na = sum(1 for r in rows if r[1] == "A")
    print(f"TEP{tp}: decode {len(dpoints)} pts, prefill {len(ppoints)} pts (A={na}, B/off-grid={nb})")
