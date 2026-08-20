# In-pod prefill burst driver (L3_PLAN_V2 §3): fires each plan cell as bp
# simultaneous token-id requests, validates the produced step coordinate from
# the FPM stream, retries split bursts, and logs per-cell stream windows.
# Scoring happens offline from the stream + windows.tsv.
import asyncio
import csv
import json
import random

import sharegpt_ids
import sys
import time

import aiohttp

PLAN = sys.argv[1]                       # prefill_plan.csv
STREAM = sys.argv[2]                     # /results/fpm_stream.jsonl
OUT = sys.argv[3]                        # /results/burst_windows.tsv
URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "MiniMaxAI/MiniMax-M2.7"
VOCAB_LO, VOCAB_HI = 1, 199000
BUDGET = 8192

def stream_lines():
    try:
        with open(STREAM) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def stream_has(start, bp, tok_total, kv_total):
    with open(STREAM) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            try:
                s = json.loads(line)["scheduled_requests"]
            except Exception:
                continue
            if (s["num_prefill_requests"] == bp and s["sum_prefill_tokens"] == tok_total
                    and s["sum_prefill_kv_tokens"] == kv_total):
                return True
    return False

PFX_CACHE = {}
def prefix_ids(kv):
    if kv not in PFX_CACHE:
        PFX_CACHE[kv] = sharegpt_ids.ids((10_000 + kv), kv)
    return PFX_CACHE[kv]

async def post(session, ids):
    async with session.post(URL, json={"model": MODEL, "prompt": ids, "max_tokens": 1,
                                       "temperature": 0.0}, timeout=aiohttp.ClientTimeout(total=1800)) as r:
        await r.read()
        return r.status

BLOCKER = None
async def fire(session, bp, n, kv, salt):
    global BLOCKER
    pf = prefix_ids(kv) if kv else []
    prompts = [pf + sharegpt_ids.ids((salt * 1000 + i), n)
               for i in range(bp)]
    if bp >= 6:  # 拦路石:大 prefill 挡一步,让 burst 全员到齐后整批入场
        if BLOCKER is None:
            BLOCKER = sharegpt_ids.ids((424242), 8192)
        blocker_task = asyncio.create_task(post(session, sharegpt_ids.ids((salt), 8192)))
        await asyncio.sleep(0.05)
        res = await asyncio.gather(*[post(session, p) for p in prompts])
        await blocker_task
        return res
    return await asyncio.gather(*[post(session, p) for p in prompts])

async def main():
    rows = list(csv.DictReader(open(PLAN)))
    out = open(OUT, "a")
    t_start = time.time()
    async with aiohttp.ClientSession() as session:
        for ci, r in enumerate(rows):
            grp, bp, n, kv = r["grp"], int(r["bp"]), int(r["n"]), int(r["kv"])
            reps = int(r.get("repeats", 5) or 5)
            single = n <= BUDGET
            # prime prefix once per kv value (cheap for repeats; cache persists)
            if kv and kv not in PFX_CACHE:
                await post(session, prefix_ids(kv) + [7])   # +1 token past prefix
                await asyncio.sleep(0.5)
            got, attempts = 0, 0
            while got < reps and attempts < reps * 3:
                attempts += 1
                s0 = stream_lines()
                try:
                    await fire(session, bp, n, kv, salt=ci * 100 + attempts)
                except Exception as exc:
                    print(f"cell {ci} attempt {attempts} error {type(exc).__name__}", flush=True)
                    await asyncio.sleep(2)
                    continue
                await asyncio.sleep(0.4)
                s1 = stream_lines()
                ok = True
                if single:  # 坐标校验(单步格才可判;长请求逐步离线打分)
                    ok = stream_has(s0, bp, bp * n, bp * kv)
                if ok:
                    got += 1
                    out.write(f"{grp}\t{bp}\t{n}\t{kv}\t{got}\t{s0}\t{s1}\n")
                    out.flush()
            status = "OK" if got >= reps else f"SHORT({got}/{reps})"
            print(f"[{ci+1}/{len(rows)}] {grp} bp{bp} n{n} kv{kv}: {status} "
                  f"({attempts} attempts, {time.time()-t_start:.0f}s elapsed)", flush=True)
    print(f"BURST-DRIVER-DONE in {time.time()-t_start:.0f}s", flush=True)

asyncio.run(main())
