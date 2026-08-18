# In-pod mixed-step driver (L3_PLAN_V2 §5, pool-first):
#   pool  = Bd ignore-eos requests decoding steadily (launched by wrapper via bench serve)
#   inject = single token-id requests: prompt = primed_prefix(kvp) + chunk tokens
#            -> one mixed step at exactly (bp, chunk, kvp) + Bd decodes
# Windows logged to mixed_windows.tsv; scoring offline (steady filter = exact coords).
import asyncio
import csv
import json
import random

import sharegpt_ids
import sys
import time

import aiohttp

PLAN, STREAM, OUT, KVP_MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]  # KVP_MODE: zero|chunk
URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "MiniMaxAI/MiniMax-M2.7"
REPS = 15
# probe 盐必须全局唯一:同 salt 的探针 token 序列跨 Bd/跨调用重复,会被引擎
# prefix cache 整段命中,坐标塌成 (16, chunk-16)(tep4 战役 zero 模式 27/51 窗阵亡)
NONCE = int(time.time())

def lines():
    try:
        with open(STREAM) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

PFX = {}
def prefix_ids(kv):
    if kv not in PFX:
        PFX[kv] = sharegpt_ids.ids((50_000 + kv), kv)
    return PFX[kv]

async def post(s, ids, max_tokens=1):
    async with s.post(URL, json={"model": MODEL, "prompt": ids, "max_tokens": max_tokens,
                                 "temperature": 0.0}, timeout=aiohttp.ClientTimeout(total=600)) as r:
        await r.read()

async def main():
    rows = [r for r in csv.DictReader(open(PLAN))]
    out = open(OUT, "a")
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        for ri, r in enumerate(rows):
            grp, bp, Bd, chunk = r["grp"], int(r["bp"]), int(r["Bd"]), int(r["chunk"])
            kvp = chunk if (KVP_MODE == "chunk" and grp in ("A", "B")) else 0
            if kvp:
                pfl = kvp // 16 * 16
                if pfl and pfl not in PFX:
                    await post(s, prefix_ids(pfl) + [7])
                    await asyncio.sleep(0.5)
            else:
                pfl = 0
            s0 = lines()
            for k in range(REPS):
                if bp == 1:
                    ids = (prefix_ids(pfl) if pfl else []) + \
                          sharegpt_ids.ids((f"{NONCE}:{ri}:{k}"), chunk)
                    await post(s, ids)
                else:  # C 组:bp 条短请求齐发 + 池
                    per = chunk // bp
                    await asyncio.gather(*[
                        post(s, sharegpt_ids.ids((f"{NONCE}:{ri}:{k}:{i}"), per))
                        for i in range(bp)])
                await asyncio.sleep(0.25)
            s1 = lines()
            out.write(f"{grp}\t{bp}\t{Bd}\t{chunk}\t{pfl}\t{s0}\t{s1}\n"); out.flush()
            print(f"[{ri+1}/{len(rows)}] {grp} Bd{Bd} chunk{chunk} kvp{pfl}: {s0}..{s1} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"MIXED-DRIVER-DONE in {time.time()-t0:.0f}s", flush=True)

asyncio.run(main())
