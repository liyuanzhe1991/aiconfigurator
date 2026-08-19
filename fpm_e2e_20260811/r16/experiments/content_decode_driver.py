# 内容对拍版 decode 驱动:l3v2_decode_driver.py 原样,仅加 L3_CONTENT 开关
# (sharegpt=奇数池真实文本 | random=均匀随机 token,r14 randtok 同约定
#  VOCAB 1..199000)。拦路石两臂都用 ShareGPT(变量只留池 prompt 内容)。
# 窗口 tag 前缀 L3_PASS,便于离线分臂。用法同 l3v2:plan stream windows [dp]
import asyncio
import csv
import os
import random as _random
import sys

import aiohttp

sys.path.insert(0, "/tmp/fpm-serve")
import sharegpt_ids

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = os.environ.get("L3_MODEL_ID", "MiniMaxAI/MiniMax-M2.7")
CONTENT = os.environ.get("L3_CONTENT", "sharegpt")
PASS = os.environ.get("L3_PASS", CONTENT)
PLAN, STREAM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DP = int(sys.argv[4]) if len(sys.argv) > 4 else 1
assert CONTENT in ("sharegpt", "random"), CONTENT

def prompt_ids(seed, k):
    if CONTENT == "sharegpt":
        return sharegpt_ids.ids(seed, k)
    return _random.Random(str(seed)).choices(range(1, 199000), k=k)

def lines():
    try:
        with open(STREAM) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

async def post(s, ids, mt):
    async with s.post(URL, json={"model": MODEL, "prompt": ids, "max_tokens": mt,
                                 "ignore_eos": True},
                      timeout=aiohttp.ClientTimeout(total=14400)) as r:
        await r.read()
        return r.status

def entry_clean(s0, s1, C, isl):
    import json as _json
    total = 0
    with open(STREAM) as f:
        for i, ln in enumerate(f):
            if i < s0:
                continue
            if i >= s1:
                break
            try:
                d = _json.loads(ln)
                sc = d["scheduled_requests"]
            except Exception:
                continue
            if (sc["num_prefill_requests"] == 0 and sc["num_decode_requests"] >= 1
                    and sc["sum_decode_kv_tokens"]
                        == sc["num_decode_requests"] * isl):
                total += sc["num_decode_requests"]
    return total == C

async def pool(s, tag, C, isl, osl, rep):
    wtag = f"{PASS}:{tag}"
    for attempt in range(3):
        s0 = lines()
        blk_isl = max(8192, min(65536, C * 32))
        blockers = [asyncio.create_task(
            post(s, sharegpt_ids.ids(f"blk{i}:{wtag}:{rep}:{attempt}", blk_isl), 1))
            for i in range(DP)]
        await asyncio.sleep(0.15)
        prompts = [prompt_ids(f"{wtag}:{rep}:{j}", isl) for j in range(C)]
        codes = await asyncio.gather(*[post(s, p, osl) for p in prompts],
                                     return_exceptions=True)
        blk_codes = await asyncio.gather(*blockers, return_exceptions=True)
        blk_ok = all(c == 200 for c in blk_codes)
        if not blk_ok:
            print(f"[{wtag} rep{rep}] 拦路石异常: {blk_codes}", flush=True)
        await asyncio.sleep(0.4)
        s1 = lines()
        ok = sum(1 for c in codes if c == 200)
        if isl == 1:
            clean = blk_ok and entry_clean(s0, s1, C, isl)
            mark = "lockstep" if clean else "ragged"
        else:
            clean, mark = True, "na"
        if clean or attempt == 2:
            with open(OUT, "a") as f:
                f.write(f"{wtag}\t{C}\t{isl}\t{osl}\t{rep}\t{s0}\t{s1}\t{ok}\t{mark}\n")
            print(f"[{wtag} rep{rep}] C={C} isl={isl}: {ok}/{C} {mark}", flush=True)
            return ok
        print(f"[{wtag} rep{rep}] 进场参差,重试 {attempt+1}/3", flush=True)
    return 0

async def main():
    rows = list(csv.DictReader(open(PLAN)))
    conn = aiohttp.TCPConnector(limit=0)
    dead = 0
    async with aiohttp.ClientSession(connector=conn) as s:
        for r in rows:
            for rep in range(int(r["reps"])):
                got = await pool(s, r["tag"], int(r["C"]), int(r["isl"]),
                                 int(r["osl"]), rep)
                dead = dead + 1 if got == 0 else 0
                if dead >= 2:
                    print("CONTENT-DRIVER-ABORT: 连续全灭", flush=True)
                    sys.exit(7)
    print(f"CONTENT-DRIVER-DONE pass={PASS}", flush=True)

asyncio.run(main())
