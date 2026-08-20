# kit decode 真值驱动(ShareGPT 版):替代 vllm bench serve random。
# 依据:内容 ABA 实验(2026-08-19)random 池系统性偏快 -1.51%;本驱动
# 沿 l3v2 血统(奇数池 token id 直发 /v1/completions、DP 拦路石、
# isl==1 锁步进场校验),窗口写 v3 九列(tag,C,isl,osl,rep,s0,s1,ok,mark),
# 打分器对 lockstep 窗逐坐标直录。
# 用法: decode_driver.py plan.csv stream.jsonl windows.tsv [dp]
# plan 列:grp,C,isl,osl,kind,boots(kit 既有格式;C 为每 rank 目标,
# dp>1 时实发 C*dp;boots 当 reps 用)。环境:L3_TOKENIZER、
# L3_SHAREGPT_PATH(sharegpt_ids 用)、L3_CONTENT=sharegpt|random(默认
# sharegpt,保留 random 供对照实验)、L3_CAPACITY_GUARD(默认 9e6 token)。
import asyncio
import csv
import os
import random as _random
import sys

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/fpm-serve")
import sharegpt_ids

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = os.environ.get("L3_MODEL_ID", "MiniMaxAI/MiniMax-M2.7")
CONTENT = os.environ.get("L3_CONTENT", "sharegpt")
CAP = int(float(os.environ.get("L3_CAPACITY_GUARD", "9000000")))
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

async def wait_route(s):
    # 路由恢复等待:frontend 对已注册模型也会在 worker 心跳闪断时短暂 404
    # (实测 burst 尾部巨长 prefill 后有闪断窗);小探针等到 200 再继续。
    for _ in range(120):
        try:
            async with s.post(URL, json={"model": MODEL, "prompt": "hi",
                                         "max_tokens": 1},
                              timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(5)
    return False

async def post(s, ids, mt):
    async with s.post(URL, json={"model": MODEL, "prompt": ids, "max_tokens": mt,
                                 "ignore_eos": True},
                      timeout=aiohttp.ClientTimeout(total=14400)) as r:
        await r.read()
        return r.status

def entry_clean(s0, s1, total, isl):
    import json as _json
    seen = 0
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
                seen += sc["num_decode_requests"]
    return seen == total

async def pool(s, tag, C, isl, osl, rep):
    total = C * DP
    if total * (isl + osl) > CAP:
        print(f"[{tag} rep{rep}] 容量守卫跳过:{total}x({isl}+{osl}) > {CAP}", flush=True)
        return -1
    waits = 0
    attempt = 0
    while attempt < 3:
        s0 = lines()
        blk_isl = max(8192, min(65536, total * 32))
        blockers = [asyncio.create_task(
            post(s, sharegpt_ids.ids(f"blk{i}:{tag}:{rep}:{attempt}", blk_isl), 1))
            for i in range(DP)]
        await asyncio.sleep(0.15)
        prompts = [prompt_ids(f"{tag}:{rep}:{j}", isl) for j in range(total)]
        codes = await asyncio.gather(*[post(s, p, osl) for p in prompts],
                                     return_exceptions=True)
        blk_codes = await asyncio.gather(*blockers, return_exceptions=True)
        blk_ok = all(c == 200 for c in blk_codes)
        await asyncio.sleep(0.4)
        s1 = lines()
        ok = sum(1 for c in codes if c == 200)
        if ok == 0 and waits < 2:
            waits += 1
            print(f"[{tag} rep{rep}] 全部非200(路由闪断?),等待恢复后重试", flush=True)
            if not await wait_route(s):
                print(f"[{tag} rep{rep}] 路由 10 分钟未恢复", flush=True)
                break
            continue
        if isl == 1:
            clean = blk_ok and entry_clean(s0, s1, total, isl)
            mark = "lockstep" if clean else "ragged"
        else:
            clean, mark = True, "na"
        if clean or attempt == 2:
            with open(OUT, "a") as f:
                f.write(f"{tag}\t{total}\t{isl}\t{osl}\t{rep}\t{s0}\t{s1}\t{ok}\t{mark}\n")
            print(f"[{tag} rep{rep}] total={total} isl={isl}: {ok}/{total} {mark}", flush=True)
            return ok
        print(f"[{tag} rep{rep}] 进场参差,重试 {attempt+1}/3", flush=True)
        attempt += 1
        await asyncio.sleep(2)
    return 0

async def main():
    rows = list(csv.DictReader(open(PLAN)))
    conn = aiohttp.TCPConnector(limit=0)
    dead = 0
    async with aiohttp.ClientSession(connector=conn) as s:
        if not await wait_route(s):
            print("DECODE-DRIVER-ABORT: 路由 10 分钟未就绪", flush=True)
            sys.exit(7)
        for r in rows:
            reps = int(r.get("boots") or r.get("reps") or 1)
            for rep in range(reps):
                got = await pool(s, r["grp"], int(r["C"]), int(r["isl"]),
                                 int(r["osl"]), rep)
                if got == -1:
                    continue
                dead = dead + 1 if got == 0 else 0
                if dead >= 2:
                    print("DECODE-DRIVER-ABORT: 连续全灭,引擎疑似死亡", flush=True)
                    sys.exit(7)
    print("DECODE-DRIVER-DONE", flush=True)

asyncio.run(main())
