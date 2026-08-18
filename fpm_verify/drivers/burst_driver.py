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
import os
DP_MODE = os.environ.get("L3_DP_MODE") == "1"
DP_SIZE = int(os.environ.get("L3_DP_SIZE", "4"))

PLAN = sys.argv[1]                       # prefill_plan.csv
STREAM = sys.argv[2]                     # /results/fpm_stream.jsonl
OUT = sys.argv[3]                        # /results/burst_windows.tsv
URL = "http://127.0.0.1:8000/v1/completions"
MODEL = os.environ.get("L3_MODEL_ID", "MiniMaxAI/MiniMax-M2.7")
VOCAB_LO, VOCAB_HI = 1, 199000
BUDGET = 8192

def stream_lines():
    try:
        with open(STREAM) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def stream_has(start, bp, tok_total, kv_total, end=None):
    with open(STREAM) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            try:
                s = json.loads(line)["scheduled_requests"]
            except Exception:
                continue
            if (s["num_prefill_requests"] == bp and s["sum_prefill_tokens"] == tok_total
                    and s["sum_prefill_kv_tokens"] == kv_total):
                return True
    return False

def dp_balanced(start, end, bp, n, kv):
    """DP 均衡校验(镜像打分门):窗 [start,end) 内 DP_SIZE 个不同 rank 各出现
    "每 rank 恰 bp/DP 份"的同形纯 prefill 步,且四者 wall 差 ≤3%(同拍证据,
    与 score_r13_burst 的投票门一致)。不达标判 False 换 seed 重试,
    把废票挡在采集时而非打分时。"""
    tgt_bp = bp // DP_SIZE
    tgt = (tgt_bp, tgt_bp * n, tgt_bp * kv)
    walls = {}
    with open(STREAM) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            try:
                d = json.loads(line)
                s = d["scheduled_requests"]
            except Exception:
                continue
            if (s["num_decode_requests"] == 0 and s["num_prefill_requests"] >= 1
                    and (s["num_prefill_requests"], s["sum_prefill_tokens"],
                         s["sum_prefill_kv_tokens"]) == tgt
                    and d.get("dp_rank", 0) not in walls):
                walls[d.get("dp_rank", 0)] = d["wall_time"]
                if len(walls) >= DP_SIZE:
                    v = list(walls.values())
                    return max(v) <= min(v) * 1.03
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

async def fire(session, bp, n, kv, salt):
    pf = prefix_ids(kv) if kv else []
    prompts = [pf + sharegpt_ids.ids((salt * 1000 + i), n)
               for i in range(bp)]
    if DP_MODE:
        # DP 拦路石 ×DP_SIZE(与 decode 锁步同源):把全部 rank 占住,
        # burst 份额在拦路石释放后同一拍收编——单石只遮一个 rank,
        # 其余 rank 参差入场,四 rank 同形步 wall 双峰(r13 实测 1.03 门 2% 通过)
        blockers = [asyncio.create_task(
            post(session, sharegpt_ids.ids((salt * 7 + i + 1), 8192)))
            for i in range(DP_SIZE)]
        await asyncio.sleep(0.1)
        res = await asyncio.gather(*[post(session, p) for p in prompts])
        await asyncio.gather(*blockers, return_exceptions=True)
        return res
    if bp >= 2:  # 单调度器:一颗拦路石挡一步,burst 全员到齐后整批入场
        blocker_task = asyncio.create_task(post(session, sharegpt_ids.ids((salt), 8192)))
        await asyncio.sleep(0.05)
        res = await asyncio.gather(*[post(session, p) for p in prompts])
        await blocker_task
        return res
    return await asyncio.gather(*[post(session, p) for p in prompts])

async def main():
    rows = list(csv.DictReader(open(PLAN)))
    if DP_MODE:
        # ×4 均衡计划契约:旧 ×1 计划混入时 tgt_bp=0 会被 idle 步伪命中,fail-loud
        bad = [r for r in rows if int(r["bp"]) % DP_SIZE or int(r["bp"]) < DP_SIZE]
        assert not bad, f"DP_MODE 计划须 bp%{DP_SIZE}==0 且 ≥{DP_SIZE}: {bad[:3]}"
    done = {}
    try:  # 断点续跑:已有 ≥reps 条窗记录的格直接跳过(pod 被杀后重启不重采)
        for line in open(OUT):
            f = line.split("\t")
            k = (f[0], int(f[1]), int(f[2]), int(f[3]))
            done[k] = done.get(k, 0) + 1
    except FileNotFoundError:
        pass
    out = open(OUT, "a")
    t_start = time.time()
    # limit=0:×4 计划最大 bp=256 + DP 拦路石,默认 100 连接上限会切碎 burst
    conn = aiohttp.TCPConnector(limit=0)
    short_streak = 0
    async with aiohttp.ClientSession(connector=conn) as session:
        for ci, r in enumerate(rows):
            grp, bp, n, kv = r["grp"], int(r["bp"]), int(r["n"]), int(r["kv"])
            reps = int(r.get("repeats", 5) or 5)
            if done.get((grp, bp, n, kv), 0) >= reps:
                print(f"[{ci+1}/{len(rows)}] {grp} bp{bp} n{n} kv{kv}: SKIP(resume)", flush=True)
                continue
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
                    ok = (dp_balanced(s0, s1, bp, n, kv) if DP_MODE
                          else stream_has(s0, bp, bp * n, bp * kv, s1))
                if ok:
                    got += 1
                    out.write(f"{grp}\t{bp}\t{n}\t{kv}\t{got}\t{s0}\t{s1}\n")
                    out.flush()
            status = "OK" if got >= reps else f"SHORT({got}/{reps})"
            # 熔断只数全灭格(got==0)。连续全灭有两种成因,须体检分流:
            #   引擎死 → 熔断;引擎活 → 结构性不可行格(大 bp 突发入队散布
            #   超过单步时长,r13/r14 复现同一批格恒 0/5),如实记 SHORT 继续
            short_streak = short_streak + 1 if got == 0 else 0
            if short_streak >= 5:
                try:
                    async with session.get(
                            "http://127.0.0.1:8000/v1/models",
                            timeout=aiohttp.ClientTimeout(total=10)) as hr:
                        alive = hr.status == 200
                except Exception:
                    alive = False
                if alive:
                    print("连续全灭但引擎健康:判结构性不可行区,继续", flush=True)
                    short_streak = 0
                else:
                    print("BURST-DRIVER-ABORT: 连续全灭且引擎失联", flush=True)
                    sys.exit(7)
            print(f"[{ci+1}/{len(rows)}] {grp} bp{bp} n{n} kv{kv}: {status} "
                  f"({attempts} attempts, {time.time()-t_start:.0f}s elapsed)", flush=True)
    print(f"BURST-DRIVER-DONE in {time.time()-t_start:.0f}s", flush=True)

asyncio.run(main())
