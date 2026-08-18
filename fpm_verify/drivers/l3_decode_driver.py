# L3 v2 decode 驱动:按格点计划跑池;isl 灌注跳位,ShareGPT 奇数池,
# 无连接上限,窗口按流行数记账。用法: driver.py plan.csv stream.jsonl windows.tsv
import asyncio
import csv
import os
import sys

import aiohttp

sys.path.insert(0, "/tmp/fpm-kvwarm")
import sharegpt_ids

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = os.environ.get("L3_MODEL_ID", "MiniMaxAI/MiniMax-M2.7")
PLAN, STREAM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DP = int(sys.argv[4]) if len(sys.argv) > 4 else 1

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
    """锁步自检:窗内所有"每请求恰 isl 个 past-kv"的纯 decode 步(即各调度器
    的首步,kv 均匀 == isl)所含请求数合计 == C。对 dep 多 rank 拆分与流交错
    稳健:参差进场时早入请求已走到 isl+k,与晚入者混合的步 sum != num*isl,
    合计凑不满 C;而每 rank 的首步无论先后都恰被计入一次。"""
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
    for attempt in range(3):
        s0 = lines()
        # 遮蔽窗按池规模伸缩:大池的 HTTP 到达散布更长,拦路石要占住引擎更久。
        # DP>1 时发 DP 个拦路石(不同 seed):单石只会被路由到一个 rank,
        # 其余 rank 不设防必然参差;多石靠路由均摊逐 rank 遮蔽,偶发同 rank
        # 堆叠由 entry_clean 判 ragged 后换 seed 重试兜住。
        blk_isl = max(8192, min(65536, C * 32))
        blockers = [asyncio.create_task(
            post(s, sharegpt_ids.ids(f"blk{i}:{tag}:{rep}:{attempt}", blk_isl), 1))
            for i in range(DP)]
        await asyncio.sleep(0.15)
        prompts = [sharegpt_ids.ids(f"{tag}:{rep}:{j}", isl) for j in range(C)]
        # return_exceptions:单请求瞬断只记失败,不炸穿整个计划(无人值守必需)
        codes = await asyncio.gather(*[post(s, p, osl) for p in prompts],
                                     return_exceptions=True)
        blk_codes = await asyncio.gather(*blockers, return_exceptions=True)
        blk_ok = all(c == 200 for c in blk_codes)
        if not blk_ok:
            print(f"[{tag} rep{rep}] 拦路石异常: {blk_codes}", flush=True)
        await asyncio.sleep(0.4)
        s1 = lines()
        ok = sum(1 for c in codes if c == 200)
        # 仅浅池(isl==1)强校验;isl>1 的进场可能被 token 预算切多拍,不核验不背书。
        # 拦路石非 200 时遮蔽失效,浅池不得记 lockstep
        if isl == 1:
            clean = blk_ok and entry_clean(s0, s1, C, isl)
            mark = "lockstep" if clean else "ragged"
        else:
            clean, mark = True, "na"
        if clean or attempt == 2:
            with open(OUT, "a") as f:
                f.write(f"{tag}\t{C}\t{isl}\t{osl}\t{rep}\t{s0}\t{s1}\t{ok}\t{mark}\n")
            print(f"[{tag} rep{rep}] C={C} isl={isl}: {ok}/{C} {mark}"
                  f"{'' if clean else '(3次未锁步,如实记档)'}", flush=True)
            return ok
        print(f"[{tag} rep{rep}] 进场参差,重试 {attempt+1}/3", flush=True)
    return 0

async def main():
    rows = list(csv.DictReader(open(PLAN)))
    conn = aiohttp.TCPConnector(limit=0)
    dead = 0  # 熔断:连续全灭说明引擎已死,快速失败而非烧完计划假装 DONE
    async with aiohttp.ClientSession(connector=conn) as s:
        for r in rows:
            for rep in range(int(r["reps"])):
                got = await pool(s, r["tag"], int(r["C"]), int(r["isl"]),
                                 int(r["osl"]), rep)
                dead = dead + 1 if got == 0 else 0
                if dead >= 2:
                    print("L3V2-DRIVER-ABORT: 连续全灭,引擎疑似死亡", flush=True)
                    sys.exit(7)
    print("L3V2-DECODE-DONE", flush=True)

asyncio.run(main())
