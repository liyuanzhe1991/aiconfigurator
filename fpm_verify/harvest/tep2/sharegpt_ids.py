# L3 驱动器 prompt 源:ShareGPT 奇数池(采集侧用偶数池,考卷/练习册零重叠)。
# 接口与 random.Random(seed).choices(...) 等价:ids(seed, k) -> k 个 token id,
# 同 seed 同结果(跨进程/跨 rep 可复现)。
import hashlib
import json
import os
import random

_DATASET = os.environ.get("L3_SHAREGPT_PATH") or os.path.abspath(
    os.path.join(
        os.environ.get("HF_HOME", "/tmp"), os.pardir, "fpm_datasets",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
    )
)
_TOKENIZER = os.environ.get("L3_TOKENIZER") or os.environ.get("HF_HOME")

_texts = None
_tok = None
_enc_cache: dict = {}


def _load():
    global _texts, _tok
    if _texts is None:
        data = json.loads(open(_DATASET, encoding="utf-8").read())
        pool = []
        for item in data:
            convs = item.get("conversations") or []
            body = "\n".join(
                str(t.get("value", "")) for t in convs if t.get("value")
            )
            if len(body) < 64:
                continue
            if hashlib.sha256(body.encode("utf-8", "ignore")).digest()[0] % 2 == 1:
                pool.append(body)
        if not pool:
            raise RuntimeError(f"sharegpt_ids: no odd-pool conversations in {_DATASET}")
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(_TOKENIZER, trust_remote_code=True)
        _texts = pool
    return _texts, _tok


def _encode(idx):
    ids = _enc_cache.get(idx)
    if ids is None:
        texts, tok = _load()
        ids = tok.encode(texts[idx], add_special_tokens=False)
        _enc_cache[idx] = ids
    return ids


def ids(seed, k):
    texts, _ = _load()
    rng = random.Random(str(seed))
    out = []
    while len(out) < k:
        out.extend(_encode(rng.randrange(len(texts))))
    return out[:k]
