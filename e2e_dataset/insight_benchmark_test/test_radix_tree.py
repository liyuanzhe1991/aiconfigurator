from insight_benchmark.simulation.schedule_emulator.types import FakeRequest
from insight_benchmark.simulation.schedule_emulator.kvcache_simulation import (
    RadixCache,
    RadixKey,
    ReqToTokenPoolHost,
    KVCachePool,
)


class req_factory:
    def __init__(self):
        self.req_id = 0

    def get_req(
        self, req_to_token_pool: ReqToTokenPoolHost = None, token_ids=None
    ) -> FakeRequest:
        if token_ids is None:
            token_ids = [1, 2, 3]
        ret = FakeRequest(
            id=self.req_id,
            input_token_length=len(token_ids),
            output_token_length=5,
            origin_input_ids=token_ids,
            fill_ids=token_ids,
        )
        self.req_id += 1
        req_pool_idx = req_to_token_pool.alloc(1)
        ret.req_pool_idx = req_pool_idx[0]
        return ret


def test_radix_tree_insert():
    tree = RadixCache(page_size=1, disable=False)

    # Example token id sequences (as lists of ints)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    ret = tree.insert(RadixKey(token_ids=[1, 2, 4, 5], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    tree.insert(RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    tree.pretty_print()
    print(f"{ret=}")
    print(tree.match_prefix(RadixKey(token_ids=[1, 2, 4, 5, 13, 14], extra_key=None)))


def test_radix_tree_evict():
    tree = RadixCache(page_size=1, disable=False)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 5], extra_key=None))
    tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    tree.pretty_print()

    tree.evict(3)
    print("after evict:")
    tree.pretty_print()


def test_radix_tree_with_req():
    req_pool = ReqToTokenPoolHost(size=10, max_context_len=10)
    kv_pool = KVCachePool(size=20, page_size=1)
    tree_cache = RadixCache(
        req_to_token_pool=req_pool, kv_pool=kv_pool, page_size=1, disable=False
    )
    factory = req_factory()
    req0 = factory.get_req(req_pool, [1, 2])
    req1 = factory.get_req(req_pool, [1, 2, 3, 4, 5])
    req2 = factory.get_req(req_pool, [1, 2, 3, 6])

    def req_match_prefix(req: FakeRequest):
        match_result = tree_cache.match_prefix(
            key=RadixKey(token_ids=req.origin_input_ids, extra_key=None)
        )
        prefix_indices = match_result.device_indices
        req.last_node = match_result.last_device_node
        print(f"{prefix_indices=}")
        req.prefix_indices_len = len(prefix_indices)

    # 实际推理前先match
    req_match_prefix(req0)
    # 推理获取output_ids
    req0.output_ids = [3, 7, 8]
    # 推理完成插入
    tree_cache.cache_finished_req(req0)

    # 新请求匹配
    req_match_prefix(req1)
    req_match_prefix(req2)

    tree_cache.pretty_print()


if __name__ == "__main__":
    test_radix_tree_insert()
    test_radix_tree_evict()
    test_radix_tree_with_req()
