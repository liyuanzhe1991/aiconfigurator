from typing import List

from insight_benchmark.simulation.schedule_emulator.types import FakeRequest
from insight_benchmark.simulation.schedule_emulator.kvcache_simulation import (
    SimHiRadixCache,
    RadixKey,
    ReqToTokenPoolHost,
    KVCachePool,
)


class req_factory:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPoolHost = None,
        mem_pool: KVCachePool = None,
        tree_cache: SimHiRadixCache = None,
    ):
        self.req_id = 0

        self.req_to_token_pool = req_to_token_pool
        self.mem_pool = mem_pool
        self.tree_cache = tree_cache

    def prefix_loadback(
        self,
        req: FakeRequest,
    ):
        loading_len, last_node = self.tree_cache.init_load_back(req.last_host_node)
        req.prefix_indices_len = req.prefix_indices_len + loading_len
        req.last_node = last_node
        req.last_matched_prefix_len = req.prefix_indices_len
        return None

    def get_req(
        self,
        # req_to_token_pool: ReqToTokenPoolHost = None,
        # mem_pool: KVCachePool = None,
        # tree_cache: SimHiRadixCache = None,
        token_ids=None,
    ) -> FakeRequest:
        if token_ids is None:
            token_ids = [1, 2, 3]
        req_len = len(token_ids)
        req = FakeRequest(
            id=self.req_id,
            input_token_length=len(token_ids),
            output_token_length=5,
            origin_input_ids=token_ids,
            fill_ids=token_ids,
        )
        self.req_id += 1
        # 前缀匹配
        if not self.tree_cache:
            return req
        match_result = self.tree_cache.match_prefix(
            key=RadixKey(token_ids=req.origin_input_ids, extra_key=None)
        )
        req.prefix_indices_len = len(match_result.device_indices)
        req.last_node = match_result.last_device_node
        req.last_host_node = match_result.last_host_node
        req.host_hit_len = match_result.host_hit_length
        req.last_matched_prefix_len = len(match_result.device_indices)
        if req.host_hit_len > 0:
            self.prefix_loadback(req)

        new_token_len = req_len - req.prefix_indices_len

        # 一级缓存分配，对应alloc_req_slots
        req_pool_idx = self.req_to_token_pool.alloc(1)
        req.req_pool_idx = req_pool_idx[0]
        self.req_to_token_pool.write(req.req_pool_idx, req_len)

        # 二级内存分配，对应alloc_token_slots/alloc_paged_token_slots_extend
        if self.mem_pool is not None:
            # tree_cache.pretty_print()
            # 对应_evict_tree_cache_if_needed，判断kv内存池容量是否足够
            if self.mem_pool.available_size < new_token_len:
                print(
                    f"prefill stage: try to evict {new_token_len}, {self.mem_pool.available_size=}, {self.mem_pool.evictable_size=}"
                )
                self.tree_cache.evict(new_token_len)
                if self.mem_pool.available_size < new_token_len:
                    print("prefill stage: OOM Error!! No enough space in mem pool")
                    self.tree_cache.pretty_print()
                    return None
            print(
                f"prefill stage: alloc {new_token_len}, {self.mem_pool.available_size=}, {self.mem_pool.evictable_size=}"
            )
            self.mem_pool.alloc(new_token_len)

        # 模拟prefill推理，需要将req相关节点上锁，避免释放
        # 在cache_finished_req中释放锁
        self.tree_cache.inc_lock_ref(req.last_node)
        return req


def decode_stage(
    req: FakeRequest,
    new_token_ids: List[int],
    tree_cache: SimHiRadixCache,
    mem_pool: KVCachePool,
):
    if not tree_cache or not mem_pool:
        req.output_ids = new_token_ids
        return req

    # 对应alloc_token_slots
    num_tokens = len(new_token_ids)
    if mem_pool.available_size < num_tokens:
        print(
            f"decode stage: try to evict {num_tokens}, {mem_pool.available_size=}, {mem_pool.evictable_size=}"
        )
        tree_cache.evict(num_tokens)
        if mem_pool.available_size < num_tokens:
            print("decode stage: OOM Error!! no enough space in mem pool")
            tree_cache.pretty_print()
            return req
    print(
        f"decode stage: alloc {num_tokens}, {mem_pool.available_size=}, {mem_pool.evictable_size=}"
    )
    mem_pool.alloc(num_tokens)

    if req.output_ids is None:
        req.output_ids = new_token_ids
    else:
        req.output_ids.extend(new_token_ids)
    print(f"{req.output_ids=}")
    return req


def process_finished_req(
    req: FakeRequest,
    tree_cache: SimHiRadixCache,
    mem_pool: KVCachePool,
):
    # 处理最后一个token，即eos token需要单独释放，当前默认page_size=1
    # TODO：需要通过page_size来判断是否手动释放
    mem_pool.free(1)
    req.fill_ids = req.origin_input_ids + req.output_ids
    # 存入前缀树中
    tree_cache.cache_finished_req(req)
    # 打印前缀树状况
    tree_cache.pretty_print()


def test_radix_tree_base():
    req_pool = ReqToTokenPoolHost(size=20, max_context_len=10)
    kv_pool = KVCachePool(size=20, page_size=1)
    tree = SimHiRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_pool,
        page_size=1,
        hicache_size=50,
        hicache_write_policy="write_through",
        eviction_policy="lru",
        hicache_storage_backend=None,
        hicache_storage_prefetch_policy="best_effort",
        storage_backend_extra_config=(256, 1, 0.25),
        is_eagle=False,
    )
    factory = req_factory(req_pool, kv_pool, tree)
    req0 = factory.get_req([1, 2])
    # 推理获取output_ids
    output_ids = [3, 7, 100]
    tree.pretty_print()
    decode_stage(req0, output_ids, tree, kv_pool)
    # 推理完成插入
    tree.cache_finished_req(req0)
    tree.pretty_print()

    req1 = factory.get_req([1, 2, 3, 4, 5])
    output_ids = [6, 7, 100]
    tree.pretty_print()
    decode_stage(req1, output_ids, tree, kv_pool)
    tree.cache_finished_req(req1)
    tree.pretty_print()

    req2 = factory.get_req([1, 2, 8, 7, 5])
    output_ids = [3, 3, 4, 100]
    tree.pretty_print()
    decode_stage(req2, output_ids, tree, kv_pool)
    tree.cache_finished_req(req2)
    tree.pretty_print()


def test_radix_tree_evict():
    req_pool = ReqToTokenPoolHost(size=10, max_context_len=10)
    kv_pool = KVCachePool(size=13, page_size=1)
    tree = SimHiRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_pool,
        page_size=1,
        hicache_size=50,
        hicache_write_policy="write_through",
        eviction_policy="lru",
        hicache_storage_backend=None,
        hicache_storage_prefetch_policy="best_effort",
        storage_backend_extra_config=(256, 1, 0.25),
        is_eagle=False,
    )
    factory = req_factory(req_pool, kv_pool, tree)

    req0 = factory.get_req([1, 2])
    # 推理获取output_ids
    output_ids = [3, 7, 100]
    # tree.pretty_print()
    decode_stage(req0, output_ids, tree, kv_pool)
    # 推理完成的处理
    process_finished_req(req0, tree, kv_pool)

    req1 = factory.get_req([1, 2, 3, 4, 5])
    output_ids = [6, 7, 100]
    decode_stage(req1, output_ids, tree, kv_pool)
    process_finished_req(req1, tree, kv_pool)

    req2 = factory.get_req([1, 2, 3, 4])
    output_ids = [9, 10, 100]
    decode_stage(req2, output_ids, tree, kv_pool)
    process_finished_req(req2, tree, kv_pool)

    req3 = factory.get_req([4, 6, 2, 7, 8, 9, 10, 11, 12])
    output_ids = [13, 14, 100]
    decode_stage(req3, output_ids, tree, kv_pool)
    process_finished_req(req3, tree, kv_pool)


def test_radix_tree_loadback():
    req_pool = ReqToTokenPoolHost(size=10, max_context_len=10)
    kv_pool = KVCachePool(size=15, page_size=1)
    tree = SimHiRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_pool,
        page_size=1,
        hicache_size=50,
        hicache_write_policy="write_through",
        eviction_policy="lru",
        hicache_storage_backend=None,
        hicache_storage_prefetch_policy="best_effort",
        storage_backend_extra_config=(256, 1, 0.25),
        is_eagle=False,
    )
    factory = req_factory(req_pool, kv_pool, tree)

    req0 = factory.get_req([1, 2])
    print(
        f"req0 prefix_indices_len is {req0.prefix_indices_len}, host_hit_len is {req0.host_hit_len}"
    )
    # 推理获取output_ids
    output_ids = [3, 7, 100]
    # tree.pretty_print()
    decode_stage(req0, output_ids, tree, kv_pool)
    # 推理完成的处理
    process_finished_req(req0, tree, kv_pool)

    req1 = factory.get_req([1, 2, 3, 4, 5, 6, 7, 8])
    print(
        f"req1 prefix_indices_len is {req1.prefix_indices_len}, host_hit_len is {req1.host_hit_len}"
    )
    output_ids = [9, 10, 100]
    decode_stage(req1, output_ids, tree, kv_pool)
    process_finished_req(req1, tree, kv_pool)

    req2 = factory.get_req([1, 2, 3, 4])
    print(
        f"req2 prefix_indices_len is {req2.prefix_indices_len}, host_hit_len is {req2.host_hit_len}"
    )
    output_ids = [11, 12, 100]
    decode_stage(req2, output_ids, tree, kv_pool)
    process_finished_req(req2, tree, kv_pool)

    req3 = factory.get_req([4, 4, 4, 4, 4, 4, 4, 4, 4])
    print(
        f"req3 prefix_indices_len is {req3.prefix_indices_len}, host_hit_len is {req3.host_hit_len}"
    )
    output_ids = [4, 4, 4, 100]
    decode_stage(req3, output_ids, tree, kv_pool)
    process_finished_req(req3, tree, kv_pool)

    req4 = factory.get_req([1, 2, 3, 4, 5, 6])
    print(
        f"req4 prefix_indices_len is {req4.prefix_indices_len}, host_hit_len is {req4.host_hit_len}"
    )
    output_ids = [13, 14, 100]
    decode_stage(req4, output_ids, tree, kv_pool)
    process_finished_req(req4, tree, kv_pool)


def test_radix_tree_evict_host():
    req_pool = ReqToTokenPoolHost(size=10, max_context_len=10)
    kv_pool = KVCachePool(size=15, page_size=1)
    tree = SimHiRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_pool,
        page_size=1,
        hicache_size=30,
        hicache_write_policy="write_through",
        eviction_policy="lru",
        hicache_storage_backend=None,
        hicache_storage_prefetch_policy="best_effort",
        storage_backend_extra_config=(256, 1, 0.25),
        is_eagle=False,
    )
    factory = req_factory(req_pool, kv_pool, tree)

    req0 = factory.get_req([1, 2])
    print(
        f"req0 prefix_indices_len is {req0.prefix_indices_len}, host_hit_len is {req0.host_hit_len}"
    )
    # 推理获取output_ids
    output_ids = [3, 7, 100]
    # tree.pretty_print()
    decode_stage(req0, output_ids, tree, kv_pool)
    # 推理完成的处理
    process_finished_req(req0, tree, kv_pool)

    req1 = factory.get_req([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(
        f"req1 prefix_indices_len is {req1.prefix_indices_len}, host_hit_len is {req1.host_hit_len}"
    )
    output_ids = [11, 12, 13, 14, 100]
    decode_stage(req1, output_ids, tree, kv_pool)
    process_finished_req(req1, tree, kv_pool)

    req2 = factory.get_req([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    print(
        f"req2 prefix_indices_len is {req2.prefix_indices_len}, host_hit_len is {req2.host_hit_len}"
    )
    output_ids = [26, 27, 100]
    decode_stage(req2, output_ids, tree, kv_pool)
    process_finished_req(req2, tree, kv_pool)

    req3 = factory.get_req([4, 4, 4, 4, 4, 4, 4, 4, 4])
    print(
        f"req3 prefix_indices_len is {req3.prefix_indices_len}, host_hit_len is {req3.host_hit_len}"
    )
    output_ids = [4, 4, 4, 100]
    decode_stage(req3, output_ids, tree, kv_pool)
    process_finished_req(req3, tree, kv_pool)


if __name__ == "__main__":
    test_radix_tree_base()
    test_radix_tree_evict()
    test_radix_tree_loadback()
    test_radix_tree_evict_host()
