# KV 预热补丁(kvwarm)— decode 采集从伪造 KV 改为真实内容 + 倒带测量。
# 设计:FIX_DESIGN_DECODE_SWEEP.md §4.5;基底:gc-steady-randtok2-20260812
# 提取件(md5 a338ec5f62f2e5126931a809dd8e136f)。
#
# 保证(与用户约定):
#  1. 只动 self-benchmark 代码(_bench_*/新增 _kvwarm_*);生产路径零改动;
#  2. DYN_BENCH_KV_WARMUP=off 时行为与原文件逐字节等价(所有手术都是
#     flag 门控的新增分支;off 路径无行为 diff);默认 on;
#  3. large-kv 修复并入:巨点(总 kv≥阈值)稳态步重复取中位;可行性块预算
#     在 flag on 时 +1 块余量(修 §6.1 的 ctx+2 写越界预算缺口);
#  4. dense / moe_tp 拓扑自动跳过预热(物理免疫),元数据记账;
#  5. 零 hardcode:链数/深度/日程全部由网格+池导出;数据集 URL/sha/阈值
#     为 env 可覆盖的配置常量。
import ast
import os
import sys

P = os.environ.get(
    "SCHED_PATH",
    "/usr/local/lib/python3.12/dist-packages/dynamo/vllm/instrumented_scheduler.py",
)
src = open(P).read()
if "DYN_BENCH_KV_WARMUP" in src:
    print("kvwarm: already patched")
    sys.exit(0)

# ---------------------------------------------------------------------------
# S1 新方法块:插在 _bench_make_steady_step 之前(类体内)
# ---------------------------------------------------------------------------
METHODS = '''
    # ------------------------------------------------------------------
    # KVWARM — 真实内容 KV 预热(设计:FIX_DESIGN_DECODE_SWEEP.md §4.5)
    # 机制:每个 decode 批量档一群互不相同的真实文本链(ShareGPT 偶数池,
    # chunked prefill 建成,建成后停车驻留);测量点 (B, kv) 用影子请求
    # 1:1 借用链的块表(只读、零分配、免管理器),走原两拍测量。
    # 链住独立注册表(_kvwarm_chain_ids),对现有 benchmark 簿记零干涉。
    # ------------------------------------------------------------------

    _KVWARM_DEFAULT_DATASET_URL = (
        "https://huggingface.co/datasets/anon8231489123/"
        "ShareGPT_Vicuna_unfiltered/resolve/main/"
        "ShareGPT_V3_unfiltered_cleaned_split.json"
    )

    def _kvwarm_flag_on(self) -> bool:
        return os.environ.get("DYN_BENCH_KV_WARMUP", "on").lower() not in (
            "off", "0", "false",
        )

    def _kvwarm_giant_threshold(self) -> int:
        return int(os.environ.get("DYN_BENCH_GIANT_KV_THRESHOLD", "1000000"))

    def _kvwarm_giant_repeats(self) -> int:
        return max(1, int(os.environ.get("DYN_BENCH_GIANT_KV_REPEATS", "3")))

    def _kvwarm_meta_init(self) -> dict:
        meta = getattr(self, "_kvwarm_meta", None)
        if meta is None:
            meta = {
                "enabled": self._kvwarm_flag_on(),
                "warm_eligible": None,
                "skip_reason": None,
                "dataset": None,
                "stages": [],
                "points_real_kv": 0,
                "points_fake_fallback": 0,
                "giant_kv_threshold": self._kvwarm_giant_threshold(),
                "giant_kv_repeats": self._kvwarm_giant_repeats(),
            }
            self._kvwarm_meta = meta
        return meta

    def _kvwarm_warm_eligible(self) -> bool:
        """预热只对 EP 切分的 MoE 有一阶意义;dense/moe_tp 物理免疫,跳过。"""
        cached = getattr(self, "_kvwarm_eligible_cache", None)
        if cached is not None:
            return cached
        meta = self._kvwarm_meta_init()
        eligible = False
        reason = None
        if not self._kvwarm_flag_on():
            reason = "flag_off"
        else:
            parallel = getattr(self.vllm_config, "parallel_config", None)
            model = getattr(self.vllm_config, "model_config", None)
            hf = getattr(model, "hf_config", None)
            hf_text = getattr(model, "hf_text_config", hf)
            has_experts = any(
                bool(getattr(cfg, key, 0))
                for cfg in (hf, hf_text)
                if cfg is not None
                for key in (
                    "num_local_experts",
                    "num_experts",
                    "n_routed_experts",
                    "moe_num_experts",
                )
            )
            ep_enabled = bool(getattr(parallel, "enable_expert_parallel", False))
            prefix_on = bool(
                getattr(self.cache_config, "enable_prefix_caching", False)
            )
            if not has_experts:
                reason = "dense_model_content_insensitive"
            elif not ep_enabled:
                reason = "moe_tp_balanced_by_construction"
            elif not prefix_on:
                # 103 个批量档靠 prefix cache 换代增量续深(~16M token);
                # 关着时整链重刷要 ~230M token,不可接受,宁跳过。
                reason = "prefix_caching_disabled"
            else:
                eligible = True
        meta["warm_eligible"] = eligible
        meta["skip_reason"] = reason
        self._kvwarm_eligible_cache = eligible
        if not eligible:
            logger.info("KVWARM: warm-up skipped (%s)", reason)
        return eligible

    # ---------------- 数据集:三级解析 + 偶数池 + 懒 tokenize ----------------

    def _kvwarm_resolve_dataset(self) -> str:
        spec = os.environ.get(
            "DYN_BENCH_KV_WARMUP_DATASET", self._KVWARM_DEFAULT_DATASET_URL
        )
        if not spec.startswith(("http://", "https://")):
            if not os.path.exists(spec):
                raise RuntimeError(f"KVWARM dataset path does not exist: {spec}")
            return spec
        cache_root = os.environ.get("DYN_BENCH_KV_WARMUP_CACHE_DIR") or (
            os.path.join(os.environ["HF_HOME"], os.pardir, "fpm_datasets")
            if os.environ.get("HF_HOME")
            else "/tmp/fpm_datasets"
        )
        cache_root = os.path.abspath(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        name = os.path.basename(spec.split("?")[0]) or "kvwarm_dataset.json"
        cached = os.path.join(cache_root, name)
        expected_sha = os.environ.get("DYN_BENCH_KV_WARMUP_SHA256")
        if os.path.exists(cached):
            digest = self._kvwarm_sha256(cached)
            if expected_sha and digest != expected_sha:
                raise RuntimeError(
                    f"KVWARM dataset cache sha mismatch: {digest} != {expected_sha}"
                )
            return cached
        import urllib.request

        logger.info("KVWARM: downloading dataset %s -> %s", spec, cached)
        tmp = cached + ".part"
        urllib.request.urlretrieve(spec, tmp)
        digest = self._kvwarm_sha256(tmp)
        if expected_sha and digest != expected_sha:
            os.unlink(tmp)
            raise RuntimeError(
                f"KVWARM dataset download sha mismatch: {digest} != {expected_sha}"
            )
        os.replace(tmp, cached)
        logger.info("KVWARM: dataset cached (sha256=%s)", digest)
        return cached

    @staticmethod
    def _kvwarm_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _kvwarm_load_texts(self) -> list:
        """ShareGPT 对话 → 文本;按对话 hash 偶数半给采集(奇数半留 L3)。"""
        texts = getattr(self, "_kvwarm_texts", None)
        if texts is not None:
            return texts
        path = self._kvwarm_resolve_dataset()
        data = json.loads(open(path, encoding="utf-8").read())
        texts = []
        for item in data:
            convs = item.get("conversations") or []
            body = "\\n".join(
                str(turn.get("value", "")) for turn in convs if turn.get("value")
            )
            if len(body) < 64:
                continue
            digest = hashlib.sha256(body.encode("utf-8", "ignore")).digest()
            if digest[0] % 2 == 0:  # 偶数池 = 采集;奇数池 = L3
                texts.append(body)
        if not texts:
            raise RuntimeError("KVWARM: dataset yielded no usable conversations")
        meta = self._kvwarm_meta_init()
        meta["dataset"] = {
            "path": path,
            "sha256": self._kvwarm_sha256(path),
            "collection_pool": "conversation_sha256_even",
            "conversations": len(texts),
        }
        self._kvwarm_texts = texts
        return texts

    def _kvwarm_tokenizer(self):
        tok = getattr(self, "_kvwarm_tok", None)
        if tok is None:
            from transformers import AutoTokenizer

            model = self.vllm_config.model_config
            tok = AutoTokenizer.from_pretrained(
                model.tokenizer,
                trust_remote_code=bool(getattr(model, "trust_remote_code", False)),
            )
            self._kvwarm_tok = tok
        return tok

    def _kvwarm_chain_token_ids(self, chain_index: int, depth: int) -> list:
        """确定性拼链:种子 = (网格digest, dp_rank, chain);对话级洗牌打包。
        逐链缓存单调增长——同链跨换代只延伸不重算(prefix cache 命中的前提
        也是同链前缀逐字节稳定)。"""
        cache = getattr(self, "_kvwarm_token_cache", None)
        if cache is None:
            cache = {}
            self._kvwarm_token_cache = cache
        tokens, cursor, order = cache.get(chain_index, ([], 0, None))
        if order is None:
            texts = self._kvwarm_load_texts()
            seed = f"{self._bench_grid_digest}:{self._fpm_dp_rank}:{chain_index}"
            rng = __import__("random").Random(seed)
            order = list(range(len(texts)))
            rng.shuffle(order)
        if len(tokens) < depth:
            texts = self._kvwarm_load_texts()
            tok = self._kvwarm_tokenizer()
            while len(tokens) < depth and cursor < len(order):
                tokens.extend(
                    tok.encode(texts[order[cursor]], add_special_tokens=False)
                )
                cursor += 1
            if len(tokens) < depth:
                raise RuntimeError(
                    f"KVWARM: dataset too small for chain depth {depth} "
                    f"(got {len(tokens)} tokens)"
                )
        cache[chain_index] = (tokens, cursor, order)
        return tokens[:depth]

    # ---------------- 阶梯计划:全部由网格 + 池导出 ----------------

    def _kvwarm_prepare(self, mode: str) -> None:
        if mode not in ("decode", "agg"):
            return
        if not self._kvwarm_flag_on():
            return
        self._kvwarm_meta_init()
        if not self._kvwarm_warm_eligible():
            return
        # 点位重排:decode 段按 (批量降序, kv 降序)。执行顺序与 benchmark_id
        # 顺序解耦是既定契约(见网格编号处注释),重排合规。
        points = list(self._bench_grid)
        decode_pts = [p for p in points if p.point_type == "decode"]
        other_pts = [p for p in points if p.point_type != "decode"]
        decode_pts.sort(
            key=lambda p: (-p.batch_size, -p.total_kv_read_tokens)
        )
        self._bench_grid = deque(other_pts + decode_pts)
        # 每批量档的预热深度 = 该档最深可预热点的 max(ctx) + 1 + 稳态写余量
        # (余量 = 巨点重复步数,保证多拍写不越链块);受池可行性封顶。
        giant_thr = self._kvwarm_giant_threshold()
        repeats = self._kvwarm_giant_repeats()
        plan: dict = {}
        for p in decode_pts:
            ctxs = self._bench_decode_context_lengths(
                p.total_kv_read_tokens, p.batch_size
            )
            margin = 1 + (repeats if p.total_kv_read_tokens >= giant_thr else 1)
            # 上限 -4:prompt=max_len-1 的链在 prefill 完成步即被长度停止
            # 回收(完成步自带首个采样 token);留漂移余量。
            want = min(max(ctxs) + margin, self.max_model_len - 4)
            plan[p.batch_size] = max(plan.get(p.batch_size, 0), want)
        for batch, depth in list(plan.items()):
            while depth > 8 and (
                self._bench_blocks_per_req(depth) * batch
                > self._bench_usable_blocks(batch, reserve_watermark=True)
            ):
                depth -= 1
            plan[batch] = depth
        self._kvwarm_plan = plan
        # 二次重排:warmed 全体在前、fake 回退殿后——fake 注入会写满整池、
        # 逐出链的缓存块,插在换代之间会把增量续深打回全量重刷(实测 57s/段)。
        warmed_pts = [p for p in decode_pts if self._kvwarm_plan_covers(p)]
        fake_pts = [p for p in decode_pts if not self._kvwarm_plan_covers(p)]
        self._bench_grid = deque(other_pts + warmed_pts + fake_pts)
        self._kvwarm_chain_ids: list = []
        self._kvwarm_chain_prompts: dict = {}
        self._kvwarm_borrowed_ids: set = set()
        self._kvwarm_stage_batch = None
        self._kvwarm_building = False
        self._kvwarm_seq = 0
        logger.info(
            "KVWARM: prepared %d stage plans over %d decode points",
            len(plan),
            len(decode_pts),
        )

    # ---------------- 预热状态机(相位分发前拦截) ----------------

    def _kvwarm_point_need(self, point) -> int:
        """链深须 ≥ max(injected)+need。非巨点写透 injected+1(admission 写
        ctx-1、steady 写 ctx);巨点多拍再 +repeats-1。"""
        giant = point.total_kv_read_tokens >= self._kvwarm_giant_threshold()
        return 2 + (self._kvwarm_giant_repeats() - 1 if giant else 0)

    def _kvwarm_plan_covers(self, point) -> bool:
        """计划层覆盖判定(不依赖活链):建链决策与注入分派的共同事实源。"""
        plan = getattr(self, "_kvwarm_plan", None)
        if not plan:
            return False
        depth = plan.get(point.batch_size, 0)
        if not depth:
            return False
        ctxs = self._bench_decode_context_lengths(
            point.total_kv_read_tokens, point.batch_size
        )
        need = self._kvwarm_point_need(point)
        return max(max(1, c - 1) for c in ctxs) + need <= depth

    def _kvwarm_covers(self, point, injected_lengths) -> bool:
        """注入层覆盖判定 = 计划覆盖 + 活链就绪(数量与逐链深度)。"""
        if not self._kvwarm_plan_covers(point):
            return False
        if self._kvwarm_building or self._kvwarm_stage_batch is None:
            return False
        chains = self._kvwarm_chain_ids
        if len(chains) < point.batch_size:
            return False
        need = self._kvwarm_point_need(point)
        return all(
            max(1, injected) + need
            <= len(self._kvwarm_chain_prompts[chains[i]])
            for i, injected in enumerate(injected_lengths)
        )

    def _kvwarm_step_busy(self) -> bool:
        """DECODE_SWEEP 相位:链群建设/停车/换代。True = 本步交还真调度器。"""
        if not getattr(self, "_kvwarm_plan", None):
            return False
        if self._bench_active_req_ids or self._bench_current_point is not None:
            return False
        grid = self._bench_grid
        nxt = grid[0] if grid and grid[0].point_type == "decode" else None
        if nxt is None:
            self._kvwarm_shed_chains()
            return False
        if not self._kvwarm_plan_covers(nxt):
            # fake 回退点需要整池:先甩链还池,再放行 fake 注入。
            if self._kvwarm_chain_ids:
                self._kvwarm_shed_chains()
            return False
        if self._kvwarm_building:
            return self._kvwarm_monitor_build()
        if self._kvwarm_stage_batch != nxt.batch_size:
            self._kvwarm_shed_chains()
            self._kvwarm_start_stage(
                nxt.batch_size, self._kvwarm_plan[nxt.batch_size]
            )
            return True
        return False

    def _kvwarm_start_stage(self, batch: int, depth: int) -> None:
        t0 = time.monotonic()
        for i in range(batch):
            tokens = self._kvwarm_chain_token_ids(i, depth)
            req_id = f"__kvwarm_chain_{self._kvwarm_seq}"
            self._kvwarm_seq += 1
            req = Request(
                request_id=req_id,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(
                    max_tokens=100_000, ignore_eos=True
                ),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                # 盐按 (rank, 链号) 稳定:换代新链命中旧链缓存块,只算延伸段
                cache_salt=f"__kvwarm_{self._fpm_dp_rank}_{i}",
            )
            self.add_request(req)
            self._kvwarm_chain_ids.append(req_id)
            self._kvwarm_chain_prompts[req_id] = tokens
        self._kvwarm_stage_batch = batch
        self._kvwarm_building = True
        self._kvwarm_stage_t0 = t0
        logger.info("KVWARM: stage build batch=%d depth=%d", batch, depth)

    def _kvwarm_monitor_build(self) -> bool:
        pending = False
        vanished = []
        for req_id in self._kvwarm_chain_ids:
            req = self.requests.get(req_id)
            if req is None:
                # 长度停止/异常回收:剔链降级(失覆盖点走 fake),不致命。
                logger.warning("KVWARM: chain %s vanished during build; degrading", req_id)
                vanished.append(req_id)
                continue
            if req.num_computed_tokens >= len(self._kvwarm_chain_prompts[req_id]):
                if any(r.request_id == req_id for r in self.running):
                    self.running = [
                        r for r in self.running if r.request_id != req_id
                    ]  # 停车:退出调度视野,块与请求驻留
            else:
                pending = True
        if vanished:
            self._kvwarm_chain_ids = [
                r for r in self._kvwarm_chain_ids if r not in set(vanished)
            ]
            for r in vanished:
                self._kvwarm_chain_prompts.pop(r, None)
        if pending:
            return True
        self._kvwarm_building = False
        secs = time.monotonic() - getattr(self, "_kvwarm_stage_t0", time.monotonic())
        self._kvwarm_meta_init()["stages"].append(
            {
                "batch": self._kvwarm_stage_batch,
                "depth": max(
                    len(v) for v in self._kvwarm_chain_prompts.values()
                ),
                "build_seconds": round(secs, 3),
            }
        )
        logger.info(
            "KVWARM: stage ready batch=%s (%.1fs)",
            self._kvwarm_stage_batch,
            secs,
        )
        return True  # 再空转一步,下一拍进入正常点位流

    def _kvwarm_shed_chains(self) -> None:
        for req_id in getattr(self, "_kvwarm_chain_ids", []):
            req = self.requests.pop(req_id, None)
            if req is not None:
                self.kv_cache_manager.free(req)
                self.finished_req_ids.add(req_id)
        self.running = [
            r
            for r in self.running
            if r.request_id not in set(getattr(self, "_kvwarm_chain_ids", []))
        ]
        self._kvwarm_chain_ids = []
        self._kvwarm_chain_prompts = {}
        self._kvwarm_stage_batch = None
        self._kvwarm_building = False

    # ---------------- 影子注入:借链块,原两拍 ----------------

    def _kvwarm_point_need(self, point) -> int:
        # 非巨点:admission 写 ctx-1、steady 写 ctx → 需覆盖到 injected+1,
        # 即链深 ≥ injected+2;巨点多拍再加 repeats-1。
        giant = point.total_kv_read_tokens >= self._kvwarm_giant_threshold()
        return 2 + (self._kvwarm_giant_repeats() - 1 if giant else 0)

    def _kvwarm_covers(self, point, injected_lengths) -> bool:
        if not getattr(self, "_kvwarm_plan", None):
            return False
        if self._kvwarm_building or self._kvwarm_stage_batch is None:
            return False
        chains = self._kvwarm_chain_ids
        if len(chains) < point.batch_size:
            return False
        need = self._kvwarm_point_need(point)
        return all(
            injected + need <= len(self._kvwarm_chain_prompts[chains[i]])
            for i, injected in enumerate(injected_lengths)
        )

    def _kvwarm_inject_borrowed(self, context_lengths) -> "SchedulerOutput":
        """_bench_inject_fake_decode 的真实内容版:prompt=链真 token 前缀,
        块表=链自己的物理块(完整表,只读借用,零分配、免管理器登记)。"""
        new_reqs_data: list = []
        num_scheduled_tokens: dict = {}
        for index, ctx_len in enumerate(context_lengths):
            chain_id = self._kvwarm_chain_ids[index]
            chain_req = self.requests[chain_id]
            chain_tokens = self._kvwarm_chain_prompts[chain_id]
            block_ids = self.kv_cache_manager.get_block_ids(chain_id)
            # 按需截块表:整条链的块表在深链(131k+ = 1.28 万块/请求)上给
            # 测量步带来 ~11ms 簿记附加(r11 实测 b<=9 x per-req 131k 段
            # +45~63%,真值平滑而采集跳变)。只借覆盖写入位的前缀块。
            _bs = int(getattr(self.cache_config, "block_size", 16))
            _need_tokens = ctx_len + 1 + max(2, self._kvwarm_giant_repeats())
            _need_blocks = -(-_need_tokens // _bs) + 1
            block_ids = tuple(ids[:_need_blocks] for ids in block_ids)
            req_id = f"__bench_{self._bench_seq}"
            self._bench_seq += 1
            prompt = list(chain_tokens[: ctx_len + 1])
            req = Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=100_000),
                pooling_params=None,
                block_hasher=self._bench_block_hasher,
                cache_salt=req_id,
            )
            req.num_computed_tokens = ctx_len
            req.status = RequestStatus.RUNNING
            self.requests[req_id] = req
            self.running.append(req)  # type: ignore[has-type]
            self._bench_active_req_ids.add(req_id)
            self._kvwarm_borrowed_ids.add(req_id)
            new_reqs_data.append(
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    mm_features=[],
                    sampling_params=req.sampling_params,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=ctx_len,
                    lora_request=None,
                    prefill_token_ids=req._all_token_ids,
                )
            )
            num_scheduled_tokens[req_id] = 1
            del chain_req  # 只借块,不动链请求本体
        output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=len(new_reqs_data),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=(
                [0] * self.kv_cache_manager.num_kv_cache_groups
            ),
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=None,
        )
        if self.connector is not None:
            output.kv_connector_metadata = self.connector.build_connector_meta(
                output
            )
        if self.ec_connector is not None:
            output.ec_connector_metadata = self.ec_connector.build_connector_meta(
                output
            )
        return output

'''

SURGERIES = [
    # S11 DONE/超时路径:清合成前缀缓存之前必须甩链还池,否则 reset 失败致命
    (
        "S11-clear-cache-shed",
        """    def _bench_clear_prefix_cache(self) -> None:
        \"\"\"Remove all synthetic prefix entries before normal serving starts.\"\"\"
        if self._bench_prefix_cache_cleared:
            return""",
        """    def _bench_clear_prefix_cache(self) -> None:
        \"\"\"Remove all synthetic prefix entries before normal serving starts.\"\"\"
        if self._bench_prefix_cache_cleared:
            return
        if getattr(self, "_kvwarm_chain_ids", None):
            # KVWARM 停车链仍钉着块(超时/异常路径未经 busy 甩链),先还池。
            self._kvwarm_shed_chains()""",
    ),
    # S1 新方法块
    (
        "S1-methods",
        "    def _bench_make_steady_step(self) -> SchedulerOutput | None:",
        METHODS
        + "    def _bench_make_steady_step(self) -> SchedulerOutput | None:",
    ),
    # S2a 稳态步:借块影子跳过分配
    (
        "S2a-steady-alloc",
        """        new_blocks = {}
        for request in reqs:
            blocks = self.kv_cache_manager.allocate_slots(""",
        """        kvwarm_borrowed = getattr(self, "_kvwarm_borrowed_ids", set())
        new_blocks = {}
        for request in reqs:
            if request.request_id in kvwarm_borrowed:
                # KVWARM 影子:块借自停车链,深度余量已覆盖稳态写,零分配。
                new_blocks[request.request_id] = None
                continue
            blocks = self.kv_cache_manager.allocate_slots(""",
    ),
    # S2b 稳态步:new_block_ids 允许 None 条目
    (
        "S2b-steady-blockids",
        """            new_block_ids=[
                new_blocks[request.request_id].get_block_ids(allow_none=True)
                for request in reqs
            ],""",
        """            new_block_ids=[
                (
                    new_blocks[request.request_id].get_block_ids(allow_none=True)
                    if new_blocks[request.request_id] is not None
                    else None
                )
                for request in reqs
            ],""",
    ),
    # S3 清理:影子不走管理器 free(会把链块还池)
    (
        "S3-cleanup",
        """        for req_id in list(self._bench_active_req_ids):
            req = self.requests.get(req_id)
            if req:
                self.kv_cache_manager.free(req)
                self.finished_req_ids.add(req_id)
                del self.requests[req_id]""",
        """        kvwarm_borrowed = getattr(self, "_kvwarm_borrowed_ids", set())
        for req_id in list(self._bench_active_req_ids):
            req = self.requests.get(req_id)
            if req:
                if req_id in kvwarm_borrowed:
                    # KVWARM 影子借链块;管理器 free 会误还链的块。
                    kvwarm_borrowed.discard(req_id)
                    self.finished_req_ids.add(req_id)
                    del self.requests[req_id]
                    continue
                self.kv_cache_manager.free(req)
                self.finished_req_ids.add(req_id)
                del self.requests[req_id]""",
    ),
    # S4 巨点中位设置 + 点位注解
    (
        "S4-giant-and-tag",
        """        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_extra_steps_left = 1
        self._bench_expected_fpms = 2
        logger.info(
            "Benchmark decode: total_kv_reads=%d batch_size=%d",""",
        """        kvwarm_real = self._kvwarm_covers(point, injected_lengths)
        if self._kvwarm_flag_on():
            meta = self._kvwarm_meta_init()
            if kvwarm_real:
                meta["points_real_kv"] += 1
            else:
                meta["points_fake_fallback"] += 1
            point = replace(
                point,
                sample_reasons=[
                    *point.sample_reasons,
                    "kvwarm_real_kv" if kvwarm_real else "kvwarm_fake_fallback",
                ],
            )
        self._bench_current_point = point
        self._bench_current_fpms = []
        self._bench_extra_steps_left = 1
        self._bench_expected_fpms = 2
        if self._kvwarm_flag_on() and (
            kvwarm_real
            or point.total_kv_read_tokens >= self._kvwarm_giant_threshold()
        ):
            # warmed 点稳态零分配,多拍近乎免费——中位保护全覆盖,
            # 根治散发计时搬移(r8 实测 1/1457 打中非巨点);fake 点仍仅巨点。
            # 巨 KV 计时守卫(HANDOVER §6.1):稳态步重复取中位。
            # fake 注入的巨点在池边界装不下多拍时降级 legacy 两拍
            # (warmed 影子零分配,不受限)。
            repeats = self._kvwarm_giant_repeats()
            # 模型长度封顶:第 k 拍后 total = ctx+k ≤ max_model_len,
            # 且 runner 簿记写透 +1(顶点自动降回 legacy 单拍)。
            max_ctx = max(injected_lengths) + 1
            repeats = min(repeats, max(1, self.max_model_len - 1 - max_ctx))
            if not kvwarm_real:
                multi = sum(
                    self._bench_blocks_per_req(max(c, 2) + repeats)
                    for c in injected_lengths
                )
                if multi > self._bench_usable_blocks(
                    point.batch_size, reserve_watermark=True
                ):
                    repeats = 1
            self._bench_extra_steps_left = repeats
            self._bench_expected_fpms = repeats + 1
        logger.info(
            "Benchmark decode: total_kv_reads=%d batch_size=%d",""",
    ),
    # S5 注入分派:影子 vs fake
    (
        "S5-inject-dispatch",
        "        output = self._bench_inject_fake_decode(injected_lengths)",
        """        output = (
            self._kvwarm_inject_borrowed(injected_lengths)
            if kvwarm_real
            else self._bench_inject_fake_decode(injected_lengths)
        )""",
    ),
    # S6 保存:巨点稳态中位(坐标取首个稳态)
    (
        "S6-save-median",
        """            expected_fpms = getattr(self, "_bench_expected_fpms", 1)
            if expected_fpms > 1 and len(local_fpms) >= expected_fpms:""",
        """            expected_fpms = getattr(self, "_bench_expected_fpms", 1)
            if expected_fpms > 2 and len(local_fpms) >= 2:
                # 巨 KV 中位:多个相邻稳态步,有几拍算几拍(池边界点的
                # 后续稳态可能分配失败提前断拍,median_of 如实记录);
                # 坐标取首个稳态;只剩 admission 时回落原路径由形状校验跳点。
                steadies = local_fpms[1:expected_fpms]
                walls = sorted(
                    float(f.get("wall_time", 0.0)) for f in steadies
                )
                chosen = dict(steadies[0])
                chosen["wall_time"] = walls[len(walls) // 2]
                chosen["kvwarm_giant_median_of"] = len(steadies)
                local_fpms = [chosen]
            elif expected_fpms > 1 and len(local_fpms) >= expected_fpms:""",
    ),
    # S7 相位机拦截:预热状态机
    (
        "S7-phase-hook",
        """        if self._bench_phase == _BenchPhase.WARMUP:
            return self._bench_step_warmup()""",
        """        if (
            self._bench_phase == _BenchPhase.DECODE_SWEEP
            and self._kvwarm_step_busy()
        ):
            return None  # 链群建设中:交还真调度器 chunked prefill
        if self._bench_phase == _BenchPhase.WARMUP:
            return self._bench_step_warmup()""",
    ),
    # S8 网格构建尾:重排 + 计划
    (
        "S8-grid-prepare",
        '        logger.info("Benchmark grid: %d points (%s mode)", len(self._bench_grid), mode)',
        '        logger.info("Benchmark grid: %d points (%s mode)", len(self._bench_grid), mode)\n'
        "        self._kvwarm_prepare(mode)",
    ),
    # S10 结果元数据(off 时无此键,输出与原版一致)
    (
        "S10-results-meta",
        """        output = {
            "schema_version": 2,""",
        """        output = {
            "schema_version": 2,
            **(
                {"kvwarm": self._kvwarm_meta}
                if getattr(self, "_kvwarm_meta", None) is not None
                else {}
            ),""",
    ),
]

applied = src
for name, old, new in SURGERIES:
    count = applied.count(old)
    assert count == 1, f"{name}: anchor count={count} (需要恰好 1)"
    applied = applied.replace(old, new)

ast.parse(applied)
open(P, "w").write(applied)
print(f"kvwarm: {len(SURGERIES)} surgeries applied, ast OK -> {P}")
