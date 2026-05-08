from pathlib import Path


def replace_once(path: Path, old: str, new: str):
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"pattern not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


def patch_scheduler_metrics(root: Path):
    path = root / "srt/managers/scheduler_metrics_mixin.py"

    replace_once(
        path,
        "from typing import TYPE_CHECKING, Dict, List, Optional, Union\n\n"
        "from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch\n",
        "from typing import TYPE_CHECKING, Dict, List, Optional, Union\n\n"
        "try:\n"
        "    import torch.cuda.nvtx as cuda_nvtx\n"
        "except Exception:\n"
        "    cuda_nvtx = None\n\n"
        "from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch\n",
    )

    replace_once(
        path,
        "RECORD_STEP_TIME = get_bool_env_var(\"SGLANG_RECORD_STEP_TIME\")\n"
        "LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()\n"
        "ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()\n",
        "RECORD_STEP_TIME = get_bool_env_var(\"SGLANG_RECORD_STEP_TIME\")\n"
        "LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()\n"
        "SCHEDULER_NVTX = get_bool_env_var(\"SGLANG_SCHEDULER_NVTX\")\n"
        "ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()\n",
    )

    replace_once(
        path,
        "        self.stats = SchedulerStats()\n\n"
        "        # Metrics\n",
        "        self.stats = SchedulerStats()\n"
        "        self.scheduler_nvtx_enabled = SCHEDULER_NVTX and cuda_nvtx is not None\n\n"
        "        # Metrics\n",
    )

    replace_once(
        path,
        "        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create()\n\n"
        "    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):\n",
        "        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create()\n\n"
        "    def _scheduler_nvtx_rank_suffix(self: Scheduler) -> str:\n"
        "        ranks = []\n"
        "        for name in (\"dp_rank\", \"tp_rank\", \"pp_rank\", \"moe_ep_rank\"):\n"
        "            value = getattr(self, name, None)\n"
        "            if value is not None:\n"
        "                ranks.append(f\"{name[:-5]}={value}\")\n"
        "        return \" \".join(ranks)\n\n"
        "    def _scheduler_nvtx_batch_label(self: Scheduler, batch: ScheduleBatch) -> str:\n"
        "        mode = batch.forward_mode.name.lower() if batch.forward_mode else \"none\"\n"
        "        bs = batch.batch_size()\n"
        "        parts = [f\"sgl_sched mode={mode}\", f\"iter={self.forward_ct}\", f\"bs={bs}\"]\n\n"
        "        if batch.forward_mode and batch.forward_mode.is_extend():\n"
        "            parts.append(f\"new_token={batch.extend_num_tokens}\")\n"
        "            if batch.extend_lens:\n"
        "                parts.append(f\"max_extend={max(batch.extend_lens)}\")\n"
        "        elif batch.forward_mode and batch.forward_mode.is_decode():\n"
        "            parts.append(f\"running={bs}\")\n\n"
        "        seq_lens_sum = getattr(batch, \"seq_lens_sum\", None)\n"
        "        if seq_lens_sum is not None:\n"
        "            parts.append(f\"seq_sum={seq_lens_sum}\")\n"
        "        parts.append(f\"queue={len(self.waiting_queue)}\")\n\n"
        "        rank_suffix = self._scheduler_nvtx_rank_suffix()\n"
        "        if rank_suffix:\n"
        "            parts.append(rank_suffix)\n"
        "        if self.disaggregation_mode != DisaggregationMode.NULL:\n"
        "            parts.append(f\"disagg={self.disaggregation_mode.name.lower()}\")\n"
        "        return \" \".join(parts)\n\n"
        "    @contextmanager\n"
        "    def scheduler_nvtx_range(self: Scheduler, batch: ScheduleBatch):\n"
        "        if not getattr(self, \"scheduler_nvtx_enabled\", False):\n"
        "            yield\n"
        "            return\n\n"
        "        pushed = False\n"
        "        try:\n"
        "            cuda_nvtx.range_push(self._scheduler_nvtx_batch_label(batch))\n"
        "            pushed = True\n"
        "        except Exception:\n"
        "            logger.exception(\"Failed to push scheduler NVTX range\")\n\n"
        "        try:\n"
        "            yield\n"
        "        finally:\n"
        "            if pushed:\n"
        "                try:\n"
        "                    cuda_nvtx.range_pop()\n"
        "                except Exception:\n"
        "                    logger.exception(\"Failed to pop scheduler NVTX range\")\n\n"
        "    def scheduler_nvtx_mark(self: Scheduler, label: str):\n"
        "        if not getattr(self, \"scheduler_nvtx_enabled\", False):\n"
        "            return\n"
        "        try:\n"
        "            cuda_nvtx.mark(label)\n"
        "        except Exception:\n"
        "            logger.exception(\"Failed to emit scheduler NVTX mark\")\n\n"
        "    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):\n",
    )

    replace_once(
        path,
        "        logger.info(f)\n\n"
        "        if self.enable_metrics:\n",
        "        logger.info(f)\n"
        "        self.scheduler_nvtx_mark(f\"sgl_prefill_stats iter={self.forward_ct + 1} {f}\")\n\n"
        "        if self.enable_metrics:\n",
    )

    replace_once(
        path,
        "        logger.info(msg)\n"
        "        if self.enable_metrics:\n",
        "        logger.info(msg)\n"
        "        self.scheduler_nvtx_mark(f\"sgl_decode_stats iter={self.forward_ct} {msg}\")\n\n"
        "        if self.enable_metrics:\n",
    )


def patch_scheduler(root: Path):
    path = root / "srt/managers/scheduler.py"

    replace_once(
        path,
        "                    with self.record_forward_metrics(batch):\n"
        "                        batch_result = self.model_worker.forward_batch_generation(\n"
        "                            model_worker_batch\n"
        "                            # here pp is not compatible with overlap\n"
        "                        )\n",
        "                    with self.scheduler_nvtx_range(batch):\n"
        "                        with self.record_forward_metrics(batch):\n"
        "                            batch_result = self.model_worker.forward_batch_generation(\n"
        "                                model_worker_batch\n"
        "                                # here pp is not compatible with overlap\n"
        "                            )\n",
    )

    replace_once(
        path,
        "            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():\n"
        "                batch_result = self.tp_worker.forward_batch_split_prefill(batch)\n"
        "                future_indices_or_next_token_ids = batch_result.next_token_ids\n",
        "            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():\n"
        "                with self.scheduler_nvtx_range(batch):\n"
        "                    batch_result = self.tp_worker.forward_batch_split_prefill(batch)\n"
        "                future_indices_or_next_token_ids = batch_result.next_token_ids\n",
    )

    replace_once(
        path,
        "                with self.record_forward_metrics(batch):\n"
        "                    batch_result = self.model_worker.forward_batch_generation(\n"
        "                        worker_batch_or_batch, **kwargs\n"
        "                    )\n",
        "                with self.scheduler_nvtx_range(batch):\n"
        "                    with self.record_forward_metrics(batch):\n"
        "                        batch_result = self.model_worker.forward_batch_generation(\n"
        "                            worker_batch_or_batch, **kwargs\n"
        "                        )\n",
    )

    replace_once(
        path,
        "                    embeddings = self.tp_worker.forward_batch_embedding(\n"
        "                        model_worker_batch\n"
        "                    )\n",
        "                    with self.scheduler_nvtx_range(batch):\n"
        "                        embeddings = self.tp_worker.forward_batch_embedding(\n"
        "                            model_worker_batch\n"
        "                        )\n",
    )

    replace_once(
        path,
        "                embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)\n",
        "                with self.scheduler_nvtx_range(batch):\n"
        "                    embeddings = self.tp_worker.forward_batch_embedding(\n"
        "                        model_worker_batch\n"
        "                    )\n",
    )


def main():
    candidates = [
        Path("/workspace/sglang/python/sglang"),
        Path("/sgl-workspace/sglang/python/sglang"),
    ]
    root = next((p for p in candidates if (p / "__init__.py").exists()), None)
    if root is None:
        raise RuntimeError(f"cannot find sglang package root in {candidates}")

    patch_scheduler_metrics(root)
    patch_scheduler(root)
    print(f"applied scheduler NVTX patch under {root}", flush=True)


if __name__ == "__main__":
    main()
