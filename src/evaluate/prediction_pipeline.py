"""Asynchronous pipeline responsible for pruning and LLM matching."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiohttp
import numpy as np
import pandas as pd

from config import ParallelismConfig

from ..embedding import Embedder
from ..matching import MatchRequest, match_items_to_tree
from ..pruning import AsyncTreePruner, PrunedTreeResult
from .metrics import build_result_row
from .types import PredictionJob, ProgressHook


class PredictionPipeline:
    """Coordinates asynchronous pruning and prediction requests."""

    SENTINEL = object()

    def __init__(
        self,
        *,
        jobs: Sequence[PredictionJob],
        pruning_cfg,
        llm_cfg,
        parallel_cfg: ParallelismConfig,
        keywords: pd.DataFrame,
        graph,
        undirected_graph,
        depth_map: Mapping[str, Optional[int]],
        embedder: Embedder,
        tax_names: Sequence[str],
        tax_embs_unit: np.ndarray,
        hnsw_index,
        name_to_id: Dict[str, str],
        name_to_path: Dict[str, str],
        gloss_map: Dict[str, str],
        session: aiohttp.ClientSession,
        progress_hook: ProgressHook | None,
    ) -> None:
        self.jobs = list(jobs)
        self.pruning_cfg = pruning_cfg
        self.llm_cfg = llm_cfg
        self.parallel_cfg = parallel_cfg
        self._prune_workers = max(1, int(getattr(parallel_cfg, "pruning_workers", 1)))
        self._prune_batch_size = max(
            1, int(getattr(parallel_cfg, "pruning_batch_size", 1))
        )
        queue_size = getattr(parallel_cfg, "pruning_queue_size", self._prune_batch_size)
        self._prune_queue_size = max(1, int(queue_size))
        self.keywords = keywords
        self.graph = graph
        self.undirected_graph = undirected_graph
        self.depth_map = depth_map
        self.embedder = embedder
        self.tax_names = list(tax_names)
        self.tax_embs_unit = tax_embs_unit
        self.hnsw_index = hnsw_index
        self.name_to_id = name_to_id
        self.name_to_path = name_to_path
        self.gloss_map = gloss_map
        self.session = session
        self.progress_hook = progress_hook

        self.total = len(self.jobs)
        self.rows: List[Optional[Dict[str, Any]]] = [None] * self.total
        self.start_time = time.time()
        self.correct_sum = 0
        self.completed = 0
        self.gold_progress_seen = False

        self.encode_lock = threading.Lock()
        self.index_lock = threading.Lock()

        self.pruner = AsyncTreePruner(
            graph=self.graph,
            frame=self.keywords,
            embedder=self.embedder,
            tax_names=self.tax_names,
            tax_embs_unit=self.tax_embs_unit,
            hnsw_index=self.hnsw_index,
            pruning_cfg=self.pruning_cfg,
            name_col="name",
            order_col="order",
            gloss_map=self.gloss_map,
            max_workers=self._prune_workers,
            encode_lock=self.encode_lock,
            index_lock=self.index_lock,
        )

        self.prune_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=self._prune_queue_size
        )
        self.prune_tasks: List[asyncio.Task[None]] = []
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.pruner.close()
        self._closed = True

    async def cancel_workers(self) -> None:
        for task in self.prune_tasks:
            task.cancel()
        if self.prune_tasks:
            await asyncio.gather(*self.prune_tasks, return_exceptions=True)
        self.close()

    def run_prune_workers(self) -> List[asyncio.Task[None]]:
        if not self.prune_tasks:
            self.prune_tasks = [
                asyncio.create_task(self._prune_worker())
                for _ in range(self._prune_workers)
            ]
        return self.prune_tasks

    async def queue_prune_jobs(self) -> None:
        for idx, job in enumerate(self.jobs):
            await self.prune_queue.put((idx, job))
        for _ in range(self._prune_workers):
            await self.prune_queue.put(self.SENTINEL)

    async def finalise_results(self) -> List[Dict[str, Any]]:
        try:
            await self.prune_queue.join()
            if self.prune_tasks:
                await asyncio.gather(*self.prune_tasks)
        finally:
            self.close()
        return [row for row in self.rows if row is not None]

    async def _prune_worker(self) -> None:
        pending: List[Tuple[int, PredictionJob]] = []
        while True:
            item = await self.prune_queue.get()
            if item is self.SENTINEL:
                await self._flush_prune(pending)
                self.prune_queue.task_done()
                break
            pending.append(item)
            if len(pending) >= self._prune_batch_size:
                await self._flush_prune(pending)
        if pending:
            await self._flush_prune(pending)

    async def _flush_prune(self, pending: List[Tuple[int, PredictionJob]]) -> None:
        if not pending:
            return
        batch = list(pending)
        pending.clear()

        try:
            results = await self.pruner.prune_many(
                [(idx, job.item) for idx, job in batch]
            )
        except Exception:
            for _ in batch:
                self.prune_queue.task_done()
            raise

        prompt_batch: List[Tuple[int, PredictionJob, PrunedTreeResult]] = []
        for (idx, job), (result_idx, pruned) in zip(batch, results):
            assert idx == result_idx
            prompt_batch.append((idx, job, pruned))
            self.prune_queue.task_done()

        if prompt_batch:
            await self._process_prompt_batch(prompt_batch)

    async def _process_prompt_batch(
        self, prompt_batch: List[Tuple[int, PredictionJob, PrunedTreeResult]]
    ) -> None:
        for idx, job, pruned in prompt_batch:
            await self._handle_single_prediction(idx, job, pruned)

    async def _handle_single_prediction(
        self, idx: int, job: PredictionJob, pruned: PrunedTreeResult
    ) -> None:
        request = self._build_request(job, pruned)
        predictions = await self._fetch_predictions(request, idx, job)
        if not predictions:
            raise RuntimeError(
                "LLM matching returned no predictions for request "
                f"at index {idx} (slot_id={job.slot_id})."
            )

        result, correct_increment, has_gold = build_result_row(
            job,
            pruned,
            predictions[0],
            graph=self.graph,
            undirected_graph=self.undirected_graph,
            depth_map=self.depth_map,
        )
        self.rows[idx] = result
        self.completed += 1
        self._update_progress(correct_increment, has_gold)

    def _build_request(
        self, job: PredictionJob, pruned: PrunedTreeResult
    ) -> MatchRequest:
        return MatchRequest(
            item=job.item,
            tree_markdown=pruned.markdown,
            allowed_labels=tuple(pruned.allowed_labels),
            allowed_children=pruned.allowed_children,
            slot_id=job.slot_id,
        )

    async def _fetch_predictions(
        self, request: MatchRequest, idx: int, job: PredictionJob
    ) -> Sequence[Mapping[str, Any]]:
        try:
            return await match_items_to_tree(
                [request],
                name_to_id=self.name_to_id,
                name_to_path=self.name_to_path,
                tax_names=self.tax_names,
                tax_embs=self.tax_embs_unit,
                embedder=self.embedder,
                hnsw_index=self.hnsw_index,
                llm_config=self.llm_cfg,
                session=self.session,
                encode_lock=self.encode_lock,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                "LLM matching failed for prompt request at "
                f"index {idx} (slot_id={job.slot_id})."
            ) from exc

    def _update_progress(self, correct_increment: int, has_gold: bool) -> None:
        if has_gold:
            self.gold_progress_seen = True
            self.correct_sum += correct_increment

        if self.progress_hook is None:
            return

        elapsed = max(time.time() - self.start_time, 0.0)
        hook_correct = self.correct_sum if self.gold_progress_seen else None
        self.progress_hook(self.completed, self.total, hook_correct, elapsed)
