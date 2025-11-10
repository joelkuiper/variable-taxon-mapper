from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import PruningConfig
from ..embedding import (
    Embedder,
    build_hnsw_index,
    collect_item_texts,
    encode_item_texts,
)
from .tree import PrunedTreeResult, TreePruner


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_PROCESS_TREE_PRUNER: Optional[TreePruner] = None


@dataclass(frozen=True)
class _PrunerState:
    graph: nx.DiGraph
    frame: pd.DataFrame
    tax_names: Tuple[str, ...]
    tax_embs_unit: np.ndarray
    pruning_cfg: Optional[PruningConfig]
    name_col: str
    order_col: str
    gloss_map: Optional[Dict[str, str]]
    snake_case_to_title: bool
    embedder_kwargs: Optional[Dict[str, Any]]
    hnsw_kwargs: Dict[str, Any]
    worker_devices: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class PrunePayload:
    idx: int
    item: Dict[str, Optional[str]]
    item_texts: Tuple[str, ...]
    item_embs: np.ndarray


def _init_tree_pruner(state: _PrunerState) -> None:
    """Process initializer that builds a :class:`TreePruner` once per worker."""

    global _PROCESS_TREE_PRUNER
    if _PROCESS_TREE_PRUNER is not None:
        return

    try:
        from tokenizers import parallelism as _parallelism  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    else:  # pragma: no cover - exercised when tokenizers is installed
        try:
            _parallelism.set_parallelism(False)  # type: ignore[attr-defined]
        except AttributeError:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    embedder_kwargs = (
        dict(state.embedder_kwargs) if state.embedder_kwargs is not None else None
    )
    embedder: Optional[Embedder] = None
    if embedder_kwargs:
        worker_devices = state.worker_devices
        if worker_devices:
            process = mp.current_process()
            identity = getattr(process, "_identity", None) or ()
            if identity:
                ordinal = (identity[0] - 1) % len(worker_devices)
            else:
                ordinal = 0
            device_override = worker_devices[ordinal]
            if device_override:
                embedder_kwargs["device"] = device_override

        embedder = Embedder(**embedder_kwargs)

    hnsw_kwargs = dict(state.hnsw_kwargs)
    if not hnsw_kwargs:
        raise RuntimeError("TreePruner initializer requires HNSW build kwargs")
    hnsw_index = build_hnsw_index(state.tax_embs_unit, **hnsw_kwargs)

    _PROCESS_TREE_PRUNER = TreePruner(
        graph=state.graph,
        frame=state.frame,
        embedder=embedder,
        tax_names=state.tax_names,
        tax_embs_unit=state.tax_embs_unit,
        hnsw_index=hnsw_index,
        pruning_cfg=state.pruning_cfg,
        name_col=state.name_col,
        order_col=state.order_col,
        gloss_map=state.gloss_map,
        snake_case_to_title=state.snake_case_to_title,
    )


def _ensure_process_pruner() -> TreePruner:
    pruner = _PROCESS_TREE_PRUNER
    if pruner is None:
        raise RuntimeError("Process pool TreePruner has not been initialized")
    return pruner


def prune_single(idx: int, item: Dict[str, Optional[str]]) -> Tuple[int, PrunedTreeResult]:
    """Worker entry point used by :class:`AsyncTreePruner` clients."""

    pruner = _ensure_process_pruner()
    return idx, pruner.prune(item)


def prune_batch(
    batch: Sequence[Tuple[int, Dict[str, Optional[str]]]]
) -> List[Tuple[int, PrunedTreeResult]]:
    pruner = _ensure_process_pruner()
    results: List[Tuple[int, PrunedTreeResult]] = []
    for idx, item in batch:
        results.append((idx, pruner.prune(item)))
    return results


def _normalise_start_method(
    candidate: Optional[str], embedder: Optional[Embedder]
) -> str:
    device = getattr(embedder, "device", "cpu") if embedder is not None else "cpu"
    if not device:
        device = "cpu"
    default = "spawn" if str(device).lower().startswith("cuda") else "fork"
    if candidate is None:
        return default
    value = candidate.strip().lower()
    if value in {"", "auto"}:
        return default
    if value not in {"fork", "spawn"}:
        raise ValueError(
            f"Unsupported pruning start method '{candidate}'. Expected 'fork', 'spawn',"
            " or 'auto'."
        )
    return value


class AsyncTreePruner:
    """Parallel pruning helper that adapts to available resources.

    When worker-side embedders are requested the class spins up a
    :class:`~concurrent.futures.ProcessPoolExecutor` and relies on
    :func:`_init_tree_pruner` to construct a ``TreePruner`` in each process. In
    the default configuration, embeddings are precomputed on the coordinator and
    pruning is dispatched to a lightweight thread pool backed by a single
    ``TreePruner`` instance. This avoids cross-process serialisation of large
    objects and keeps GPU-bound models single-instanced while still allowing CPU
    stages to overlap. Callers must ensure this module is importable in
    subprocesses (use ``if __name__ == "__main__"`` guards for entry points) and
    that referenced objects are compatible with ``multiprocessing`` fork/spawn
    semantics. Tokenizer parallelism is disabled within worker processes to
    avoid Hugging Face fork warnings. When the embedder uses CUDA, the start
    method automatically switches to ``spawn`` and worker devices can be
    overridden via ``worker_devices``.
    """

    def __init__(
        self,
        *,
        graph: nx.DiGraph,
        frame: pd.DataFrame,
        embedder: Optional[Embedder],
        tax_names: Sequence[str],
        tax_embs_unit: np.ndarray,
        hnsw_index,
        pruning_cfg: Optional[PruningConfig] = None,
        name_col: str = "name",
        order_col: str = "order",
        gloss_map: Optional[Dict[str, str]] = None,
        max_workers: int = 1,
        snake_case_to_title: bool = True,
        hnsw_build_kwargs: Optional[Mapping[str, Any]] = None,
        embedder_init_kwargs: Optional[Mapping[str, Any]] = None,
        start_method: Optional[str] = None,
        worker_devices: Optional[Sequence[str]] = None,
    ) -> None:
        workers = max(1, int(max_workers))
        self._parent_embedder = embedder
        self._precompute_embeddings = embedder_init_kwargs is None
        if self._precompute_embeddings and embedder is None:
            raise ValueError(
                "AsyncTreePruner requires an embedder when running in precompute"
                " mode."
            )

        self._encode_lock: Optional[asyncio.Lock] = None
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        self._local_pruner: Optional[TreePruner] = None

        if self._precompute_embeddings:
            self._local_pruner = TreePruner(
                graph=graph,
                frame=frame,
                embedder=embedder,
                tax_names=tax_names,
                tax_embs_unit=tax_embs_unit,
                hnsw_index=hnsw_index,
                pruning_cfg=pruning_cfg,
                name_col=name_col,
                order_col=order_col,
                gloss_map=gloss_map,
                snake_case_to_title=snake_case_to_title,
            )
            if workers > 1:
                self._thread_executor = ThreadPoolExecutor(max_workers=workers)
        else:
            resolved_method = _normalise_start_method(start_method, embedder)

            embedder_kwargs: Optional[Dict[str, Any]]
            embedder_kwargs = dict(embedder_init_kwargs or {})
            if not embedder_kwargs:
                if embedder is None:
                    raise ValueError(
                        "Embedder init kwargs were requested but no embedder"
                        " instance was provided to export defaults."
                    )
                embedder_kwargs = embedder.export_init_kwargs()
            if not embedder_kwargs:
                raise ValueError(
                    "AsyncTreePruner requires embedder initialization kwargs to"
                    " spawn worker-local model instances."
                )

            hnsw_kwargs: Dict[str, Any] = dict(hnsw_build_kwargs or {})
            if not hnsw_kwargs:
                raise ValueError(
                    "AsyncTreePruner requires HNSW build kwargs to construct"
                    " worker-local indices."
                )

            device_tuple: Optional[Tuple[str, ...]] = None
            if worker_devices:
                device_tuple = tuple(str(device) for device in worker_devices)

            state = _PrunerState(
                graph=graph,
                frame=frame,
                tax_names=tuple(tax_names),
                tax_embs_unit=tax_embs_unit,
                pruning_cfg=pruning_cfg,
                name_col=name_col,
                order_col=order_col,
                gloss_map=dict(gloss_map or {}),
                snake_case_to_title=snake_case_to_title,
                embedder_kwargs=embedder_kwargs,
                hnsw_kwargs=hnsw_kwargs,
                worker_devices=device_tuple,
            )

            context = mp.get_context(resolved_method)
            self._process_executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_tree_pruner,
                initargs=(state,),
                mp_context=context,
            )

    @property
    def executor(self) -> Executor:
        executor: Optional[Executor]
        if self._precompute_embeddings:
            executor = self._thread_executor
        else:
            executor = self._process_executor
        if executor is None:
            raise RuntimeError("AsyncTreePruner has been closed")
        return executor

    def close(self) -> None:
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=True, cancel_futures=True)
            self._thread_executor = None
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=True, cancel_futures=True)
            self._process_executor = None
        self._local_pruner = None

    async def prune_many(
        self, items: Iterable[Tuple[int, Dict[str, Optional[str]]]]
    ) -> List[Tuple[int, PrunedTreeResult]]:
        batch = list(items)
        if not batch:
            return []

        loop = asyncio.get_running_loop()
        if self._precompute_embeddings:
            payloads = await self._encode_batch(loop, batch)
            if not payloads:
                return []
            executor = self._thread_executor
            if executor is None:
                return await loop.run_in_executor(
                    None, self._run_payload_batch, payloads
                )
            return await loop.run_in_executor(executor, self._run_payload_batch, payloads)

        return await loop.run_in_executor(self.executor, prune_batch, batch)

    async def _encode_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        batch: Sequence[Tuple[int, Dict[str, Optional[str]]]],
    ) -> List[PrunePayload]:
        lock = self._ensure_encode_lock()
        async with lock:
            return await loop.run_in_executor(
                None, self._prepare_payloads, batch
            )

    def _ensure_encode_lock(self) -> asyncio.Lock:
        lock = self._encode_lock
        if lock is None:
            lock = asyncio.Lock()
            self._encode_lock = lock
        return lock

    def _prepare_payloads(
        self, batch: Sequence[Tuple[int, Dict[str, Optional[str]]]]
    ) -> List[PrunePayload]:
        embedder = self._parent_embedder
        if embedder is None:
            raise RuntimeError(
                "AsyncTreePruner is configured for worker-side embedding"
                " but attempted to precompute embeddings on the coordinator."
            )

        payloads: List[PrunePayload] = []
        for idx, item in batch:
            texts = tuple(collect_item_texts(item))
            embs = encode_item_texts(item, embedder, texts=texts)
            embs = np.asarray(embs, dtype=np.float32)
            payloads.append(
                PrunePayload(
                    idx=idx,
                    item=item,
                    item_texts=texts,
                    item_embs=embs,
                )
            )
        return payloads

    def _ensure_local_pruner(self) -> TreePruner:
        pruner = self._local_pruner
        if pruner is None:
            raise RuntimeError(
                "AsyncTreePruner does not have a local TreePruner; configure"
                " worker embedders or ensure the pruner has not been closed."
            )
        return pruner

    def _run_payload_batch(
        self, payloads: Sequence[PrunePayload]
    ) -> List[Tuple[int, PrunedTreeResult]]:
        pruner = self._ensure_local_pruner()
        results: List[Tuple[int, PrunedTreeResult]] = []
        for payload in payloads:
            results.append(
                (
                    payload.idx,
                    pruner.prune_with_embeddings(
                        item=payload.item,
                        item_texts=payload.item_texts,
                        item_embs=payload.item_embs,
                    ),
                )
            )
        return results
