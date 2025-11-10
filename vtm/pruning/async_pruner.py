from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import PruningConfig
from ..embedding import Embedder
from .tree import PrunedTreeResult, TreePruner


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_PROCESS_TREE_PRUNER: Optional[TreePruner] = None


def _init_tree_pruner(state) -> None:
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

    (
        graph,
        frame,
        embedder,
        tax_names,
        tax_embs_unit,
        hnsw_index,
        pruning_cfg,
        name_col,
        order_col,
        gloss_map,
        snake_case_to_title,
    ) = state

    _PROCESS_TREE_PRUNER = TreePruner(
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


class AsyncTreePruner:
    """Process-pooled wrapper around :class:`TreePruner`.

    The underlying pool is backed by :class:`~concurrent.futures.ProcessPoolExecutor`
    and relies on :func:`_init_tree_pruner` to construct a ``TreePruner`` in each
    worker. Callers must ensure this module is importable in subprocesses (use
    ``if __name__ == "__main__"`` guards for entry points) and that referenced
    objects are compatible with ``multiprocessing`` fork/spawn semantics. Tokenizer
    parallelism is disabled within worker processes to avoid Hugging Face fork
    warnings.
    """

    def __init__(
        self,
        *,
        graph: nx.DiGraph,
        frame: pd.DataFrame,
        embedder: Embedder,
        tax_names: Sequence[str],
        tax_embs_unit: np.ndarray,
        hnsw_index,
        pruning_cfg: Optional[PruningConfig] = None,
        name_col: str = "name",
        order_col: str = "order",
        gloss_map: Optional[Dict[str, str]] = None,
        max_workers: int = 1,
        encode_lock=None,
        index_lock=None,
        snake_case_to_title: bool = True,
    ) -> None:
        del encode_lock, index_lock  # Locks are local to each process.

        workers = max(1, int(max_workers))
        self._executor: Optional[ProcessPoolExecutor]
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_tree_pruner,
            initargs=(
                (
                    graph,
                    frame,
                    embedder,
                    tuple(tax_names),
                    tax_embs_unit,
                    hnsw_index,
                    pruning_cfg,
                    name_col,
                    order_col,
                    gloss_map,
                    snake_case_to_title,
                ),
            ),
        )

    @property
    def executor(self) -> ProcessPoolExecutor:
        executor = self._executor
        if executor is None:
            raise RuntimeError("AsyncTreePruner has been closed")
        return executor

    def close(self) -> None:
        executor = self._executor
        if executor is None:
            return
        executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None

    async def prune_many(
        self, items: Iterable[Tuple[int, Dict[str, Optional[str]]]]
    ) -> List[Tuple[int, PrunedTreeResult]]:
        batch = list(items)
        if not batch:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, prune_batch, batch)
