from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import PruningConfig
from ..embedding import Embedder
from .tree import PrunedTreeResult, TreePruner


class AsyncTreePruner:
    """Thread-pooled wrapper around :class:`TreePruner`."""

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
    ) -> None:
        self._pruner = TreePruner(
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
            encode_lock=encode_lock,
            index_lock=index_lock,
        )
        workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    async def prune_many(
        self, items: Iterable[Tuple[int, Dict[str, Optional[str]]]]
    ) -> List[Tuple[int, PrunedTreeResult]]:
        batch = list(items)
        if not batch:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_batch, batch)

    # ------------------------------------------------------------------
    def _run_batch(
        self, batch: Sequence[Tuple[int, Dict[str, Optional[str]]]]
    ) -> List[Tuple[int, PrunedTreeResult]]:
        results: List[Tuple[int, PrunedTreeResult]] = []
        for idx, item in batch:
            result = self._pruner.prune(item)
            results.append((idx, result))
        return results
