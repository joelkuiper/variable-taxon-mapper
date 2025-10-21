"""Helpers to orchestrate asynchronous prediction collection."""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Sequence

import aiohttp
import numpy as np
import pandas as pd

from config import HttpConfig, LLMConfig, ParallelismConfig, PruningConfig

from ..embedding import Embedder
from ..graph_utils import compute_node_depths, get_undirected_taxonomy
from .parallelism import sock_read_timeout
from .prediction_pipeline import PredictionPipeline
from .types import PredictionJob, ProgressHook


def collect_predictions(
    jobs: Sequence[PredictionJob],
    *,
    pruning_cfg: PruningConfig,
    llm_cfg: LLMConfig,
    parallel_cfg: ParallelismConfig,
    http_cfg: HttpConfig,
    keywords: pd.DataFrame,
    graph,
    embedder: Embedder,
    tax_names: Sequence[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    gloss_map: Dict[str, str],
    progress_hook: ProgressHook | None,
) -> List[Dict[str, Any]]:
    undirected_graph = get_undirected_taxonomy(graph) if graph is not None else None
    depth_map = compute_node_depths(graph) if graph is not None else {}

    async def _predict_all() -> List[Dict[str, Any]]:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None,
            sock_connect=float(http_cfg.sock_connect),
            sock_read=sock_read_timeout(http_cfg, llm_cfg),
        )
        pool_limit = max(1, int(getattr(parallel_cfg, "pool_maxsize", 1)))
        connector = aiohttp.TCPConnector(
            limit=pool_limit,
            limit_per_host=pool_limit,
        )

        async with aiohttp.ClientSession(timeout=timeout_cfg, connector=connector) as session:
            pipeline = PredictionPipeline(
                jobs=jobs,
                pruning_cfg=pruning_cfg,
                llm_cfg=llm_cfg,
                parallel_cfg=parallel_cfg,
                keywords=keywords,
                graph=graph,
                undirected_graph=undirected_graph,
                depth_map=depth_map,
                embedder=embedder,
                tax_names=tax_names,
                tax_embs_unit=tax_embs_unit,
                hnsw_index=hnsw_index,
                name_to_id=name_to_id,
                name_to_path=name_to_path,
                gloss_map=gloss_map,
                session=session,
                progress_hook=progress_hook,
            )

            pipeline.run_prune_workers()
            producer_task = asyncio.create_task(pipeline.queue_prune_jobs())

            try:
                await producer_task
                return await pipeline.finalise_results()
            except Exception:
                await pipeline.cancel_workers()
                raise

    try:
        rows = asyncio.run(_predict_all())
    except KeyboardInterrupt:
        sys.stderr.write("\nEvaluation cancelled.\n")
        raise
    except RuntimeError as exc:
        if "asyncio.run()" in str(exc) and "running event loop" in str(exc):
            raise RuntimeError(
                "run_label_benchmark must be called from a synchronous context "
                "without an active event loop."
            ) from exc
        raise

    return list(rows)
