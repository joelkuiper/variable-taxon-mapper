"""Helpers to orchestrate asynchronous prediction collection."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from vtm.config import (
    FieldMappingConfig,
    HNSWConfig,
    HttpConfig,
    LLMConfig,
    LlamaCppConfig,
    PostprocessingConfig,
    ParallelismConfig,
    PruningConfig,
)

from ..embedding import Embedder
from ..graph_utils import compute_node_depths, get_undirected_taxonomy
from ..prompts import PromptRenderer
from .prediction_pipeline import PredictionPipeline
from .types import PredictionJob, ProgressHook


logger = logging.getLogger(__name__)


async def async_collect_predictions(
    jobs: Sequence[PredictionJob],
    *,
    pruning_cfg: PruningConfig,
    llm_cfg: LLMConfig,
    llama_cpp_cfg: LlamaCppConfig,
    postprocessing_cfg: PostprocessingConfig,
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
    hnsw_config: HNSWConfig,
    progress_hook: ProgressHook | None,
    prompt_renderer: PromptRenderer,
    field_mapping: FieldMappingConfig | None = None,
    output_column_prefix: str | None = None,
) -> List[Dict[str, Any]]:
    """Asynchronously collect predictions for ``jobs`` using the pipeline."""

    logger.info("Collecting predictions for %d jobs", len(jobs))
    undirected_graph = get_undirected_taxonomy(graph) if graph is not None else None
    depth_map = compute_node_depths(graph) if graph is not None else {}

    pipeline = PredictionPipeline(
        jobs=jobs,
        pruning_cfg=pruning_cfg,
        llm_cfg=llm_cfg,
        llama_cpp_cfg=llama_cpp_cfg,
        postprocessing_cfg=postprocessing_cfg,
        http_cfg=http_cfg,
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
        hnsw_config=hnsw_config,
        progress_hook=progress_hook,
        prompt_renderer=prompt_renderer,
        field_mapping=field_mapping,
        output_column_prefix=output_column_prefix,
    )

    pipeline.run_prune_workers()
    logger.debug(
        "Started %d prune workers with batch size %d",
        pipeline._prune_workers,
        pipeline._prune_batch_size,
    )
    producer_task = asyncio.create_task(pipeline.queue_prune_jobs())
    logger.debug("Queued prune jobs task created")

    try:
        await producer_task
        logger.debug("All prune jobs queued; awaiting final results")
        rows = await pipeline.finalise_results()
    except Exception:
        logger.exception("Prediction pipeline failed; cancelling workers")
        await pipeline.cancel_workers()
        raise

    logger.info("Finished collecting predictions (%d rows)", len(rows))
    return list(rows)


def collect_predictions(
    jobs: Sequence[PredictionJob],
    *,
    pruning_cfg: PruningConfig,
    llm_cfg: LLMConfig,
    llama_cpp_cfg: LlamaCppConfig,
    postprocessing_cfg: PostprocessingConfig,
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
    hnsw_config: HNSWConfig,
    progress_hook: ProgressHook | None,
    prompt_renderer: PromptRenderer,
    field_mapping: FieldMappingConfig | None = None,
    output_column_prefix: str | None = None,
) -> List[Dict[str, Any]]:
    """Synchronously collect predictions, delegating to ``async_collect_predictions``."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        raise RuntimeError(
            "collect_predictions detected an active event loop. "
            "Await async_collect_predictions(...) instead of using the synchronous helper."
        )

    try:
        return asyncio.run(
            async_collect_predictions(
                jobs,
                pruning_cfg=pruning_cfg,
                llm_cfg=llm_cfg,
                llama_cpp_cfg=llama_cpp_cfg,
                postprocessing_cfg=postprocessing_cfg,
                parallel_cfg=parallel_cfg,
                http_cfg=http_cfg,
                keywords=keywords,
                graph=graph,
                embedder=embedder,
                tax_names=tax_names,
                tax_embs_unit=tax_embs_unit,
                hnsw_index=hnsw_index,
                name_to_id=name_to_id,
                name_to_path=name_to_path,
                gloss_map=gloss_map,
                hnsw_config=hnsw_config,
                progress_hook=progress_hook,
                prompt_renderer=prompt_renderer,
                field_mapping=field_mapping,
                output_column_prefix=output_column_prefix,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Evaluation cancelled by user")
        raise
