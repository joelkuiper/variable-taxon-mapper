"""Helpers to orchestrate asynchronous prediction collection."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import HttpConfig, LLMConfig, ParallelismConfig, PruningConfig

try:  # pragma: no cover - tqdm provides a visual aid only
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing/broken
    tqdm = None  # type: ignore[assignment]

from ..embedding import Embedder
from ..graph_utils import compute_node_depths, get_undirected_taxonomy
from ..prompts import PromptRenderer
from .prediction_pipeline import PredictionPipeline
from .types import PredictionJob, ProgressHook


logger = logging.getLogger(__name__)


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
    prompt_renderer: PromptRenderer,
) -> List[Dict[str, Any]]:
    logger.info("Collecting predictions for %d jobs", len(jobs))
    undirected_graph = get_undirected_taxonomy(graph) if graph is not None else None
    depth_map = compute_node_depths(graph) if graph is not None else {}

    progress_hook, progress_cleanup = _build_progress_hook(progress_hook, len(jobs))

    async def _predict_all() -> List[Dict[str, Any]]:
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
            progress_hook=progress_hook,
            prompt_renderer=prompt_renderer,
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
            return await pipeline.finalise_results()
        except Exception:
            logger.exception("Prediction pipeline failed; cancelling workers")
            await pipeline.cancel_workers()
            raise

    try:
        rows = asyncio.run(_predict_all())
    except KeyboardInterrupt:
        logger.warning("Evaluation cancelled by user")
        raise
    except RuntimeError as exc:
        if "asyncio.run()" in str(exc) and "running event loop" in str(exc):
            raise RuntimeError(
                "run_label_benchmark must be called from a synchronous context "
                "without an active event loop."
            ) from exc
        raise
    finally:
        progress_cleanup()

    logger.info("Finished collecting predictions (%d rows)", len(rows))
    return list(rows)


def _build_progress_hook(
    progress_hook: ProgressHook | None, total_jobs: int
) -> Tuple[ProgressHook | None, Callable[[], None]]:
    """Create a logging progress hook when one isn't provided."""

    if progress_hook is not None or total_jobs <= 0:
        return progress_hook, lambda: None

    log_interval = max(1, total_jobs // 20)
    last_logged = {"count": 0, "time": time.time()}

    bar = tqdm(total=total_jobs, desc="Collecting predictions", unit="job") if tqdm else None

    def _cleanup() -> None:
        if bar is not None:
            bar.close()

    def _hook(completed: int, total: int, correct: int | None, elapsed: float) -> None:
        if bar is not None:
            bar.n = completed
            bar.refresh()

        should_log = completed == total
        if not should_log:
            delta = completed - last_logged["count"]
            time_delta = time.time() - last_logged["time"]
            if delta >= log_interval or time_delta >= 30.0:
                should_log = True

        if should_log:
            pct = (completed / total) * 100 if total else 0.0
            correct_part = (
                f", gold_correct={correct}" if correct is not None else ""
            )
            logger.info(
                "Prediction progress: %d/%d (%.1f%%) elapsed=%.1fs%s",
                completed,
                total,
                pct,
                elapsed,
                correct_part,
            )
            last_logged["count"] = completed
            last_logged["time"] = time.time()

    return _hook, _cleanup
