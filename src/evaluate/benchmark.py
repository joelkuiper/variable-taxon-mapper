"""Public entry points for running the evaluation benchmark."""

from __future__ import annotations

import random
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    EvaluationConfig,
    HttpConfig,
    LLMConfig,
    ParallelismConfig,
    PruningConfig,
    coerce_config,
    coerce_eval_config,
)

from ..embedding import Embedder
from ..utils import clean_str_or_none, split_keywords_comma
from .collector import collect_predictions
from .metrics import summarise_dataframe
from .types import PredictionJob, ProgressHook


logger = logging.getLogger(__name__)


def _dedupe_variables(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    dedupe_on = [col for col in columns if col]
    if not dedupe_on:
        return df.copy()

    missing = [col for col in dedupe_on if col not in df.columns]
    if missing:
        raise KeyError(f"dedupe_on columns missing: {missing}")

    return df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(drop=True)


def _select_indices_for_evaluation(
    eligible_idxs: List[int],
    *,
    seed: Optional[int],
    limit: Optional[int],
) -> List[int]:
    if not eligible_idxs:
        return []
    rng = random.Random(seed)
    rng.shuffle(eligible_idxs)
    if isinstance(limit, int) and limit > 0:
        return eligible_idxs[: min(limit, len(eligible_idxs))]
    return eligible_idxs


def _iter_prediction_jobs(
    df: pd.DataFrame,
    idxs: Sequence[int],
    *,
    token_sets: Optional[pd.Series],
    known_labels: Set[str],
    parallel_cfg: ParallelismConfig,
    evaluate: bool,
) -> List[PredictionJob]:
    jobs: List[PredictionJob] = []
    slot_base = max(1, parallel_cfg.num_slots)

    for j, idx in enumerate(idxs):
        row = df.loc[idx]
        item = {
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "name": row.get("name"),
            "description": row.get("description"),
        }

        gold_labels: Optional[Sequence[str]] = None
        if evaluate and token_sets is not None:
            gold_tokens = token_sets.loc[idx]
            gold_labels = sorted(gold_tokens & known_labels)

        slot_id = j % slot_base
        metadata: Dict[str, Any] = {
            "dataset": item.get("dataset"),
            "label": item.get("label"),
            "name": item.get("name"),
            "description": item.get("description"),
            "_idx": idx,
            "_j": j,
            "_slot": slot_id,
            "_error": None,
        }

        jobs.append(
            PredictionJob(
                item=item,
                slot_id=slot_id,
                metadata=metadata,
                gold_labels=gold_labels,
            )
        )

    return jobs


def _create_progress_hook(total_jobs: int) -> tuple[ProgressHook, tqdm]:
    progress_bar = tqdm(total=total_jobs, desc="Evaluating", unit="rows")
    last_done = 0

    def _hook(
        done: int, total: int, correct_sum: Optional[int], elapsed: float
    ) -> None:
        nonlocal last_done
        if total == 0:
            return
        increment = done - last_done
        if increment > 0:
            progress_bar.update(increment)
            last_done = done
        if done and correct_sum is not None:
            accuracy = (correct_sum or 0) / done
            rows_per_sec = done / elapsed if elapsed > 0 else 0.0
            progress_bar.set_postfix(
                {"acc≈": f"{accuracy:.3f}", "rows/s": f"{rows_per_sec:.1f}"}
            )
        if done >= total:
            progress_bar.close()

    return _hook, progress_bar


def run_label_benchmark(
    variables: pd.DataFrame,
    keywords: pd.DataFrame,
    *,
    G,
    embedder: Embedder,
    tax_names: Sequence[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    gloss_map: Dict[str, str],
    eval_config: EvaluationConfig | Dict[str, Any] | None = None,
    pruning_config: PruningConfig | Dict[str, Any] | None = None,
    llm_config: LLMConfig | Dict[str, Any] | None = None,
    parallel_config: ParallelismConfig | Dict[str, Any] | None = None,
    http_config: HttpConfig | Dict[str, Any] | None = None,
    evaluate: bool = True,
    progress_hook: ProgressHook | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    logger.info(
        "Preparing benchmark: total_rows=%d, evaluate=%s",
        len(variables),
        evaluate,
    )
    cfg = coerce_eval_config(eval_config)
    pruning_cfg = coerce_config(pruning_config, PruningConfig, "pruning_config")
    llm_cfg = coerce_config(llm_config, LLMConfig, "llm_config")
    parallel_cfg = coerce_config(parallel_config, ParallelismConfig, "parallel_config")
    http_cfg = coerce_config(http_config, HttpConfig, "http_config")

    work_df = _dedupe_variables(variables, cfg.dedupe_on or [])
    total_rows = len(work_df)
    if total_rows != len(variables):
        logger.debug(
            "Deduplicated variables frame from %d to %d rows using columns %s",
            len(variables),
            total_rows,
            cfg.dedupe_on,
        )
    cleaned_kw = (
        work_df["keywords"].map(clean_str_or_none)
        if "keywords" in work_df.columns
        else pd.Series(index=work_df.index, dtype=object)
    )

    known_labels: Set[str] = set(tax_names)
    token_sets: Optional[pd.Series] = None
    total_with_any_keyword = int(cleaned_kw.notna().sum()) if total_rows else 0
    n_excluded_not_in_taxonomy = 0

    if evaluate:
        if "keywords" not in work_df.columns:
            raise KeyError("variables must have a 'keywords' column")
        token_lists = cleaned_kw.map(split_keywords_comma)
        token_sets = token_lists.map(lambda values: {val for val in values if val})
        eligible_mask = token_sets.map(lambda values: len(values & known_labels) > 0)
        n_eligible = int(eligible_mask.sum())
        n_excluded_not_in_taxonomy = total_with_any_keyword - n_eligible
        if n_eligible == 0:
            raise ValueError(
                "No eligible rows: no comma-split keywords present in the taxonomy."
            )
        eligible_idxs = list(work_df.index[eligible_mask])
        idxs = _select_indices_for_evaluation(eligible_idxs, seed=cfg.seed, limit=cfg.n)
        logger.info(
            "Selected %d eligible rows for evaluation (%d excluded, limit=%s)",
            len(idxs),
            n_excluded_not_in_taxonomy,
            cfg.n,
        )
    else:
        idxs = list(work_df.index)
        if isinstance(cfg.n, int) and cfg.n > 0:
            idxs = idxs[: min(cfg.n, len(idxs))]
        n_eligible = len(idxs)
        logger.info("Evaluation disabled; predicting %d rows", n_eligible)

    jobs = _iter_prediction_jobs(
        work_df,
        idxs,
        token_sets=token_sets,
        known_labels=known_labels,
        parallel_cfg=parallel_cfg,
        evaluate=evaluate,
    )
    logger.info("Created %d prediction jobs", len(jobs))

    default_progress_hook: ProgressHook | None = None
    progress_bar: tqdm | None = None
    if progress_hook is None and evaluate:
        default_progress_hook, progress_bar = _create_progress_hook(len(jobs))
        logger.debug("Initialized default progress hook for %d jobs", len(jobs))

    rows = collect_predictions(
        jobs,
        pruning_cfg=pruning_cfg,
        llm_cfg=llm_cfg,
        parallel_cfg=parallel_cfg,
        http_cfg=http_cfg,
        keywords=keywords,
        graph=G,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs_unit=tax_embs_unit,
        hnsw_index=hnsw_index,
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        gloss_map=gloss_map,
        progress_hook=progress_hook or default_progress_hook,
    )

    if progress_bar is not None:
        progress_bar.close()

    rows.sort(key=lambda row: row.get("_j", 0))
    for row in rows:
        row.pop("_j", None)
        row.pop("_idx", None)
        row.pop("_slot", None)

    df = pd.DataFrame(rows)
    logger.info("Collected %d prediction rows", len(df))

    metrics = summarise_dataframe(
        df,
        evaluate=evaluate,
        total_rows=total_rows,
        total_with_any_keyword=total_with_any_keyword,
        n_eligible=n_eligible,
        n_excluded_not_in_taxonomy=n_excluded_not_in_taxonomy,
    )
    logger.debug("Computed metrics: %s", metrics)

    return df, metrics
