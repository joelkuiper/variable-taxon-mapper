"""Evaluation helpers for running llama.cpp benchmarks."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

import aiohttp
import numpy as np
import pandas as pd

from config import EvaluationConfig

from .embedding import Embedder
from .matching import match_item_to_tree
from .taxonomy_pruning import pruned_tree_markdown_for_item
from .taxonomy import is_ancestor_of
from .utils import clean_str_or_none, split_keywords_comma


ProgressHook = Callable[[int, int, Optional[int], float], None]


@dataclass
class PredictionJob:
    item: Dict[str, Optional[str]]
    slot_id: int
    metadata: Dict[str, Any]
    gold_labels: Optional[List[str]] = None


def _sock_read_timeout(cfg: EvaluationConfig) -> float:
    base = float(cfg.http_sock_read_floor)
    return max(base, float(cfg.n_predict))


def _collect_predictions(
    jobs: Sequence[PredictionJob],
    *,
    cfg: EvaluationConfig,
    keywords: pd.DataFrame,
    G,
    embedder: Embedder,
    tax_names: List[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    gloss_map: Dict[str, str],
    progress_hook: ProgressHook | None,
) -> List[Dict[str, Any]]:
    async def _predict_all() -> List[Dict[str, Any]]:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None,
            sock_connect=float(cfg.http_sock_connect),
            sock_read=_sock_read_timeout(cfg),
        )
        connector = aiohttp.TCPConnector(
            limit=max(1, cfg.pool_maxsize), limit_per_host=max(1, cfg.pool_maxsize)
        )

        rows: List[Dict[str, Any]] = []
        start = time.time()
        correct_sum = 0
        total = len(jobs)

        async with aiohttp.ClientSession(
            timeout=timeout_cfg, connector=connector
        ) as session:
            for job in jobs:
                tree_markdown, allowed_labels = pruned_tree_markdown_for_item(
                    job.item,
                    G=G,
                    df=keywords,
                    embedder=embedder,
                    tax_names=tax_names,
                    tax_embs_unit=tax_embs_unit,
                    hnsw_index=hnsw_index,
                    name_col="name",
                    order_col="order",
                    top_k_nodes=cfg.top_k_nodes,
                    desc_max_depth=cfg.desc_max_depth,
                    max_total_nodes=cfg.max_total_nodes,
                    gloss_map=gloss_map,
                    anchor_overfetch_mult=cfg.anchor_overfetch_mult,
                    anchor_min_overfetch=cfg.anchor_min_overfetch,
                    candidate_list_max_items=cfg.candidate_list_max_items,
                    lexical_anchor_count=cfg.lexical_anchor_count,
                    community_k=cfg.community_k,
                    community_max_size=cfg.community_max_size,
                )

                allowed_has_gold: Optional[bool] = None
                if job.gold_labels is not None:
                    if allowed_labels:
                        allowed_lookup = set(allowed_labels)
                        allowed_has_gold = any(
                            g in allowed_lookup for g in job.gold_labels if g
                        )
                    else:
                        allowed_has_gold = False

                match_kwargs = {}
                if cfg.llm_grammar is not None:
                    match_kwargs["grammar"] = cfg.llm_grammar

                pred = await match_item_to_tree(
                    job.item,
                    tree_markdown=tree_markdown,
                    allowed_labels=allowed_labels,
                    name_to_id=name_to_id,
                    name_to_path=name_to_path,
                    tax_names=tax_names,
                    tax_embs=tax_embs_unit,
                    embedder=embedder,
                    hnsw_index=hnsw_index,
                    endpoint=cfg.endpoint,
                    n_predict=cfg.n_predict,
                    temperature=cfg.temperature,
                    slot_id=job.slot_id,
                    cache_prompt=cfg.llm_cache_prompt,
                    n_keep=cfg.llm_n_keep,
                    top_k=cfg.llm_top_k,
                    top_p=cfg.llm_top_p,
                    min_p=cfg.llm_min_p,
                    session=session,
                    **match_kwargs,
                )

                resolved_label = pred.get("resolved_label")
                result: Dict[str, Any] = dict(job.metadata)
                result.update(
                    {
                        "pred_label_raw": pred.get("pred_label_raw"),
                        "resolved_label": resolved_label,
                        "resolved_id": pred.get("resolved_id"),
                        "resolved_path": pred.get("resolved_path"),
                        "_error": job.metadata.get("_error"),
                    }
                )

                if "raw" in pred:
                    result["raw"] = pred["raw"]

                if job.gold_labels is not None:
                    match_type = determine_match_type(
                        resolved_label, job.gold_labels, G=G
                    )
                    correct = match_type != "none"
                    result["gold_labels"] = job.gold_labels
                    result["match_type"] = match_type
                    result["correct"] = bool(correct)
                    allowed_has_gold_flag = bool(allowed_has_gold)
                    result["possible_correct_under_allowed"] = allowed_has_gold_flag
                    result["allowed_subtree_contains_gold"] = allowed_has_gold_flag
                    correct_sum += 1 if correct else 0

                rows.append(result)

                if progress_hook is not None:
                    elapsed = max(time.time() - start, 0.0)
                    hook_correct = correct_sum if job.gold_labels is not None else None
                    progress_hook(len(rows), total, hook_correct, elapsed)

        return rows

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


def determine_match_type(
    pred_label: Optional[str], gold_labels: List[str], *, G
) -> str:
    if not isinstance(pred_label, str):
        return "none"
    for g in gold_labels:
        if not isinstance(g, str):
            continue
        if pred_label == g:
            return "exact"
        if is_ancestor_of(G, pred_label, g):
            return "ancestor"
        if is_ancestor_of(G, g, pred_label):
            return "descendant"
    return "none"


def is_correct_prediction(
    pred_label: Optional[str], gold_labels: List[str], *, G
) -> bool:
    return determine_match_type(pred_label, gold_labels, G=G) != "none"


def _coerce_eval_config(
    config: EvaluationConfig | Mapping[str, Any] | None,
) -> EvaluationConfig:
    if config is None:
        return EvaluationConfig()
    if isinstance(config, EvaluationConfig):
        return config
    if isinstance(config, Mapping):
        return EvaluationConfig(**config)
    raise TypeError(
        "eval_config must be an EvaluationConfig or a mapping of keyword arguments"
    )


def run_label_benchmark(
    variables: pd.DataFrame,
    keywords: pd.DataFrame,
    *,
    G,
    embedder: Embedder,
    tax_names: List[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    gloss_map: Dict[str, str],
    eval_config: EvaluationConfig | Mapping[str, Any] | None = None,
    evaluate: bool = True,
    progress_hook: ProgressHook | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = _coerce_eval_config(eval_config)

    dedupe_on = [c for c in (cfg.dedupe_on or []) if c]

    work_df = variables.copy()
    if dedupe_on:
        missing = [c for c in dedupe_on if c not in work_df.columns]
        if missing:
            raise KeyError(f"dedupe_on columns missing: {missing}")
        work_df = work_df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(
            drop=True
        )

    total_rows = len(work_df)
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
        token_sets = token_lists.map(lambda lst: set(t for t in lst if t))
        eligible_mask = token_sets.map(lambda st: len(st & known_labels) > 0)
        n_eligible = int(eligible_mask.sum())
        n_excluded_not_in_taxonomy = total_with_any_keyword - n_eligible
        if n_eligible == 0:
            raise ValueError(
                "No eligible rows: no comma-split keywords present in the taxonomy."
            )
        eligible_idxs = list(work_df.index[eligible_mask])
        rnd = random.Random(cfg.seed)
        rnd.shuffle(eligible_idxs)
        limit = (
            min(cfg.n, len(eligible_idxs))
            if isinstance(cfg.n, int) and cfg.n > 0
            else len(eligible_idxs)
        )
        idxs = eligible_idxs[:limit]
    else:
        token_sets = None
        idxs = list(work_df.index)
        if isinstance(cfg.n, int) and cfg.n > 0:
            idxs = idxs[: min(cfg.n, len(idxs))]
        n_eligible = len(idxs)

    jobs: List[PredictionJob] = []
    slot_base = max(1, cfg.num_slots)

    for j, i in enumerate(idxs):
        row = work_df.loc[i]
        item = {
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "name": row.get("name"),
            "description": row.get("description"),
        }

        gold_labels: Optional[List[str]] = None
        if evaluate and token_sets is not None:
            gold_tokens = token_sets.loc[i]
            gold_labels = sorted(gold_tokens & known_labels)

        slot_id = j % slot_base
        metadata: Dict[str, Any] = {
            "dataset": item.get("dataset"),
            "label": item.get("label"),
            "name": item.get("name"),
            "description": item.get("description"),
            "_idx": i,
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

    default_progress_hook: ProgressHook | None = None
    if progress_hook is None and evaluate:
        interval = max(1, int(cfg.progress_log_interval))

        def _default_progress(
            done: int, total: int, correct_sum: Optional[int], elapsed: float
        ) -> None:
            if not total:
                return
            if done % interval != 0 and done != total:
                return
            acc = (
                (correct_sum or 0) / done if correct_sum is not None and done else 0.0
            )
            rps = done / elapsed if elapsed > 0 else 0.0
            sys.stderr.write(
                f"\rEvaluating: {done}/{total} (acc≈{acc:.3f}, {rps:.1f} rows/s)"
            )
            if done == total:
                sys.stderr.write("\n")
            sys.stderr.flush()

        default_progress_hook = _default_progress

    rows = _collect_predictions(
        jobs,
        cfg=cfg,
        keywords=keywords,
        G=G,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs_unit=tax_embs_unit,
        hnsw_index=hnsw_index,
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        gloss_map=gloss_map,
        progress_hook=progress_hook or default_progress_hook,
    )

    rows.sort(key=lambda r: r.get("_j", 0))
    for r in rows:
        r.pop("_j", None)
        r.pop("_idx", None)
        r.pop("_slot", None)

    df = pd.DataFrame(rows)
    total_processed = int(len(df))

    metrics: Dict[str, Any] = {
        "n_total_rows_after_dedupe": int(total_rows),
        "n_with_any_keyword": int(total_with_any_keyword),
        "n_eligible": int(n_eligible),
        "n_excluded_not_in_taxonomy": int(n_excluded_not_in_taxonomy),
        "n_evaluated": total_processed,
        "n_errors": int(df["_error"].notna().sum()) if "_error" in df.columns else 0,
    }

    if evaluate and "correct" in df.columns and total_processed:
        metrics["n_correct"] = int(df["correct"].sum())
        metrics["label_accuracy_any_match"] = float(df["correct"].mean())

        if "possible_correct_under_allowed" in df.columns:
            possible_series = (
                df["possible_correct_under_allowed"].fillna(False).astype(bool)
            )
            metrics["n_possible_correct_under_allowed"] = int(possible_series.sum())
            metrics["possible_correct_under_allowed_rate"] = float(
                possible_series.mean()
            )

        if "match_type" in df.columns:
            match_counts_series = df["match_type"].value_counts(sort=False)
            match_counts = {str(k): int(v) for k, v in match_counts_series.items()}
            metrics["match_type_counts"] = match_counts
            metrics["match_type_rates"] = {
                str(k): float(v / total_processed) for k, v in match_counts.items()
            }
            metrics["label_accuracy_exact_only"] = float(
                match_counts.get("exact", 0) / total_processed
            )
            metrics["label_accuracy_ancestor_only"] = float(
                match_counts.get("ancestor", 0) / total_processed
            )
            metrics["label_accuracy_descendant_only"] = float(
                match_counts.get("descendant", 0) / total_processed
            )
            metrics["n_unmatched"] = int(match_counts.get("none", 0))

    return df, metrics
