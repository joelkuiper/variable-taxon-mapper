"""Evaluation helpers for running llama.cpp benchmarks."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

import aiohttp
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import EvaluationConfig

from .embedding import Embedder
from .matching import match_item_to_tree
from .taxonomy_pruning import pruned_tree_markdown_for_item
from .taxonomy import is_ancestor_of
from .utils import clean_str_or_none, split_keywords_comma


ProgressHook = Callable[[int, int, Optional[int], float], None]


def _compute_node_depths(G) -> Dict[str, Optional[int]]:
    depth_map: Dict[str, Optional[int]] = {}
    if G is None:
        return depth_map

    try:
        topo_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo_nodes = list(G.nodes())

    for node in topo_nodes:
        preds = list(G.predecessors(node))
        if not preds:
            depth_map[node] = 0
            continue
        parent = preds[0]
        parent_depth = depth_map.get(parent)
        depth_map[node] = parent_depth + 1 if parent_depth is not None else None

    for node in G.nodes():
        if node not in depth_map:
            depth_map[node] = 0

    return depth_map


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
    undirected_G = nx.Graph(G) if G is not None else None
    depth_map = _compute_node_depths(G) if G is not None else {}

    async def _predict_all() -> List[Dict[str, Any]]:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None,
            sock_connect=float(cfg.http_sock_connect),
            sock_read=_sock_read_timeout(cfg),
        )
        connector = aiohttp.TCPConnector(
            limit=max(1, cfg.pool_maxsize), limit_per_host=max(1, cfg.pool_maxsize)
        )

        total = len(jobs)
        rows: List[Optional[Dict[str, Any]]] = [None] * total
        start = time.time()
        correct_sum = 0
        gold_progress_seen = False

        slot_limit = max(1, getattr(cfg, "num_slots", 1) or 1)
        pool_limit = max(1, getattr(cfg, "pool_maxsize", slot_limit) or slot_limit)
        concurrency_limit = max(1, min(slot_limit, pool_limit))
        semaphore = asyncio.Semaphore(concurrency_limit)

        async with aiohttp.ClientSession(
            timeout=timeout_cfg, connector=connector
        ) as session:
            async def _predict_job(
                idx: int, job: PredictionJob
            ) -> tuple[int, Dict[str, Any], int, bool]:
                async with semaphore:
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
                        anchor_top_k=cfg.anchor_top_k,
                        max_descendant_depth=cfg.max_descendant_depth,
                        node_budget=cfg.node_budget,
                        gloss_map=gloss_map,
                        anchor_overfetch_multiplier=cfg.anchor_overfetch_multiplier,
                        anchor_min_overfetch=cfg.anchor_min_overfetch,
                        suggestion_list_limit=cfg.suggestion_list_limit,
                        lexical_anchor_limit=cfg.lexical_anchor_limit,
                        community_clique_size=cfg.community_clique_size,
                        max_community_size=cfg.max_community_size,
                        pagerank_damping=cfg.pagerank_damping,
                        pagerank_score_floor=cfg.pagerank_score_floor,
                        pagerank_candidate_limit=cfg.pagerank_candidate_limit,
                        enable_taxonomy_pruning=cfg.enable_taxonomy_pruning,
                        tree_sort_mode=cfg.tree_sort_mode,
                        pruning_mode=cfg.pruning_mode,
                        similarity_threshold=cfg.similarity_threshold,
                        pruning_radius=cfg.pruning_radius,
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
                            "match_strategy": pred.get("match_strategy"),
                        }
                    )

                    if "raw" in pred:
                        result["raw"] = pred["raw"]

                    correct_increment = 0
                    has_gold_labels = job.gold_labels is not None
                    if has_gold_labels:
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
                        correct_increment = 1 if correct else 0

                        min_distance: Optional[int] = None
                        min_depth_delta: Optional[int] = None
                        min_gold_depth: Optional[int] = None
                        min_gold_label: Optional[str] = None
                        pred_depth: Optional[int] = None

                        if (
                            undirected_G is not None
                            and isinstance(resolved_label, str)
                            and undirected_G.has_node(resolved_label)
                        ):
                            pred_depth = depth_map.get(resolved_label)
                            gold_candidates = [
                                g
                                for g in (job.gold_labels or [])
                                if isinstance(g, str) and undirected_G.has_node(g)
                            ]

                            for gold_label in gold_candidates:
                                try:
                                    distance = nx.shortest_path_length(
                                        undirected_G, resolved_label, gold_label
                                    )
                                except (nx.NetworkXNoPath, nx.NodeNotFound):
                                    continue

                                gold_depth = depth_map.get(gold_label)
                                new_depth_delta: Optional[int] = None
                                if (
                                    pred_depth is not None
                                    and gold_depth is not None
                                ):
                                    new_depth_delta = pred_depth - gold_depth

                                prefer = False
                                if min_distance is None or distance < min_distance:
                                    prefer = True
                                elif min_distance is not None and distance == min_distance:
                                    if (
                                        new_depth_delta is not None
                                        and (
                                            min_depth_delta is None
                                            or abs(new_depth_delta)
                                            < abs(min_depth_delta)
                                        )
                                    ):
                                        prefer = True

                                if prefer:
                                    min_distance = int(distance)
                                    min_depth_delta = (
                                        int(new_depth_delta)
                                        if new_depth_delta is not None
                                        else None
                                    )
                                    min_gold_depth = (
                                        int(gold_depth)
                                        if gold_depth is not None
                                        else None
                                    )
                                    min_gold_label = gold_label

                        result["hierarchical_distance_min"] = (
                            int(min_distance) if min_distance is not None else None
                        )
                        result["hierarchical_distance_pred_depth"] = (
                            int(pred_depth) if pred_depth is not None else None
                        )
                        result["hierarchical_distance_gold_depth_at_min"] = (
                            int(min_gold_depth) if min_gold_depth is not None else None
                        )
                        result["hierarchical_distance_depth_delta"] = (
                            int(min_depth_delta)
                            if min_depth_delta is not None
                            else None
                        )
                        result["hierarchical_distance_gold_label_at_min"] = (
                            str(min_gold_label) if min_gold_label is not None else None
                        )

                    return idx, result, correct_increment, has_gold_labels

            tasks = [
                asyncio.create_task(_predict_job(idx, job)) for idx, job in enumerate(jobs)
            ]

            completed = 0

            for task in asyncio.as_completed(tasks):
                idx, result, correct_increment, has_gold_labels = await task
                rows[idx] = result
                completed += 1
                if has_gold_labels:
                    gold_progress_seen = True
                    correct_sum += correct_increment

                if progress_hook is not None:
                    elapsed = max(time.time() - start, 0.0)
                    hook_correct = correct_sum if gold_progress_seen else None
                    progress_hook(completed, total, hook_correct, elapsed)

            for task in tasks:
                task.result()

        return [r for r in rows if r is not None]

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
    progress_bar: tqdm | None = None
    if progress_hook is None and evaluate:
        total_jobs = len(jobs)
        progress_bar = tqdm(total=total_jobs, desc="Evaluating", unit="rows")
        last_done = 0

        def _default_progress(
            done: int, total: int, correct_sum: Optional[int], elapsed: float
        ) -> None:
            nonlocal last_done
            if progress_bar is None or total == 0:
                return
            increment = done - last_done
            if increment > 0:
                progress_bar.update(increment)
                last_done = done
            if done and correct_sum is not None:
                acc = (correct_sum or 0) / done
                rps = done / elapsed if elapsed > 0 else 0.0
                progress_bar.set_postfix({"acc≈": f"{acc:.3f}", "rows/s": f"{rps:.1f}"})
            if done >= total:
                progress_bar.close()

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

    if progress_bar is not None:
        progress_bar.close()

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

    if "match_strategy" in df.columns:
        strategy_series = df["match_strategy"].fillna("unknown").astype(str)
        df["match_strategy"] = strategy_series
        strategy_counts = strategy_series.value_counts(sort=False)
        metrics["match_strategy_volume"] = {
            str(k): int(v) for k, v in strategy_counts.items()
        }
    else:
        strategy_series = None

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

        if strategy_series is not None:
            strategy_stats: Dict[str, Dict[str, float | int]] = {}
            grouped = df.groupby("match_strategy", dropna=False)
            for strat, group in grouped:
                correct_series = group["correct"].dropna()
                correct_count = int(correct_series.sum())
                denom = len(correct_series)
                accuracy = float(correct_count / denom) if denom else 0.0
                strategy_stats[str(strat)] = {
                    "n": int(len(group)),
                    "n_correct": correct_count,
                    "accuracy": accuracy,
                }

            metrics["match_strategy_performance"] = strategy_stats

            total_correct = float(df["correct"].sum())
            if total_correct > 0:
                metrics["match_strategy_correct_share"] = {
                    strat: float(stats["n_correct"] / total_correct)
                    for strat, stats in strategy_stats.items()
                }

    if "hierarchical_distance_min" in df.columns:
        distance_series = pd.to_numeric(
            df["hierarchical_distance_min"], errors="coerce"
        )
        distance_nonnull = distance_series.dropna()
        metrics["hierarchical_distance_count"] = int(distance_nonnull.shape[0])
        if not distance_nonnull.empty:
            metrics["hierarchical_distance_min_mean"] = float(distance_nonnull.mean())
            metrics["hierarchical_distance_min_median"] = float(
                distance_nonnull.median()
            )
            metrics["hierarchical_distance_within_1_rate"] = float(
                (distance_nonnull <= 1).mean()
            )
            metrics["hierarchical_distance_within_2_rate"] = float(
                (distance_nonnull <= 2).mean()
            )

        if "correct" in df.columns:
            incorrect_mask = df["correct"] == False
            incorrect_distances = distance_series[incorrect_mask].dropna()
            metrics["hierarchical_distance_error_count"] = int(
                incorrect_distances.shape[0]
            )
            if not incorrect_distances.empty:
                metrics["hierarchical_distance_error_mean"] = float(
                    incorrect_distances.mean()
                )
                metrics["hierarchical_distance_error_median"] = float(
                    incorrect_distances.median()
                )
                metrics["hierarchical_distance_error_within_1_rate"] = float(
                    (incorrect_distances <= 1).mean()
                )
                metrics["hierarchical_distance_error_within_2_rate"] = float(
                    (incorrect_distances <= 2).mean()
                )

    return df, metrics
