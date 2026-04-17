"""Metrics and scoring helpers for evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import networkx as nx
import pandas as pd

from ..taxonomy import is_ancestor_of
from ..utils import apply_output_prefix
from ..pruning import PrunedTreeResult
from ..graph_utils import lookup_direct_parent
from .types import PredictionJob


def determine_match_type(
    pred_label: Optional[str], gold_labels: Sequence[str], *, G
) -> str:
    if not isinstance(pred_label, str):
        return "none"
    for gold in gold_labels:
        if not isinstance(gold, str):
            continue
        if pred_label == gold:
            return "exact"
        if is_ancestor_of(G, pred_label, gold):
            return "ancestor"
        if is_ancestor_of(G, gold, pred_label):
            return "descendant"
    return "none"


def determine_match_type_any(
    pred_labels: Sequence[str], gold_labels: Sequence[str], *, G
) -> str:
    for pred_label in pred_labels:
        match_type = determine_match_type(pred_label, gold_labels, G=G)
        if match_type == "exact":
            return "exact"
    for pred_label in pred_labels:
        match_type = determine_match_type(pred_label, gold_labels, G=G)
        if match_type == "ancestor":
            return "ancestor"
    for pred_label in pred_labels:
        match_type = determine_match_type(pred_label, gold_labels, G=G)
        if match_type == "descendant":
            return "descendant"
    return "none"


def is_correct_prediction(
    pred_label: Optional[str], gold_labels: Sequence[str], *, G
) -> bool:
    return determine_match_type(pred_label, gold_labels, G=G) != "none"


def _compute_hierarchical_distance(
    pred_label: Optional[str],
    gold_labels: Sequence[str],
    *,
    undirected_graph,
    depth_map: Mapping[str, Optional[int]],
) -> Dict[str, Optional[int | str]]:
    if (
        undirected_graph is None
        or not isinstance(pred_label, str)
        or not undirected_graph.has_node(pred_label)
    ):
        return {
            "hierarchical_distance_min": None,
            "hierarchical_distance_pred_depth": None,
            "hierarchical_distance_gold_depth_at_min": None,
            "hierarchical_distance_depth_delta": None,
            "hierarchical_distance_gold_label_at_min": None,
        }

    pred_depth = depth_map.get(pred_label)
    min_distance: Optional[int] = None
    min_depth_delta: Optional[int] = None
    min_gold_depth: Optional[int] = None
    min_gold_label: Optional[str] = None

    gold_candidates = [
        gold
        for gold in gold_labels
        if isinstance(gold, str) and undirected_graph.has_node(gold)
    ]
    for gold in gold_candidates:
        try:
            distance = nx.shortest_path_length(undirected_graph, pred_label, gold)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        gold_depth = depth_map.get(gold)
        depth_delta: Optional[int] = None
        if pred_depth is not None and gold_depth is not None:
            depth_delta = pred_depth - gold_depth

        prefer = False
        if min_distance is None or distance < min_distance:
            prefer = True
        elif (
            min_distance is not None
            and distance == min_distance
            and depth_delta is not None
            and (min_depth_delta is None or abs(depth_delta) < abs(min_depth_delta))
        ):
            prefer = True

        if prefer:
            min_distance = int(distance)
            min_depth_delta = int(depth_delta) if depth_delta is not None else None
            min_gold_depth = int(gold_depth) if gold_depth is not None else None
            min_gold_label = gold

    return {
        "hierarchical_distance_min": int(min_distance)
        if min_distance is not None
        else None,
        "hierarchical_distance_pred_depth": int(pred_depth)
        if pred_depth is not None
        else None,
        "hierarchical_distance_gold_depth_at_min": min_gold_depth,
        "hierarchical_distance_depth_delta": min_depth_delta,
        "hierarchical_distance_gold_label_at_min": min_gold_label,
    }


def build_result_row(
    job: PredictionJob,
    pruned: PrunedTreeResult,
    prediction: Mapping[str, Any],
    *,
    graph,
    undirected_graph,
    depth_map: Mapping[str, Optional[int]],
    column_prefix: Optional[str] = None,
) -> tuple[Dict[str, Any], int, bool]:
    allowed_labels = pruned.allowed_labels
    allowed_has_gold: Optional[bool] = None

    gold_labels_seq = list(job.gold_labels or [])
    if gold_labels_seq:
        if allowed_labels:
            allowed_lookup = set(allowed_labels)
            allowed_has_gold = any(
                gold in allowed_lookup for gold in gold_labels_seq if gold
            )
        else:
            allowed_has_gold = False

    resolved_raw = prediction.get("resolved_keywords")
    resolved_keywords: list[str] = []
    if isinstance(resolved_raw, (list, tuple, set)):
        resolved_keywords = [
            str(item).strip() for item in resolved_raw if item is not None
        ]
    elif isinstance(resolved_raw, str):
        resolved_keywords = [resolved_raw.strip()] if resolved_raw.strip() else []
    elif resolved_raw is not None:
        resolved_keywords = [str(resolved_raw).strip()]
    resolved_keywords = [label for label in resolved_keywords if label]
    primary_keyword = resolved_keywords[0] if resolved_keywords else None

    result: Dict[str, Any] = {}

    # Preserve the original input columns so the output dataframe is a superset
    # of the source data. ``item_columns`` contains the raw row values (plus a
    # dataset alias) whereas ``metadata`` may only carry a configured subset.
    if job.item_columns:
        result.update(job.item_columns)

    result.update(job.metadata)
    generated: Dict[str, Any] = {
        "resolved_keywords": resolved_keywords,
        "resolved_ids": prediction.get("resolved_ids"),
        "resolved_paths": prediction.get("resolved_paths"),
        "resolved_definitions": prediction.get("resolved_definitions"),
        "resolved_direct_parent": lookup_direct_parent(graph, primary_keyword),
        "_error": job.metadata.get("_error"),
        "match_strategy": prediction.get("match_strategy"),
        "embedding_similarity": prediction.get("embedding_similarity"),
        "embedding_score": prediction.get("embedding_score"),
    }

    has_gold_labels = bool(gold_labels_seq)
    correct_increment = 0
    if has_gold_labels:
        match_type = determine_match_type_any(
            resolved_keywords, gold_labels_seq, G=graph
        )
        correct = match_type != "none"
        generated.update(
            {
                "gold_labels": list(gold_labels_seq),
                "match_type": match_type,
                "correct": bool(correct),
                "possible_correct_under_allowed": bool(allowed_has_gold),
                "allowed_subtree_contains_gold": bool(allowed_has_gold),
            }
        )
        correct_increment = 1 if correct else 0

        generated.update(
            _compute_hierarchical_distance(
                primary_keyword,
                gold_labels_seq,
                undirected_graph=undirected_graph,
                depth_map=depth_map,
            )
        )
    if column_prefix:
        generated = {
            apply_output_prefix(column_prefix, key): value
            for key, value in generated.items()
        }
    result.update(generated)
    return result, correct_increment, has_gold_labels


def add_parent_column(
    df: pd.DataFrame, graph, *, column_prefix: Optional[str] = None
) -> None:
    resolved_col = apply_output_prefix(column_prefix, "resolved_keywords")
    if resolved_col not in df.columns:
        return
    output_col = apply_output_prefix(column_prefix, "direct_parent")
    df[output_col] = df[resolved_col].map(
        lambda labels: lookup_direct_parent(graph, labels[0])
        if isinstance(labels, list) and labels
        else lookup_direct_parent(graph, labels)
    )


def summarise_dataframe(
    df: pd.DataFrame,
    *,
    evaluate: bool,
    total_rows: int,
    total_with_any_gold_label: int,
    n_eligible: int,
    n_excluded_not_in_taxonomy: int,
    column_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    total_processed = int(len(df))
    metrics: Dict[str, Any] = {
        "n_total_rows_after_dedupe": int(total_rows),
        "n_with_any_gold_label": int(total_with_any_gold_label),
        "n_eligible": int(n_eligible),
        "n_excluded_not_in_taxonomy": int(n_excluded_not_in_taxonomy),
        "n_evaluated": total_processed,
        "n_errors": int(df["_error"].notna().sum()) if "_error" in df.columns else 0,
    }

    strategy_series = None
    strategy_col = apply_output_prefix(column_prefix, "match_strategy")
    if strategy_col in df.columns:
        strategy_series = df[strategy_col].fillna("unknown").astype(str)
        df[strategy_col] = strategy_series
        strategy_counts = strategy_series.value_counts(sort=False)
        metrics["match_strategy_volume"] = {
            str(key): int(value) for key, value in strategy_counts.items()
        }

    correct_col = apply_output_prefix(column_prefix, "correct")
    if evaluate and correct_col in df.columns and total_processed:
        metrics["n_correct"] = int(df[correct_col].sum())
        metrics["label_accuracy_any_match"] = float(df[correct_col].mean())

        possible_col = apply_output_prefix(column_prefix, "possible_correct_under_allowed")
        if possible_col in df.columns:
            possible_series = (
                df[possible_col].fillna(False).astype(bool)
            )
            metrics["n_possible_correct_under_allowed"] = int(possible_series.sum())
            metrics["possible_correct_under_allowed_rate"] = float(
                possible_series.mean()
            )

        match_col = apply_output_prefix(column_prefix, "match_type")
        if match_col in df.columns:
            match_counts_series = df[match_col].value_counts(sort=False)
            match_counts = {
                str(key): int(value) for key, value in match_counts_series.items()
            }
            metrics["match_type_counts"] = match_counts
            metrics["match_type_rates"] = {
                str(key): float(value / total_processed)
                for key, value in match_counts.items()
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
            grouped = df.groupby(strategy_col, dropna=False)
            for strat, group in grouped:
                correct_series = group[correct_col].dropna()
                correct_count = int(correct_series.sum())
                denom = len(correct_series)
                accuracy = float(correct_count / denom) if denom else 0.0
                strategy_stats[str(strat)] = {
                    "n": int(len(group)),
                    "n_correct": correct_count,
                    "accuracy": accuracy,
                }

            metrics["match_strategy_performance"] = strategy_stats

            total_correct = float(df[correct_col].sum())
            if total_correct > 0:
                metrics["match_strategy_correct_share"] = {
                    strat: float(stats["n_correct"] / total_correct)
                    for strat, stats in strategy_stats.items()
                }

    distance_col = apply_output_prefix(column_prefix, "hierarchical_distance_min")
    if distance_col in df.columns:
        distance_series = pd.to_numeric(
            df[distance_col], errors="coerce"
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

        if correct_col in df.columns:
            incorrect_mask = df[correct_col] == False  # noqa: E712
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

    return metrics
