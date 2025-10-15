from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from config import AppConfig, load_config
from main import _prepare_keywords
from src.embedding import Embedder, build_hnsw_index, build_taxonomy_embeddings_composed
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from src.taxonomy_pruning import pruned_tree_markdown_for_item
from src.utils import clean_str_or_none, split_keywords_comma


def _parse_limit(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("Row limit override must be an integer or 'none'.")
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return 0
    try:
        parsed = int(text)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("Row limit override must be an integer or 'none'.") from exc
    if parsed < 0:
        raise ValueError("Row limit override must be non-negative or 'none'.")
    return parsed


def _compute_effective_subset(
    variables: pd.DataFrame,
    tax_names: Sequence[str],
    *,
    cfg,
    row_limit_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    dedupe_on = [c for c in (cfg.dedupe_on or []) if c]
    work_df = variables.copy()
    if dedupe_on:
        missing = [c for c in dedupe_on if c not in work_df.columns]
        if missing:
            raise KeyError(f"dedupe_on columns missing: {missing}")
        work_df = work_df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(
            drop=True
        )

    total_rows = int(len(work_df))
    cleaned_kw = (
        work_df["keywords"].map(clean_str_or_none)
        if "keywords" in work_df.columns
        else pd.Series(index=work_df.index, dtype=object)
    )

    known_labels = set(tax_names)
    total_with_any_keyword = int(cleaned_kw.notna().sum()) if total_rows else 0
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
    rnd = pd.Series(eligible_idxs)
    rnd = rnd.sample(frac=1.0, random_state=int(cfg.seed) if cfg.seed is not None else None)
    effective_limit = row_limit_override
    if effective_limit is None:
        effective_limit = cfg.n
    if isinstance(effective_limit, int) and effective_limit > 0:
        idxs = rnd.iloc[: min(effective_limit, len(rnd))].tolist()
    else:
        idxs = rnd.tolist()

    metadata = {
        "n_total_rows_after_dedupe": total_rows,
        "n_with_any_keyword": total_with_any_keyword,
        "n_eligible": n_eligible,
        "n_excluded_not_in_taxonomy": n_excluded_not_in_taxonomy,
    }

    subset_df = work_df.loc[idxs].reset_index(drop=True)
    subset_tokens = token_sets.loc[idxs].reset_index(drop=True)
    metadata["n_rows_selected"] = int(len(subset_df))

    return subset_df, subset_tokens, metadata


def _format_metrics(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether gold keywords fall inside the allowed subtree produced "
            "by taxonomy pruning for each variable."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    parser.add_argument(
        "--variables",
        type=Path,
        default=None,
        help="Optional override for the variables CSV file.",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=None,
        help="Optional override for the taxonomy keywords CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for the row-level evaluation results.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV output path for per-dataset summary metrics.",
    )
    parser.add_argument(
        "--limit",
        dest="row_limit_override",
        metavar="VALUE",
        default=None,
        help="Limit how many rows to evaluate (integer) or use 'none' for no limit.",
    )

    args = parser.parse_args(argv)
    if args.row_limit_override is not None:
        try:
            args.row_limit_override = _parse_limit(str(args.row_limit_override))
        except ValueError as exc:
            parser.error(str(exc))
    return args


def _resolve_path(base_path: Path, override: Optional[Path], default: Path) -> Path:
    if override is None:
        return default
    return override.resolve() if override.is_absolute() else (base_path / override).resolve()


def _prepare_outputs(base_path: Path, *paths: Optional[Path]) -> Tuple[Optional[Path], ...]:
    resolved: List[Optional[Path]] = []
    for path in paths:
        if path is None:
            resolved.append(None)
            continue
        if not path.is_absolute():
            path = (base_path / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved.append(path)
    return tuple(resolved)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    config_path = args.config.resolve()
    base_path = config_path.parent
    config: AppConfig = load_config(config_path)

    variables_default, keywords_default = config.data.to_paths(base_path)
    variables_path = _resolve_path(base_path, args.variables, variables_default)
    keywords_path = _resolve_path(base_path, args.keywords, keywords_default)

    variables = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)
    keywords, summary_df = _prepare_keywords(keywords_raw)

    G = build_taxonomy_graph(
        keywords,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    _name_to_id, name_to_path = build_name_maps_from_graph(G)

    embedder = Embedder(**config.embedder.to_kwargs())
    taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
    summary_source = summary_df if summary_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        G,
        embedder,
        summaries=summary_source,
        **taxonomy_kwargs,
    )
    hnsw_index = build_hnsw_index(tax_embs, **config.hnsw.to_kwargs())
    gloss_map = build_gloss_map(summary_source)

    work_df, token_sets, meta = _compute_effective_subset(
        variables,
        tax_names,
        cfg=config.evaluation,
        row_limit_override=args.row_limit_override,
    )

    evaluation_rows: List[Dict[str, Any]] = []
    tax_name_set = set(tax_names)
    iterator = range(len(work_df))
    total_nodes_in_graph = int(G.number_of_nodes())

    for idx in tqdm(iterator, desc="Evaluating", unit="item"):
        row = work_df.iloc[idx]
        token_set = token_sets.iloc[idx]
        item = {
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "name": row.get("name"),
            "description": row.get("description"),
        }

        gold_labels = sorted(set(token_set) & tax_name_set)

        _, allowed_ranked = pruned_tree_markdown_for_item(
            item,
            G=G,
            df=keywords,
            embedder=embedder,
            tax_names=tax_names,
            tax_embs_unit=tax_embs,
            hnsw_index=hnsw_index,
            name_col="name",
            order_col="order",
            anchor_top_k=config.evaluation.anchor_top_k,
            max_descendant_depth=config.evaluation.max_descendant_depth,
            node_budget=config.evaluation.node_budget,
            gloss_map=gloss_map,
            anchor_overfetch_multiplier=config.evaluation.anchor_overfetch_multiplier,
            anchor_min_overfetch=config.evaluation.anchor_min_overfetch,
            suggestion_list_limit=config.evaluation.suggestion_list_limit,
            lexical_anchor_limit=config.evaluation.lexical_anchor_limit,
            community_clique_size=config.evaluation.community_clique_size,
            max_community_size=config.evaluation.max_community_size,
            pagerank_damping=config.evaluation.pagerank_damping,
            pagerank_score_floor=config.evaluation.pagerank_score_floor,
            pagerank_candidate_limit=config.evaluation.pagerank_candidate_limit,
            enable_taxonomy_pruning=config.evaluation.enable_taxonomy_pruning,
            tree_sort_mode=config.evaluation.tree_sort_mode,
            pruning_mode=config.evaluation.pruning_mode,
            similarity_threshold=config.evaluation.similarity_threshold,
            pruning_radius=config.evaluation.pruning_radius,
        )

        allowed_set = set(allowed_ranked)
        allowed_count = len(allowed_set)
        pruned_count = (
            max(total_nodes_in_graph - allowed_count, 0)
            if total_nodes_in_graph
            else 0
        )
        pct_saved = (
            float(allowed_count) / float(total_nodes_in_graph)
            if total_nodes_in_graph
            else 0.0
        )
        pct_pruned = (
            float(pruned_count) / float(total_nodes_in_graph)
            if total_nodes_in_graph
            else 0.0
        )
        gold_in_allowed = sorted(set(gold_labels) & allowed_set)
        allowed_has_gold_flag = bool(gold_in_allowed)

        evaluation_rows.append(
            {
                "dataset": row.get("dataset"),
                "label": row.get("label"),
                "name": row.get("name"),
                "description": row.get("description"),
                "keywords": row.get("keywords"),
                "gold_labels": gold_labels,
                "allowed_labels": allowed_ranked,
                "n_allowed_labels": allowed_count,
                "n_pruned_labels": pruned_count,
                "pct_saved": pct_saved,
                "pct_pruned": pct_pruned,
                "n_gold_labels": len(gold_labels),
                "n_gold_in_allowed": len(gold_in_allowed),
                "gold_in_allowed": gold_in_allowed,
                "allowed_subtree_contains_gold": allowed_has_gold_flag,
                "possible_correct_under_allowed": allowed_has_gold_flag,
                "resolved_path_candidates": [name_to_path.get(lbl) for lbl in gold_in_allowed],
            }
        )

    result_df = pd.DataFrame(evaluation_rows)

    n_evaluated = int(len(result_df))
    n_allowed_contains = int(result_df["allowed_subtree_contains_gold"].sum()) if n_evaluated else 0
    metrics: Dict[str, Any] = {
        **meta,
        "n_evaluated": n_evaluated,
        "n_allowed_subtree_contains_gold": n_allowed_contains,
        "allowed_subtree_contains_gold_rate": float(
            n_allowed_contains / n_evaluated if n_evaluated else 0.0
        ),
        "n_without_gold_in_allowed": int(n_evaluated - n_allowed_contains),
        "taxonomy_total_nodes": total_nodes_in_graph,
        "mean_nodes_saved": float(result_df["n_allowed_labels"].mean())
        if n_evaluated
        else 0.0,
        "mean_nodes_pruned": float(result_df["n_pruned_labels"].mean())
        if n_evaluated
        else 0.0,
        "mean_saved_percentage": float(result_df["pct_saved"].mean())
        if n_evaluated
        else 0.0,
        "mean_pruned_percentage": float(result_df["pct_pruned"].mean())
        if n_evaluated
        else 0.0,
    }

    summary_cols = [
        "dataset",
        "allowed_subtree_contains_gold",
        "n_allowed_labels",
        "n_pruned_labels",
        "pct_saved",
        "pct_pruned",
        "n_gold_labels",
    ]
    summary_present = [c for c in summary_cols if c in result_df.columns]
    summary_df = pd.DataFrame()
    if summary_present and not result_df.empty:
        summary_df = (
            result_df.groupby("dataset", dropna=False)
            .agg(
                total_rows=("allowed_subtree_contains_gold", "size"),
                contains_gold=("allowed_subtree_contains_gold", "sum"),
                contains_gold_rate=("allowed_subtree_contains_gold", "mean"),
                mean_allowed_labels=("n_allowed_labels", "mean"),
                mean_pruned_labels=("n_pruned_labels", "mean"),
                mean_saved_percentage=("pct_saved", "mean"),
                mean_pruned_percentage=("pct_pruned", "mean"),
                mean_gold_labels=("n_gold_labels", "mean"),
            )
            .reset_index()
        )

    metrics_text = _format_metrics(metrics)
    print("Allowed subtree containment metrics:")
    print(metrics_text)

    if not summary_df.empty:
        print("\nPer-dataset summary:")
        print(summary_df.to_string(index=False))

    output_path, summary_output_path = _prepare_outputs(
        base_path, args.output, args.summary_output
    )
    if output_path is not None:
        result_df.to_csv(output_path, index=False)
        print(f"Saved row-level results to {output_path}")
    if summary_output_path is not None and not summary_df.empty:
        summary_df.to_csv(summary_output_path, index=False)
        print(f"Saved summary results to {summary_output_path}")


if __name__ == "__main__":
    main()
