from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
import typer
from tqdm.auto import tqdm

from vtm.embedding import Embedder, build_hnsw_index, build_taxonomy_embeddings_composed
from vtm.pipeline.service import prepare_keywords_dataframe
from vtm.pruning import pruned_tree_markdown_for_item
from vtm.taxonomy import build_gloss_map, build_name_maps_from_graph, build_taxonomy_graph
from vtm.utils import (
    clean_str_or_none,
    ensure_file_exists,
    resolve_path,
    split_keywords_comma,
)

from .app import app, logger
from .common import (
    ConfigArgument,
    FieldMappingConfig,
    RowLimitOption,
    coerce_config,
    load_app_config,
)


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


def _compute_effective_subset(
    variables: pd.DataFrame,
    tax_names: Sequence[str],
    *,
    cfg,
    field_mapping: FieldMappingConfig | Dict[str, Any] | None = None,
    row_limit_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    field_cfg = coerce_config(field_mapping, FieldMappingConfig, "field_mapping")

    dedupe_columns: List[str] = []
    for column in cfg.dedupe_on or []:
        resolved = field_cfg.resolve_column(column)
        if isinstance(resolved, str) and resolved and resolved not in dedupe_columns:
            dedupe_columns.append(resolved)

    work_df = variables.copy()
    if dedupe_columns:
        missing = [c for c in dedupe_columns if c not in work_df.columns]
        if missing:
            raise KeyError(f"dedupe_on columns missing: {missing}")
        work_df = work_df.drop_duplicates(subset=dedupe_columns, keep="first").reset_index(
            drop=True
        )

    total_rows = int(len(work_df))
    resolved_gold_column = field_cfg.resolve_column("gold_labels")
    gold_column: Optional[str] = (
        resolved_gold_column if isinstance(resolved_gold_column, str) else None
    )
    cleaned_gold = (
        work_df[gold_column].map(clean_str_or_none)
        if isinstance(gold_column, str) and gold_column in work_df.columns
        else pd.Series(index=work_df.index, dtype=object)
    )

    known_labels = set(tax_names)
    total_with_any_keyword = int(cleaned_gold.notna().sum()) if total_rows else 0
    if gold_column is None or gold_column not in work_df.columns:
        configured_name = field_cfg.gold_labels
        if configured_name:
            raise KeyError(
                "variables must include the column configured in "
                f"fields.gold_labels ('{configured_name}')"
            )
        raise KeyError("Evaluation requires a gold label column; set fields.gold_labels")

    token_lists = cleaned_gold.map(split_keywords_comma)
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
    rnd = rnd.sample(
        frac=1.0, random_state=int(cfg.seed) if cfg.seed is not None else None
    )
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
        "gold_column": gold_column,
    }

    subset_df = work_df.loc[idxs].reset_index(drop=True)
    subset_tokens = token_sets.loc[idxs].reset_index(drop=True)
    metadata["n_rows_selected"] = int(len(subset_df))

    return subset_df, subset_tokens, metadata


def _format_metrics(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


@app.command("prune-check")
def prune_check_command(
    config: Path = ConfigArgument,
    variables: Optional[Path] = typer.Option(
        None,
        "--variables",
        help="Optional override for the variables CSV file.",
        path_type=Path,
    ),
    keywords: Optional[Path] = typer.Option(
        None,
        "--keywords",
        help="Optional override for the taxonomy keywords CSV file.",
        path_type=Path,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional CSV output path for the row-level evaluation results.",
        path_type=Path,
    ),
    summary_output: Optional[Path] = typer.Option(
        None,
        "--summary-output",
        help="Optional CSV output path for per-dataset summary metrics.",
        path_type=Path,
    ),
    row_limit: Optional[int] = RowLimitOption,
) -> None:
    """Evaluate taxonomy pruning coverage against gold labels."""

    config_path = config.resolve()
    base_path = config_path.parent
    config_obj = load_app_config(config_path)
    logger.info("Loaded configuration from %s", config_path)

    variables_default, keywords_default = config_obj.data.to_paths(base_path)
    variables_path = resolve_path(base_path, variables_default, variables)
    keywords_path = resolve_path(base_path, keywords_default, keywords)

    ensure_file_exists(variables_path, "variables CSV")
    ensure_file_exists(keywords_path, "keywords CSV")

    variables_df = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)
    logger.info(
        "Loaded %d variables rows and %d keyword rows",
        len(variables_df),
        len(keywords_raw),
    )
    keywords_df, definition_df = prepare_keywords_dataframe(
        keywords_raw, config_obj.taxonomy_fields
    )

    graph = build_taxonomy_graph(
        keywords_df,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    _name_to_id, name_to_path = build_name_maps_from_graph(graph)

    embedder = Embedder(**config_obj.embedder.to_kwargs())
    taxonomy_kwargs = config_obj.taxonomy_embeddings.to_kwargs()
    definition_source = definition_df if definition_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        graph,
        embedder,
        definitions=definition_source,
        **taxonomy_kwargs,
    )
    hnsw_index = build_hnsw_index(tax_embs, **config_obj.hnsw.to_kwargs())
    gloss_map = build_gloss_map(definition_source)
    logger.debug("Constructed resources for %d taxonomy labels", len(tax_names))

    field_cfg = config_obj.fields
    dataset_col = field_cfg.resolve_column("dataset")
    label_col = field_cfg.resolve_column("label")
    name_col = field_cfg.resolve_column("name")
    desc_col = field_cfg.resolve_column("description")
    resolved_gold_column = field_cfg.resolve_column("gold_labels")
    gold_column = resolved_gold_column if isinstance(resolved_gold_column, str) else None
    text_keys = field_cfg.item_text_keys()

    work_df, token_sets, meta = _compute_effective_subset(
        variables_df,
        tax_names,
        cfg=config_obj.evaluation,
        field_mapping=field_cfg,
        row_limit_override=row_limit,
    )
    logger.info(
        "Prepared %d candidate rows for evaluation (eligible=%d)",
        len(work_df),
        meta.get("n_eligible"),
    )

    evaluation_rows: List[Dict[str, Any]] = []
    tax_name_set = set(tax_names)
    iterator = range(len(work_df))
    total_nodes_in_graph = int(graph.number_of_nodes())
    logger.info(
        "Evaluating pruning coverage across %d items and %d taxonomy nodes",
        len(work_df),
        total_nodes_in_graph,
    )

    for idx in tqdm(iterator, desc="Evaluating", unit="item"):
        row = work_df.iloc[idx]
        token_set = token_sets.iloc[idx]
        item = {
            "dataset": row.get(dataset_col) if dataset_col else None,
            "label": row.get(label_col) if label_col else None,
            "name": row.get(name_col) if name_col else None,
            "description": row.get(desc_col) if desc_col else None,
        }
        if text_keys:
            item["_text_fields"] = tuple(text_keys)

        gold_labels = sorted(set(token_set) & tax_name_set)

        _markdown, allowed_ranked = pruned_tree_markdown_for_item(
            item,
            graph=graph,
            frame=keywords_df,
            embedder=embedder,
            tax_names=tax_names,
            tax_embs_unit=tax_embs,
            hnsw_index=hnsw_index,
            pruning_cfg=config_obj.pruning,
            name_col="name",
            order_col="order",
            gloss_map=gloss_map,
        )

        allowed_set = set(allowed_ranked)
        allowed_count = len(allowed_set)
        pruned_count = (
            max(total_nodes_in_graph - allowed_count, 0) if total_nodes_in_graph else 0
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

        gold_parent_chain_matches: List[Dict[str, Any]] = []
        gold_parent_chain_nodes: List[str] = []
        n_gold_or_parent_in_allowed = 0
        allowed_has_gold_or_parent_flag = allowed_has_gold_flag

        if gold_labels and graph is not None:
            digraph = graph
            for gold_label in gold_labels:
                if not isinstance(gold_label, str):
                    continue
                matches: List[str] = []
                if gold_label in allowed_set:
                    matches.append(gold_label)
                if digraph.has_node(gold_label):
                    ancestors = nx.ancestors(digraph, gold_label)
                    ancestor_matches = sorted(allowed_set & ancestors)
                    if ancestor_matches:
                        matches.extend(ancestor_matches)
                if matches:
                    unique_matches = list(dict.fromkeys(matches))
                    gold_parent_chain_matches.append(
                        {"gold_label": gold_label, "matches": unique_matches}
                    )
                    gold_parent_chain_nodes.extend(unique_matches)
                    n_gold_or_parent_in_allowed += 1

            if gold_parent_chain_nodes and not allowed_has_gold_or_parent_flag:
                allowed_has_gold_or_parent_flag = True

        gold_parent_chain_nodes_sorted = sorted(dict.fromkeys(gold_parent_chain_nodes))

        evaluation_rows.append(
            {
                "dataset": row.get(dataset_col) if dataset_col else None,
                "label": row.get(label_col) if label_col else None,
                "name": row.get(name_col) if name_col else None,
                "description": row.get(desc_col) if desc_col else None,
                "keywords": row.get(gold_column) if gold_column else None,
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
                "gold_or_parent_in_allowed": gold_parent_chain_nodes_sorted,
                "gold_parent_chain_matches": gold_parent_chain_matches,
                "n_gold_or_parent_in_allowed": n_gold_or_parent_in_allowed,
                "allowed_subtree_contains_gold_or_parent": allowed_has_gold_or_parent_flag,
                "possible_correct_under_allowed_including_parents": allowed_has_gold_or_parent_flag,
                "resolved_path_candidates": [
                    name_to_path.get(lbl) for lbl in gold_in_allowed
                ],
            }
        )

    result_df = pd.DataFrame(evaluation_rows)

    n_evaluated = int(len(result_df))
    n_allowed_contains = (
        int(result_df["allowed_subtree_contains_gold"].sum()) if n_evaluated else 0
    )
    n_allowed_contains_parent = (
        int(result_df["allowed_subtree_contains_gold_or_parent"].sum())
        if n_evaluated
        else 0
    )
    metrics: Dict[str, Any] = {
        **meta,
        "n_evaluated": n_evaluated,
        "n_allowed_subtree_contains_gold": n_allowed_contains,
        "allowed_subtree_contains_gold_rate": float(
            n_allowed_contains / n_evaluated if n_evaluated else 0.0
        ),
        "n_allowed_subtree_contains_gold_or_parent": n_allowed_contains_parent,
        "allowed_subtree_contains_gold_or_parent_rate": float(
            n_allowed_contains_parent / n_evaluated if n_evaluated else 0.0
        ),
        "n_without_gold_in_allowed": int(n_evaluated - n_allowed_contains),
        "n_without_gold_or_parent_in_allowed": int(
            n_evaluated - n_allowed_contains_parent
        ),
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
        "allowed_subtree_contains_gold_or_parent",
        "n_allowed_labels",
        "n_pruned_labels",
        "pct_saved",
        "pct_pruned",
        "n_gold_labels",
        "n_gold_or_parent_in_allowed",
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
                contains_gold_or_parent=(
                    "allowed_subtree_contains_gold_or_parent",
                    "sum",
                ),
                contains_gold_or_parent_rate=(
                    "allowed_subtree_contains_gold_or_parent",
                    "mean",
                ),
                mean_allowed_labels=("n_allowed_labels", "mean"),
                mean_pruned_labels=("n_pruned_labels", "mean"),
                mean_saved_percentage=("pct_saved", "mean"),
                mean_pruned_percentage=("pct_pruned", "mean"),
                mean_gold_labels=("n_gold_labels", "mean"),
                mean_gold_or_parent_in_allowed=(
                    "n_gold_or_parent_in_allowed",
                    "mean",
                ),
            )
            .reset_index()
        )

    metrics_text = _format_metrics(metrics)
    logger.info("Allowed subtree containment metrics:\n%s", metrics_text)

    if not summary_df.empty:
        logger.info("Per-dataset summary:\n%s", summary_df.to_string(index=False))

    output_path, summary_output_path = _prepare_outputs(
        base_path, output, summary_output
    )
    if output_path is not None:
        result_df.to_csv(output_path, index=False)
        logger.info("Saved row-level results to %s", output_path)
    if summary_output_path is not None and not summary_df.empty:
        summary_df.to_csv(summary_output_path, index=False)
        logger.info("Saved summary results to %s", summary_output_path)
