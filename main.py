from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from config import AppConfig, load_config
from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.evaluate import ProgressHook, run_label_benchmark
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)


def _prepare_keywords(keywords: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    required_columns = {"name", "parent", "order"}
    missing = sorted(required_columns - set(keywords.columns))
    if missing:
        raise KeyError(f"Keywords data missing required columns: {missing}")

    summary_df = keywords if "definition_summary" in keywords.columns else None

    return keywords, summary_df


def run_pipeline(
    config: AppConfig,
    *,
    base_path: Path | None = None,
    variables_csv: Path | None = None,
    evaluate: bool = True,
    progress_hook: ProgressHook | None = None,
) -> Tuple[pd.DataFrame, dict[str, object]]:
    variables_default, keywords_path = config.data.to_paths(base_path)
    variables_path = variables_csv or variables_default

    variables = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)

    keywords, summary_df = _prepare_keywords(keywords_raw)

    G = build_taxonomy_graph(
        keywords,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    name_to_id, name_to_path = build_name_maps_from_graph(G)

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

    df, metrics = run_label_benchmark(
        variables,
        keywords,
        G=G,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs_unit=tax_embs,
        hnsw_index=hnsw_index,
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        gloss_map=gloss_map,
        eval_config=config.evaluation,
        pruning_config=config.pruning,
        llm_config=config.llm,
        parallel_config=config.parallelism,
        http_config=config.http,
        evaluate=evaluate,
        progress_hook=progress_hook,
    )

    return df, metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    key_order = [
        "n_total_rows_after_dedupe",
        "n_with_any_keyword",
        "n_eligible",
        "n_excluded_not_in_taxonomy",
        "n_evaluated",
        "n_correct",
        "n_unmatched",
        "label_accuracy_any_match",
        "label_accuracy_exact_only",
        "label_accuracy_ancestor_only",
        "label_accuracy_descendant_only",
        "n_errors",
    ]

    lines: list[str] = ["Evaluation Metrics:"]
    for key in key_order:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")

    for aggregate_key in ("match_type_counts", "match_type_rates"):
        if aggregate_key in metrics:
            lines.append(f"  - {aggregate_key}:")
            submap = metrics[aggregate_key]
            if isinstance(submap, dict):
                for sub_key, sub_val in sorted(submap.items()):
                    if isinstance(sub_val, float):
                        lines.append(f"      • {sub_key}: {sub_val:.4f}")
                    else:
                        lines.append(f"      • {sub_key}: {sub_val}")

    remaining = {
        k: v
        for k, v in metrics.items()
        if k not in key_order and k not in {"match_type_counts", "match_type_rates"}
    }
    if remaining:
        lines.append("  - other_metrics:")
        pretty_rest = json.dumps(remaining, indent=4, sort_keys=True)
        for line in pretty_rest.splitlines():
            lines.append(f"      {line}")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the variable taxonomy mapper")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)
    variables_path, _ = config.data.to_paths(base_path)
    df, metrics = run_pipeline(
        config,
        base_path=base_path,
        variables_csv=variables_path,
    )

    display_cols = [
        "label",
        "name",
        "description",
        "gold_labels",
        "resolved_label",
        "correct",
        "match_type",
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    if present_cols:
        print(df[present_cols])
    else:
        print(df)
    results_path = config.evaluation.resolve_results_path(
        base_path=base_path,
        variables_path=variables_path,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    metrics_report = format_metrics(metrics)
    print(metrics_report)


if __name__ == "__main__":
    main()
