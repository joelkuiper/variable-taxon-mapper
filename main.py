from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import AppConfig, load_config
from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.evaluate import run_label_benchmark
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


def run_pipeline(config: AppConfig) -> Tuple[pd.DataFrame, dict[str, object]]:
    variables_path, keywords_path = config.data.to_paths()

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
    )

    return df, metrics


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
    config = load_config(args.config)
    df, metrics = run_pipeline(config)

    display_cols = [
        "label",
        "name",
        "description",
        "gold_labels",
        "resolved_label",
        "correct",
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    if present_cols:
        print(df[present_cols])
    else:
        print(df)
    print(metrics)


if __name__ == "__main__":
    main()
