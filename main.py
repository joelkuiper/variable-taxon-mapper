from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

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
from src.utils import configure_logging, ensure_file_exists, set_global_seed
from src.reporting import report_results


logger = logging.getLogger(__name__)


def _prepare_keywords(
    keywords: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
    logger.debug("Setting global seed to %s", config.seed)
    set_global_seed(config.seed)

    variables_default, keywords_default = config.data.to_paths(base_path)

    def _resolve_input(default: Path, override: Path | None) -> Path:
        if override is None:
            return default.resolve()
        if override.is_absolute():
            return override.resolve()
        if base_path is not None:
            return (base_path / override).resolve()
        return override.resolve()

    variables_path = _resolve_input(variables_default, variables_csv)
    keywords_path = _resolve_input(keywords_default, None)

    logger.debug("Resolved variables path: %s", variables_path)
    logger.debug("Resolved keywords path: %s", keywords_path)

    ensure_file_exists(variables_path, "variables CSV")
    ensure_file_exists(keywords_path, "keywords CSV")

    parallel_cfg = config.parallelism
    logger.info(
        "[pipeline] concurrency settings: pruning_workers=%s, pruning_batch=%s",
        parallel_cfg.pruning_workers,
        parallel_cfg.pruning_batch_size,
    )

    variables = pd.read_csv(variables_path, low_memory=False)
    logger.info(
        "Loaded variables frame with %d rows and %d columns", len(variables), len(variables.columns)
    )
    keywords_raw = pd.read_csv(keywords_path)
    logger.info(
        "Loaded keywords frame with %d rows and %d columns",
        len(keywords_raw),
        len(keywords_raw.columns),
    )

    keywords, summary_df = _prepare_keywords(keywords_raw)
    logger.debug(
        "Prepared keywords dataframe; summaries_present=%s",
        summary_df is not None,
    )

    G = build_taxonomy_graph(
        keywords,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    logger.info("Constructed taxonomy graph with %d nodes", len(G))
    name_to_id, name_to_path = build_name_maps_from_graph(G)
    logger.debug("Generated taxonomy name mappings: %d entries", len(name_to_id))

    embedder = Embedder(**config.embedder.to_kwargs())
    logger.info("Initialized embedder %s", embedder.__class__.__name__)

    taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
    summary_source = summary_df if summary_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        G,
        embedder,
        summaries=summary_source,
        **taxonomy_kwargs,
    )
    logger.info("Built taxonomy embeddings for %d labels", len(tax_names))

    hnsw_index = build_hnsw_index(tax_embs, **config.hnsw.to_kwargs())
    logger.debug("Constructed HNSW index for %d-dimensional embeddings", tax_embs.shape[1])
    gloss_map = build_gloss_map(summary_source)
    logger.debug("Constructed gloss map with %d entries", len(gloss_map))

    logger.info(
        "Starting label benchmark evaluation; evaluate=%s, total_variables=%d",
        evaluate,
        len(variables),
    )
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
    logger.info("Completed label benchmark evaluation")

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
    configure_logging(level=os.getenv("LOG_LEVEL", logging.INFO))

    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)
    logger.info("Loaded configuration from %s", config_path)
    set_global_seed(config.seed)
    variables_path, _ = config.data.to_paths(base_path)
    df, metrics = run_pipeline(
        config,
        base_path=base_path,
        variables_csv=variables_path,
    )

    results_path = config.evaluation.resolve_results_path(
        base_path=base_path,
        variables_path=variables_path,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)

    metrics_path = results_path.with_name(f"{results_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    logger.info("Metrics saved to %s", metrics_path)

    report_results(df, metrics)


if __name__ == "__main__":
    main()
