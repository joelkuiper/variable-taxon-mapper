from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from config import AppConfig
from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.evaluate import ProgressHook, run_label_benchmark
from src.prompts import PromptRenderer, create_prompt_renderer
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from src.utils import ensure_file_exists


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _KeywordArtifacts:
    keywords: pd.DataFrame
    summaries: Optional[pd.DataFrame]


class VariableTaxonMapper:
    """High-level service for running the variable taxonomy pipeline."""

    def __init__(
        self,
        config: AppConfig,
        *,
        keywords: pd.DataFrame,
        graph,
        embedder: Embedder,
        taxonomy_names,
        taxonomy_embeddings,
        hnsw_index,
        name_to_id,
        name_to_path,
        gloss_map,
        prompt_renderer: PromptRenderer,
    ) -> None:
        self.config = config
        self.keywords = keywords
        self.graph = graph
        self.embedder = embedder
        self.taxonomy_names = taxonomy_names
        self.taxonomy_embeddings = taxonomy_embeddings
        self.hnsw_index = hnsw_index
        self.name_to_id = name_to_id
        self.name_to_path = name_to_path
        self.gloss_map = gloss_map
        self.prompt_renderer = prompt_renderer

    @staticmethod
    def _prepare_keywords(keywords: pd.DataFrame) -> _KeywordArtifacts:
        required_columns = {"name", "parent", "order"}
        missing = sorted(required_columns - set(keywords.columns))
        if missing:
            raise KeyError(f"Keywords data missing required columns: {missing}")

        summaries = keywords if "definition_summary" in keywords.columns else None
        return _KeywordArtifacts(keywords=keywords, summaries=summaries)

    @staticmethod
    def _resolve_input(
        default: Path,
        override: Path | None,
        *,
        base_path: Path | None,
    ) -> Path:
        if override is None:
            return default.resolve()
        if override.is_absolute():
            return override.resolve()
        if base_path is not None:
            return (base_path / override).resolve()
        return override.resolve()

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        base_path: Path | None = None,
        keywords: pd.DataFrame | None = None,
        keywords_path: Path | None = None,
        embedder: Embedder | None = None,
    ) -> "VariableTaxonMapper":
        """Factory that assembles taxonomy dependencies from configuration."""

        if keywords is None:
            variables_default, keywords_default = config.data.to_paths(base_path)
            resolved_path = cls._resolve_input(
                keywords_default,
                keywords_path,
                base_path=base_path,
            )
            logger.debug("Resolved keywords path: %s", resolved_path)
            ensure_file_exists(resolved_path, "keywords CSV")
            keywords = pd.read_csv(resolved_path)
            logger.info(
                "Loaded keywords frame with %d rows and %d columns",
                len(keywords),
                len(keywords.columns),
            )
        else:
            logger.debug("Using in-memory keywords dataframe with %d rows", len(keywords))

        if base_path is not None:
            config.prompts.set_config_root(base_path)

        artifacts = cls._prepare_keywords(keywords.copy())
        logger.debug(
            "Prepared keywords dataframe; summaries_present=%s",
            artifacts.summaries is not None,
        )

        graph = build_taxonomy_graph(
            artifacts.keywords,
            name_col="name",
            parent_col="parent",
            order_col="order",
        )
        logger.info("Constructed taxonomy graph with %d nodes", len(graph))
        name_to_id, name_to_path = build_name_maps_from_graph(graph)
        logger.debug(
            "Generated taxonomy name mappings: %d entries", len(name_to_id)
        )

        mapper_embedder = embedder or Embedder(**config.embedder.to_kwargs())
        if embedder is None:
            logger.info("Initialized embedder %s", mapper_embedder.__class__.__name__)
        else:
            logger.debug(
                "Using provided embedder instance %s", mapper_embedder.__class__.__name__
            )

        taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
        summaries = artifacts.summaries if artifacts.summaries is not None else None
        taxonomy_names, taxonomy_embeddings = build_taxonomy_embeddings_composed(
            graph,
            mapper_embedder,
            summaries=summaries,
            **taxonomy_kwargs,
        )
        logger.info(
            "Built taxonomy embeddings for %d labels", len(taxonomy_names)
        )

        hnsw_index = build_hnsw_index(
            taxonomy_embeddings, **config.hnsw.to_kwargs()
        )
        logger.debug(
            "Constructed HNSW index for %d-dimensional embeddings",
            taxonomy_embeddings.shape[1] if taxonomy_embeddings.size else 0,
        )
        gloss_map = build_gloss_map(summaries)
        logger.debug("Constructed gloss map with %d entries", len(gloss_map))

        prompt_renderer = create_prompt_renderer(
            config.prompts, base_dir=base_path
        )
        logger.debug("Initialized prompt renderer for taxonomy matching")

        return cls(
            config,
            keywords=artifacts.keywords,
            graph=graph,
            embedder=mapper_embedder,
            taxonomy_names=taxonomy_names,
            taxonomy_embeddings=taxonomy_embeddings,
            hnsw_index=hnsw_index,
            name_to_id=name_to_id,
            name_to_path=name_to_path,
            gloss_map=gloss_map,
            prompt_renderer=prompt_renderer,
        )

    def predict(
        self,
        variables: pd.DataFrame,
        *,
        evaluate: bool = True,
        progress_hook: ProgressHook | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Generate predictions (and optionally metrics) for ``variables``."""

        logger.info(
            "Starting label benchmark evaluation; evaluate=%s, total_variables=%d",
            evaluate,
            len(variables),
        )
        df, metrics = run_label_benchmark(
            variables,
            self.keywords,
            G=self.graph,
            embedder=self.embedder,
            tax_names=self.taxonomy_names,
            tax_embs_unit=self.taxonomy_embeddings,
            hnsw_index=self.hnsw_index,
            name_to_id=self.name_to_id,
            name_to_path=self.name_to_path,
            gloss_map=self.gloss_map,
            eval_config=self.config.evaluation,
            pruning_config=self.config.pruning,
            llm_config=self.config.llm,
            parallel_config=self.config.parallelism,
            http_config=self.config.http,
            evaluate=evaluate,
            progress_hook=progress_hook,
            field_mapping=self.config.fields,
            prompt_config=self.config.prompts,
            prompt_renderer=self.prompt_renderer,
            prompt_base_path=self.config.prompts.get_config_root(),
        )
        logger.info("Completed label benchmark evaluation")
        return df, metrics
