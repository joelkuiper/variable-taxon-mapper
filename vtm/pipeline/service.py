from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from pandas._libs.missing import NAType

from config import AppConfig, TaxonomyFieldMappingConfig
from vtm.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from vtm.evaluate import ProgressHook, run_label_benchmark
from vtm.prompts import PromptRenderer, create_prompt_renderer
from vtm.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from vtm.utils import ensure_file_exists


logger = logging.getLogger(__name__)


def prepare_keywords_dataframe(
    keywords: pd.DataFrame,
    taxonomy_fields: TaxonomyFieldMappingConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Return canonical keywords and optional summary frames based on config."""

    if not isinstance(keywords, pd.DataFrame):
        raise TypeError("keywords must be a pandas DataFrame")

    name_col = taxonomy_fields.require_column("name")
    parent_col = taxonomy_fields.require_column("parent")
    order_col = taxonomy_fields.resolve_column("order")
    summary_col = taxonomy_fields.resolve_column("definition_summary")
    definition_col = taxonomy_fields.resolve_column("definition")
    label_col = taxonomy_fields.resolve_column("label")

    rename_map: dict[str, str] = {}
    if name_col != "name":
        rename_map[name_col] = "name"
    if parent_col != "parent":
        rename_map[parent_col] = "parent"
    if order_col:
        if order_col not in keywords.columns:
            raise KeyError(f"Keywords data missing required column: '{order_col}'")
        if order_col != "order":
            rename_map[order_col] = "order"
    if summary_col and summary_col in keywords.columns and summary_col != "definition_summary":
        rename_map[summary_col] = "definition_summary"
    if definition_col and definition_col in keywords.columns and definition_col != "definition":
        rename_map[definition_col] = "definition"
    if label_col and label_col in keywords.columns and label_col != "label":
        rename_map[label_col] = "label"

    canonical = keywords.rename(columns=rename_map).copy()

    def _is_na(value: Any) -> bool:
        try:
            return bool(pd.isna(value))
        except TypeError:
            return False

    def _clean_str(value: Any) -> str | NAType:
        if _is_na(value):
            return pd.NA
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if not text:
            return pd.NA
        return text

    if "name" in canonical.columns:
        canonical["name"] = canonical["name"].map(_clean_str)

    lookup_columns: list[str] = []
    if "identifier" in canonical.columns:
        lookup_columns.append("identifier")
    if name_col != "name" and name_col in canonical.columns:
        lookup_columns.append(name_col)
    if label_col and label_col in canonical.columns and label_col not in {"label", "name"}:
        lookup_columns.append(label_col)

    if lookup_columns:
        lookup_columns = list(dict.fromkeys(lookup_columns))

    identifier_to_name: dict[str, str] = {}
    if lookup_columns and "name" in canonical.columns:
        relevant_cols = ["name", *lookup_columns]
        for values in canonical[relevant_cols].itertuples(index=False, name=None):
            canonical_name = values[0]
            if _is_na(canonical_name):
                continue
            for column, raw_identifier in zip(lookup_columns, values[1:]):
                cleaned_identifier = _clean_str(raw_identifier)
                if _is_na(cleaned_identifier):
                    continue
                identifier_to_name[str(cleaned_identifier)] = str(canonical_name)

    def _normalize_parent(value: Any) -> Any:
        cleaned = _clean_str(value)
        if _is_na(cleaned):
            return pd.NA
        return identifier_to_name.get(str(cleaned), cleaned)

    if "parent" in canonical.columns:
        canonical["parent"] = canonical["parent"].map(_normalize_parent)

    def _normalize_multi_parent(value: Any) -> Any:
        if _is_na(value):
            return pd.NA
        text = str(value) if not isinstance(value, str) else value
        parts = [part.strip() for part in text.split("|")]
        normalized: list[str] = []
        for part in parts:
            if not part:
                continue
            normalized_part = identifier_to_name.get(part, part)
            normalized.append(normalized_part)
        if not normalized:
            return pd.NA
        return "|".join(normalized)

    if "parents" in canonical.columns:
        canonical["parents"] = canonical["parents"].map(_normalize_multi_parent)

    missing = sorted({"name", "parent"} - set(canonical.columns))
    if missing:
        raise KeyError(f"Keywords data missing required columns: {missing}")

    if order_col is None and "order" not in canonical.columns:
        canonical["order"] = pd.NA
    elif order_col is not None and "order" not in canonical.columns:
        raise KeyError("Keywords data missing required column: 'order'")

    summaries: Optional[pd.DataFrame] = None
    resolved_summary_col = "definition_summary"
    if summary_col and summary_col in keywords.columns:
        if resolved_summary_col not in canonical.columns and summary_col != "definition_summary":
            # summary column was only present in the original frame; rename and copy over
            canonical[resolved_summary_col] = keywords[summary_col]
        if resolved_summary_col in canonical.columns:
            summaries = canonical[["name", resolved_summary_col]].copy()
            summaries.fillna("", inplace=True)

    return canonical, summaries


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
    def _prepare_keywords(
        keywords: pd.DataFrame, taxonomy_fields: TaxonomyFieldMappingConfig
    ) -> _KeywordArtifacts:
        canonical, summaries = prepare_keywords_dataframe(keywords, taxonomy_fields)
        return _KeywordArtifacts(keywords=canonical, summaries=summaries)

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

        artifacts = cls._prepare_keywords(keywords.copy(), config.taxonomy_fields)
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
