from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import pandas as pd
from pandas._libs.missing import NAType

from vtm.config import AppConfig, TaxonomyFieldMappingConfig
from vtm.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from vtm.utils import ensure_file_exists, load_table, resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vtm.evaluate import ProgressHook
    from vtm.embedding import Embedder
    from vtm.prompts import PromptRenderer


logger = logging.getLogger(__name__)


def _is_na(value: Any) -> bool:
    """Return True when a value should be treated as missing."""

    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _clean_str(value: Any) -> str | NAType:
    """Normalize arbitrary values into stripped strings or pandas.NA."""

    if _is_na(value):
        return pd.NA
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return pd.NA
    return text


def _rename_taxonomy_columns(
    keywords: pd.DataFrame,
    *,
    name_col: str,
    parent_col: str,
    parents_col: str | None,
    order_col: str | None,
    label_col: str | None,
) -> pd.DataFrame:
    """Return a canonical keyword frame with taxonomy-aligned column names."""

    rename_map: dict[str, str] = {}
    if name_col != "name":
        rename_map[name_col] = "name"
    if parent_col != "parent":
        rename_map[parent_col] = "parent"
    if parents_col:
        if parents_col not in keywords.columns:
            raise KeyError(
                f"Keywords data missing required column for multi-parent mapping: '{parents_col}'"
            )
        if parents_col != "parents":
            rename_map[parents_col] = "parents"
    if order_col:
        if order_col not in keywords.columns:
            raise KeyError(f"Keywords data missing required column: '{order_col}'")
        if order_col != "order":
            rename_map[order_col] = "order"
    if (
        label_col
        and label_col in keywords.columns
        and label_col != "label"
        and label_col != name_col
    ):
        rename_map[label_col] = "label"

    canonical = keywords.rename(columns=rename_map).copy()
    if "name" in canonical.columns:
        canonical["name"] = canonical["name"].map(_clean_str)

    missing = sorted({"name", "parent"} - set(canonical.columns))
    if missing:
        raise KeyError(f"Keywords data missing required columns: {missing}")

    if order_col is None and "order" not in canonical.columns:
        canonical["order"] = pd.NA
    elif order_col is not None and "order" not in canonical.columns:
        raise KeyError("Keywords data missing required column: 'order'")

    return canonical


def _build_identifier_map(
    canonical: pd.DataFrame,
    *,
    name_col: str,
    label_col: str | None,
) -> dict[str, str]:
    """Construct a lookup from identifiers to canonical keyword names."""

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

    return identifier_to_name


def _normalize_parent_fields(
    canonical: pd.DataFrame,
    identifier_to_name: dict[str, str],
    *,
    parents_col: str | None = None,
) -> pd.DataFrame:
    """Normalize parent fields using the identifier lookup."""

    normalized = canonical.copy()

    def _normalize_parent(value: Any) -> Any:
        cleaned = _clean_str(value)
        if _is_na(cleaned):
            return pd.NA
        return identifier_to_name.get(str(cleaned), cleaned)

    if "parent" in normalized.columns:
        normalized["parent"] = normalized["parent"].map(_normalize_parent)

    def _normalize_multi_parent(value: Any) -> Any:
        if _is_na(value):
            return pd.NA
        text = str(value) if not isinstance(value, str) else value
        parts = [part.strip() for part in text.split("|")]
        normalized_parts: list[str] = []
        for part in parts:
            if not part:
                continue
            normalized_parts.append(identifier_to_name.get(part, part))
        if not normalized_parts:
            return pd.NA
        return "|".join(normalized_parts)

    resolved_parents_col = parents_col or ("parents" if "parents" in normalized.columns else None)
    if resolved_parents_col and resolved_parents_col in normalized.columns:
        normalized[resolved_parents_col] = normalized[resolved_parents_col].map(
            _normalize_multi_parent
        )

    return normalized


def _extract_definitions(
    keywords: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    definition_col: str | None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Attach definition metadata and return optional definition frame."""

    resolved_definition_col = "definition"
    definition_source: Optional[pd.Series] = None
    if definition_col and definition_col in keywords.columns:
        definition_source = keywords[definition_col]
    elif definition_col and definition_col not in keywords.columns:
        logger.debug(
            "Configured definition column '%s' not found; falling back to existing 'definition' values",
            definition_col,
        )
    if definition_source is None and resolved_definition_col in canonical.columns:
        definition_source = canonical[resolved_definition_col]

    canonical_with_definitions = canonical.copy()
    definitions: Optional[pd.DataFrame] = None
    if definition_source is not None:
        canonical_with_definitions[resolved_definition_col] = definition_source.map(
            lambda value: ""
            if _is_na(value)
            else str(value).strip()
        )
        definitions = canonical_with_definitions[["name", resolved_definition_col]].copy()
        definitions.fillna("", inplace=True)
    elif resolved_definition_col in canonical_with_definitions.columns:
        canonical_with_definitions.drop(columns=[resolved_definition_col], inplace=True)

    return canonical_with_definitions, definitions


def prepare_keywords_dataframe(
    keywords: pd.DataFrame,
    taxonomy_fields: TaxonomyFieldMappingConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, Tuple[str, ...]]]:
    """Return canonical keywords, definitions, and multi-parent mappings."""

    if not isinstance(keywords, pd.DataFrame):
        raise TypeError("keywords must be a pandas DataFrame")

    name_col = taxonomy_fields.require_column("name")
    parent_col = taxonomy_fields.require_column("parent")
    parents_col = taxonomy_fields.resolve_parents_column()
    order_col = taxonomy_fields.resolve_column("order")
    definition_col = taxonomy_fields.resolve_column("definition")
    label_col = taxonomy_fields.resolve_column("label")

    canonical = _rename_taxonomy_columns(
        keywords,
        name_col=name_col,
        parent_col=parent_col,
        parents_col=parents_col,
        order_col=order_col,
        label_col=label_col,
    )

    identifier_to_name = _build_identifier_map(
        canonical,
        name_col=name_col,
        label_col=label_col,
    )

    resolved_parents_col = None
    if parents_col or "parents" in canonical.columns:
        resolved_parents_col = "parents"

    canonical = _normalize_parent_fields(
        canonical,
        identifier_to_name,
        parents_col=resolved_parents_col,
    )
    multi_parent_map: dict[str, Tuple[str, ...]] = {}
    if (
        resolved_parents_col
        and resolved_parents_col in canonical.columns
        and "name" in canonical.columns
    ):
        for row in canonical[["name", resolved_parents_col]].itertuples(index=False):
            raw_name, raw_parents = row
            cleaned_name = _clean_str(raw_name)
            if _is_na(cleaned_name):
                continue
            parents: list[str] = []
            if not _is_na(raw_parents):
                text = raw_parents if isinstance(raw_parents, str) else str(raw_parents)
                parts = text.split("|")
                for part in parts:
                    cleaned_parent = _clean_str(part)
                    if _is_na(cleaned_parent):
                        continue
                    parent_str = str(cleaned_parent)
                    if parent_str not in parents:
                        parents.append(parent_str)
            if parents:
                multi_parent_map[str(cleaned_name)] = tuple(parents)
    canonical, definitions = _extract_definitions(
        keywords,
        canonical,
        definition_col=definition_col,
    )

    return canonical, definitions, multi_parent_map


@dataclass(slots=True)
class _KeywordArtifacts:
    keywords: pd.DataFrame
    definitions: Optional[pd.DataFrame]
    multi_parents: dict[str, Tuple[str, ...]]


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
        canonical, definitions, multi_parents = prepare_keywords_dataframe(
            keywords, taxonomy_fields
        )
        return _KeywordArtifacts(
            keywords=canonical, definitions=definitions, multi_parents=multi_parents
        )

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
            resolved_path = resolve_path(base_path, keywords_default, keywords_path)
            logger.debug("Resolved keywords path: %s", resolved_path)
            ensure_file_exists(resolved_path, "keywords data file")
            keywords = load_table(resolved_path)
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
            "Prepared keywords dataframe; definitions_present=%s",
            artifacts.definitions is not None,
        )

        graph = build_taxonomy_graph(
            artifacts.keywords,
            name_col="name",
            parent_col="parent",
            order_col="order",
            multi_parents=artifacts.multi_parents,
        )
        logger.info("Constructed taxonomy graph with %d nodes", len(graph))
        name_to_id, name_to_path = build_name_maps_from_graph(graph)
        logger.debug(
            "Generated taxonomy name mappings: %d entries", len(name_to_id)
        )

        from vtm.embedding import (
            Embedder as EmbedderImpl,
            build_hnsw_index as build_hnsw_index_impl,
            build_taxonomy_embeddings_composed as build_taxonomy_embeddings_composed_impl,
        )

        mapper_embedder = embedder or EmbedderImpl(**config.embedder.to_kwargs())
        if embedder is None:
            logger.info("Initialized embedder %s", mapper_embedder.__class__.__name__)
        else:
            logger.debug(
                "Using provided embedder instance %s", mapper_embedder.__class__.__name__
            )

        taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
        definitions = artifacts.definitions if artifacts.definitions is not None else None
        taxonomy_names, taxonomy_embeddings = build_taxonomy_embeddings_composed_impl(
            graph,
            mapper_embedder,
            definitions=definitions,
            **taxonomy_kwargs,
        )
        logger.info(
            "Built taxonomy embeddings for %d labels", len(taxonomy_names)
        )

        hnsw_index = build_hnsw_index_impl(
            taxonomy_embeddings, **config.hnsw.to_kwargs()
        )
        logger.debug(
            "Constructed HNSW index for %d-dimensional embeddings",
            taxonomy_embeddings.shape[1] if taxonomy_embeddings.size else 0,
        )
        gloss_map = build_gloss_map(definitions)
        logger.debug("Constructed gloss map with %d entries", len(gloss_map))

        from vtm.prompts import create_prompt_renderer

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

        from vtm.evaluate import run_label_benchmark

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
            hnsw_config=self.config.hnsw,
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
