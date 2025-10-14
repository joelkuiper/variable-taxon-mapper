from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import tomllib


@dataclass
class DataConfig:
    """Configuration for loading raw data assets."""

    variables_csv: str = "data/Variables.csv"
    keywords_csv: str = "data/Keywords_summarized.csv"

    def to_paths(self, base_path: Optional[Path] = None) -> tuple[Path, Path]:
        base = base_path or Path.cwd()
        return base / self.variables_csv, base / self.keywords_csv


@dataclass
class EmbedderConfig:
    """Model and batching configuration for the embedding model."""

    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    device: str | None = None
    max_length: int = 256
    batch_size: int = 128
    fp16: bool = True
    mean_pool: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TaxonomyEmbeddingConfig:
    """Hyperparameters for composing taxonomy embeddings."""

    gamma: float = 0.3
    summary_weight: float = 0.25

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HNSWConfig:
    """Parameters for building the HNSW index over taxonomy embeddings."""

    space: str = "cosine"
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    num_threads: int = 0

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Runtime options for the label benchmarking pipeline."""

    endpoint: str = "http://127.0.0.1:8080/completions"
    n: int = 1000
    seed: int = 37
    n_predict: int = 512
    temperature: float = 0.0
    dedupe_on: list[str] = field(default_factory=lambda: ["name"])
    anchor_top_k: int = 32
    max_descendant_depth: int = 3
    node_budget: int = 800
    max_workers: int = 4
    num_slots: int = 4
    pool_maxsize: int = 64
    suggestion_list_limit: int = 40
    anchor_overfetch_multiplier: int = 3
    anchor_min_overfetch: int = 128
    lexical_anchor_limit: int = 3
    community_clique_size: int = 2
    max_community_size: Optional[int] = 400
    pagerank_damping: float = 0.85
    pagerank_score_floor: float = 0.0
    pagerank_candidate_limit: int = 256
    llm_top_k: int = 20
    llm_top_p: float = 0.8
    llm_min_p: float = 0.0
    llm_cache_prompt: bool = True
    llm_n_keep: int = -1
    llm_grammar: Optional[str] = None
    http_sock_connect: float = 10.0
    http_sock_read_floor: float = 30.0
    progress_log_interval: int = 10
    results_csv: Optional[str] = None
    enable_taxonomy_pruning: bool = True
    tree_sort_mode: str = "relevance"  # e.g., "relevance", "topological", "alphabetical", "proximity"
    pruning_mode: str = "dominant_forest"
    similarity_threshold: float = 0.0
    pruning_radius: int = 2

    def to_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data["dedupe_on"] = list(self.dedupe_on)
        return data

    def resolve_results_path(
        self,
        *,
        base_path: Optional[Path] = None,
        variables_path: Optional[Path] = None,
    ) -> Path:
        base = base_path or Path.cwd()
        if self.results_csv:
            path = Path(self.results_csv)
            if not path.is_absolute():
                path = base / path
            return path
        if variables_path is not None:
            return variables_path.with_name(f"{variables_path.stem}_results.csv")
        return base / "results.csv"


@dataclass
class AppConfig:
    """Full application configuration tree."""

    data: DataConfig = field(default_factory=DataConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    taxonomy_embeddings: TaxonomyEmbeddingConfig = field(
        default_factory=TaxonomyEmbeddingConfig
    )
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _coerce_section(section: Mapping[str, Any] | None, cls: type[Any]) -> Any:
    if section is None:
        return cls()
    if not isinstance(section, Mapping):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(section)!r}")
    kwargs: MutableMapping[str, Any] = dict(section)
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a TOML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as fh:
        raw: Mapping[str, Any] = tomllib.load(fh)

    data_section = raw.get("data")
    embedder_section = raw.get("embedder")
    taxonomy_section = raw.get("taxonomy_embeddings")
    hnsw_section = raw.get("hnsw")
    evaluation_section = raw.get("evaluation")

    app_config = AppConfig(
        data=_coerce_section(data_section, DataConfig),
        embedder=_coerce_section(embedder_section, EmbedderConfig),
        taxonomy_embeddings=_coerce_section(taxonomy_section, TaxonomyEmbeddingConfig),
        hnsw=_coerce_section(hnsw_section, HNSWConfig),
        evaluation=_coerce_section(evaluation_section, EvaluationConfig),
    )

    return app_config


__all__ = [
    "AppConfig",
    "DataConfig",
    "EmbedderConfig",
    "EvaluationConfig",
    "HNSWConfig",
    "TaxonomyEmbeddingConfig",
    "load_config",
]
