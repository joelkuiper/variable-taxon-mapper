from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Type, TypeVar

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

    n: int = 1000
    seed: int = 37
    dedupe_on: list[str] = field(default_factory=lambda: ["name"])
    progress_log_interval: int = 10
    results_csv: Optional[str] = None

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
class PruningConfig:
    """Configuration controlling taxonomy pruning and candidate generation."""

    enable_taxonomy_pruning: bool = True
    tree_sort_mode: str = "relevance"
    suggestion_sort_mode: str = "relevance"
    pruning_mode: str = "dominant_forest"  # options: dominant_forest, anchor_hull,
    # similarity_threshold, radius, community_pagerank, steiner_similarity
    surrogate_root_label: Optional[str] = None
    similarity_threshold: float = 0.0
    pruning_radius: int = 2
    anchor_top_k: int = 32
    max_descendant_depth: int = 3
    node_budget: int = 800
    suggestion_list_limit: int = 40
    lexical_anchor_limit: int = 3
    community_clique_size: int = 2
    max_community_size: Optional[int] = 400
    anchor_overfetch_multiplier: int = 3
    anchor_min_overfetch: int = 128
    pagerank_damping: float = 0.85
    pagerank_score_floor: float = 0.0
    pagerank_candidate_limit: Optional[int] = 256

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMConfig:
    """Settings for LLM matching requests."""

    endpoint: str = "http://127.0.0.1:8080/completions"
    n_predict: int = 512
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    cache_prompt: bool = True
    n_keep: int = -1
    grammar: Optional[str] = None
    embedding_remap_threshold: float = 0.45
    snap_to_child: bool = False
    snap_margin: float = 0.1
    snap_similarity: str = "token_sort"
    snap_descendant_depth: int = 1
    force_slot_id: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelismConfig:
    """Client-side concurrency controls for pruning and prompting."""

    num_slots: int = 4
    pool_maxsize: int = 64
    pruning_workers: int = 2
    pruning_batch_size: int = 4
    pruning_queue_size: int = 16

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HttpConfig:
    """HTTP client timeout tuning."""

    sock_connect: float = 10.0
    sock_read_floor: float = 30.0

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AppConfig:
    """Full application configuration tree."""

    seed: int = 37
    data: DataConfig = field(default_factory=DataConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    taxonomy_embeddings: TaxonomyEmbeddingConfig = field(
        default_factory=TaxonomyEmbeddingConfig
    )
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    http: HttpConfig = field(default_factory=HttpConfig)


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

    global_section = raw.get("global")
    if global_section is None:
        global_section = {}
    elif not isinstance(global_section, Mapping):
        raise TypeError("Config 'global' section must be a mapping if provided.")

    seed_default = AppConfig.__dataclass_fields__["seed"].default
    seed_raw = global_section.get("seed", raw.get("seed", seed_default))
    try:
        seed_value = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Config 'seed' must be an integer.") from exc

    data_section = raw.get("data")
    embedder_section = raw.get("embedder")
    taxonomy_section = raw.get("taxonomy_embeddings")
    hnsw_section = raw.get("hnsw")
    evaluation_section = raw.get("evaluation")
    pruning_section = raw.get("pruning")
    llm_section = raw.get("llm")
    parallel_section = raw.get("parallelism")
    http_section = raw.get("http")

    evaluation_cfg = _coerce_section(evaluation_section, EvaluationConfig)
    if not isinstance(evaluation_section, Mapping) or "seed" not in evaluation_section:
        evaluation_cfg.seed = seed_value

    app_config = AppConfig(
        seed=seed_value,
        data=_coerce_section(data_section, DataConfig),
        embedder=_coerce_section(embedder_section, EmbedderConfig),
        taxonomy_embeddings=_coerce_section(taxonomy_section, TaxonomyEmbeddingConfig),
        hnsw=_coerce_section(hnsw_section, HNSWConfig),
        evaluation=evaluation_cfg,
        pruning=_coerce_section(pruning_section, PruningConfig),
        llm=_coerce_section(llm_section, LLMConfig),
        parallelism=_coerce_section(parallel_section, ParallelismConfig),
        http=_coerce_section(http_section, HttpConfig),
    )

    return app_config


T = TypeVar("T")


def coerce_eval_config(
    config: EvaluationConfig | Mapping[str, Any] | None,
) -> EvaluationConfig:
    """Normalize evaluation configuration inputs."""

    if config is None:
        return EvaluationConfig()
    if isinstance(config, EvaluationConfig):
        return config
    if isinstance(config, Mapping):
        return EvaluationConfig(**config)
    raise TypeError(
        "eval_config must be an EvaluationConfig or a mapping of keyword arguments"
    )


def coerce_config(config: Any, cls: Type[T], label: str) -> T:
    """Normalise arbitrary configuration inputs into dataclass instances."""

    if config is None:
        return cls()
    if isinstance(config, cls):
        return config
    if isinstance(config, Mapping):
        return cls(**config)
    raise TypeError(
        f"{label} must be a {cls.__name__} or a mapping of keyword arguments"
    )


__all__ = [
    "AppConfig",
    "DataConfig",
    "EmbedderConfig",
    "EvaluationConfig",
    "HttpConfig",
    "HNSWConfig",
    "LLMConfig",
    "ParallelismConfig",
    "PruningConfig",
    "TaxonomyEmbeddingConfig",
    "coerce_config",
    "coerce_eval_config",
    "load_config",
]
