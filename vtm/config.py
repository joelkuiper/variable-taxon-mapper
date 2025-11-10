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
class FieldMappingConfig:
    """Logical field to column mapping for variable datasets."""

    dataset: Optional[str] = "dataset"
    label: Optional[str] = "label"
    name: Optional[str] = "name"
    description: Optional[str] = "description"
    gold_labels: Optional[str] = "keywords"

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "dataset": self.dataset,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "gold_labels": self.gold_labels,
        }

    def resolve_column(self, key: str) -> Optional[str]:
        mapping = self.as_dict()
        if key in mapping:
            value = mapping[key]
            return value if value else None
        return key

    def item_text_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for key in ("label", "name", "description"):
            value = self.resolve_column(key)
            if isinstance(value, str) and value and key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def item_text_columns(self) -> list[str]:
        columns: list[str] = []
        seen: set[str] = set()
        for key in self.item_text_keys():
            column = self.resolve_column(key)
            if isinstance(column, str) and column not in seen:
                columns.append(column)
                seen.add(column)
        return columns


@dataclass
class TaxonomyFieldMappingConfig:
    """Column mapping for taxonomy keyword metadata."""

    name: str = "name"
    parent: str = "parent"
    parents: Optional[str] = None
    order: Optional[str] = "order"
    definition: Optional[str] = "definition_summary"
    label: Optional[str] = "label"

    def resolve_column(self, key: str) -> Optional[str]:
        value = getattr(self, key, None)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return str(value)

    def require_column(self, key: str) -> str:
        value = self.resolve_column(key)
        if not value:
            raise KeyError(f"Taxonomy field '{key}' is not configured")
        return value

    def resolve_parents_column(self) -> Optional[str]:
        """Return the configured multi-parent column, if available."""

        return self.resolve_column("parents")

    def require_parents_column(self) -> str:
        """Return the multi-parent column name or raise if missing."""

        value = self.resolve_parents_column()
        if not value:
            raise KeyError("Taxonomy field 'parents' is not configured")
        return value


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

    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    n_predict: int = 512
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    cache_prompt: bool = True
    n_keep: int = -1
    grammar: Optional[str] = None
    embedding_remap_threshold: float = 0.45
    alias_similarity_threshold: float = 0.9
    snap_to_child: bool = False
    snap_margin: float = 0.1
    snap_similarity: str = "token_sort"
    snap_descendant_depth: int = 1
    force_slot_id: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptTemplateConfig:
    """Configuration for rendering LLM prompt templates."""

    system_template_path: Optional[str] = "templates/match_system_prompt.j2"
    user_template_path: Optional[str] = "templates/match_user_prompt.j2"
    system_template: Optional[str] = None
    user_template: Optional[str] = None
    encoding: str = "utf-8"
    _config_root: Optional[Path] = field(default=None, repr=False, compare=False)

    def set_config_root(self, root: Optional[Path]) -> None:
        """Record the directory used to resolve relative template paths."""

        self._config_root = root

    def get_config_root(self) -> Optional[Path]:
        return self._config_root

    def resolve_path(self, template_path: str, *, base_dir: Optional[Path] = None) -> Path:
        """Resolve ``template_path`` relative to ``base_dir`` or the config root."""

        candidate = Path(template_path)
        if candidate.is_absolute():
            return candidate

        root = base_dir or self._config_root or Path.cwd()
        return (root / candidate).resolve()


@dataclass
class ParallelismConfig:
    """Client-side concurrency controls for pruning and prompting."""

    num_slots: int = 4
    pool_maxsize: int = 64
    pruning_workers: int = 2
    pruning_batch_size: int = 4
    pruning_queue_size: int = 16
    pruning_start_method: str | None = None
    pruning_embed_on_workers: bool = False
    pruning_worker_devices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.pruning_worker_devices is not None:
            devices = tuple(self.pruning_worker_devices)
            self.pruning_worker_devices = tuple(str(device) for device in devices)

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
    fields: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    taxonomy_fields: TaxonomyFieldMappingConfig = field(
        default_factory=TaxonomyFieldMappingConfig
    )
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    taxonomy_embeddings: TaxonomyEmbeddingConfig = field(
        default_factory=TaxonomyEmbeddingConfig
    )
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompts: PromptTemplateConfig = field(default_factory=PromptTemplateConfig)
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
    fields_section = raw.get("fields")
    taxonomy_fields_section = raw.get("taxonomy_fields")
    embedder_section = raw.get("embedder")
    taxonomy_section = raw.get("taxonomy_embeddings")
    hnsw_section = raw.get("hnsw")
    evaluation_section = raw.get("evaluation")
    pruning_section = raw.get("pruning")
    llm_section = raw.get("llm")
    prompts_section = raw.get("prompts")
    parallel_section = raw.get("parallelism")
    http_section = raw.get("http")

    evaluation_cfg = _coerce_section(evaluation_section, EvaluationConfig)
    if not isinstance(evaluation_section, Mapping) or "seed" not in evaluation_section:
        evaluation_cfg.seed = seed_value

    app_config = AppConfig(
        seed=seed_value,
        data=_coerce_section(data_section, DataConfig),
        fields=_coerce_section(fields_section, FieldMappingConfig),
        taxonomy_fields=_coerce_section(
            taxonomy_fields_section, TaxonomyFieldMappingConfig
        ),
        embedder=_coerce_section(embedder_section, EmbedderConfig),
        taxonomy_embeddings=_coerce_section(taxonomy_section, TaxonomyEmbeddingConfig),
        hnsw=_coerce_section(hnsw_section, HNSWConfig),
        evaluation=evaluation_cfg,
        pruning=_coerce_section(pruning_section, PruningConfig),
        llm=_coerce_section(llm_section, LLMConfig),
        prompts=_coerce_section(prompts_section, PromptTemplateConfig),
        parallelism=_coerce_section(parallel_section, ParallelismConfig),
        http=_coerce_section(http_section, HttpConfig),
    )

    app_config.prompts.set_config_root(config_path.parent)

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
    "FieldMappingConfig",
    "TaxonomyFieldMappingConfig",
    "EmbedderConfig",
    "EvaluationConfig",
    "HttpConfig",
    "HNSWConfig",
    "LLMConfig",
    "PromptTemplateConfig",
    "ParallelismConfig",
    "PruningConfig",
    "TaxonomyEmbeddingConfig",
    "coerce_config",
    "coerce_eval_config",
    "load_config",
]
