from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Type, TypeVar

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
    """Column selection for variable datasets."""

    embedding_columns: Sequence[str] = ("label", "name", "description")
    metadata_columns: Sequence[str] = ("label", "name", "dataset", "description")
    gold_labels_column: Optional[str] = "keywords"
    dataset_column: Optional[str] = "dataset"
    embedding_chunk_chars: Optional[int] = None
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        self.embedding_columns = tuple(self._normalise_sequence(self.embedding_columns))
        self.metadata_columns = tuple(self._normalise_sequence(self.metadata_columns))
        self.gold_labels_column = self._normalise_optional(self.gold_labels_column)
        dataset_normalised = self._normalise_optional(self.dataset_column)
        self.dataset_column = dataset_normalised
        if dataset_normalised and dataset_normalised not in self.metadata_columns:
            # Ensure downstream consumers always see the dataset column value.
            self.metadata_columns = tuple((*self.metadata_columns, dataset_normalised))

        chunk_chars = self.embedding_chunk_chars
        if chunk_chars is not None:
            try:
                chunk_chars = int(chunk_chars)
            except (TypeError, ValueError):
                chunk_chars = None
            else:
                if chunk_chars <= 0:
                    chunk_chars = None
        self.embedding_chunk_chars = chunk_chars

        overlap = self.chunk_overlap
        try:
            overlap = int(overlap)
        except (TypeError, ValueError):
            overlap = 0
        if overlap < 0:
            overlap = 0
        if self.embedding_chunk_chars is None:
            overlap = 0
        elif overlap >= self.embedding_chunk_chars:
            overlap = max(0, self.embedding_chunk_chars - 1)
        self.chunk_overlap = overlap

    @staticmethod
    def _normalise_sequence(values: Iterable[str] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values_iter: Iterable[str] = [values]
        else:
            try:
                values_iter = list(values)
            except TypeError:
                values_iter = [str(values)]
        normalised: list[str] = []
        seen: set[str] = set()
        for raw in values_iter:
            if raw is None:
                continue
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalised.append(text)
        return normalised

    @staticmethod
    def _normalise_optional(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def embedding_columns_list(self) -> list[str]:
        return list(self.embedding_columns)

    def metadata_columns_list(self) -> list[str]:
        return list(self.metadata_columns)

    def gold_column(self) -> Optional[str]:
        return self.gold_labels_column

    def dataset_column_name(self) -> Optional[str]:
        return self.dataset_column

    def build_item_payload(
        self, row: Mapping[str, Any]
    ) -> tuple[dict[str, Optional[str]], dict[str, Any], dict[str, Any]]:
        """Return item + metadata dictionaries for ``row``.

        The returned tuple contains ``(item, metadata, item_columns)``. The
        ``item`` mapping is suitable for embedding and prompting, ``metadata``
        captures the configured metadata columns, and ``item_columns`` provides a
        normalised view of the raw row including helper aliases (e.g. ``dataset``).
        """

        row_columns: dict[str, Any] = {str(key): value for key, value in row.items()}

        embed_cols = self.embedding_columns_list()
        metadata_cols = self.metadata_columns_list()

        item_keys: list[str] = []
        for column in (*embed_cols, *metadata_cols):
            if column and column not in item_keys:
                item_keys.append(column)

        item: dict[str, Optional[str]] = {
            column: row_columns.get(column) for column in item_keys if column
        }

        dataset_value: Any | None = None
        dataset_col = self.dataset_column_name()
        if dataset_col:
            dataset_value = row_columns.get(dataset_col)
        if dataset_value is None:
            dataset_value = row_columns.get("dataset")

        # Persist the dataset alias even when the source column is renamed.
        row_columns.setdefault("dataset", dataset_value)
        if dataset_value is not None:
            item.setdefault("dataset", dataset_value)

        metadata: dict[str, Any] = {
            column: row_columns.get(column) for column in metadata_cols
        }
        metadata.setdefault("dataset", dataset_value)
        if dataset_col:
            metadata.setdefault(dataset_col, row_columns.get(dataset_col))

        display_columns: list[str] = []
        for column in metadata_cols:
            if column and column not in display_columns:
                display_columns.append(column)
        if dataset_value is not None and "dataset" not in display_columns:
            display_columns.append("dataset")

        if embed_cols:
            item["_text_fields"] = tuple(embed_cols)
        if display_columns:
            item["_display_fields"] = tuple(display_columns)

        return item, metadata, row_columns


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
    models: Sequence[str] | None = None
    device: str | None = None
    max_length: int = 256
    batch_size: int = 128
    fp16: bool = True
    mean_pool: bool = True
    pca_components: Optional[int] = None
    pca_whiten: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TaxonomyEmbeddingConfig:
    """Hyperparameters for composing taxonomy embeddings."""

    gamma: float = 0.3
    summary_weight: float = 0.25
    child_aggregation_weight: float = 0.0
    child_aggregation_depth: Optional[int] = None

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
    output_column_prefix: Optional[str] = None

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
    """Connection and generation settings for LLM matching requests."""

    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    n_predict: int = 512
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    response_format: Optional[Dict[str, Any]] = None
    json_schema: Optional[Dict[str, Any]] = None

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LlamaCppConfig:
    """llama.cpp-specific toggles that may not be portable."""

    cache_prompt: bool = True
    n_keep: int = -1
    force_slot_id: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PostprocessingConfig:
    """Controls for interpreting and refining LLM outputs."""

    embedding_remap_threshold: float = 0.45
    alias_similarity_threshold: float = 0.9
    snap_to_child: bool = False
    snap_margin: float = 0.1
    snap_similarity: str = "token_sort"
    snap_descendant_depth: int = 1

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
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
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


def _coerce_optional_mapping(section: Mapping[str, Any] | None, label: str):
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise TypeError(f"Config '{label}' section must be a mapping if provided.")
    return dict(section)


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
    llm_section = _coerce_optional_mapping(raw.get("llm"), "llm")
    postprocessing_section = _coerce_optional_mapping(
        raw.get("postprocessing"), "postprocessing"
    )
    llama_cpp_section = _coerce_optional_mapping(raw.get("llama_cpp"), "llama_cpp")
    prompts_section = raw.get("prompts")
    parallel_section = raw.get("parallelism")
    http_section = raw.get("http")

    # Backwards compatibility: split legacy llm keys into new sections when
    # explicit sections are not provided.
    postprocessing_keys = {
        "embedding_remap_threshold",
        "alias_similarity_threshold",
        "snap_to_child",
        "snap_margin",
        "snap_similarity",
        "snap_descendant_depth",
    }
    llama_cpp_keys = {"cache_prompt", "n_keep", "force_slot_id"}
    if llm_section is None:
        llm_section = {}
    if postprocessing_section is None:
        postprocessing_section = {}
    if llama_cpp_section is None:
        llama_cpp_section = {}
    for key in postprocessing_keys:
        if key in llm_section and key not in postprocessing_section:
            postprocessing_section[key] = llm_section.pop(key)
    for key in llama_cpp_keys:
        if key in llm_section and key not in llama_cpp_section:
            llama_cpp_section[key] = llm_section.pop(key)

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
        llama_cpp=_coerce_section(llama_cpp_section, LlamaCppConfig),
        postprocessing=_coerce_section(
            postprocessing_section, PostprocessingConfig
        ),
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
