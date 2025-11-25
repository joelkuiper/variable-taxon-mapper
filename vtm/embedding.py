"""Embedding utilities including SapBERT wrappers and HNSW helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from typing import Literal


import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


import pandas as pd

from .taxonomy import ensure_traversal_cache, taxonomy_node_texts
from .utils import clean_text
from vtm.config import FieldMappingConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemTextChunk:
    """A chunk of item text derived from a specific source field."""

    text: str
    field: str
    chunk_index: int
    chunk_count: int

    def __post_init__(self) -> None:  # pragma: no cover - dataclass post-init
        normalized = self.text if isinstance(self.text, str) else str(self.text)
        object.__setattr__(self, "text", normalized)
        object.__setattr__(self, "field", str(self.field))


def l2_normalize(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, eps)


def _topological_order(G) -> List[str]:
    indegree = {node: G.in_degree(node) for node in G.nodes}
    stack = [node for node, deg in indegree.items() if deg == 0]
    order: List[str] = []
    while stack:
        node = stack.pop()
        order.append(node)
        for succ in G.successors(node):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                stack.append(succ)
    if len(order) != len(indegree):
        raise ValueError("Taxonomy graph contains a cycle; expected a DAG.")
    return order


class Embedder:
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        *,
        models: Sequence[str] | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 128,
        fp16: bool = True,
        mean_pool: bool = False,  # False = [CLS]
        pca_components: Optional[int] = None,
        pca_whiten: bool = False,
    ):
        model_candidates = [
            str(name).strip()
            for name in (models if models is not None else [model_name])
            if str(name).strip()
        ]
        if not model_candidates:
            raise ValueError("At least one embedding model must be provided")

        self.model_names = tuple(model_candidates)
        self.model_name = self.model_names[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16)
        self.max_length = max_length
        self.batch_size = batch_size
        self.mean_pool = mean_pool
        if pca_components is None:
            validated_components = None
        else:
            try:
                validated_components = int(pca_components)
            except (TypeError, ValueError):
                validated_components = None
            else:
                if validated_components <= 0:
                    validated_components = None

        self._pca_components = validated_components
        self._pca_whiten = bool(pca_whiten)
        self._pca_model: PCA | None = None
        self._pca_fitted = False

        self.tokenizers: list[Any] = []
        self.models: list[Any] = []
        self.hidden_sizes: list[int] = []

        for name in self.model_names:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            if self.fp16 and self.device.startswith("cuda"):
                model.half()
            model.to(self.device).eval()

            self.tokenizers.append(tok)
            self.models.append(model)
            self.hidden_sizes.append(int(getattr(model.config, "hidden_size", 768)))

    def _encode_with_model(
        self, model: Any, tok: Any, texts: Sequence[str], *, model_label: str
    ) -> np.ndarray:
        n_texts = len(texts)
        hidden_size = int(getattr(model.config, "hidden_size", 768))
        if n_texts == 0:
            return np.zeros((0, hidden_size), dtype=np.float32)

        out = np.empty((n_texts, hidden_size), dtype=np.float32)
        bs = self.batch_size
        total_tokens = 0
        offset = 0
        log_progress = n_texts >= 1000
        progress_interval = max(bs * 10, 1000)
        next_progress = progress_interval
        start_ts = time.perf_counter()
        if log_progress:
            logger.info(
                "Encoding %d texts with %s (batch size %d) on %s",
                n_texts,
                model_label,
                bs,
                self.device,
            )

        for i in range(0, n_texts, bs):
            batch = texts[i : i + bs]
            toks = tok.batch_encode_plus(
                batch,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            total_tokens += int(toks["attention_mask"].sum().item())
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                last_hidden = model(**toks)[0]
                if self.mean_pool:
                    attn = (toks["attention_mask"].unsqueeze(-1)).float()
                    summed = (last_hidden * attn).sum(dim=1)
                    denom = attn.sum(dim=1).clamp(min=1e-6)
                    rep = summed / denom
                else:
                    rep = last_hidden[:, 0, :]
            batch_size = rep.shape[0]
            out[offset : offset + batch_size] = rep.float().cpu().numpy()
            offset += batch_size

            if log_progress and offset >= next_progress:
                logger.info(
                    "Encoded %d/%d texts with %s (%.0f%%)",
                    offset,
                    n_texts,
                    model_label,
                    100 * offset / n_texts,
                )
                next_progress += progress_interval

        if log_progress:
            duration = time.perf_counter() - start_ts
            logger.info(
                "Finished encoding %d texts with %s in %.2fs (%.1f tokens/s)",
                n_texts,
                model_label,
                duration,
                total_tokens / duration if duration > 0 else float("inf"),
            )
        return l2_normalize(out)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        n_texts = len(texts)
        total_dim = int(sum(self.hidden_sizes)) if self.hidden_sizes else 0
        if n_texts == 0:
            if self._pca_components is not None:
                total_dim = min(total_dim, self._pca_components)
            return np.zeros((0, total_dim), dtype=np.float32)

        model_vecs = [
            self._encode_with_model(model, tok, texts, model_label=name)
            for name, model, tok in zip(self.model_names, self.models, self.tokenizers)
        ]

        combined = model_vecs[0] if len(model_vecs) == 1 else np.concatenate(model_vecs, axis=1)

        if self._pca_components is not None:
            if self._pca_model is None:
                n_components = min(
                    self._pca_components,
                    combined.shape[1],
                    combined.shape[0],
                )
                self._pca_model = PCA(
                    n_components=n_components,
                    whiten=self._pca_whiten,
                )
            if self._pca_fitted:
                combined = self._pca_model.transform(combined)
            else:
                combined = self._pca_model.fit_transform(combined)
                self._pca_fitted = True
                logger.info(
                    "Fitted PCA on %d embeddings (n_components=%d, whiten=%s)",
                    combined.shape[0],
                    self._pca_model.n_components_,
                    self._pca_model.whiten,
                )

        return l2_normalize(combined.astype(np.float32, copy=False))

    def export_init_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments that can rebuild this embedder."""

        return {
            "model_name": self.model_name,
            "models": list(self.model_names),
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "fp16": self.fp16,
            "mean_pool": self.mean_pool,
            "pca_components": self._pca_components,
            "pca_whiten": self._pca_whiten,
        }


def _resolve_default_fields(
    item: Mapping[str, Optional[str]],
    field_mapping: FieldMappingConfig | None,
) -> List[str]:
    if field_mapping is not None:
        columns = field_mapping.embedding_columns_list()
        if columns:
            return columns
    candidate = item.get("_text_fields")
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        candidate_seq: Sequence[object] = list(candidate)
        fields = [
            str(key)
            for key in candidate_seq
            if isinstance(key, str) and key and not key.startswith("_")
        ]
        if fields:
            seen: set[str] = set()
            unique: List[str] = []
            for key in fields:
                if key not in seen:
                    seen.add(key)
                    unique.append(key)
            return unique
    return FieldMappingConfig().embedding_columns_list()


def _iter_text_chunks(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    if chunk_overlap < 0:
        chunk_overlap = 0
    step = chunk_chars - min(chunk_overlap, chunk_chars - 1)
    if step <= 0:
        step = 1
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_chars
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step
    if not chunks and text:
        chunks.append(text)
    return chunks


def _normalize_chunk_params(
    *,
    field_mapping: FieldMappingConfig | None,
    chunk_chars: Optional[int],
    chunk_overlap: Optional[int],
) -> tuple[Optional[int], int]:
    mapping_chunk = field_mapping.embedding_chunk_chars if field_mapping else None
    mapping_overlap = field_mapping.chunk_overlap if field_mapping else 0

    if chunk_chars is None:
        chunk_chars = mapping_chunk
    if chunk_overlap is None:
        chunk_overlap = mapping_overlap

    if chunk_chars is not None:
        try:
            chunk_chars = int(chunk_chars)
        except (TypeError, ValueError):
            chunk_chars = None
        else:
            if chunk_chars <= 0:
                chunk_chars = None

    if chunk_chars is None:
        chunk_overlap = 0
    else:
        try:
            chunk_overlap = int(chunk_overlap)
        except (TypeError, ValueError):
            chunk_overlap = 0
        if chunk_overlap < 0:
            chunk_overlap = 0
        if chunk_overlap >= chunk_chars:
            chunk_overlap = chunk_chars - 1

    return chunk_chars, chunk_overlap


def collect_item_texts(
    item: Mapping[str, Optional[str]],
    *,
    fields: Sequence[str] | None = None,
    field_mapping: FieldMappingConfig | None = None,
    clean: bool = True,
    max_length: int = 256,
    chunk_chars: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[ItemTextChunk]:
    """Extract candidate text fields from an item for embedding."""

    resolved_fields: Sequence[str]
    if fields is None:
        resolved_fields = _resolve_default_fields(item, field_mapping)
    else:
        resolved_fields = list(fields)

    try:
        normalized_max_length = int(max_length)
    except (TypeError, ValueError):
        normalized_max_length = 0
    normalized_max_length = max(0, normalized_max_length)
    resolved_chunk_chars, resolved_overlap = _normalize_chunk_params(
        field_mapping=field_mapping,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[ItemTextChunk] = []
    for key in resolved_fields:
        raw = item.get(key)
        if clean:
            text = clean_text(raw)
            if not text or text == "(empty)":
                continue
        else:
            if isinstance(raw, str):
                text = raw.strip()
            elif raw is None:
                text = ""
            else:
                text = str(raw).strip()
            if not text:
                continue

        if resolved_chunk_chars is None:
            truncated = text[:normalized_max_length] if normalized_max_length else text
            if not truncated:
                continue
            chunks.append(
                ItemTextChunk(
                    text=truncated,
                    field=key,
                    chunk_index=0,
                    chunk_count=1,
                )
            )
            continue

        field_chunks = _iter_text_chunks(text, resolved_chunk_chars, resolved_overlap)
        total = len(field_chunks)
        for idx, chunk in enumerate(field_chunks):
            if not chunk:
                continue
            chunks.append(
                ItemTextChunk(
                    text=chunk,
                    field=key,
                    chunk_index=idx,
                    chunk_count=total,
                )
            )

    return chunks


def encode_item_texts(
    item: Mapping[str, Optional[str]],
    embedder: Embedder,
    *,
    fields: Sequence[str] | None = None,
    field_mapping: FieldMappingConfig | None = None,
    clean: bool = True,
    max_length: int = 256,
    chunk_chars: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    texts: Optional[Sequence[ItemTextChunk | str]] = None,
) -> np.ndarray:
    """Encode selected text fields from ``item`` using ``embedder``."""

    if texts is None:
        texts = collect_item_texts(
            item,
            fields=fields,
            field_mapping=field_mapping,
            clean=clean,
            max_length=max_length,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
    else:
        texts = list(texts)

    resolved_chunks: List[ItemTextChunk] = []
    for entry in texts:
        if isinstance(entry, ItemTextChunk):
            resolved_chunks.append(entry)
        elif isinstance(entry, str):
            resolved_chunks.append(
                ItemTextChunk(text=entry, field="", chunk_index=0, chunk_count=1)
            )
        else:
            raise TypeError(
                "texts must contain strings or ItemTextChunk instances, "
                f"got {type(entry)!r}"
            )

    if not resolved_chunks:
        return np.zeros((0, 768), dtype=np.float32)
    return embedder.encode([chunk.text for chunk in resolved_chunks])


def cosine_match_maxpool(
    node_vecs: np.ndarray, item_embs: np.ndarray
) -> Tuple[int, float]:
    sims = node_vecs @ item_embs.T
    best = sims.max(axis=1)
    j = int(np.argmax(best))
    return j, float(best[j])


def _normalize_hnsw_space(space: str) -> Literal["l2", "ip", "cosine"]:
    candidate = space.strip().lower()
    if candidate not in {"l2", "ip", "cosine"}:
        raise ValueError(
            "HNSW space must be one of {'l2', 'ip', 'cosine'}, got %r" % space
        )
    return cast(Literal["l2", "ip", "cosine"], candidate)


def build_hnsw_index(
    embs_unit: np.ndarray,
    *,
    space: str = "cosine",
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    num_threads: int = 0,
) -> Any:
    if embs_unit.dtype != np.float32:
        embs_unit = embs_unit.astype(np.float32, copy=False)
    N, D = embs_unit.shape
    import hnswlib

    normalized_space = _normalize_hnsw_space(space)
    index = hnswlib.Index(space=normalized_space, dim=D)
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    index.add_items(embs_unit, np.arange(N), num_threads=num_threads)
    index.set_ef(ef_search)
    return index


def _extract_definition_map(
    definitions: Optional[object],
) -> Dict[str, str]:
    if definitions is None:
        return {}

    definition_map: Dict[str, str] = {}

    if isinstance(definitions, Mapping):
        items: Iterable[Tuple[object, object]] = definitions.items()
    elif isinstance(definitions, pd.Series):
        items = definitions.items()
    elif isinstance(definitions, pd.DataFrame):
        items = definitions[["name", "definition"]].itertuples(
            index=False, name=None
        )
    else:
        raise TypeError("definitions must be a mapping, pandas Series/DataFrame, or None")

    for entry in items:
        if isinstance(entry, tuple):
            if not entry:
                continue
            key = entry[0]
            value = entry[1] if len(entry) > 1 else None
        elif isinstance(entry, Mapping):
            # Some pandas iterators yield dict-like rows instead of tuples.
            key = entry.get("name")
            value = entry.get("definition")
        else:
            continue

        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.strip()
        definition = value.strip()
        if name and definition and name not in definition_map:
            definition_map[name] = definition

    return definition_map


def build_taxonomy_embeddings_composed(
    G,
    embedder: Embedder,
    gamma: float = 0.3,
    *,
    definitions: Optional[object] = None,
    summary_weight: float = 1.0,
    child_aggregation_weight: float = 0.0,
    child_aggregation_depth: Optional[int] = None,
    multi_parents: Optional[Mapping[str, Sequence[str]]] = None,
    taxonomy_text_transform: Optional[Callable[[str], str]] = None,
) -> Tuple[List[str], np.ndarray]:
    start_ts = time.perf_counter()
    names = taxonomy_node_texts(G)
    label2idx = {n: i for i, n in enumerate(names)}

    topo = _topological_order(G)
    cache = ensure_traversal_cache(G)
    all_parents = cache.get("all_parents", {}) if cache is not None else {}
    parent_indices: List[List[int]] = [[] for _ in names]
    if all_parents:
        for node, parents in all_parents.items():
            idx = label2idx.get(node)
            if idx is None or not parents:
                continue
            parent_indices[idx] = [
                label2idx[p] for p in parents if p in label2idx
            ]
    else:
        for node in names:
            idx = label2idx[node]
            preds = tuple(G.predecessors(node))
            if preds:
                parent_indices[idx] = [
                    label2idx[p] for p in preds if p in label2idx
                ]

    if multi_parents:
        for child_name, parent_names in multi_parents.items():
            child_idx = label2idx.get(child_name)
            if child_idx is None:
                continue
            for parent_name in parent_names:
                parent_idx = label2idx.get(parent_name)
                if parent_idx is None:
                    continue
                if parent_idx not in parent_indices[child_idx]:
                    parent_indices[child_idx].append(parent_idx)

    child_weight = float(child_aggregation_weight)
    logger.info(
        "Composing taxonomy embeddings for %d nodes (gamma=%.2f, child_weight=%.2f)",
        len(names),
        gamma,
        child_weight,
    )

    if taxonomy_text_transform is not None:
        texts = [taxonomy_text_transform(n) for n in names]
    else:
        texts = names

    encode_start = time.perf_counter()
    name_vecs = embedder.encode(texts)
    logger.info(
        "Encoded taxonomy label texts in %.2fs",
        time.perf_counter() - encode_start,
    )

    definition_map = _extract_definition_map(definitions)
    summary_vecs = np.zeros_like(name_vecs)
    if definition_map:
        definition_start = time.perf_counter()
        texts: List[str] = []
        idxs: List[int] = []
        for i, n in enumerate(names):
            s = definition_map.get(n)
            if s:
                texts.append(s)
                idxs.append(i)
        if texts:
            encoded = embedder.encode(texts)
            for j, idx in enumerate(idxs):
                summary_vecs[idx] = encoded[j]
        logger.info(
            "Encoded %d taxonomy definitions in %.2fs",
            len(texts),
            time.perf_counter() - definition_start,
        )

    composed_base = l2_normalize(name_vecs + summary_weight * summary_vecs)

    D = composed_base.shape[1] if composed_base.size else 0
    out = np.zeros((len(names), D), dtype=np.float32)
    compose_start = time.perf_counter()
    total_nodes = len(topo)
    for processed, node in enumerate(topo, start=1):
        idx = label2idx[node]
        parents = parent_indices[idx]
        if not parents:
            out[idx] = composed_base[idx]
        elif len(parents) == 1:
            out[idx] = composed_base[idx] + gamma * out[parents[0]]
        else:
            parent_vec = out[list(parents)].mean(axis=0, dtype=np.float32)
            out[idx] = composed_base[idx] + gamma * parent_vec

        if processed % 5000 == 0 or processed == total_nodes:
            logger.info(
                "Composed embeddings for %d/%d taxonomy nodes", processed, total_nodes
            )

    depth_limit = child_aggregation_depth
    apply_child_agg = child_weight > 0.0
    if apply_child_agg and depth_limit is not None and depth_limit <= 0:
        apply_child_agg = False

    if apply_child_agg:
        aggregated = out.copy()
        contrib = out.copy()
        depth = 0
        while True:
            depth += 1
            parent_buffer = np.zeros_like(out)
            active = False
            for child_idx, parents in enumerate(parent_indices):
                if not parents:
                    continue
                vec = contrib[child_idx]
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-9:
                    continue
                normalised_child = (vec / norm).astype(np.float32, copy=False)
                for parent_idx in parents:
                    parent_buffer[parent_idx] += normalised_child
                    active = True
            if not active:
                break
            for parent_idx in range(len(parent_indices)):
                vec = parent_buffer[parent_idx]
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-9:
                    parent_buffer[parent_idx].fill(0.0)
                    continue
                normalised_parent = (vec / norm).astype(np.float32, copy=False)
                aggregated[parent_idx] += child_weight * normalised_parent
                parent_buffer[parent_idx] = normalised_parent
            if depth_limit is not None and depth >= depth_limit:
                break
            contrib = parent_buffer
        out = aggregated

    out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    logger.info(
        "Finished composing taxonomy embeddings in %.2fs",
        time.perf_counter() - compose_start,
    )
    logger.info(
        "Completed taxonomy embedding pipeline in %.2fs",
        time.perf_counter() - start_ts,
    )
    return names, out
