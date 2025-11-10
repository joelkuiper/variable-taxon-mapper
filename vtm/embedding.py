"""Embedding utilities including SapBERT wrappers and HNSW helpers."""

from __future__ import annotations

import logging
import time
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
from transformers import AutoModel, AutoTokenizer


import pandas as pd

from .taxonomy import ensure_traversal_cache, taxonomy_node_texts
from .utils import clean_text
from config import FieldMappingConfig


logger = logging.getLogger(__name__)


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
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 128,
        fp16: bool = True,
        mean_pool: bool = False,  # False = [CLS]
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.mean_pool = mean_pool

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        n_texts = len(texts)
        if n_texts == 0:
            hidden_size = getattr(self.model.config, "hidden_size", 768)
            return np.zeros((0, hidden_size), dtype=np.float32)

        hidden_size = getattr(self.model.config, "hidden_size", 768)
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
                "Encoding %d texts with batch size %d on %s",
                n_texts,
                bs,
                self.device,
            )

        for i in range(0, n_texts, bs):
            batch = texts[i : i + bs]
            toks = self.tok.batch_encode_plus(
                batch,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            total_tokens += int(toks["attention_mask"].sum().item())
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                last_hidden = self.model(**toks)[0]
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
                    "Encoded %d/%d texts (%.0f%%)",
                    offset,
                    n_texts,
                    100 * offset / n_texts,
                )
                next_progress += progress_interval

        if log_progress:
            duration = time.perf_counter() - start_ts
            logger.info(
                "Finished encoding %d texts in %.2fs (%.1f tokens/s)",
                n_texts,
                duration,
                total_tokens / duration if duration > 0 else float("inf"),
            )
        return l2_normalize(out)


def _resolve_default_fields(
    item: Mapping[str, Optional[str]],
    field_mapping: FieldMappingConfig | None,
) -> List[str]:
    if field_mapping is not None:
        return field_mapping.item_text_keys()
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
    return FieldMappingConfig().item_text_keys()


def collect_item_texts(
    item: Mapping[str, Optional[str]],
    *,
    fields: Sequence[str] | None = None,
    field_mapping: FieldMappingConfig | None = None,
    clean: bool = True,
    max_length: int = 256,
) -> List[str]:
    """Extract candidate text fields from an item for embedding."""

    resolved_fields: Sequence[str]
    if fields is None:
        resolved_fields = _resolve_default_fields(item, field_mapping)
    else:
        resolved_fields = list(fields)

    texts: List[str] = []
    for key in resolved_fields:
        raw = item.get(key)
        if clean:
            text = clean_text(raw)
            if text and text != "(empty)":
                texts.append(text[:max_length])
        else:
            if isinstance(raw, str):
                text = raw.strip()
            elif raw is None:
                text = ""
            else:
                text = str(raw).strip()
            if text:
                texts.append(text[:max_length])
    return texts


def encode_item_texts(
    item: Mapping[str, Optional[str]],
    embedder: Embedder,
    *,
    fields: Sequence[str] | None = None,
    field_mapping: FieldMappingConfig | None = None,
    clean: bool = True,
    max_length: int = 256,
    texts: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Encode selected text fields from ``item`` using ``embedder``."""

    if texts is None:
        texts = collect_item_texts(
            item,
            fields=fields,
            field_mapping=field_mapping,
            clean=clean,
            max_length=max_length,
        )
    else:
        texts = list(texts)

    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    return embedder.encode(texts)


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


def _extract_summary_map(
    summaries: Optional[object],
) -> Dict[str, str]:
    if summaries is None:
        return {}

    summary_map: Dict[str, str] = {}

    if isinstance(summaries, Mapping):
        items: Iterable[Tuple[object, object]] = summaries.items()
    elif isinstance(summaries, pd.Series):
        items = summaries.items()
    elif isinstance(summaries, pd.DataFrame):
        items = summaries[["name", "definition_summary"]].itertuples(
            index=False, name=None
        )
    else:
        raise TypeError("summaries must be a mapping, pandas Series/DataFrame, or None")

    for entry in items:
        if isinstance(entry, tuple):
            if not entry:
                continue
            key = entry[0]
            value = entry[1] if len(entry) > 1 else None
        elif isinstance(entry, Mapping):
            # Some pandas iterators yield dict-like rows instead of tuples.
            key = entry.get("name")
            value = entry.get("definition_summary")
        else:
            continue

        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.strip()
        summary = value.strip()
        if name and summary and name not in summary_map:
            summary_map[name] = summary

    return summary_map


def build_taxonomy_embeddings_composed(
    G,
    embedder: Embedder,
    gamma: float = 0.3,
    *,
    summaries: Optional[object] = None,
    summary_weight: float = 1.0,
    taxonomy_text_transform: Optional[Callable[[str], str]] = None,
) -> Tuple[List[str], np.ndarray]:
    start_ts = time.perf_counter()
    names = taxonomy_node_texts(G)
    label2idx = {n: i for i, n in enumerate(names)}

    topo = _topological_order(G)
    cache = ensure_traversal_cache(G)
    all_parents = cache.get("all_parents", {}) if cache is not None else {}
    parent_indices: List[Tuple[int, ...]] = [tuple() for _ in names]
    if all_parents:
        for node, parents in all_parents.items():
            idx = label2idx.get(node)
            if idx is None or not parents:
                continue
            parent_indices[idx] = tuple(
                label2idx[p] for p in parents if p in label2idx
            )
    else:
        for node in names:
            idx = label2idx[node]
            preds = tuple(G.predecessors(node))
            if preds:
                parent_indices[idx] = tuple(
                    label2idx[p] for p in preds if p in label2idx
                )

    logger.info(
        "Composing taxonomy embeddings for %d nodes (gamma=%.2f)", len(names), gamma
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

    summary_map = _extract_summary_map(summaries)
    summary_vecs = np.zeros_like(name_vecs)
    if summary_map:
        summary_start = time.perf_counter()
        texts: List[str] = []
        idxs: List[int] = []
        for i, n in enumerate(names):
            s = summary_map.get(n)
            if s:
                texts.append(s)
                idxs.append(i)
        if texts:
            encoded = embedder.encode(texts)
            for j, idx in enumerate(idxs):
                summary_vecs[idx] = encoded[j]
        logger.info(
            "Encoded %d taxonomy summaries in %.2fs",
            len(texts),
            time.perf_counter() - summary_start,
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
