"""Snap-to-child helper utilities for taxonomy matching."""

from __future__ import annotations

import threading
from typing import Mapping, Optional, Sequence

from config import LLMConfig
from .embedding import Embedder
from .string_similarity import (
    normalized_token_set_ratio,
    normalized_token_sort_ratio,
)


def _token_set_similarity(a: str, b: str) -> float:
    return normalized_token_set_ratio(a or "", b or "")


def _token_sort_similarity(a: str, b: str) -> float:
    return normalized_token_sort_ratio(a or "", b or "")


def _normalize_snap_similarity(name: Optional[str]) -> str:
    if not name:
        return "token_sort"

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"token_set", "set"}:
        return "token_set"
    if normalized in {"sapbert", "sapbert_cosine", "embedding", "embedding_cosine"}:
        return "embedding"
    return "token_sort"


def _snap_with_string_similarity(
    parent: str,
    children: Sequence[str],
    item_text: str,
    *,
    similarity: str,
    margin: float,
) -> str:
    sim_fn = {
        "token_set": _token_set_similarity,
        "token_sort": _token_sort_similarity,
    }.get(similarity, _token_sort_similarity)

    parent_score = sim_fn(parent, item_text)
    best_child = None
    best_score = parent_score
    for child in children:
        score = sim_fn(child, item_text)
        if score > best_score:
            best_child = child
            best_score = score

    if best_child and (best_score - parent_score) >= margin:
        return best_child
    return parent


def _snap_with_embedding(
    parent: str,
    children: Sequence[str],
    item_text: str,
    *,
    embedder: Embedder,
    encode_lock: threading.Lock,
    margin: float,
) -> str:
    if not parent or not item_text or not children:
        return parent

    label_texts = [parent]
    child_labels: list[str] = []
    for child in children:
        if child:
            label_texts.append(child)
            child_labels.append(child)

    if len(label_texts) == 1:
        return parent

    with encode_lock:
        vectors = embedder.encode([item_text] + label_texts)

    if vectors.size == 0 or vectors.shape[0] != len(label_texts) + 1:
        return parent

    item_vec = vectors[0]
    label_vecs = vectors[1:]
    parent_vec = label_vecs[0]
    child_vecs = label_vecs[1:]

    parent_score = float(parent_vec @ item_vec)
    best_child = None
    best_score = parent_score
    for child, vec in zip(child_labels, child_vecs):
        score = float(vec @ item_vec)
        if score > best_score:
            best_child = child
            best_score = score

    if best_child and (best_score - parent_score) >= margin:
        return best_child
    return parent


def maybe_snap_to_child(
    label: Optional[str],
    *,
    item_text: str,
    allowed_children: Mapping[str, Sequence[str]] | None,
    llm_config: LLMConfig,
    embedder: Embedder,
    encode_lock: threading.Lock | None,
) -> Optional[str]:
    if not label or not getattr(llm_config, "snap_to_child", False):
        return label

    if not allowed_children:
        return label

    children = allowed_children.get(label)
    if not children:
        return label

    margin = float(getattr(llm_config, "snap_margin", 0.0))
    if margin <= 0:
        margin = 0.0

    similarity_mode = _normalize_snap_similarity(
        getattr(llm_config, "snap_similarity", None)
    )

    if similarity_mode == "embedding":
        lock = encode_lock or threading.Lock()
        return _snap_with_embedding(
            label,
            children,
            item_text,
            embedder=embedder,
            encode_lock=lock,
            margin=margin,
        )

    return _snap_with_string_similarity(
        label,
        children,
        item_text,
        similarity=similarity_mode,
        margin=margin,
    )
