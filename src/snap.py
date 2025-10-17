"""Snap-to-child helper utilities for taxonomy matching."""

from __future__ import annotations

import threading
from typing import Mapping, Optional, Sequence

import textdistance

from config import LLMConfig
from .embedding import Embedder


def _textdistance_cosine(a: str, b: str) -> float:
    return float(textdistance.cosine.normalized_similarity(a or "", b or ""))


def _entropy_ncd(a: str, b: str) -> float:
    return float(textdistance.entropy_ncd.normalized_similarity(a or "", b or ""))


def _normalize_snap_similarity(name: Optional[str]) -> str:
    if not name:
        return "entropy_ncd"

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"textdistance", "textdistance_cosine", "cosine"}:
        return "textdistance_cosine"
    if normalized in {
        "entropy",
        "entropy_ncd",
        "ncd",
        "token_overlap",
        "token_overlap_cosine",
    }:
        return "entropy_ncd"
    if normalized in {"sapbert", "sapbert_cosine", "embedding", "embedding_cosine"}:
        return "embedding"
    return "entropy_ncd"


def _snap_with_string_similarity(
    parent: str,
    children: Sequence[str],
    item_text: str,
    *,
    similarity: str,
    margin: float,
) -> str:
    sim_fn = {
        "textdistance_cosine": _textdistance_cosine,
        "entropy_ncd": _entropy_ncd,
    }.get(similarity, _entropy_ncd)

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
