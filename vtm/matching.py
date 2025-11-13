"""Helpers for matching items to taxonomy nodes via the LLM."""

from __future__ import annotations

import json
import re
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from vtm.config import HttpConfig, LLMConfig
from openai.types.chat import ChatCompletionMessageParam
from .embedding import Embedder, collect_item_texts
from .snap import maybe_snap_to_child
from .llm_chat import (
    DEFAULT_MATCH_RESPONSE_FORMAT,
    json_schema_response_format,
    llama_completion_many,
)
from .prompts import PromptRenderer, create_prompt_renderer
from .string_similarity import normalized_score, normalized_token_set_ratio
from .evaluate.parallelism import sock_read_timeout

_PROMPT_DEBUG_SHOWN = False

logger = logging.getLogger(__name__)


def _format_prompt(messages: Sequence[ChatCompletionMessageParam]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message.get("role", "?")).upper()
        content = str(message.get("content", ""))
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _print_prompt_once(messages: Sequence[ChatCompletionMessageParam]) -> None:
    """Print the first LLM prompt for debugging."""

    global _PROMPT_DEBUG_SHOWN
    if not _PROMPT_DEBUG_SHOWN:
        _PROMPT_DEBUG_SHOWN = True
        logger.info(
            "\n====== LLM PROMPT (one-time) ======\n%s\n====== END PROMPT ======\n",
            _format_prompt(messages),
        )


def _build_allowed_index_map(
    allowed_labels: Sequence[str],
    name_to_idx: Mapping[str, int],
) -> Dict[int, str]:
    """Map taxonomy indices to their label for the allowed subset."""

    idx_map: Dict[int, str] = {}
    for label in allowed_labels:
        idx = name_to_idx.get(label)
        if idx is not None:
            idx_map[idx] = label
    return idx_map


def _normalize_similarity_cutoff(value: float) -> float:
    """Clamp similarity cutoffs into the ``[0, 1]`` interval."""

    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return float(value)
    limited = min(float(value), 100.0)
    return min(1.0, normalized_score(limited))


def _canonicalize_label_text(
    pred_text: Optional[str],
    *,
    allowed_labels: Sequence[str],
    similarity_cutoff: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize the free-form text and case-fold into the allowed label set."""

    if not isinstance(pred_text, str):
        return None, None

    allowed_lookup = {label.lower(): label for label in allowed_labels}
    alias_lookup = {}
    for label in allowed_labels:
        alias = re.sub(r"\W+", " ", label).lower().strip()
        if alias and alias not in alias_lookup:
            alias_lookup[alias] = label

    normalized = pred_text.strip()
    if not normalized:
        return None, None

    direct_resolved = allowed_lookup.get(normalized.lower()) if normalized else None
    if direct_resolved:
        return normalized, direct_resolved

    while True:
        trimmed = re.sub(r"\s*\[[^()]*\]\s*$", "", normalized)
        if trimmed == normalized:
            break
        normalized = trimmed.strip()
        if not normalized:
            return None, None

    quotes = {"'", '"'}
    while (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in quotes
    ):
        normalized = normalized[1:-1].strip()
        if not normalized:
            return None, None

    resolved = allowed_lookup.get(normalized.lower()) if normalized else None

    if not resolved and normalized:
        normalized_alias = re.sub(r"\W+", " ", normalized).lower().strip()
        if normalized_alias:
            resolved = alias_lookup.get(normalized_alias)
            if not resolved:
                cutoff = _normalize_similarity_cutoff(similarity_cutoff)
                best_label: Optional[str] = None
                best_score = 0.0
                for alias_key, candidate_label in alias_lookup.items():
                    if not alias_key:
                        continue
                    score = normalized_token_set_ratio(normalized_alias, alias_key)
                    if score > best_score:
                        best_score = score
                        best_label = candidate_label
                if best_label and best_score >= cutoff:
                    resolved = best_label

    return normalized if normalized else None, resolved


def _compose_item_text(item: Mapping[str, Optional[str]]) -> str:
    parts = collect_item_texts(item, clean=True)
    return " ".join(chunk.text for chunk in parts if chunk.text)


def _build_match_result(
    req: MatchRequest,
    *,
    node_label_raw: Optional[str],
    raw: str,
    resolved_label: Optional[str],
    name_to_id: Mapping[str, str],
    name_to_path: Mapping[str, str],
    match_strategy: str,
    matched: bool,
    no_match: bool,
) -> Dict[str, Any]:
    """Construct a standard match result payload."""

    return {
        "input_item": req.item,
        "pred_label_raw": node_label_raw,
        "resolved_label": resolved_label,
        "resolved_id": name_to_id.get(resolved_label) if resolved_label else None,
        "resolved_path": name_to_path.get(resolved_label) if resolved_label else None,
        "matched": matched,
        "no_match": no_match,
        "match_strategy": match_strategy,
        "raw": raw,
    }


@dataclass(frozen=True)
class MatchRequest:
    item: Dict[str, Optional[str]]
    tree_markdown: str
    allowed_labels: Sequence[str]
    allowed_children: Mapping[str, Sequence[Sequence[str]] | Sequence[str]] | None = (
        None
    )
    slot_id: int = 0
    item_columns: Mapping[str, Any] | None = None


def _llm_kwargs_for_config(cfg: LLMConfig, *, slot_id: int) -> Dict[str, Any]:
    use_explicit_slots = getattr(cfg, "force_slot_id", False)

    kwargs: Dict[str, Any] = {
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "min_p": cfg.min_p,
        "cache_prompt": cfg.cache_prompt,
        "n_keep": cfg.n_keep,
    }
    if cfg.response_format is not None:
        kwargs["response_format"] = cfg.response_format
    elif cfg.json_schema is not None:
        kwargs["response_format"] = json_schema_response_format(
            "match_response", cfg.json_schema
        )
    else:
        kwargs["response_format"] = deepcopy(DEFAULT_MATCH_RESPONSE_FORMAT)
    if use_explicit_slots:
        kwargs["slot_id"] = slot_id
    else:
        kwargs["slot_id"] = -1
    kwargs["n_predict"] = max(int(cfg.n_predict), 64)
    return kwargs


async def match_items_to_tree(
    requests: Sequence[MatchRequest],
    *,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    tax_names: Sequence[str],
    tax_embs: np.ndarray,
    embedder: Embedder,
    hnsw_index,
    llm_config: LLMConfig,
    http_config: HttpConfig | None = None,
    prompt_renderer: PromptRenderer | None = None,
    encode_lock: Optional[threading.Lock] = None,
) -> List[Dict[str, Any]]:
    """Resolve ``requests`` to taxonomy nodes via the LLM and embedding remap."""

    if not requests:
        return []

    renderer = prompt_renderer or create_prompt_renderer()

    message_payloads: List[
        Tuple[Sequence[ChatCompletionMessageParam], Dict[str, Any]]
    ] = []
    for req in requests:
        messages = renderer.render_messages(
            req.tree_markdown, req.item, item_columns=req.item_columns
        )
        _print_prompt_once(messages)
        message_payloads.append(
            (messages, _llm_kwargs_for_config(llm_config, slot_id=req.slot_id))
        )

    resolved_timeout: float | None = None
    if http_config is None:
        resolved_timeout = max(float(llm_config.n_predict), 64.0)
    else:
        resolved_timeout = float(sock_read_timeout(http_config, llm_config))

    raw_responses = await llama_completion_many(
        message_payloads,
        llm_config.endpoint,
        model=llm_config.model,
        timeout=resolved_timeout,
        api_key=llm_config.api_key,
        http_cfg=http_config,
        llm_cfg=llm_config,
    )

    encode_guard = encode_lock or threading.Lock()
    embedding_remap_threshold = getattr(llm_config, "embedding_remap_threshold", 0.45)

    item_texts = [_compose_item_text(req.item) for req in requests]

    name_to_idx = {name: i for i, name in enumerate(tax_names)}

    results: List[Dict[str, Any]] = []
    for req, raw, item_text in zip(requests, raw_responses, item_texts):
        if raw is None:
            raise RuntimeError(
                "LLM returned no response for slot "
                f"{req.slot_id}; expected JSON payload."
            )

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Failed to parse JSON from LLM response for "
                f"slot {req.slot_id}: {raw!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"LLM response for slot {req.slot_id} is not a JSON object: {payload!r}"
            )

        node_label_raw: Optional[str] = payload.get("concept_label")

        normalized_text, canonical_label = _canonicalize_label_text(
            node_label_raw,
            allowed_labels=req.allowed_labels,
            similarity_cutoff=getattr(
                llm_config, "alias_similarity_threshold", 0.9
            ),
        )

        if canonical_label:
            snapped_label = maybe_snap_to_child(
                canonical_label,
                item_text=item_text,
                allowed_children=req.allowed_children,
                llm_config=llm_config,
                embedder=embedder,
                encode_lock=encode_guard,
            )
            snapped = bool(snapped_label and snapped_label != canonical_label)
            match_strategy = "llm_direct_and_snapped" if snapped else "llm_direct"
            results.append(
                _build_match_result(
                    req,
                    node_label_raw=node_label_raw,
                    raw=raw,
                    resolved_label=snapped_label,
                    name_to_id=name_to_id,
                    name_to_path=name_to_path,
                    match_strategy=match_strategy,
                    matched=True,
                    no_match=False,
                )
            )
            continue

        if normalized_text:
            allowed_idx_map = _build_allowed_index_map(req.allowed_labels, name_to_idx)
            if allowed_idx_map:
                allowed_indices = list(allowed_idx_map.keys())

                with encode_guard:
                    query_vecs = embedder.encode([normalized_text])
                if query_vecs.size:
                    query_vec = query_vecs[0]
                    best_idx: Optional[int] = None
                    best_similarity: float = float("-inf")

                    if hnsw_index is not None:
                        allowed_idx_set = set(allowed_indices)
                        query = query_vec.astype(np.float32, copy=False)
                        query = query[np.newaxis, :]
                        k = max(1, min(len(allowed_idx_set), 32))
                        labels = None
                        try:
                            labels, _ = hnsw_index.knn_query(
                                query,
                                k=k,
                                filter=lambda idx: idx in allowed_idx_set,
                            )
                        except TypeError:
                            labels, _ = hnsw_index.knn_query(query, k=k)

                        if labels is not None and len(labels):
                            for idx_val in labels[0]:
                                idx_int = int(idx_val)
                                if idx_int < 0 or idx_int not in allowed_idx_set:
                                    continue
                                best_idx = idx_int
                                best_similarity = float(tax_embs[idx_int] @ query_vec)
                                break

                    if best_idx is None:
                        allowed_items = list(allowed_idx_map.items())
                        allowed_embs = tax_embs[allowed_indices]
                        sims = allowed_embs @ query_vec
                        best_local_idx = int(np.argmax(sims))
                        best_similarity = float(sims[best_local_idx])
                        best_idx = allowed_items[best_local_idx][0]

                    if (
                        best_idx is not None
                        and best_similarity >= embedding_remap_threshold
                    ):
                        resolved_label = allowed_idx_map[best_idx]
                        snapped_label = maybe_snap_to_child(
                            resolved_label,
                            item_text=item_text,
                            allowed_children=req.allowed_children,
                            llm_config=llm_config,
                            embedder=embedder,
                            encode_lock=encode_guard,
                        )
                        snapped = bool(
                            snapped_label
                            and resolved_label
                            and snapped_label != resolved_label
                        )
                        match_strategy = (
                            "embedding_remap_and_snapped"
                            if snapped
                            else "embedding_remap"
                        )
                        results.append(
                            _build_match_result(
                                req,
                                node_label_raw=node_label_raw,
                                raw=raw,
                                resolved_label=snapped_label,
                                name_to_id=name_to_id,
                                name_to_path=name_to_path,
                                match_strategy=match_strategy,
                                matched=True,
                                no_match=False,
                            )
                        )
                        continue

        results.append(
            _build_match_result(
                req,
                node_label_raw=node_label_raw,
                raw=raw,
                resolved_label=None,
                name_to_id=name_to_id,
                name_to_path=name_to_path,
                match_strategy="no_match",
                matched=False,
                no_match=True,
            )
        )

    return results


async def match_item_to_tree(
    item: Dict[str, Optional[str]],
    *,
    tree_markdown: str,
    allowed_labels: Sequence[str],
    allowed_children: Mapping[str, Sequence[Sequence[str]] | Sequence[str]]
    | None = None,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    tax_names: Sequence[str],
    tax_embs: np.ndarray,
    embedder: Embedder,
    hnsw_index,
    llm_config: LLMConfig,
    slot_id: int = 0,
    item_columns: Mapping[str, Any] | None = None,
    prompt_renderer: PromptRenderer | None = None,
    encode_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for single-item matching."""

    result = await match_items_to_tree(
        [
            MatchRequest(
                item=item,
                tree_markdown=tree_markdown,
                allowed_labels=tuple(allowed_labels),
                allowed_children=allowed_children,
                slot_id=slot_id,
                item_columns=item_columns,
            )
        ],
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        tax_names=tax_names,
        tax_embs=tax_embs,
        embedder=embedder,
        hnsw_index=hnsw_index,
        llm_config=llm_config,
        prompt_renderer=prompt_renderer,
        encode_lock=encode_lock,
    )
    return result[0]
