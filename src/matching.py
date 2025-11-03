"""Helpers for matching items to taxonomy nodes via the LLM."""

from __future__ import annotations

import difflib
import json
import re
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import aiohttp
import numpy as np
from config import LLMConfig
from .embedding import Embedder, collect_item_texts
from .snap import maybe_snap_to_child
from .llm_chat import (
    GRAMMAR_RESPONSE,
    llama_completion_many,
    make_tree_match_prompt,
)

_PROMPT_DEBUG_SHOWN = False

logger = logging.getLogger(__name__)


def _print_prompt_once(prompt: str) -> None:
    """Print the first LLM prompt for debugging."""

    global _PROMPT_DEBUG_SHOWN
    if not _PROMPT_DEBUG_SHOWN:
        _PROMPT_DEBUG_SHOWN = True
        logger.debug("\n====== LLM PROMPT (one-time) ======\n%s\n====== END PROMPT ======\n", prompt)


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


def _canonicalize_label_text(
    pred_text: Optional[str],
    *,
    allowed_labels: Sequence[str],
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
                alias_keys = list(alias_lookup.keys())
                close_matches = difflib.get_close_matches(
                    normalized_alias, alias_keys, n=1, cutoff=0.9
                )
                if close_matches:
                    resolved = alias_lookup.get(close_matches[0])

    return normalized if normalized else None, resolved


def _compose_item_text(item: Mapping[str, Optional[str]]) -> str:
    parts = collect_item_texts(item, clean=True)
    return " ".join(part for part in parts if part)


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


def _llm_kwargs_for_config(cfg: LLMConfig, *, slot_id: int) -> Dict[str, Any]:
    use_explicit_slots = getattr(cfg, "force_slot_id", False)

    kwargs: Dict[str, Any] = {
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "min_p": cfg.min_p,
        "grammar": cfg.grammar if cfg.grammar is not None else GRAMMAR_RESPONSE,
        "cache_prompt": cfg.cache_prompt,
        "n_keep": cfg.n_keep,
    }
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
    session: Optional[aiohttp.ClientSession] = None,
    encode_lock: Optional[threading.Lock] = None,
) -> List[Dict[str, Any]]:
    """Resolve ``requests`` to taxonomy nodes via the LLM and embedding remap."""

    if not requests:
        return []

    prompt_payloads: List[Tuple[str, Dict[str, Any]]] = []
    for req in requests:
        prompt = make_tree_match_prompt(req.tree_markdown, req.item)
        _print_prompt_once(prompt)
        prompt_payloads.append(
            (prompt, _llm_kwargs_for_config(llm_config, slot_id=req.slot_id))
        )

    raw_responses = await llama_completion_many(
        prompt_payloads,
        llm_config.endpoint,
        timeout=max(float(llm_config.n_predict), 64.0),
        session=session,
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
            node_label_raw, allowed_labels=req.allowed_labels
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
                allowed_items = list(allowed_idx_map.items())
                allowed_indices = [idx for idx, _ in allowed_items]
                allowed_embs = tax_embs[allowed_indices]

                with encode_guard:
                    query_vecs = embedder.encode([normalized_text])
                if query_vecs.size:
                    query_vec = query_vecs[0]
                    sims = allowed_embs @ query_vec
                    best_local_idx = int(np.argmax(sims))
                    best_similarity = float(sims[best_local_idx])

                    if best_similarity >= embedding_remap_threshold:
                        _, resolved_label = allowed_items[best_local_idx]
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
    session: Optional[aiohttp.ClientSession] = None,
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
            )
        ],
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        tax_names=tax_names,
        tax_embs=tax_embs,
        embedder=embedder,
        hnsw_index=hnsw_index,
        llm_config=llm_config,
        session=session,
        encode_lock=encode_lock,
    )
    return result[0]
