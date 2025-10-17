from __future__ import annotations

import difflib
import json
import re
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import aiohttp
import numpy as np

from config import LLMConfig
from .embedding import Embedder
from .llm_chat import (
    GRAMMAR_RESPONSE,
    llama_completion_async,
    make_tree_match_prompt,
)

_PROMPT_DEBUG_SHOWN = False

def _print_prompt_once(prompt: str) -> None:
    """Print the first LLM prompt for debugging."""

    global _PROMPT_DEBUG_SHOWN
    if not _PROMPT_DEBUG_SHOWN:
        _PROMPT_DEBUG_SHOWN = True
        print("\n====== LLM PROMPT (one-time) ======\n")
        print(prompt)
        print("\n====== END PROMPT ======\n")
        sys.stdout.flush()


def _build_allowed_index_map(
    allowed_labels: Sequence[str],
    tax_names: Sequence[str],
) -> Dict[int, str]:
    """Map taxonomy indices to their label for the allowed subset."""

    idx_map: Dict[int, str] = {}
    name_to_idx = {name: i for i, name in enumerate(tax_names)}
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


async def match_item_to_tree(
    item: Dict[str, Optional[str]],
    *,
    tree_markdown: str,
    allowed_labels: Sequence[str],
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    tax_names: Sequence[str],
    tax_embs: np.ndarray,
    embedder: Embedder,
    hnsw_index,
    llm_config: LLMConfig,
    slot_id: int = 0,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    """Resolve ``item`` to the best taxonomy node via the LLM with embedding remap."""

    prompt = make_tree_match_prompt(tree_markdown, item)
    _print_prompt_once(prompt)

    llama_kwargs: Dict[str, Any] = {
        "temperature": llm_config.temperature,
        "top_k": llm_config.top_k,
        "top_p": llm_config.top_p,
        "min_p": llm_config.min_p,
        "grammar": (
            llm_config.grammar if llm_config.grammar is not None else GRAMMAR_RESPONSE
        ),
        "cache_prompt": llm_config.cache_prompt,
        "n_keep": llm_config.n_keep,
        "slot_id": slot_id,
        "session": session,
    }
    llama_kwargs["n_predict"] = max(int(llm_config.n_predict), 64)

    raw = await llama_completion_async(
        prompt,
        llm_config.endpoint,
        **llama_kwargs,
    )

    node_label_raw: Optional[str] = None
    try:
        payload = json.loads(raw)
        node_label_raw = payload.get("concept_label")
    except Exception:
        node_label_raw = None

    normalized_text, canonical_label = _canonicalize_label_text(
        node_label_raw, allowed_labels=allowed_labels
    )

    if canonical_label:
        return {
            "input_item": item,
            "pred_label_raw": node_label_raw,
            "resolved_label": canonical_label,
            "resolved_id": name_to_id.get(canonical_label),
            "resolved_path": name_to_path.get(canonical_label),
            "matched": True,
            "no_match": False,
            "match_strategy": "llm_direct",
        }

    embedding_remap_threshold = getattr(llm_config, "embedding_remap_threshold", 0.45)

    if normalized_text:
        allowed_idx_map = _build_allowed_index_map(allowed_labels, tax_names)
        if allowed_idx_map:
            allowed_items = list(allowed_idx_map.items())
            allowed_indices = [idx for idx, _ in allowed_items]
            allowed_embs = tax_embs[allowed_indices]

            query_vecs = embedder.encode([normalized_text])
            if query_vecs.size:
                query_vec = query_vecs[0]
                sims = allowed_embs @ query_vec
                best_local_idx = int(np.argmax(sims))
                best_similarity = float(sims[best_local_idx])

                if best_similarity >= embedding_remap_threshold:
                    _, resolved_label = allowed_items[best_local_idx]
                    return {
                        "input_item": item,
                        "pred_label_raw": node_label_raw,
                        "resolved_label": resolved_label,
                        "resolved_id": name_to_id.get(resolved_label),
                        "resolved_path": name_to_path.get(resolved_label),
                        "matched": True,
                        "no_match": False,
                        "match_strategy": "embedding_remap",
                    }

    return {
        "input_item": item,
        "pred_label_raw": node_label_raw,
        "resolved_label": None,
        "resolved_id": None,
        "resolved_path": None,
        "matched": False,
        "no_match": True,
        "raw": raw,
        "match_strategy": "no_match",
    }
