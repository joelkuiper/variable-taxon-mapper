"""Inference helpers: pruning, prompt assembly, and llama.cpp calls."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import networkx as nx
import numpy as np
import pandas as pd

from .embedding import Embedder, l2_normalize
from .llm_chat import (
    GRAMMAR_RESPONSE,
    llama_completion_async,
    make_tree_match_prompt,
)
from .taxonomy import (
    make_label_display,
    sort_key_factory,
    ancestors_to_root,
    roots_in_order,
)


def clean_text(v) -> str:
    if v is None:
        return "(empty)"
    if isinstance(v, str):
        s = v.strip()
        return s if s else "(empty)"
    try:
        if isinstance(v, float) and math.isnan(v):
            return "(empty)"
    except Exception:
        pass
    s = str(v).strip()
    return s if s else "(empty)"


def encode_item_parts(item: Dict[str, Optional[str]], embedder: Embedder) -> np.ndarray:
    fields: List[str] = []
    for k in ("label", "name", "description"):
        s = clean_text(item.get(k))
        if s and s != "(empty)":
            fields.append(s[:256])
    embs = embedder.encode(fields)
    return l2_normalize(embs)


def _hnsw_anchor_indices(
    item_embs: np.ndarray,
    hnsw_index,
    *,
    N: int,
    top_k_nodes: int,
    overfetch_mult: int = 3,
    min_overfetch: int = 128,
) -> List[int]:
    """
    Query HNSW for each item-part embedding; return top-k node indices after
    max-pooling similarities across parts.
    """
    if item_embs.size == 0:
        return list(range(min(top_k_nodes, N)))

    Kq = min(max(min_overfetch, top_k_nodes * overfetch_mult), N)

    # Pool max(sim) over all query parts
    scores = np.full(N, -np.inf, dtype=np.float32)
    seen = np.zeros(N, dtype=bool)
    for q in item_embs:
        labels, dists = hnsw_index.knn_query(q[np.newaxis, :].astype(np.float32), k=Kq)
        labels, dists = labels[0], dists[0]
        sims = 1.0 - dists.astype(np.float32)  # cosine dist -> sim
        for idx, sim in zip(labels, sims):
            if idx < 0:
                continue
            seen[idx] = True
            if sim > scores[idx]:
                scores[idx] = sim

    if not seen.any():
        return list(range(min(top_k_nodes, N)))

    scores[~seen] = -np.inf

    # Top-k by pooled score
    idxs = np.argpartition(-scores, kth=min(top_k_nodes - 1, N - 1))[:top_k_nodes]
    idxs = idxs[np.argsort(-scores[idxs])]
    filtered = [int(i) for i in idxs if seen[int(i)]]
    if filtered:
        return filtered
    return [int(i) for i in idxs]


def _expand_allowed_nodes(
    G: nx.DiGraph,
    anchors: List[str],
    *,
    desc_max_depth: int,
    max_total_nodes: int,
) -> Set[str]:
    """Assemble an allowed-node set while respecting a hard cap.

    Nodes are added by iterating anchors in order and including their ancestor
    paths plus a down-sampled selection of descendants. Descendants are added in
    breadth-first order up to ``desc_max_depth`` while ensuring that at least
    one descendant survives for each retained anchor (when any exist) without
    exceeding ``max_total_nodes``.
    """

    def _descendant_order(node: str) -> List[str]:
        if desc_max_depth <= 0:
            return []
        order: List[str] = []
        seen: Set[str] = {node}
        queue: deque[Tuple[str, int]] = deque([(node, 0)])
        while queue:
            cur, depth = queue.popleft()
            if depth >= desc_max_depth:
                continue
            for child in G.successors(cur):
                if child in seen:
                    continue
                seen.add(child)
                order.append(child)
                queue.append((child, depth + 1))
        return order

    allowed: Set[str] = set()
    if max_total_nodes <= 0:
        return allowed

    total_anchors = len(anchors)

    for idx, anchor in enumerate(anchors):
        if len(allowed) >= max_total_nodes:
            break

        ancestors = ancestors_to_root(G, anchor)
        new_ancestors = [n for n in ancestors if n not in allowed]

        if len(allowed) + len(new_ancestors) > max_total_nodes:
            # Cannot fit this anchor and its ancestors; skip it entirely.
            continue

        for n in new_ancestors:
            allowed.add(n)

        descendants_order = _descendant_order(anchor)
        has_descendants = bool(descendants_order)

        if not has_descendants:
            continue

        remaining_capacity = max_total_nodes - len(allowed)
        if remaining_capacity <= 0:
            # Roll back the ancestors we just added and skip this anchor so we
            # don't violate the descendant guarantee.
            for n in new_ancestors:
                allowed.remove(n)
            continue

        remaining_anchors = max(total_anchors - (idx + 1), 0)
        if remaining_anchors > 0:
            per_anchor_base = remaining_capacity // (remaining_anchors + 1)
            extra = remaining_capacity % (remaining_anchors + 1)
            desc_budget = per_anchor_base + (1 if extra > 0 else 0)
        else:
            desc_budget = remaining_capacity
        desc_budget = max(1, min(desc_budget, remaining_capacity))

        added_descendants: Set[str] = set()
        for node in descendants_order:
            if len(allowed) >= max_total_nodes or len(added_descendants) >= desc_budget:
                break
            if node in allowed:
                continue
            if not any(pred in allowed for pred in G.predecessors(node)):
                continue
            allowed.add(node)
            added_descendants.add(node)

        if not added_descendants:
            # No descendant could be added; undo new nodes for this anchor.
            for node in added_descendants:
                allowed.remove(node)
            for node in new_ancestors:
                allowed.remove(node)

    return allowed


def _rank_allowed_nodes(anchors: List[str], allowed: Set[str]) -> List[str]:
    """
    Rank nodes: anchors first (deduped, in order), then remaining allowed in alpha.
    """
    allowed_ranked: List[str] = []
    seen: Set[str] = set()
    for a in anchors:
        if a in allowed and a not in seen:
            allowed_ranked.append(a)
            seen.add(a)
    for n in sorted(allowed, key=lambda s: s.lower()):
        if n not in seen:
            allowed_ranked.append(n)
            seen.add(n)
    return allowed_ranked


def _render_tree_markdown(
    G: nx.DiGraph,
    allowed: Set[str],
    df: pd.DataFrame,
    *,
    name_col: str,
    order_col: str,
    gloss_map: Optional[Dict[str, str]],
) -> Tuple[str, List[str]]:
    """
    Render the (allowed) subtree as markdown. If allowed produces nothing,
    fall back to rendering the full tree and return full-node alpha order.
    """
    order_map = df.groupby(name_col)[order_col].min().to_dict()
    sort_key = sort_key_factory(order_map)

    lines: List[str] = []

    def _walk(node: str, depth: int):
        if node not in allowed:
            return
        label_display = make_label_display(node, gloss_map or {})
        lines.append("  " * depth + f"- {label_display}")
        for c in sorted(G.successors(node), key=sort_key):
            if c in allowed:
                _walk(c, depth + 1)

    for r in roots_in_order(G, sort_key):
        if r in allowed:
            _walk(r, 0)

    if lines:
        tree_md = "\n".join(lines)
        return tree_md, None

    # Fallback: render the full taxonomy without filtering and provide
    # an alphabetical list of all labels for downstream consumers.
    fallback_lines: List[str] = []

    def _walk_full(node: str, depth: int):
        label_display = make_label_display(node, gloss_map or {})
        fallback_lines.append("  " * depth + f"- {label_display}")
        for c in sorted(G.successors(node), key=sort_key):
            _walk_full(c, depth + 1)

    for r in roots_in_order(G, sort_key):
        _walk_full(r, 0)

    tree_md = "\n".join(fallback_lines)
    fallback_ranked = sorted(G.nodes, key=lambda s: s.lower())

    return tree_md, fallback_ranked


def pruned_tree_markdown_for_item(
    item: Dict[str, Optional[str]],
    *,
    G: nx.DiGraph,
    df: pd.DataFrame,
    embedder: "Embedder",
    tax_names: List[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_col: str = "name",
    order_col: str = "order",
    top_k_nodes: int = 128,
    desc_max_depth: int = 3,
    max_total_nodes: int = 1200,
    gloss_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    """
    Build a pruned TREE (markdown) and an ordered candidate list for the ITEM.
    Uses HNSW to pick anchor nodes via max-pooled cosine similarity across
    {label,name,description} embeddings.
    """
    # 1) Encode item parts
    item_embs = encode_item_parts(item, embedder)

    # 2) Anchor selection (HNSW or fallback to head of taxonomy)
    if item_embs.size == 0:
        anchor_idxs = list(range(min(top_k_nodes, len(tax_names))))
    else:
        anchor_idxs = _hnsw_anchor_indices(
            item_embs,
            hnsw_index,
            N=len(tax_names),
            top_k_nodes=top_k_nodes,
            overfetch_mult=3,
            min_overfetch=128,
        )
    anchors = [tax_names[i] for i in anchor_idxs]

    # 3) Allowed node set (anchors + relatives) with size cap
    allowed = _expand_allowed_nodes(
        G,
        anchors,
        desc_max_depth=desc_max_depth,
        max_total_nodes=max_total_nodes,
    )

    # 4) Rank allowed candidates
    allowed_ranked = _rank_allowed_nodes(anchors, allowed)

    # 5) Render subtree markdown (fallback to full tree if empty)
    tree_md, fallback_ranked = _render_tree_markdown(
        G,
        allowed,
        df,
        name_col=name_col,
        order_col=order_col,
        gloss_map=gloss_map,
    )
    if fallback_ranked is not None:
        allowed_ranked = fallback_ranked

    # 6) Compose concise "Candidates" + full "Taxonomy" markdown
    top_show = min(40, len(allowed_ranked))
    md_lines: List[str] = ["### Candidates \n"]
    for lbl in allowed_ranked[:top_show]:
        md_lines.append(
            f"- {make_label_display(lbl, gloss_map or {}, use_summary=False)}"
        )
    md_lines.append("\n### Taxonomy \n")
    md_lines.append(tree_md)

    return "\n".join(md_lines), allowed_ranked


# ---- helpers: prompt / LLM ---------------------------------------------------

_PROMPT_DEBUG_SHOWN = False


def _print_prompt_once(prompt: str) -> None:
    """Print the LLM prompt once for inspection."""
    global _PROMPT_DEBUG_SHOWN
    if not _PROMPT_DEBUG_SHOWN:
        _PROMPT_DEBUG_SHOWN = True
        print("\n====== LLM PROMPT (one-time) ======\n")
        print(prompt)
        print("\n====== END PROMPT ======\n")
        sys.stdout.flush()


# ---- helpers: embeddings / ANN fallback -------------------------------------
def _encode_item_parts_local(
    item: Dict[str, Optional[str]],
    embedder: "Embedder",
) -> np.ndarray:
    """
    Encode non-empty item parts (label/name/description) using the provided embedder.
    L2-normalized; may return (0, D) if no parts.
    """
    fields: List[str] = []
    for k in ("label", "name", "description"):
        s = item.get(k)
        if isinstance(s, str):
            s = s.strip()
        if s:
            fields.append(str(s)[:256])
    if not fields:
        return np.zeros((0, 768), dtype=np.float32)
    embs = embedder.encode(fields)  # expected L2-normalized by embedder
    return embs.astype(np.float32, copy=False)


def _build_allowed_index_map(
    allowed_labels: List[str],
    tax_names: List[str],
) -> Dict[int, str]:
    """Map taxonomy indices -> label, restricted to allowed labels present in taxonomy."""
    idx_map: Dict[int, str] = {}
    name_to_idx = {n: i for i, n in enumerate(tax_names)}
    for lbl in allowed_labels:
        idx = name_to_idx.get(lbl)
        if idx is not None:
            idx_map[idx] = lbl
    return idx_map


def _exact_maxpool_within(
    item_embs: np.ndarray,
    sub_embs: np.ndarray,
) -> Optional[int]:
    """
    Exact max-pool over (K x D) sub_embs vs (m x D) item_embs.
    Returns best row index in sub_embs or None if not computable.
    """
    if item_embs.size == 0 or sub_embs.size == 0:
        return None
    sims = sub_embs @ item_embs.T  # (K, m)
    best = sims.max(axis=1)  # (K,)
    return int(np.argmax(best)) if best.size else None


def _hnsw_top_match_for_query_vec(
    q_vec: np.ndarray,
    *,
    allowed_idx_map: Dict[int, str],
    tax_embs: np.ndarray,
    hnsw_index,
    fanout_k: int,
) -> Optional[str]:
    """
    Single-vector ANN search with HNSW, then restrict to allowed indices and pick the best by cosine sim.
    Falls back to exact within-allowed if ANN result doesn't intersect allowed.
    """
    if q_vec.size == 0 or not allowed_idx_map:
        return None

    N = tax_embs.shape[0]
    Kq = min(max(64, fanout_k), N)

    # HNSW returns cosine *distance* (1 - sim); convert back to similarity.
    labels, dists = hnsw_index.knn_query(q_vec[np.newaxis, :].astype(np.float32), k=Kq)
    labels, dists = labels[0], dists[0]

    # Keep only allowed; pick the best by highest similarity.
    best_label = None
    best_sim = -1.0
    for idx, dist in zip(labels, dists):
        if idx < 0:
            continue
        if idx in allowed_idx_map:
            sim = 1.0 - float(dist)
            if sim > best_sim:
                best_sim = sim
                best_label = allowed_idx_map[idx]

    if best_label is not None:
        return best_label

    # No overlap in ANN results — do exact within allowed.
    mask = list(allowed_idx_map.keys())
    sub = tax_embs[mask]  # (K,D)
    j = _exact_maxpool_within(q_vec[np.newaxis, :], sub)  # treat as 1-part item
    return allowed_idx_map[mask[j]] if j is not None else None


def _canonicalize_label_text(
    pred_text: Optional[str], *, allowed_labels: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize the raw LLM text and case-fold into the allowed label set.

    Returns a tuple of (normalized_text, resolved_allowed_label). The normalized text
    trims whitespace, strips surrounding quotes, removes trailing parenthetical
    summaries (``(...)``), and leaves the text ready for downstream embedding. If the
    normalized text matches an allowed label ignoring case, the second element will be
    that label with the original allowed casing; otherwise it is ``None``.
    """

    if not isinstance(pred_text, str):
        return None, None

    normalized = pred_text.strip()
    if not normalized:
        return None, None

    # Drop trailing parenthetical summaries (possibly nested, e.g., ``Label (foo)``).
    while True:
        trimmed = re.sub(r"\s*\([^()]*\)\s*$", "", normalized)
        if trimmed == normalized:
            break
        normalized = trimmed.strip()
        if not normalized:
            return None, None

    # Remove surrounding matching quotes.
    quotes = {"'", '"'}
    while (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in quotes
    ):
        normalized = normalized[1:-1].strip()
        if not normalized:
            return None, None

    allowed_lookup = {label.lower(): label for label in allowed_labels}
    resolved = allowed_lookup.get(normalized.lower()) if normalized else None

    return normalized if normalized else None, resolved


def _map_freeform_label_to_allowed(
    pred_text: Optional[str],
    *,
    allowed_labels: List[str],
    tax_names: List[str],
    tax_embs: np.ndarray,
    embedder: "Embedder",
    hnsw_index,
) -> Optional[str]:
    """
    Embed the freeform LLM label string and map it to the nearest allowed taxonomy node.
    """
    normalized, resolved = _canonicalize_label_text(
        pred_text, allowed_labels=allowed_labels
    )
    if resolved:
        return resolved

    if not normalized or not allowed_labels:
        return None

    # Encode the single text (already L2-normalized by embedder).
    q = embedder.encode([normalized])  # (1,D)
    if q.size == 0:
        return None
    q = q.astype(np.float32, copy=False)[0]

    allowed_idx_map = _build_allowed_index_map(allowed_labels, tax_names)
    if not allowed_idx_map:
        return None

    fanout = max(256, 4 * len(allowed_idx_map))
    return _hnsw_top_match_for_query_vec(
        q,
        allowed_idx_map=allowed_idx_map,
        tax_embs=tax_embs,
        hnsw_index=hnsw_index,
        fanout_k=fanout,
    )


def _hnsw_fallback_choose_label(
    item: Dict[str, Optional[str]],
    *,
    allowed_labels: List[str],
    tax_names: List[str],
    tax_embs: np.ndarray,
    embedder: "Embedder",
    hnsw_index,
) -> Optional[str]:
    """
    ANN-backed fallback restricted to allowed labels:
    1) HNSW large-pool candidates, max-pool similarities per node.
    2) Filter to allowed; pick best by pooled score.
    3) If nothing scored, do exact max-pool within allowed subset.
    """
    if not allowed_labels:
        return None

    # Encode item parts
    item_embs = _encode_item_parts_local(item, embedder)  # (m,D)
    if item_embs.size == 0:
        return None

    # Determine query fanout
    N = len(tax_names)
    Kq = min(max(256, 4 * len(allowed_labels)), N)

    # Aggregate max similarity per retrieved node (over all parts).
    allowed_idx_map = _build_allowed_index_map(allowed_labels, tax_names)
    if not allowed_idx_map:
        return None

    scores: Dict[int, float] = {}
    for q in item_embs:
        labels, dists = hnsw_index.knn_query(q[np.newaxis, :].astype(np.float32), k=Kq)
        labels, dists = labels[0], dists[0]
        sims = 1.0 - dists.astype(np.float32)
        for idx, sim in zip(labels, sims):
            if idx < 0:
                continue
            # Keep only allowed nodes; max-pool across parts
            if idx in allowed_idx_map:
                if sim > scores.get(idx, -1.0):
                    scores[idx] = float(sim)

    if scores:
        best_idx = max(scores.items(), key=lambda kv: kv[1])[0]
        return allowed_idx_map[best_idx]

    # Final exact max-pool within allowed subset
    mask = list(allowed_idx_map.keys())
    if not mask:
        return None
    sub = tax_embs[mask]
    j = _exact_maxpool_within(item_embs, sub)
    return allowed_idx_map[mask[j]] if j is not None else None


async def match_item_to_tree(
    item: Dict[str, Optional[str]],
    *,
    tree_markdown: str,
    allowed_labels: List[str],
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    tax_names: List[str],
    tax_embs: np.ndarray,
    embedder: "Embedder",
    hnsw_index,
    endpoint: str = "http://127.0.0.1:8080/completions",
    temperature: float = 0.7,
    n_predict: int = 256,
    slot_id: int = 0,
    cache_prompt: bool = True,
    n_keep: int = -1,
    session: Optional[aiohttp.ClientSession] = None,
    grammar: str = GRAMMAR_RESPONSE,
) -> Dict[str, Any]:
    """
    LLM-first matching; if the label isn't an exact taxonomy match, embed the LLM text and
    map to the nearest allowed node (ANN→exact). If that still fails, fall back to item-based ANN.
    Returns a flat dict.
    """
    prompt = make_tree_match_prompt(tree_markdown, item)
    _print_prompt_once(prompt)

    raw = await llama_completion_async(
        prompt,
        endpoint,
        temperature=temperature,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        n_predict=max(n_predict, 64),
        grammar=grammar,
        cache_prompt=cache_prompt,
        n_keep=n_keep,
        slot_id=slot_id,
        session=session,
    )

    # Try to parse JSON and validate; keep the freeform label for fuzzy mapping.
    node_label_raw: Optional[str] = None
    try:
        payload = json.loads(raw)
        node_label_raw = payload.get("node_label")
    except Exception:
        node_label_raw = None  # keep as None; we'll fall back

    _, canonical_label = _canonicalize_label_text(
        node_label_raw, allowed_labels=allowed_labels
    )

    # Path A: exact allowed hit
    if canonical_label:
        return {
            "input_item": item,
            "pred_label_raw": node_label_raw,
            "resolved_label": canonical_label,
            "resolved_id": name_to_id.get(canonical_label),
            "resolved_path": name_to_path.get(canonical_label),
            "matched": True,
            "no_match": False,
        }

    # Path B: fuzzy-map the LLM freeform label into the allowed set
    mapped = _map_freeform_label_to_allowed(
        node_label_raw,
        allowed_labels=allowed_labels,
        tax_names=tax_names,
        tax_embs=tax_embs,
        embedder=embedder,
        hnsw_index=hnsw_index,
    )
    if mapped:
        return {
            "input_item": item,
            "pred_label_raw": node_label_raw,  # keep what LLM said
            "resolved_label": mapped,
            "resolved_id": name_to_id.get(mapped),
            "resolved_path": name_to_path.get(mapped),
            "matched": True,
            "no_match": False,
            "raw": raw,
        }

    # Path C: item-based ANN fallback within allowed
    chosen = _hnsw_fallback_choose_label(
        item,
        allowed_labels=allowed_labels,
        tax_names=tax_names,
        tax_embs=tax_embs,
        embedder=embedder,
        hnsw_index=hnsw_index,
    )
    if chosen:
        return {
            "input_item": item,
            "pred_label_raw": node_label_raw,
            "resolved_label": chosen,
            "resolved_id": name_to_id.get(chosen),
            "resolved_path": name_to_path.get(chosen),
            "matched": True,
            "no_match": False,
            "raw": raw,
        }

    # Path D: no match
    return {
        "input_item": item,
        "pred_label_raw": node_label_raw,
        "resolved_label": None,
        "resolved_id": None,
        "resolved_path": None,
        "matched": False,
        "no_match": True,
        "raw": raw,
    }
