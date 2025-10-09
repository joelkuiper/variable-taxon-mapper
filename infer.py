"""Inference helpers: pruning, prompt assembly, and llama.cpp calls."""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

import numpy as np
import requests

from embedding import Embedder, l2_normalize, maxpool_scores
from llm_chat import GRAMMAR_RESPONSE, llama_completion, make_tree_match_prompt
from taxonomy import (
    collect_descendants,
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
    scores = np.full(N, -1.0, dtype=np.float32)
    for q in item_embs:
        labels, dists = hnsw_index.knn_query(q[np.newaxis, :].astype(np.float32), k=Kq)
        labels, dists = labels[0], dists[0]
        sims = 1.0 - dists.astype(np.float32)  # cosine dist -> sim
        for idx, sim in zip(labels, sims):
            if sim > scores[idx]:
                scores[idx] = sim

    # Top-k by pooled score
    idxs = np.argpartition(-scores, kth=min(top_k_nodes - 1, N - 1))[:top_k_nodes]
    idxs = idxs[np.argsort(-scores[idxs])]
    # Filter out never-seen nodes (score < 0)
    return [int(i) for i in idxs if scores[int(i)] >= 0.0]


def _expand_allowed_nodes(
    G: nx.DiGraph,
    anchors: List[str],
    *,
    desc_max_depth: int,
    max_total_nodes: int,
) -> Set[str]:
    """
    Allowed set = anchors + their ancestors + descendants (up to depth),
    with a size cap that keeps ancestors of a core subset if needed.
    """
    allowed: Set[str] = set()
    for a in anchors:
        for n in ancestors_to_root(G, a):
            allowed.add(n)
        for n in collect_descendants(G, a, max_depth=desc_max_depth):
            allowed.add(n)

    if len(allowed) > max_total_nodes:
        core = set(anchors[: max_total_nodes // 4])
        allowed = core | {n for a in core for n in ancestors_to_root(G, a)}
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

    tree_md = "\n".join(lines)

    if tree_md.strip():
        return tree_md, None  # None => caller keeps its own allowed_ranked

    # Fallback: render full tree
    lines = []

    def _walk_all(node: str, depth: int):
        label_display = make_label_display(node, gloss_map or {})
        lines.append("  " * depth + f"- {label_display}")
        for c in sorted(G.successors(node), key=sort_key):
            _walk_all(c, depth + 1)

    for r in roots_in_order(G, sort_key):
        _walk_all(r, 0)

    tree_md = "\n".join(lines)
    full_ranked = sorted(list(G.nodes), key=lambda s: s.lower())
    return tree_md, full_ranked


# --- Refactored main ---------------------------------------------------------


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
        md_lines.append(f"- {make_label_display(lbl, gloss_map or {})}")
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


def _parse_llm_json(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse {"node_label": "...", "rationale": "..."} from model output.
    Raise ValueError on structure/validation issues.
    """
    try:
        p = json.loads(raw)
        node_label_raw = p.get("node_label")
        rationale = p.get("rationale")
        if not isinstance(node_label_raw, str) or not node_label_raw.strip():
            raise ValueError("missing_node_label")
        if not isinstance(rationale, str):
            rationale = ""
        return node_label_raw, rationale
    except Exception as e:
        raise ValueError(f"parse_failed: {e}")


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
    embs = embedder.encode(fields)  # already L2-normalized by our embedder
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

    # Encode parts
    item_embs = _encode_item_parts_local(item, embedder)
    if item_embs.size == 0:
        return None

    # Determine query fanout
    N = len(tax_names)
    Kq = min(max(256, 4 * len(allowed_labels)), N)

    # HNSW scores for all retrieved nodes
    scores = maxpool_scores(item_embs, index=hnsw_index, pool_k=Kq)

    # Keep only allowed
    allowed_idx_map = _build_allowed_index_map(allowed_labels, tax_names)
    filtered = [(idx, sc) for idx, sc in scores.items() if idx in allowed_idx_map]

    if filtered:
        best_idx = max(filtered, key=lambda kv: kv[1])[0]
        return allowed_idx_map[best_idx]

    # Final exact max-pool within allowed subset
    if allowed_idx_map:
        mask = list(allowed_idx_map.keys())
        sub = tax_embs[mask]
        j = _exact_maxpool_within(item_embs, sub)
        if j is not None:
            return allowed_idx_map[mask[j]]

    return None


def match_item_to_tree(
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
    session: Optional[requests.Session] = None,
    grammar: str = GRAMMAR_RESPONSE,
) -> Dict[str, Any]:
    """
    LLM-first matching; on parse/validation failure, fall back to HNSW + exact max-pool
    restricted to `allowed_labels`. Returns a flat dict (no _pack helpers).
    """
    prompt = make_tree_match_prompt(tree_markdown, item)

    _print_prompt_once(prompt)

    # Direct llama.cpp call (no _call_llm helper)
    raw = llama_completion(
        prompt,
        endpoint,
        session=session,
        temperature=temperature,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        n_predict=max(n_predict, 64),
        grammar=grammar,
        cache_prompt=cache_prompt,
        n_keep=n_keep,
        slot_id=slot_id,
    )

    # Try to parse JSON and validate
    node_label_raw: Optional[str] = None
    rationale: Optional[str] = None
    parse_ok = False
    try:
        payload = json.loads(raw)
        node_label_raw = payload.get("node_label")
        rationale = payload.get("rationale")
        if isinstance(node_label_raw, str) and node_label_raw in allowed_labels:
            parse_ok = True
        else:
            raise ValueError("label_not_in_allowed")
    except Exception:
        parse_ok = False

    if not parse_ok:
        # HNSW-backed fallback restricted to allowed labels
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
                "pred_label_raw": None,
                "resolved_label": chosen,
                "resolved_id": name_to_id.get(chosen),
                "resolved_path": name_to_path.get(chosen),
                "rationale": "Fallback: HNSW + max-pool among allowed.",
                "matched": True,
                "no_match": False,
                "raw": raw,  # keep original text for debugging
            }
        # Nothing matched
        return {
            "input_item": item,
            "pred_label_raw": None,
            "resolved_label": None,
            "resolved_id": None,
            "resolved_path": None,
            "rationale": "Parse/validation failed; no fallback.",
            "matched": False,
            "no_match": True,
            "raw": raw,
        }

    # Success path via LLM
    return {
        "input_item": item,
        "pred_label_raw": node_label_raw,
        "resolved_label": node_label_raw,
        "resolved_id": name_to_id.get(node_label_raw),
        "resolved_path": name_to_path.get(node_label_raw),
        "rationale": rationale if isinstance(rationale, str) else "",
        "matched": True,
        "no_match": False,
    }
