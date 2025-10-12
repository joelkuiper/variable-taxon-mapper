from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .embedding import Embedder, encode_item_texts
from .taxonomy import (
    ancestors_to_root,
    make_label_display,
    roots_in_order,
    sort_key_factory,
)


def _hnsw_anchor_indices(
    item_embs: np.ndarray,
    hnsw_index,
    *,
    N: int,
    top_k_nodes: int,
    overfetch_mult: int = 3,
    min_overfetch: int = 128,
) -> List[int]:
    """Return HNSW anchor indices using max-pooled similarities across item parts."""

    if item_embs.size == 0:
        return list(range(min(top_k_nodes, N)))

    Kq = min(max(min_overfetch, top_k_nodes * overfetch_mult), N)

    scores = np.full(N, -np.inf, dtype=np.float32)
    seen = np.zeros(N, dtype=bool)
    for q in item_embs:
        labels, dists = hnsw_index.knn_query(q[np.newaxis, :].astype(np.float32), k=Kq)
        labels, dists = labels[0], dists[0]
        sims = 1.0 - dists.astype(np.float32)
        for idx, sim in zip(labels, sims):
            if idx < 0:
                continue
            seen[idx] = True
            if sim > scores[idx]:
                scores[idx] = sim

    if not seen.any():
        return list(range(min(top_k_nodes, N)))

    scores[~seen] = -np.inf

    idxs = np.argpartition(-scores, kth=min(top_k_nodes - 1, N - 1))[:top_k_nodes]
    idxs = idxs[np.argsort(-scores[idxs])]
    filtered = [int(i) for i in idxs if seen[int(i)]]
    if filtered:
        return filtered
    return [int(i) for i in idxs]


def _expand_allowed_nodes(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    desc_max_depth: int,
    max_total_nodes: int,
) -> Set[str]:
    """Build allowed-node set containing anchors, relatives, and sampled descendants."""

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
            continue

        for n in new_ancestors:
            allowed.add(n)

        # Adding siblings and direct children before sampling deeper
        # descendants, ensuring the rendered TREE includes those makes intuitive
        # sense but it doesn't work performance wise
        # if len(allowed) >= max_total_nodes:
        #     continue

        # siblings: Set[str] = set()
        # for parent in G.predecessors(anchor):
        #     if len(allowed) >= max_total_nodes:
        #         break
        #     if parent not in allowed:
        #         allowed.add(parent)
        #     for sib in G.successors(parent):
        #         if sib == anchor or sib in allowed:
        #             continue
        #         siblings.add(sib)

        # for sib in sorted(siblings, key=lambda s: s.lower()):
        #     if len(allowed) >= max_total_nodes:
        #         break
        #     allowed.add(sib)

        # if len(allowed) >= max_total_nodes:
        #     continue

        # children = [c for c in G.successors(anchor) if c not in allowed]
        # for child in sorted(children, key=lambda s: s.lower()):
        #     if len(allowed) >= max_total_nodes:
        #         break
        #     allowed.add(child)

        descendants_order = _descendant_order(anchor)
        has_descendants = bool(descendants_order)

        if not has_descendants:
            continue

        remaining_capacity = max_total_nodes - len(allowed)
        if remaining_capacity <= 0:
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

    return allowed


def _rank_allowed_nodes(anchors: Sequence[str], allowed: Set[str]) -> List[str]:
    """Return anchors first (deduped, in order) followed by remaining allowed nodes."""

    allowed_ranked: List[str] = []
    seen: Set[str] = set()
    for anchor in anchors:
        if anchor in allowed and anchor not in seen:
            allowed_ranked.append(anchor)
            seen.add(anchor)
    for node in sorted(allowed, key=lambda s: s.lower()):
        if node not in seen:
            allowed_ranked.append(node)
            seen.add(node)
    return allowed_ranked


def _render_tree_markdown(
    G: nx.DiGraph,
    allowed: Set[str],
    df: pd.DataFrame,
    *,
    name_col: str,
    order_col: str,
    gloss_map: Optional[Dict[str, str]],
) -> Tuple[str, Optional[List[str]]]:
    """Render the allowed subtree as markdown; fallback to full taxonomy if empty."""

    order_map = df.groupby(name_col)[order_col].min().to_dict()
    sort_key = sort_key_factory(order_map)

    lines: List[str] = []

    def _walk(node: str, depth: int) -> None:
        if node not in allowed:
            return
        label_display = make_label_display(node, gloss_map or {})
        lines.append("  " * depth + f"- {label_display}")
        for child in sorted(G.successors(node), key=sort_key):
            if child in allowed:
                _walk(child, depth + 1)

    for root in roots_in_order(G, sort_key):
        if root in allowed:
            _walk(root, 0)

    if lines:
        return "\n".join(lines), None

    fallback_lines: List[str] = []

    def _walk_full(node: str, depth: int) -> None:
        label_display = make_label_display(node, gloss_map or {})
        fallback_lines.append("  " * depth + f"- {label_display}")
        for child in sorted(G.successors(node), key=sort_key):
            _walk_full(child, depth + 1)

    for root in roots_in_order(G, sort_key):
        _walk_full(root, 0)

    tree_md = "\n".join(fallback_lines)
    fallback_ranked = sorted(G.nodes, key=lambda s: s.lower())

    return tree_md, fallback_ranked


def pruned_tree_markdown_for_item(
    item: Dict[str, Optional[str]],
    *,
    G: nx.DiGraph,
    df: pd.DataFrame,
    embedder: Embedder,
    tax_names: Sequence[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_col: str = "name",
    order_col: str = "order",
    top_k_nodes: int = 128,
    desc_max_depth: int = 3,
    max_total_nodes: int = 1200,
    gloss_map: Optional[Dict[str, str]] = None,
    anchor_overfetch_mult: int = 3,
    anchor_min_overfetch: int = 128,
    candidate_list_max_items: int = 40,
) -> Tuple[str, List[str]]:
    """Return pruned tree markdown and ranked candidate labels for ``item``."""

    item_embs = encode_item_texts(item, embedder)

    if item_embs.size == 0:
        anchor_idxs = list(range(min(top_k_nodes, len(tax_names))))
    else:
        anchor_idxs = _hnsw_anchor_indices(
            item_embs,
            hnsw_index,
            N=len(tax_names),
            top_k_nodes=top_k_nodes,
            overfetch_mult=max(1, int(anchor_overfetch_mult)),
            min_overfetch=max(1, int(anchor_min_overfetch)),
        )
    anchors = [tax_names[i] for i in anchor_idxs]

    allowed = _expand_allowed_nodes(
        G,
        anchors,
        desc_max_depth=desc_max_depth,
        max_total_nodes=max_total_nodes,
    )

    allowed_ranked = _rank_allowed_nodes(anchors, allowed)

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

    max_items = int(candidate_list_max_items)
    top_show = (
        len(allowed_ranked) if max_items <= 0 else min(max_items, len(allowed_ranked))
    )

    md_lines = ["\n## TAXONOMY", tree_md, "\n## SUGGESTIONS"]
    for label in allowed_ranked[:top_show]:
        md_lines.append(
            f"- {make_label_display(label, gloss_map or {}, use_summary=False)}"
        )
    return "\n".join(md_lines), allowed_ranked
