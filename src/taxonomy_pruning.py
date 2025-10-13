from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .embedding import Embedder, collect_item_texts, encode_item_texts
from .taxonomy import (
    ancestors_to_root,
    collect_descendants,
    make_label_display,
    roots_in_order,
    sort_key_factory,
)

import textdistance


def _normalized_token_similarity(a: str, b: str) -> float:
    """Return token-sort based similarity score in ``[0, 1]`` for two strings."""

    return 1 - textdistance.entropy_ncd(a, b)


def _lexical_anchor_indices(
    item_texts: Iterable[str],
    tax_names: Sequence[str],
    *,
    existing: Sequence[int],
    max_anchors: int = 3,
) -> List[int]:
    """Return indices of lexically similar taxonomy names using token sorting."""

    if textdistance is None or max_anchors <= 0:
        return []

    normalized_item_texts = [t.lower() for t in item_texts if t]
    if not normalized_item_texts:
        return []

    existing_set = set(existing)
    scored: List[Tuple[float, int]] = []
    for idx, name in enumerate(tax_names):
        if idx in existing_set:
            continue
        norm_name = name.lower()
        best = 0.0
        for text in normalized_item_texts:
            best = max(best, _normalized_token_similarity(norm_name, text))
            if best >= 1.0:
                break
        if best > 0.0:
            scored.append((best, idx))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scored[:max_anchors]]


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


def _anchor_neighborhood(
    G: nx.DiGraph, anchors: Sequence[str], *, desc_max_depth: int
) -> Set[str]:
    """Return nodes in the union of anchor ancestors and shallow descendants."""

    if not anchors:
        return set()

    neighborhood: Set[str] = set()
    for anchor in anchors:
        if not G.has_node(anchor):
            continue
        neighborhood.update(ancestors_to_root(G, anchor))
        if desc_max_depth > 0:
            neighborhood.update(collect_descendants(G, anchor, desc_max_depth))
    return neighborhood


def _dominant_anchor_forest(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    desc_max_depth: int,
    max_total_nodes: int,
) -> Set[str]:
    """Select a covering forest rooted at anchors using a dominating-set view."""

    if max_total_nodes == 0:
        return set()

    neighborhood = _anchor_neighborhood(G, anchors, desc_max_depth=desc_max_depth)
    if not neighborhood:
        return set()

    undirected = nx.Graph()
    undirected.add_nodes_from(neighborhood)
    undirected_base = G.to_undirected()
    undirected.add_edges_from(
        (u, v)
        for u, v in undirected_base.edges()
        if u in neighborhood and v in neighborhood
    )

    if not undirected.nodes:
        return set()

    dominating = set(
        nx.algorithms.approximation.min_weighted_dominating_set(undirected)
    )
    dominating.update(anchors)

    # Include nodes in the closed neighborhood of the dominating set to capture
    # the covered regions.
    coverage_nodes: Set[str] = set()
    for node in dominating:
        if node not in undirected:
            continue
        coverage_nodes.add(node)
        coverage_nodes.update(undirected.neighbors(node))
    coverage_nodes.update(a for a in anchors if G.has_node(a))

    # Prioritize anchors, then remaining coverage nodes ordered by distance to
    # any anchor (fallback to lexical order).
    anchor_sources = [a for a in anchors if a in undirected]
    if anchor_sources:
        distances = nx.multi_source_dijkstra_path_length(
            undirected, sources=anchor_sources
        )
    else:
        distances = {}

    ordered_nodes: List[str] = []
    seen: Set[str] = set()
    for anchor in anchors:
        if anchor in coverage_nodes and anchor not in seen:
            ordered_nodes.append(anchor)
            seen.add(anchor)
    for node in sorted(
        (coverage_nodes - seen),
        key=lambda n: (distances.get(n, float("inf")), n.lower()),
    ):
        ordered_nodes.append(node)

    allowed: Set[str] = set()
    for node in ordered_nodes:
        if not G.has_node(node):
            continue
        path = ancestors_to_root(G, node)
        new_nodes = [n for n in path if n not in allowed]
        if max_total_nodes > 0 and len(allowed) + len(new_nodes) > max_total_nodes:
            continue
        allowed.update(path)

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
    lexical_anchor_count: int = 3,
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
    item_texts = collect_item_texts(item)
    lexical_anchor_idxs = _lexical_anchor_indices(
        item_texts,
        tax_names,
        existing=anchor_idxs,
        max_anchors=max(0, int(lexical_anchor_count)),
    )

    combined_anchor_idxs: List[int] = []
    for idx in list(anchor_idxs) + lexical_anchor_idxs:
        if idx not in combined_anchor_idxs:
            combined_anchor_idxs.append(idx)

    anchors = [tax_names[i] for i in combined_anchor_idxs]

    allowed = _dominant_anchor_forest(
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

    md_lines = ["\n## TAXONOMY", tree_md, "\n## TOP SUGGESTIONS"]
    for label in allowed_ranked[:top_show]:
        md_lines.append(
            f"- {make_label_display(label, gloss_map or {}, use_summary=False)}"
        )
    return "\n".join(md_lines), allowed_ranked
