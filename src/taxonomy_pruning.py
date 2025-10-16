from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .embedding import Embedder, collect_item_texts, encode_item_texts
from .taxonomy import make_label_display
from .taxonomy_pruning_utils import (
    anchor_hull_subtree,
    dominant_anchor_forest,
    get_undirected_taxonomy,
    hnsw_anchor_indices,
    lexical_anchor_indices,
    normalize_pruning_mode,
    normalize_tree_sort_mode,
    prepare_pagerank_scores,
    radius_limited_subtree,
    rank_allowed_nodes,
    render_tree_markdown,
    similarity_threshold_subtree,
    taxonomy_similarity_scores,
    tree_sort_key_factory,
)


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
    anchor_top_k: int = 128,
    max_descendant_depth: int = 3,
    node_budget: int = 1200,
    gloss_map: Optional[Dict[str, str]] = None,
    anchor_overfetch_multiplier: int = 3,
    anchor_min_overfetch: int = 128,
    suggestion_list_limit: int = 40,
    lexical_anchor_limit: int = 3,
    community_clique_size: int = 2,
    max_community_size: Optional[int] = 400,
    pagerank_damping: float = 0.85,
    pagerank_score_floor: float = 0.0,
    pagerank_candidate_limit: int = 256,
    enable_taxonomy_pruning: bool = True,
    tree_sort_mode: str = "relevance",
    pruning_mode: str = "dominant_forest",
    similarity_threshold: float = 0.0,
    pruning_radius: int = 2,
) -> Tuple[str, List[str]]:
    """Return pruned tree markdown and ranked candidate labels for ``item``."""

    item_embs = encode_item_texts(item, embedder)

    similarity_scores = taxonomy_similarity_scores(item_embs, tax_embs_unit)
    similarity_map: Dict[str, float] = {name: float("-inf") for name in tax_names}
    limit = min(len(tax_names), len(similarity_scores))
    for i in range(limit):
        similarity_map[tax_names[i]] = float(similarity_scores[i])
    order_map = df.groupby(name_col)[order_col].min().to_dict()

    normalized_tree_sort_mode = normalize_tree_sort_mode(tree_sort_mode)
    distance_map: Dict[str, float] = {}
    pagerank_map: Dict[str, float] = {}
    anchors: List[str] = []
    pagerank_data: Optional[Tuple[Set[str], Dict[str, float], Dict[str, float]]] = None

    if enable_taxonomy_pruning:
        if item_embs.size == 0:
            anchor_idxs = list(range(min(anchor_top_k, len(tax_names))))
        else:
            anchor_idxs = hnsw_anchor_indices(
                item_embs,
                hnsw_index,
                N=len(tax_names),
                anchor_top_k=anchor_top_k,
                overfetch_mult=max(1, int(anchor_overfetch_multiplier)),
                min_overfetch=max(1, int(anchor_min_overfetch)),
            )
        item_texts = collect_item_texts(item)
        lexical_anchor_idxs = lexical_anchor_indices(
            item_texts,
            tax_names,
            existing=anchor_idxs,
            max_anchors=max(0, int(lexical_anchor_limit)),
        )

        combined_anchor_idxs: List[int] = []
        for idx in list(anchor_idxs) + lexical_anchor_idxs:
            if idx not in combined_anchor_idxs:
                combined_anchor_idxs.append(idx)

        anchors = [tax_names[i] for i in combined_anchor_idxs]

        if normalized_tree_sort_mode == "proximity" and anchors:
            undirected = get_undirected_taxonomy(G)
            anchor_nodes = [a for a in anchors if a in undirected]
            if anchor_nodes:
                distances_full = nx.multi_source_dijkstra_path_length(
                    undirected, sources=anchor_nodes
                )
                distance_map = {
                    node: float(dist) for node, dist in distances_full.items()
                }
                for anchor_node in anchor_nodes:
                    distance_map.setdefault(anchor_node, 0.0)
            else:
                distance_map = {}

        if max_community_size is None:
            max_community_size_int = None
        else:
            max_community_size_int = int(max_community_size)
            if max_community_size_int <= 0:
                max_community_size_int = None

        if community_clique_size:
            community_clique_size_int = max(2, int(community_clique_size))
        else:
            community_clique_size_int = 0

        if normalized_tree_sort_mode == "pagerank" and anchors:
            pagerank_data = prepare_pagerank_scores(
                G,
                anchors,
                max_descendant_depth=max_descendant_depth,
                community_clique_size=community_clique_size_int,
                max_community_size=max_community_size_int,
                pagerank_damping=float(pagerank_damping),
                pagerank_score_floor=float(pagerank_score_floor),
                pagerank_candidate_limit=max(0, int(pagerank_candidate_limit)),
            )
            pagerank_map = pagerank_data[1]
        else:
            pagerank_map = {}

        pruning_mode_normalized = normalize_pruning_mode(pruning_mode)

        sort_key_for_budget = tree_sort_key_factory(
            tree_sort_mode,
            similarity_map=similarity_map,
            order_map=order_map,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

        if pruning_mode_normalized == "anchor_hull":
            allowed = anchor_hull_subtree(
                G,
                anchors,
                max_descendant_depth=max_descendant_depth,
                community_clique_size=community_clique_size_int,
                max_community_size=max_community_size_int,
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        elif pruning_mode_normalized == "similarity_threshold":
            allowed = similarity_threshold_subtree(
                G,
                anchors=anchors,
                similarity_map=similarity_map,
                threshold=float(similarity_threshold),
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        elif pruning_mode_normalized == "radius":
            allowed = radius_limited_subtree(
                G,
                anchors,
                radius=int(pruning_radius),
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        else:
            allowed, pagerank_map = dominant_anchor_forest(
                G,
                anchors,
                max_descendant_depth=max_descendant_depth,
                node_budget=node_budget,
                community_clique_size=community_clique_size_int,
                max_community_size=max_community_size_int,
                pagerank_damping=float(pagerank_damping),
                pagerank_score_floor=float(pagerank_score_floor),
                pagerank_candidate_limit=max(0, int(pagerank_candidate_limit)),
                precomputed=pagerank_data,
            )
    else:
        allowed = set(G.nodes)
        pagerank_map = {}

    allowed_ranked = rank_allowed_nodes(
        allowed,
        similarity_map=similarity_map,
        order_map=order_map,
        tree_sort_mode=tree_sort_mode,
        distance_map=distance_map,
        pagerank_map=pagerank_map,
    )

    tree_md, fallback_ranked = render_tree_markdown(
        G,
        allowed,
        df,
        name_col=name_col,
        order_col=order_col,
        gloss_map=gloss_map,
        similarity_map=similarity_map,
        order_map=order_map,
        tree_sort_mode=tree_sort_mode,
        distance_map=distance_map,
        pagerank_map=pagerank_map,
    )
    if fallback_ranked is not None:
        allowed_ranked = fallback_ranked

    max_items = int(suggestion_list_limit)
    top_show = (
        len(allowed_ranked) if max_items <= 0 else min(max_items, len(allowed_ranked))
    )

    md_lines = ["\n# TAXONOMY", tree_md, "\n# SUGGESTIONS"]
    for label in allowed_ranked[:top_show]:
        md_lines.append(
            f"- {make_label_display(label, gloss_map or {}, use_summary=False)}"
        )
    return "\n".join(md_lines), allowed_ranked
