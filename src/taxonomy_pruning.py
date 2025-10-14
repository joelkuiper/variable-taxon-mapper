from __future__ import annotations

from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import networkx as nx
import numpy as np
import pandas as pd

from .embedding import Embedder, collect_item_texts, encode_item_texts
from .taxonomy import (
    ancestors_to_root,
    collect_descendants,
    make_label_display,
    roots_in_order,
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
    anchor_top_k: int,
    overfetch_mult: int = 3,
    min_overfetch: int = 128,
) -> List[int]:
    """Return HNSW anchor indices using max-pooled similarities across item parts."""

    if item_embs.size == 0:
        return list(range(min(anchor_top_k, N)))

    Kq = min(max(min_overfetch, anchor_top_k * overfetch_mult), N)

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
        return list(range(min(anchor_top_k, N)))

    scores[~seen] = -np.inf

    idxs = np.argpartition(-scores, kth=min(anchor_top_k - 1, N - 1))[:anchor_top_k]
    idxs = idxs[np.argsort(-scores[idxs])]
    filtered = [int(i) for i in idxs if seen[int(i)]]
    if filtered:
        return filtered
    return [int(i) for i in idxs]


def _taxonomy_similarity_scores(
    item_embs: np.ndarray, tax_embs_unit: np.ndarray
) -> np.ndarray:
    """Return max pooled cosine similarity for each taxonomy node."""

    if tax_embs_unit.size == 0:
        return np.zeros((0,), dtype=np.float32)

    N = tax_embs_unit.shape[0]
    if item_embs.size == 0:
        return np.zeros((N,), dtype=np.float32)

    if tax_embs_unit.dtype != np.float32:
        tax_embs = tax_embs_unit.astype(np.float32, copy=False)
    else:
        tax_embs = tax_embs_unit

    if item_embs.dtype != np.float32:
        item = item_embs.astype(np.float32, copy=False)
    else:
        item = item_embs

    sims = tax_embs @ item.T
    if sims.ndim == 1:
        return sims.astype(np.float32, copy=False)

    scores = sims.max(axis=1)
    return scores.astype(np.float32, copy=False)


def _undirected_taxonomy(G: nx.DiGraph) -> nx.Graph:
    """Return a cached undirected view of ``G`` for reuse in community detection."""

    signature = (G.number_of_nodes(), G.number_of_edges())
    cached = G.graph.get("_undirected_taxonomy_cache")
    if not cached or cached.get("signature") != signature:
        undirected = nx.Graph()
        undirected.add_nodes_from(G.nodes)
        undirected.add_edges_from(G.edges())
        cached = {"signature": signature, "graph": undirected}
        G.graph["_undirected_taxonomy_cache"] = cached
    return cached["graph"]


def _node_community_memberships(
    G: nx.DiGraph, *, k: int
) -> Tuple[Dict[str, List[int]], List[Set[str]]]:
    """Return ``(node -> community ids, community id -> node set)`` for ``k``-cliques."""

    if k < 2:
        return {}, []

    signature = (G.number_of_nodes(), G.number_of_edges())
    cache: Dict[Tuple[str, int], Dict[str, object]] = G.graph.setdefault(
        "_community_cache", {}
    )
    key = ("k_clique", int(k))
    entry = cache.get(key)

    if not entry or entry.get("signature") != signature:
        undirected = _undirected_taxonomy(G)
        communities_iter = nx.algorithms.community.k_clique_communities(
            undirected, int(max(2, k))
        )
        communities = [set(comm) for comm in communities_iter]
        node_to_comms: Dict[str, List[int]] = defaultdict(list)
        for idx, nodes in enumerate(communities):
            for node in nodes:
                node_to_comms[node].append(idx)
        for node in undirected.nodes:
            node_to_comms.setdefault(node, [])
        entry = {
            "signature": signature,
            "node_to": dict(node_to_comms),
            "communities": communities,
        }
        cache[key] = entry

    node_to = cast(Dict[str, List[int]], entry["node_to"])
    communities = cast(List[Set[str]], entry["communities"])
    return node_to, communities


def _anchor_neighborhood(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    max_descendant_depth: int,
    community_clique_size: int,
    max_community_size: Optional[int],
) -> Set[str]:
    """Return nodes reachable via ancestry/descendant rules and clique communities."""

    if not anchors:
        return set()

    neighborhood: Set[str] = set()
    for anchor in anchors:
        if not G.has_node(anchor):
            continue
        neighborhood.update(ancestors_to_root(G, anchor))
        if max_descendant_depth > 0:
            neighborhood.update(collect_descendants(G, anchor, max_descendant_depth))

    if community_clique_size >= 2:
        node_to_comms, communities = _node_community_memberships(
            G, k=community_clique_size
        )
        for anchor in anchors:
            for comm_idx in node_to_comms.get(anchor, []):
                members = communities[comm_idx]
                if max_community_size and max_community_size > 0:
                    if len(members) > max_community_size:
                        continue
                neighborhood.update(members)

    return neighborhood


def _enforce_node_budget_with_ancestors(
    G: nx.DiGraph,
    candidates: Iterable[str],
    *,
    node_budget: int,
    sort_key: Callable[[str], Tuple],
) -> Set[str]:
    """Return at most ``node_budget`` nodes, keeping ancestor closure intact."""

    if node_budget <= 0:
        return set() if node_budget == 0 else {n for n in candidates if G.has_node(n)}

    ordered = sorted({n for n in candidates if G.has_node(n)}, key=sort_key)
    allowed: Set[str] = set()
    for node in ordered:
        path = [n for n in ancestors_to_root(G, node) if G.has_node(n)]
        new_nodes = [n for n in path if n not in allowed]
        if not new_nodes:
            continue
        if len(allowed) + len(new_nodes) > node_budget:
            continue
        allowed.update(new_nodes)
        if len(allowed) >= node_budget:
            break
    return allowed


def _dominant_anchor_forest(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    max_descendant_depth: int,
    node_budget: int,
    community_clique_size: int,
    max_community_size: Optional[int],
    pagerank_damping: float,
    pagerank_score_floor: float,
    pagerank_candidate_limit: int,
) -> Set[str]:
    """Select a covering forest rooted at anchors using a dominating-set view."""

    if node_budget == 0:
        return set()

    neighborhood = _anchor_neighborhood(
        G,
        anchors,
        max_descendant_depth=max_descendant_depth,
        community_clique_size=community_clique_size,
        max_community_size=max_community_size,
    )
    if not neighborhood:
        return set()

    anchor_set: Set[str] = {a for a in anchors if G.has_node(a)}
    candidate_nodes: Set[str] = set(neighborhood)
    for anchor in anchor_set:
        candidate_nodes.update(ancestors_to_root(G, anchor))

    if not candidate_nodes:
        return set()

    undirected = _undirected_taxonomy(G)
    anchor_sources = [a for a in anchors if a in undirected]
    if anchor_sources:
        distances_full = nx.multi_source_dijkstra_path_length(
            undirected, sources=anchor_sources
        )
    else:
        distances_full = {}

    if pagerank_candidate_limit and pagerank_candidate_limit > 0:
        # Trim the candidate set before running PageRank. The later ``node_budget``
        # cap still applies, so the stricter of the two limits controls the final
        # number of retained nodes.
        if len(candidate_nodes) > pagerank_candidate_limit:
            ordered_candidates = sorted(
                candidate_nodes,
                key=lambda n: (distances_full.get(n, float("inf")), n.lower()),
            )
            candidate_nodes = set(ordered_candidates[:pagerank_candidate_limit])
            for anchor in anchor_set:
                candidate_nodes.update(ancestors_to_root(G, anchor))

    subgraph = G.subgraph(candidate_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        return set()

    personalization = {n: 1.0 for n in anchor_set if n in subgraph}
    pagerank_exception = getattr(nx, "PowerIterationFailedConvergence", RuntimeError)
    try:
        pagerank_scores = nx.pagerank(
            subgraph,
            alpha=pagerank_damping,
            personalization=personalization if personalization else None,
        )
    except pagerank_exception:
        pagerank_scores = {n: 1.0 for n in subgraph}

    if pagerank_score_floor > 0.0:
        candidate_nodes = {
            n
            for n in candidate_nodes
            if pagerank_scores.get(n, 0.0) >= pagerank_score_floor or n in anchor_set
        }
        if not candidate_nodes:
            candidate_nodes = set(anchor_set)

        pagerank_scores = {n: pagerank_scores.get(n, 0.0) for n in candidate_nodes}

    if anchor_sources:
        distances = {n: distances_full.get(n, float("inf")) for n in candidate_nodes}
    else:
        distances = {}

    ordered_nodes: List[str] = []
    seen: Set[str] = set()
    for anchor in anchors:
        if anchor in candidate_nodes and anchor not in seen:
            ordered_nodes.append(anchor)
            seen.add(anchor)

    remaining_nodes = [n for n in candidate_nodes if n not in seen]
    remaining_nodes.sort(
        key=lambda n: (
            distances.get(n, float("inf")),
            -pagerank_scores.get(n, 0.0),
            n.lower(),
        )
    )
    ordered_nodes.extend(remaining_nodes)

    allowed: Set[str] = set()
    for anchor in anchors:
        if anchor not in candidate_nodes or not G.has_node(anchor):
            continue
        path = ancestors_to_root(G, anchor)
        new_nodes = [n for n in path if n not in allowed]
        if node_budget > 0 and len(allowed) + len(new_nodes) > node_budget:
            for node in path:
                if node in allowed:
                    continue
                if node_budget > 0 and len(allowed) >= node_budget:
                    break
                allowed.add(node)
            if len(allowed) >= node_budget:
                return allowed
            continue
        allowed.update(new_nodes)
        if node_budget > 0 and len(allowed) >= node_budget:
            return allowed

    for node in ordered_nodes:
        if node in allowed or not G.has_node(node):
            continue
        path = ancestors_to_root(G, node)
        new_nodes = [n for n in path if n not in allowed]
        if not new_nodes:
            continue
        if node_budget > 0 and len(allowed) + len(new_nodes) > node_budget:
            continue
        allowed.update(new_nodes)
        if node_budget > 0 and len(allowed) >= node_budget:
            break

    return allowed


def _anchor_hull_subtree(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    max_descendant_depth: int,
    community_clique_size: int,
    max_community_size: Optional[int],
    node_budget: int,
    sort_key: Callable[[str], Tuple],
) -> Set[str]:
    """Return the ancestor/descendant hull surrounding the anchors."""

    if not anchors:
        return set()

    if node_budget == 0:
        return set()

    neighborhood = _anchor_neighborhood(
        G,
        anchors,
        max_descendant_depth=max_descendant_depth,
        community_clique_size=community_clique_size,
        max_community_size=max_community_size,
    )
    for anchor in anchors:
        if G.has_node(anchor):
            neighborhood.update(ancestors_to_root(G, anchor))

    if node_budget > 0:
        return _enforce_node_budget_with_ancestors(
            G, neighborhood, node_budget=node_budget, sort_key=sort_key
        )
    return {n for n in neighborhood if G.has_node(n)}


def _similarity_threshold_subtree(
    G: nx.DiGraph,
    *,
    anchors: Sequence[str],
    similarity_map: Mapping[str, float],
    threshold: float,
    node_budget: int,
    sort_key: Callable[[str], Tuple],
) -> Set[str]:
    """Return nodes whose similarity meets ``threshold`` plus their ancestors."""

    if node_budget == 0:
        return set()

    allowed: Set[str] = {
        name
        for name, score in similarity_map.items()
        if score >= threshold and G.has_node(name)
    }

    for anchor in anchors:
        if G.has_node(anchor):
            allowed.add(anchor)

    expanded: Set[str] = set()
    for node in allowed:
        expanded.update(ancestors_to_root(G, node))

    if node_budget > 0 and expanded:
        return _enforce_node_budget_with_ancestors(
            G, expanded, node_budget=node_budget, sort_key=sort_key
        )

    return {n for n in expanded if G.has_node(n)}


def _radius_limited_subtree(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    radius: int,
    node_budget: int,
    sort_key: Callable[[str], Tuple],
) -> Set[str]:
    """Return nodes within ``radius`` undirected hops of any anchor."""

    if not anchors:
        return set()

    if radius < 0:
        return set()

    if node_budget == 0:
        return set()

    undirected = _undirected_taxonomy(G)
    anchor_nodes = [a for a in anchors if a in undirected]
    if not anchor_nodes:
        return set()

    distances = nx.multi_source_dijkstra_path_length(
        undirected, sources=anchor_nodes, cutoff=radius
    )
    neighborhood = {n for n, dist in distances.items() if dist <= radius}
    neighborhood.update(anchor_nodes)

    expanded: Set[str] = set()
    for node in neighborhood:
        if G.has_node(node):
            expanded.update(ancestors_to_root(G, node))

    if node_budget > 0 and expanded:
        return _enforce_node_budget_with_ancestors(
            G, expanded, node_budget=node_budget, sort_key=sort_key
        )

    return {n for n in expanded if G.has_node(n)}


def _normalize_tree_sort_mode(mode: Optional[str]) -> str:
    """Return normalized tree sort mode with fallbacks."""

    if not mode:
        return "relevance"
    normalized = mode.strip().lower()
    if normalized in {"topological", "topology", "order"}:
        return "topological"
    if normalized in {"alphabetical", "alpha", "name", "id"}:
        return "alphabetical"
    if normalized in {"proximity", "distance"}:
        return "proximity"
    return "relevance"


def _tree_sort_key_factory(
    mode: Optional[str],
    *,
    similarity_map: Mapping[str, float],
    order_map: Mapping[str, float],
    distance_map: Optional[Mapping[str, float]] = None,
) -> Callable[[str], Tuple]:
    """Return a key function used to order nodes in the rendered tree."""

    normalized = _normalize_tree_sort_mode(mode)

    if normalized == "topological":

        def sort_key(node_name: str) -> Tuple[float, str]:
            order_val = order_map.get(node_name, float("inf"))
            return (
                order_val if pd.notna(order_val) else float("inf"),
                node_name.lower(),
            )

        return sort_key

    if normalized == "alphabetical":

        def sort_key(node_name: str) -> Tuple[str]:
            return (node_name.lower(),)

        return sort_key

    if normalized == "proximity":

        def sort_key(node_name: str) -> Tuple[float, float, float, str]:
            distance_val = float("inf")
            if distance_map is not None:
                raw_distance = distance_map.get(node_name, float("inf"))
                distance_val = (
                    float(raw_distance)
                    if pd.notna(raw_distance)
                    else float("inf")
                )
            score = similarity_map.get(node_name, float("-inf"))
            order_val = order_map.get(node_name, float("inf"))
            return (
                distance_val,
                -score,
                order_val if pd.notna(order_val) else float("inf"),
                node_name.lower(),
            )

        return sort_key

    def sort_key(node_name: str) -> Tuple[float, float, str]:
        score = similarity_map.get(node_name, float("-inf"))
        order_val = order_map.get(node_name, float("inf"))
        return (
            -score,
            order_val if pd.notna(order_val) else float("inf"),
            node_name.lower(),
        )

    return sort_key


def _normalize_pruning_mode(mode: Optional[str]) -> str:
    """Return normalized pruning strategy identifier."""

    if not mode:
        return "dominant_forest"

    normalized = mode.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"dominant", "dominant_forest", "forest"}:
        return "dominant_forest"
    if normalized in {"anchor_hull", "hull", "anchor"}:
        return "anchor_hull"
    if normalized in {"similarity", "similarity_threshold", "threshold"}:
        return "similarity_threshold"
    if normalized in {"radius", "radius_limited", "radius_based"}:
        return "radius"
    return "dominant_forest"


def _rank_allowed_nodes(
    allowed: Set[str],
    *,
    similarity_map: Mapping[str, float],
    order_map: Mapping[str, float],
    tree_sort_mode: Optional[str],
    distance_map: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """Return allowed nodes ranked using the configured tree sort mode."""

    sort_key = _tree_sort_key_factory(
        tree_sort_mode,
        similarity_map=similarity_map,
        order_map=order_map,
        distance_map=distance_map,
    )
    return sorted(allowed, key=sort_key)


def _render_tree_markdown(
    G: nx.DiGraph,
    allowed: Set[str],
    df: pd.DataFrame,
    *,
    name_col: str,
    order_col: str,
    gloss_map: Optional[Dict[str, str]],
    similarity_map: Mapping[str, float],
    order_map: Mapping[str, float],
    tree_sort_mode: Optional[str],
    distance_map: Optional[Mapping[str, float]] = None,
) -> Tuple[str, Optional[List[str]]]:
    """Render the allowed subtree as markdown; fallback to full taxonomy if empty."""

    sort_key = _tree_sort_key_factory(
        tree_sort_mode,
        similarity_map=similarity_map,
        order_map=order_map,
        distance_map=distance_map,
    )

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
    fallback_ranked = sorted(G.nodes, key=sort_key)

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
    """Return pruned tree markdown and ranked candidate labels for ``item``.

    Available ``pruning_mode`` values are ``"dominant_forest"`` (current default),
    ``"anchor_hull"`` for a lighter ancestor/descendant hull, ``"similarity_threshold"``
    which keeps nodes at or above ``similarity_threshold``, and ``"radius"`` to retain
    nodes within ``pruning_radius`` undirected hops of any anchor.
    """

    item_embs = encode_item_texts(item, embedder)

    similarity_scores = _taxonomy_similarity_scores(item_embs, tax_embs_unit)
    similarity_map: Dict[str, float] = {name: float("-inf") for name in tax_names}
    limit = min(len(tax_names), len(similarity_scores))
    for i in range(limit):
        similarity_map[tax_names[i]] = float(similarity_scores[i])
    order_map = df.groupby(name_col)[order_col].min().to_dict()

    normalized_tree_sort_mode = _normalize_tree_sort_mode(tree_sort_mode)
    distance_map: Dict[str, float] = {}
    anchors: List[str] = []

    if enable_taxonomy_pruning:
        if item_embs.size == 0:
            anchor_idxs = list(range(min(anchor_top_k, len(tax_names))))
        else:
            anchor_idxs = _hnsw_anchor_indices(
                item_embs,
                hnsw_index,
                N=len(tax_names),
                anchor_top_k=anchor_top_k,
                overfetch_mult=max(1, int(anchor_overfetch_multiplier)),
                min_overfetch=max(1, int(anchor_min_overfetch)),
            )
        item_texts = collect_item_texts(item)
        lexical_anchor_idxs = _lexical_anchor_indices(
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
            undirected = _undirected_taxonomy(G)
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

        pruning_mode_normalized = _normalize_pruning_mode(pruning_mode)

        sort_key_for_budget = _tree_sort_key_factory(
            tree_sort_mode,
            similarity_map=similarity_map,
            order_map=order_map,
            distance_map=distance_map,
        )

        if pruning_mode_normalized == "anchor_hull":
            allowed = _anchor_hull_subtree(
                G,
                anchors,
                max_descendant_depth=max_descendant_depth,
                community_clique_size=max(2, int(community_clique_size))
                if community_clique_size
                else 0,
                max_community_size=max_community_size_int,
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        elif pruning_mode_normalized == "similarity_threshold":
            allowed = _similarity_threshold_subtree(
                G,
                anchors=anchors,
                similarity_map=similarity_map,
                threshold=float(similarity_threshold),
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        elif pruning_mode_normalized == "radius":
            allowed = _radius_limited_subtree(
                G,
                anchors,
                radius=int(pruning_radius),
                node_budget=node_budget,
                sort_key=sort_key_for_budget,
            )
        else:
            allowed = _dominant_anchor_forest(
                G,
                anchors,
                max_descendant_depth=max_descendant_depth,
                node_budget=node_budget,
                community_clique_size=max(2, int(community_clique_size))
                if community_clique_size
                else 0,
                max_community_size=max_community_size_int,
                pagerank_damping=float(pagerank_damping),
                pagerank_score_floor=float(pagerank_score_floor),
                pagerank_candidate_limit=max(0, int(pagerank_candidate_limit)),
            )
    else:
        allowed = set(G.nodes)

    allowed_ranked = _rank_allowed_nodes(
        allowed,
        similarity_map=similarity_map,
        order_map=order_map,
        tree_sort_mode=tree_sort_mode,
        distance_map=distance_map,
    )

    tree_md, fallback_ranked = _render_tree_markdown(
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
