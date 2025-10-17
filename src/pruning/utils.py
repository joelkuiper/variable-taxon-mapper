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
try:  # pragma: no cover - optional dependency in prod
    import textdistance
except ImportError:  # pragma: no cover - fallback when not installed
    textdistance = None

from ..taxonomy import (
    ancestors_to_root,
    collect_descendants,
    make_label_display,
    roots_in_order,
)


def taxonomy_similarity_scores(
    item_embs: np.ndarray, tax_embs_unit: np.ndarray
) -> np.ndarray:
    """Return max pooled cosine similarity for each taxonomy node."""

    if tax_embs_unit.size == 0:
        return np.zeros((0,), dtype=np.float32)

    N = tax_embs_unit.shape[0]
    if item_embs.size == 0:
        return np.zeros((N,), dtype=np.float32)

    tax_embs = (
        tax_embs_unit.astype(np.float32, copy=False)
        if tax_embs_unit.dtype != np.float32
        else tax_embs_unit
    )
    item = (
        item_embs.astype(np.float32, copy=False)
        if item_embs.dtype != np.float32
        else item_embs
    )

    sims = tax_embs @ item.T
    if sims.ndim == 1:
        return sims.astype(np.float32, copy=False)

    scores = sims.max(axis=1)
    return scores.astype(np.float32, copy=False)


def _normalized_token_similarity(a: str, b: str) -> float:
    """Return token-sort based similarity score in ``[0, 1]`` for two strings."""

    return 1 - textdistance.entropy_ncd(a, b)


def lexical_anchor_indices(
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


def hnsw_anchor_indices(
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


def get_undirected_taxonomy(G: nx.DiGraph) -> nx.Graph:
    """Expose the cached undirected taxonomy graph."""

    return _undirected_taxonomy(G)


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


def anchor_neighborhood(
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


def enforce_node_budget_with_ancestors(
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


def prepare_pagerank_scores(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    max_descendant_depth: int,
    community_clique_size: int,
    max_community_size: Optional[int],
    pagerank_damping: float,
    pagerank_score_floor: float,
    pagerank_candidate_limit: int,
) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
    """Return candidate nodes, PageRank scores, and anchor distances."""

    if not anchors:
        return set(), {}, {}

    neighborhood = anchor_neighborhood(
        G,
        anchors,
        max_descendant_depth=max_descendant_depth,
        community_clique_size=community_clique_size,
        max_community_size=max_community_size,
    )
    if not neighborhood:
        return set(), {}, {}

    anchor_set: Set[str] = {a for a in anchors if G.has_node(a)}
    if not anchor_set:
        return set(), {}, {}

    candidate_nodes: Set[str] = set(neighborhood)
    for anchor in anchor_set:
        candidate_nodes.update(ancestors_to_root(G, anchor))

    if not candidate_nodes:
        return set(), {}, {}

    undirected = _undirected_taxonomy(G)
    anchor_sources = [a for a in anchors if a in undirected]
    if anchor_sources:
        distances_full = nx.multi_source_dijkstra_path_length(
            undirected, sources=anchor_sources
        )
    else:
        distances_full = {}

    if pagerank_candidate_limit and pagerank_candidate_limit > 0:
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
        return set(), {}, distances_full

    personalization = {n: 1.0 for n in anchor_set if n in subgraph}
    pagerank_exception = getattr(nx, "PowerIterationFailedConvergence", RuntimeError)
    try:
        pagerank_scores = nx.pagerank(
            subgraph.reverse(copy=True),
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

    pagerank_filtered = {n: pagerank_scores.get(n, 0.0) for n in candidate_nodes}

    return candidate_nodes, pagerank_filtered, distances_full


def dominant_anchor_forest(
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
    precomputed: Optional[Tuple[Set[str], Dict[str, float], Dict[str, float]]] = None,
) -> Tuple[Set[str], Dict[str, float]]:
    """Select a covering forest rooted at anchors using a dominating-set view."""

    if node_budget == 0:
        return set(), {}

    if precomputed is not None:
        candidate_nodes, pagerank_scores, distances_full = precomputed
    else:
        (
            candidate_nodes,
            pagerank_scores,
            distances_full,
        ) = prepare_pagerank_scores(
            G,
            anchors,
            max_descendant_depth=max_descendant_depth,
            community_clique_size=community_clique_size,
            max_community_size=max_community_size,
            pagerank_damping=pagerank_damping,
            pagerank_score_floor=pagerank_score_floor,
            pagerank_candidate_limit=pagerank_candidate_limit,
        )

    if not candidate_nodes:
        return set(), pagerank_scores

    anchor_set: Set[str] = {a for a in anchors if G.has_node(a)}

    if distances_full:
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
                return allowed, pagerank_scores
            continue
        allowed.update(new_nodes)
        if node_budget > 0 and len(allowed) >= node_budget:
            return allowed, pagerank_scores

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

    return allowed, pagerank_scores


def anchor_hull_subtree(
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

    if not anchors or node_budget == 0:
        return set()

    neighborhood = anchor_neighborhood(
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
        return enforce_node_budget_with_ancestors(
            G, neighborhood, node_budget=node_budget, sort_key=sort_key
        )
    return {n for n in neighborhood if G.has_node(n)}


def similarity_threshold_subtree(
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
        return enforce_node_budget_with_ancestors(
            G, expanded, node_budget=node_budget, sort_key=sort_key
        )

    return {n for n in expanded if G.has_node(n)}


def radius_limited_subtree(
    G: nx.DiGraph,
    anchors: Sequence[str],
    *,
    radius: int,
    node_budget: int,
    sort_key: Callable[[str], Tuple],
) -> Set[str]:
    """Return nodes within ``radius`` undirected hops of any anchor."""

    if not anchors or radius < 0 or node_budget == 0:
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
        return enforce_node_budget_with_ancestors(
            G, expanded, node_budget=node_budget, sort_key=sort_key
        )

    return {n for n in expanded if G.has_node(n)}


def normalize_tree_sort_mode(mode: Optional[str]) -> str:
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
    if normalized in {"pagerank", "rank", "centrality"}:
        return "pagerank"
    return "relevance"


def tree_sort_key_factory(
    mode: Optional[str],
    *,
    similarity_map: Mapping[str, float],
    order_map: Mapping[str, float],
    distance_map: Optional[Mapping[str, float]] = None,
    pagerank_map: Optional[Mapping[str, float]] = None,
) -> Callable[[str], Tuple]:
    """Return a key function used to order nodes in the rendered tree."""

    normalized = normalize_tree_sort_mode(mode)

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
                    float(raw_distance) if pd.notna(raw_distance) else float("inf")
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

    if normalized == "pagerank":

        def sort_key(node_name: str) -> Tuple[float, float, float, str]:
            pagerank_val = 0.0
            if pagerank_map is not None:
                raw_pagerank = pagerank_map.get(node_name, 0.0)
                pagerank_val = float(raw_pagerank) if pd.notna(raw_pagerank) else 0.0
            score = similarity_map.get(node_name, float("-inf"))
            order_val = order_map.get(node_name, float("inf"))
            return (
                -pagerank_val,
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


def normalize_pruning_mode(mode: Optional[str]) -> str:
    """Return normalized pruning strategy identifier."""

    if not mode:
        return "dominant_forest"

    normalized = mode.strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "dominant_forest": {"dominant", "dominant_forest", "forest"},
        "anchor_hull": {"anchor_hull", "hull", "anchor"},
        "similarity_threshold": {"similarity", "similarity_threshold", "threshold"},
        "radius": {"radius", "radius_limited", "radius_based"},
    }
    for target, aliases in mapping.items():
        if normalized in aliases:
            return target
    return "dominant_forest"


def rank_allowed_nodes(
    allowed: Set[str],
    *,
    similarity_map: Mapping[str, float],
    order_map: Mapping[str, float],
    sort_mode: Optional[str],
    distance_map: Optional[Mapping[str, float]] = None,
    pagerank_map: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """Return allowed nodes ranked using the configured node sort mode."""

    sort_key = tree_sort_key_factory(
        sort_mode,
        similarity_map=similarity_map,
        order_map=order_map,
        distance_map=distance_map,
        pagerank_map=pagerank_map,
    )
    return sorted(allowed, key=sort_key)


def render_tree_markdown(
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
    pagerank_map: Optional[Mapping[str, float]] = None,
    surrogate_root_label: Optional[str] = None,
) -> Tuple[str, Optional[List[str]]]:
    """Render the allowed subtree as markdown; fallback to full taxonomy if empty."""

    sort_key = tree_sort_key_factory(
        tree_sort_mode,
        similarity_map=similarity_map,
        order_map=order_map,
        distance_map=distance_map,
        pagerank_map=pagerank_map,
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

    surrogate_label = (surrogate_root_label or "").strip()
    depth_offset = 0
    if surrogate_label:
        root_display = make_label_display(
            surrogate_label, gloss_map or {}, use_summary=False
        )
        lines.append(f"- {root_display}")
        depth_offset = 1

    for root in roots_in_order(G, sort_key):
        if root in allowed:
            _walk(root, depth_offset)

    if lines:
        return "\n".join(lines)
    else:
        return ""
