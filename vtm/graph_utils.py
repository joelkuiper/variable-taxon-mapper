"""Shared graph helper functions used across the project."""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


def compute_node_depths(graph: nx.DiGraph | None) -> Dict[str, Optional[int]]:
    """Return a mapping of node -> depth for the provided graph."""

    depth_map: Dict[str, Optional[int]] = {}
    if graph is None:
        return depth_map

    cache = graph.graph.get("_taxonomy_traversal_cache")
    signature = (graph.number_of_nodes(), graph.number_of_edges())
    if not cache or cache.get("signature") != signature:
        try:
            from .taxonomy import ensure_traversal_cache
        except ImportError:  # pragma: no cover - defensive fallback
            ensure_traversal_cache = None
        if ensure_traversal_cache is not None:
            cache = ensure_traversal_cache(graph)

    if cache and cache.get("depth_map"):
        depth_map.update(cache["depth_map"])  # type: ignore[index]
    else:
        try:
            topo_nodes = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            topo_nodes = list(graph.nodes())

        for node in topo_nodes:
            preds = list(graph.predecessors(node))
            if not preds:
                depth_map[node] = 0
                continue
            parent_depths = [depth_map.get(parent) for parent in preds]
            valid_depths = [d for d in parent_depths if d is not None]
            depth_map[node] = (min(valid_depths) + 1) if valid_depths else None

    for node in graph.nodes():
        depth_map.setdefault(node, 0)

    return depth_map


def lookup_direct_parent(
    graph: nx.DiGraph | None, label: Optional[str]
) -> Optional[str]:
    """Return the immediate parent of ``label`` in ``graph``, if available."""

    if graph is None or not isinstance(label, str):
        return None
    if not graph.has_node(label):
        return None

    parents = list(graph.predecessors(label))
    if not parents:
        return None
    if len(parents) == 1:
        return parents[0]

    cache = graph.graph.get("_taxonomy_traversal_cache")
    signature = (graph.number_of_nodes(), graph.number_of_edges())
    if not cache or cache.get("signature") != signature:
        try:
            from .taxonomy import ensure_traversal_cache
        except ImportError:  # pragma: no cover - defensive fallback
            ensure_traversal_cache = None
        if ensure_traversal_cache is not None:
            cache = ensure_traversal_cache(graph)

    if cache and cache.get("primary_parent"):
        parent_map = cache["primary_parent"]  # type: ignore[index]
        parent = parent_map.get(label)
        if parent is not None:
            return parent

    parents.sort(key=lambda n: (str(n).lower(), str(n)))
    return parents[0]


def _cached_undirected_graph(G: nx.DiGraph) -> nx.Graph:
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

    return _cached_undirected_graph(G)


__all__ = [
    "compute_node_depths",
    "get_undirected_taxonomy",
    "lookup_direct_parent",
]
