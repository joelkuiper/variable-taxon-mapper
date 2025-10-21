"""Shared graph helper functions used across the project."""
from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


def compute_node_depths(graph: nx.DiGraph | None) -> Dict[str, Optional[int]]:
    """Return a mapping of node -> depth for the provided graph."""

    depth_map: Dict[str, Optional[int]] = {}
    if graph is None:
        return depth_map

    try:
        topo_nodes = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        topo_nodes = list(graph.nodes())

    for node in topo_nodes:
        preds = list(graph.predecessors(node))
        if not preds:
            depth_map[node] = 0
            continue
        parent = preds[0]
        parent_depth = depth_map.get(parent)
        depth_map[node] = parent_depth + 1 if parent_depth is not None else None

    for node in graph.nodes():
        depth_map.setdefault(node, 0)

    return depth_map


def lookup_direct_parent(graph: nx.DiGraph | None, label: Optional[str]) -> Optional[str]:
    """Return the immediate parent of ``label`` in ``graph``, if available."""

    if graph is None or not isinstance(label, str):
        return None
    if not graph.has_node(label):
        return None

    parents = graph.predecessors(label)
    return next(parents, None)


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
