"""Utilities for building and working with taxonomy graphs."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd


def normalize_taxonomy_label(
    name: str,
    *,
    lowercase: bool = True,
    snake_to_title: bool = True,
) -> str:
    """Return a normalized taxonomy label for lexical or embedding use."""

    text = str(name).strip()
    if snake_to_title:
        replaced = text.replace("_", " ")
        words = [w for w in replaced.split() if w]
        if words:
            text = " ".join(word.capitalize() for word in words)
        else:
            text = replaced
    if lowercase:
        return text.lower()
    return text


def _path_id_hex8(path_parts: List[str]) -> str:
    """Deterministic 8-hex id from the full path (joined by ' | ')."""
    s = " | ".join(path_parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def sort_key_factory(order_map: Dict[str, float]):
    def sort_key(node_name: str):
        o = order_map.get(node_name, math.inf)
        return (o if pd.notna(o) else math.inf, node_name.lower())

    return sort_key


def build_taxonomy_graph(
    df: pd.DataFrame,
    name_col: str = "name",
    parent_col: str = "parent",
    order_col: str = "order",
) -> nx.DiGraph:
    """Build a directed acyclic graph (forest of trees) with edges (parent -> child)."""
    G = nx.DiGraph()

    for n in df[name_col].dropna().astype(str):
        G.add_node(n, label=n)

    for p in df[parent_col].dropna().astype(str):
        if not G.has_node(p):
            G.add_node(p, label=p, placeholder_root=True)

    for _, row in df.iterrows():
        child = row[name_col]
        if pd.isna(child):
            continue
        c = str(child)
        parent = row[parent_col]
        if not pd.isna(parent):
            G.add_edge(str(parent), c)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(
            "Taxonomy graph contains a cycle; expected a DAG (forest of trees)."
        )
    ensure_traversal_cache(G)

    return G


def ensure_traversal_cache(G: nx.DiGraph) -> Dict[str, Any]:
    """Ensure canonical traversal metadata is cached on ``G``."""

    signature = (G.number_of_nodes(), G.number_of_edges())
    cache = G.graph.get("_taxonomy_traversal_cache")
    if cache and cache.get("signature") == signature:
        return cache

    try:
        topo_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo_nodes = list(G.nodes())

    parent_map: Dict[str, Optional[str]] = {}
    path_map: Dict[str, Tuple[str, ...]] = {}
    depth_map: Dict[str, Optional[int]] = {}
    ancestor_sets: Dict[str, FrozenSet[str]] = {}

    for node in topo_nodes:
        preds = list(G.predecessors(node))
        if not preds:
            parent_map[node] = None
            path_map[node] = (node,)
            depth_map[node] = 0
            ancestor_sets[node] = frozenset()
            continue

        best_parent: Optional[str] = None
        best_path: Tuple[str, ...] | None = None
        best_key: Tuple[int, Tuple[str, ...], Tuple[str, ...]] | None = None
        combined_ancestors: Set[str] = set()
        for parent in preds:
            parent_path = path_map.get(parent)
            if parent_path is None:
                continue
            candidate_path = parent_path + (node,)
            candidate_key = (
                len(candidate_path),
                tuple(part.lower() for part in candidate_path),
                candidate_path,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_parent = parent
                best_path = candidate_path

            combined_ancestors.update(ancestor_sets.get(parent, frozenset()))
            combined_ancestors.add(parent)

        if best_key is None or best_path is None:
            parent_map[node] = None
            path_map[node] = (node,)
            depth_map[node] = 0
            ancestor_sets[node] = frozenset(combined_ancestors)
            continue

        parent_map[node] = best_parent
        path_map[node] = best_path
        depth_map[node] = len(best_path) - 1
        ancestor_sets[node] = frozenset(combined_ancestors)

    cache = {
        "signature": signature,
        "primary_parent": parent_map,
        "canonical_path": path_map,
        "depth_map": depth_map,
        "ancestor_sets": ancestor_sets,
    }
    G.graph["_taxonomy_traversal_cache"] = cache
    return cache


def roots_in_order(G: nx.DiGraph, sort_key) -> List[str]:
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    roots.sort(key=sort_key)
    return roots


def path_to_root(G: nx.DiGraph, node: str) -> List[str]:
    """Return a canonical path from a root to ``node``."""

    if node not in G:
        return []

    cache = ensure_traversal_cache(G)
    path = cache.get("canonical_path", {}).get(node)
    if not path:
        return [node]
    return list(path)


def build_name_maps_from_graph(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    """Return name -> id (hex8) and name -> full path string."""
    name_to_id: Dict[str, str] = {}
    name_to_path: Dict[str, str] = {}
    for n in G.nodes:
        path = path_to_root(G, n)
        nid = _path_id_hex8(path)
        name_to_id[n] = nid
        name_to_path[n] = " | ".join(path)
    return name_to_id, name_to_path


def taxonomy_node_texts(G: nx.DiGraph) -> List[str]:
    nodes = list(G.nodes)
    return sorted(nodes, key=lambda s: s.lower())


def ancestors_to_root(G: nx.DiGraph, node: str) -> List[str]:
    if node not in G:
        return []

    cache = ensure_traversal_cache(G)
    depth_map = cache.get("depth_map", {})
    ancestor_sets = cache.get("ancestor_sets", {})
    ancestors = set(ancestor_sets.get(node, frozenset()))
    ancestors.add(node)

    def _sort_key(label: str) -> Tuple[int, str]:
        depth = depth_map.get(label)
        if depth is None:
            return (math.inf, label.lower())
        return (depth, label.lower())

    return sorted(ancestors, key=_sort_key)


def collect_descendants(G: nx.DiGraph, node: str, max_depth: int) -> Set[str]:
    allowed = {node}
    frontier = [(node, 0)]
    while frontier:
        cur, d = frontier.pop()
        if d >= max_depth:
            continue
        for c in G.successors(cur):
            if c not in allowed:
                allowed.add(c)
                frontier.append((c, d + 1))
    return allowed


def build_gloss_map(keywords_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Build gloss."""
    gloss: Dict[str, str] = {}
    if keywords_df is None or not {"name", "definition_summary"}.issubset(
        keywords_df.columns
    ):
        return gloss
    for _, row in keywords_df[["name", "definition_summary"]].iterrows():
        n = row["name"]
        s = row["definition_summary"]
        if isinstance(n, str) and n and isinstance(s, str):
            s = s.strip()
            if s and n not in gloss:
                gloss[n] = s
    return gloss


def make_label_display(
    name: str, gloss_map: Dict[str, str], use_summary: bool = True
) -> str:
    """Render 'Label [summary]' for display; fall back to plain label."""
    if not isinstance(name, str) or not name:
        return str(name)
    if not use_summary:
        return str(name)
    else:
        g = gloss_map.get(name, "")
        g = g.strip() if isinstance(g, str) else ""
        return f"{name} [{g}]" if g else name


def ancestors_inclusive(G: nx.DiGraph, node: str) -> List[str]:
    return ancestors_to_root(G, node)


def is_ancestor_of(G: nx.DiGraph, maybe_ancestor: str, node: str) -> bool:
    if maybe_ancestor == node:
        return True
    if maybe_ancestor not in G or node not in G:
        return False

    cache = ensure_traversal_cache(G)
    ancestor_sets = cache.get("ancestor_sets", {})
    ancestors = ancestor_sets.get(node, frozenset())
    return maybe_ancestor in ancestors
