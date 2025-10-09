"""Utilities for building and working with taxonomy graphs."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd


def _path_id_hex8(path_parts: List[str]) -> str:
    """Deterministic 8-hex id from the full path (joined by ' / ')."""
    s = " / ".join(path_parts)
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
    bad = [n for n in G.nodes if G.in_degree(n) > 1]
    if bad:
        raise ValueError(
            f"Nodes with multiple parents found (≤1 expected): {', '.join(sorted(bad)[:8])}"
        )

    return G


def roots_in_order(G: nx.DiGraph, sort_key) -> List[str]:
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    roots.sort(key=sort_key)
    return roots


def path_to_root(G: nx.DiGraph, node: str) -> List[str]:
    """Unique path from root to ``node`` (since ≤1 parent per node)."""
    path = [node]
    cur = node
    while True:
        preds = list(G.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def build_name_maps_from_graph(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    """Return name -> id (hex8) and name -> full path string."""
    name_to_id: Dict[str, str] = {}
    name_to_path: Dict[str, str] = {}
    for n in G.nodes:
        path = path_to_root(G, n)
        nid = _path_id_hex8(path)
        name_to_id[n] = nid
        name_to_path[n] = " / ".join(path)
    return name_to_id, name_to_path


def tree_to_markdown(
    G: nx.DiGraph,
    df: pd.DataFrame,
    name_col: str = "name",
    order_col: str = "order",
) -> str:
    """Nested markdown list with labels only (no ids)."""
    order_map = df.groupby(name_col)[order_col].min().to_dict()
    sort_key = sort_key_factory(order_map)

    lines: List[str] = []

    def _walk(node: str, depth: int):
        lines.append("  " * depth + f"- {node}")
        for c in sorted(G.successors(node), key=sort_key):
            _walk(c, depth + 1)

    for r in roots_in_order(G, sort_key):
        _walk(r, 0)

    return "\n".join(lines)


def taxonomy_node_texts(G: nx.DiGraph) -> List[str]:
    nodes = list(G.nodes)
    return sorted(nodes, key=lambda s: s.lower())


def ancestors_to_root(G: nx.DiGraph, node: str) -> List[str]:
    path = []
    cur = node
    while True:
        path.append(cur)
        preds = list(G.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
    path.reverse()
    return path


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
    """Build {name -> <=25 word summary} from Keywords_summarized."""
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


def make_label_display(name: str, gloss_map: Dict[str, str]) -> str:
    """Render 'Label — summary' for display; fall back to plain label."""
    if not isinstance(name, str) or not name:
        return str(name)
    g = gloss_map.get(name, "")
    g = g.strip() if isinstance(g, str) else ""
    return f"{name} — {g}" if g else name


def ancestors_inclusive(G: nx.DiGraph, node: str) -> List[str]:
    out, cur, seen = [], node, set()
    while True:
        if cur in seen:
            break
        seen.add(cur)
        out.append(cur)
        preds = list(G.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
    return out[::-1]


def is_ancestor_of(G: nx.DiGraph, maybe_ancestor: str, node: str) -> bool:
    if maybe_ancestor == node:
        return True
    return (
        maybe_ancestor in ancestors_inclusive(G, node)
        if (maybe_ancestor in G and node in G)
        else False
    )
