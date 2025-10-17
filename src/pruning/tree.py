from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import PruningConfig
from ..embedding import Embedder, collect_item_texts, encode_item_texts
from ..taxonomy import make_label_display
from .errors import PrunedTreeComputationError
from .utils import (
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


@dataclass(frozen=True)
class PrunedTreeResult:
    """Lightweight container for pruning output."""

    markdown: str
    allowed_labels: List[str]
    allowed_children: Dict[str, List[str]]


class TreePruner:
    """Synchronous taxonomy pruning helper."""

    def __init__(
        self,
        *,
        graph: nx.DiGraph,
        frame: pd.DataFrame,
        embedder: Embedder,
        tax_names: Sequence[str],
        tax_embs_unit: np.ndarray,
        hnsw_index,
        pruning_cfg: Optional[PruningConfig] = None,
        name_col: str = "name",
        order_col: str = "order",
        gloss_map: Optional[Dict[str, str]] = None,
        encode_lock: Optional[threading.Lock] = None,
        index_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._graph = graph
        self._frame = frame
        self._embedder = embedder
        self._tax_names = list(tax_names)
        self._tax_embs_unit = tax_embs_unit
        self._hnsw_index = hnsw_index
        self._config = pruning_cfg or PruningConfig()
        self._name_col = name_col
        self._order_col = order_col
        self._gloss_map = dict(gloss_map or {})
        self._encode_lock = encode_lock or threading.Lock()
        self._index_lock = index_lock or threading.Lock()

        if name_col not in frame or order_col not in frame:
            raise ValueError("Pruning frame must contain name and order columns")

        order_series = frame.groupby(name_col)[order_col].min()
        self._order_map: Dict[str, int] = {
            str(label): int(value)
            for label, value in order_series.items()
            if pd.notna(value)
        }

    def prune(self, item: Dict[str, Optional[str]]) -> PrunedTreeResult:
        """Return markdown tree and allowed labels for ``item``."""

        try:
            return self._prune_internal(item)
        except Exception as exc:  # pragma: no cover - defensive
            raise PrunedTreeComputationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # internal helpers
    def _prune_internal(self, item: Dict[str, Optional[str]]) -> PrunedTreeResult:
        cfg = self._config

        with self._encode_lock:
            item_embs = encode_item_texts(item, self._embedder)

        similarity_scores = taxonomy_similarity_scores(item_embs, self._tax_embs_unit)
        similarity_map: Dict[str, float] = {
            name: float(score)
            for name, score in zip(self._tax_names, similarity_scores)
        }

        if len(similarity_map) < len(self._tax_names):
            for name in self._tax_names[len(similarity_map) :]:
                similarity_map.setdefault(name, float("-inf"))

        normalized_tree_sort_mode = normalize_tree_sort_mode(cfg.tree_sort_mode)
        normalized_suggestion_sort_mode = normalize_tree_sort_mode(
            cfg.suggestion_sort_mode
        )
        requires_proximity = (
            normalized_tree_sort_mode == "proximity"
            or normalized_suggestion_sort_mode == "proximity"
        )
        needs_pagerank = (
            normalized_tree_sort_mode == "pagerank"
            or normalized_suggestion_sort_mode == "pagerank"
        )

        distance_map: Dict[str, float] = {}
        pagerank_map: Dict[str, float] = {}
        pagerank_data: Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]] = (
            None
        )
        anchors: List[str] = []

        if cfg.enable_taxonomy_pruning:
            anchors = self._select_anchors(item_embs, item)

            if requires_proximity and anchors:
                undirected = get_undirected_taxonomy(self._graph)
                anchor_nodes = [a for a in anchors if undirected.has_node(a)]
                if anchor_nodes:
                    distances = nx.multi_source_dijkstra_path_length(
                        undirected, sources=anchor_nodes
                    )
                    distance_map = {
                        node: float(dist) for node, dist in distances.items()
                    }
                    for anchor in anchor_nodes:
                        distance_map.setdefault(anchor, 0.0)

            if needs_pagerank and anchors:
                pagerank_candidate_limit = (
                    0
                    if cfg.pagerank_candidate_limit is None
                    else max(0, int(cfg.pagerank_candidate_limit))
                )
                pagerank_data = prepare_pagerank_scores(
                    self._graph,
                    anchors,
                    max_descendant_depth=cfg.max_descendant_depth,
                    community_clique_size=(
                        0
                        if not cfg.community_clique_size
                        else max(2, int(cfg.community_clique_size))
                    ),
                    max_community_size=(
                        None
                        if cfg.max_community_size in (None, 0)
                        else max(1, int(cfg.max_community_size))
                    ),
                    pagerank_damping=float(cfg.pagerank_damping),
                    pagerank_score_floor=float(cfg.pagerank_score_floor),
                    pagerank_candidate_limit=pagerank_candidate_limit,
                )
                pagerank_map = pagerank_data[1]

            allowed = self._select_allowed_nodes(
                anchors=anchors,
                similarity_map=similarity_map,
                distance_map=distance_map,
                pagerank_map=pagerank_map,
                pagerank_data=pagerank_data,
            )
        else:
            allowed = set(self._graph.nodes)

        allowed_ranked = rank_allowed_nodes(
            allowed,
            similarity_map=similarity_map,
            order_map=self._order_map,
            sort_mode=cfg.suggestion_sort_mode,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

        tree_markdown = render_tree_markdown(
            self._graph,
            allowed,
            self._frame,
            name_col=self._name_col,
            order_col=self._order_col,
            gloss_map=self._gloss_map,
            similarity_map=similarity_map,
            order_map=self._order_map,
            tree_sort_mode=cfg.tree_sort_mode,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
            surrogate_root_label=cfg.surrogate_root_label,
        )

        suggestion_limit = int(cfg.suggestion_list_limit)
        top_k = len(allowed_ranked)
        if suggestion_limit > 0:
            top_k = min(top_k, suggestion_limit)

        suggestion_lines = [
            f"- {make_label_display(label, self._gloss_map, use_summary=False)}"
            for label in allowed_ranked[:top_k]
        ]

        markdown = "\n".join(
            [
                "\n# TREE",
                tree_markdown,
                "\n# SUGGESTIONS",
                *suggestion_lines,
            ]
        )

        allowed_lookup = set(allowed_ranked)
        allowed_children: Dict[str, List[str]] = {}
        for node in allowed_lookup:
            if not self._graph.has_node(node):
                continue
            children = [
                child
                for child in self._graph.successors(node)
                if child in allowed_lookup
            ]
            if children:
                allowed_children[node] = list(children)

        return PrunedTreeResult(
            markdown=markdown,
            allowed_labels=list(allowed_ranked),
            allowed_children=allowed_children,
        )

    def _select_anchors(
        self, item_embs: np.ndarray, item: Dict[str, Optional[str]]
    ) -> List[str]:
        cfg = self._config
        if item_embs.size == 0:
            anchor_idxs = list(range(min(cfg.anchor_top_k, len(self._tax_names))))
        else:
            with self._index_lock:
                anchor_idxs = hnsw_anchor_indices(
                    item_embs,
                    self._hnsw_index,
                    N=len(self._tax_names),
                    anchor_top_k=cfg.anchor_top_k,
                    overfetch_mult=max(1, int(cfg.anchor_overfetch_multiplier)),
                    min_overfetch=max(1, int(cfg.anchor_min_overfetch)),
                )

        lexical_idxs = lexical_anchor_indices(
            collect_item_texts(item),
            self._tax_names,
            existing=anchor_idxs,
            max_anchors=max(0, int(cfg.lexical_anchor_limit)),
        )

        combined: List[int] = []
        seen = set()
        for idx in list(anchor_idxs) + lexical_idxs:
            if idx not in seen:
                seen.add(idx)
                combined.append(idx)
        return [self._tax_names[i] for i in combined]

    def _select_allowed_nodes(
        self,
        *,
        anchors: Sequence[str],
        similarity_map: Dict[str, float],
        distance_map: Dict[str, float],
        pagerank_map: Dict[str, float],
        pagerank_data: Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]],
    ) -> set[str]:
        cfg = self._config
        pruning_mode = normalize_pruning_mode(cfg.pruning_mode)

        sort_key = tree_sort_key_factory(
            cfg.tree_sort_mode,
            similarity_map=similarity_map,
            order_map=self._order_map,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

        if pruning_mode == "anchor_hull":
            return anchor_hull_subtree(
                self._graph,
                anchors,
                max_descendant_depth=cfg.max_descendant_depth,
                community_clique_size=(
                    0
                    if not cfg.community_clique_size
                    else max(2, int(cfg.community_clique_size))
                ),
                max_community_size=(
                    None
                    if cfg.max_community_size in (None, 0)
                    else max(1, int(cfg.max_community_size))
                ),
                node_budget=cfg.node_budget,
                sort_key=sort_key,
            )

        if pruning_mode == "similarity_threshold":
            return similarity_threshold_subtree(
                self._graph,
                anchors=anchors,
                similarity_map=similarity_map,
                threshold=float(cfg.similarity_threshold),
                node_budget=cfg.node_budget,
                sort_key=sort_key,
            )

        if pruning_mode == "radius":
            return radius_limited_subtree(
                self._graph,
                anchors,
                radius=int(cfg.pruning_radius),
                node_budget=cfg.node_budget,
                sort_key=sort_key,
            )

        allowed, pagerank_map = dominant_anchor_forest(
            self._graph,
            anchors,
            max_descendant_depth=cfg.max_descendant_depth,
            node_budget=cfg.node_budget,
            community_clique_size=(
                0
                if not cfg.community_clique_size
                else max(2, int(cfg.community_clique_size))
            ),
            max_community_size=(
                None
                if cfg.max_community_size in (None, 0)
                else max(1, int(cfg.max_community_size))
            ),
            pagerank_damping=float(cfg.pagerank_damping),
            pagerank_score_floor=float(cfg.pagerank_score_floor),
            pagerank_candidate_limit=(
                0
                if cfg.pagerank_candidate_limit is None
                else max(0, int(cfg.pagerank_candidate_limit))
            ),
            precomputed=pagerank_data,
        )
        return allowed


def pruned_tree_markdown_for_item(
    item: Dict[str, Optional[str]],
    *,
    graph: nx.DiGraph,
    frame: pd.DataFrame,
    embedder: Embedder,
    tax_names: Sequence[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    pruning_cfg: Optional[PruningConfig] = None,
    name_col: str = "name",
    order_col: str = "order",
    gloss_map: Optional[Dict[str, str]] = None,
    encode_lock: Optional[threading.Lock] = None,
    index_lock: Optional[threading.Lock] = None,
) -> Tuple[str, List[str]]:
    """Prune ``item`` and return the rendered markdown and allowed labels."""

    pruner = TreePruner(
        graph=graph,
        frame=frame,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs_unit=tax_embs_unit,
        hnsw_index=hnsw_index,
        pruning_cfg=pruning_cfg,
        name_col=name_col,
        order_col=order_col,
        gloss_map=gloss_map,
        encode_lock=encode_lock,
        index_lock=index_lock,
    )
    result = pruner.prune(item)
    return result.markdown, result.allowed_labels
