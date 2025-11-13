from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from vtm.config import FieldMappingConfig, PruningConfig
from ..embedding import (
    Embedder,
    ItemTextChunk,
    collect_item_texts,
    encode_item_texts,
)
from ..taxonomy import make_label_display, normalize_taxonomy_label
from .errors import PrunedTreeComputationError
from .utils import (
    anchor_hull_subtree,
    build_lexical_extractor,
    community_pagerank_subtree,
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
    steiner_similarity_subtree,
    similarity_threshold_subtree,
    taxonomy_similarity_scores,
    tree_sort_key_factory,
)


@dataclass(frozen=True)
class PrunedTreeResult:
    """Lightweight container for pruning output."""

    markdown: str
    allowed_labels: List[str]
    allowed_children: Dict[str, List[List[str]]]


@dataclass(frozen=True)
class PruningContext:
    """Normalized configuration values used during pruning."""

    config: PruningConfig
    enable_taxonomy_pruning: bool
    tree_sort_mode: str
    suggestion_sort_mode: str
    pruning_mode: str
    requires_proximity: bool
    needs_pagerank: bool
    anchor_top_k: int
    anchor_overfetch_multiplier: int
    anchor_min_overfetch: int
    lexical_anchor_limit: int
    community_clique_size: int
    max_community_size: Optional[int]
    pagerank_candidate_limit: int
    suggestion_limit: int
    max_descendant_depth: int
    node_budget: int
    pruning_radius: int
    similarity_threshold: float
    pagerank_damping: float
    pagerank_score_floor: float
    surrogate_root_label: Optional[str]


class TreePruner:
    """Synchronous taxonomy pruning helper."""

    def __init__(
        self,
        *,
        graph: nx.DiGraph,
        frame: pd.DataFrame,
        embedder: Optional[Embedder],
        tax_names: Sequence[str],
        tax_embs_unit: np.ndarray,
        hnsw_index,
        pruning_cfg: Optional[PruningConfig] = None,
        field_mapping: Optional[FieldMappingConfig] = None,
        name_col: str = "name",
        order_col: str = "order",
        gloss_map: Optional[Dict[str, str]] = None,
        snake_case_to_title: bool = True,
    ) -> None:
        self._graph = graph
        self._frame = frame
        self._embedder = embedder
        self._tax_names = [str(name) for name in tax_names]
        self._snake_case_to_title = bool(snake_case_to_title)
        embedding_texts = [
            normalize_taxonomy_label(
                name,
                lowercase=False,
                snake_to_title=self._snake_case_to_title,
            )
            for name in self._tax_names
        ]
        self._tax_names_embedding_texts = embedding_texts
        self._tax_names_normalized = [text.lower() for text in embedding_texts]
        self._lexical_extractor = build_lexical_extractor(
            self._tax_names_normalized, scorer=fuzz.token_sort_ratio
        )
        self._tax_embs_unit = tax_embs_unit
        self._hnsw_index = hnsw_index
        self._config = pruning_cfg or PruningConfig()
        self._field_mapping = field_mapping
        self._name_col = name_col
        self._order_col = order_col
        self._gloss_map = dict(gloss_map or {})

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

    def prune_with_embeddings(
        self,
        *,
        item: Dict[str, Optional[str]],
        item_texts: Sequence[str],
        item_embs: np.ndarray,
    ) -> PrunedTreeResult:
        """Prune ``item`` using precomputed texts and embeddings."""

        try:
            return self._prune_internal(item, precomputed=(item_texts, item_embs))
        except Exception as exc:  # pragma: no cover - defensive
            raise PrunedTreeComputationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # internal helpers
    def _build_pruning_context(self) -> PruningContext:
        cfg = self._config

        tree_sort_mode = normalize_tree_sort_mode(cfg.tree_sort_mode)
        suggestion_sort_mode = normalize_tree_sort_mode(cfg.suggestion_sort_mode)
        requires_proximity = "proximity" in {tree_sort_mode, suggestion_sort_mode}
        pruning_mode = normalize_pruning_mode(cfg.pruning_mode)
        needs_pagerank = "pagerank" in {tree_sort_mode, suggestion_sort_mode}
        if pruning_mode in {"dominant_forest", "community_pagerank"}:
            needs_pagerank = True

        community_clique_size = (
            0
            if not cfg.community_clique_size
            else max(2, int(cfg.community_clique_size))
        )
        max_community_size = (
            None
            if cfg.max_community_size in (None, 0)
            else max(1, int(cfg.max_community_size))
        )
        pagerank_candidate_limit = (
            0
            if cfg.pagerank_candidate_limit is None
            else max(0, int(cfg.pagerank_candidate_limit))
        )

        return PruningContext(
            config=cfg,
            enable_taxonomy_pruning=bool(cfg.enable_taxonomy_pruning),
            tree_sort_mode=tree_sort_mode,
            suggestion_sort_mode=suggestion_sort_mode,
            pruning_mode=pruning_mode,
            requires_proximity=requires_proximity,
            needs_pagerank=needs_pagerank,
            anchor_top_k=max(0, int(cfg.anchor_top_k)),
            anchor_overfetch_multiplier=max(1, int(cfg.anchor_overfetch_multiplier)),
            anchor_min_overfetch=max(1, int(cfg.anchor_min_overfetch)),
            lexical_anchor_limit=max(0, int(cfg.lexical_anchor_limit)),
            community_clique_size=community_clique_size,
            max_community_size=max_community_size,
            pagerank_candidate_limit=pagerank_candidate_limit,
            suggestion_limit=max(0, int(cfg.suggestion_list_limit)),
            max_descendant_depth=max(0, int(cfg.max_descendant_depth)),
            node_budget=max(0, int(cfg.node_budget)),
            pruning_radius=max(0, int(cfg.pruning_radius)),
            similarity_threshold=float(cfg.similarity_threshold),
            pagerank_damping=float(cfg.pagerank_damping),
            pagerank_score_floor=float(cfg.pagerank_score_floor),
            surrogate_root_label=cfg.surrogate_root_label,
        )

    def _prune_internal(
        self,
        item: Dict[str, Optional[str]],
        *,
        precomputed: Optional[Tuple[Sequence[str], np.ndarray]] = None,
    ) -> PrunedTreeResult:
        context = self._build_pruning_context()
        if precomputed is None:
            item_texts, item_embs, similarity_map = self._compute_similarity_maps(
                context, item
            )
        else:
            item_texts, item_embs = self._normalise_precomputed_inputs(precomputed)
            similarity_map = self._build_similarity_map(item_embs)
        anchors, distance_map, pagerank_map, pagerank_data = self._prepare_anchors(
            context,
            item_embs=item_embs,
            item_texts=item_texts,
        )
        allowed, allowed_ranked = self._compute_ranking_inputs(
            context,
            anchors=anchors,
            similarity_map=similarity_map,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
            pagerank_data=pagerank_data,
        )
        return self._render_output(
            context,
            allowed=allowed,
            allowed_ranked=allowed_ranked,
            similarity_map=similarity_map,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

    def _compute_similarity_maps(
        self,
        context: PruningContext,
        item: Dict[str, Optional[str]],
    ) -> Tuple[Sequence[str], np.ndarray, Dict[str, float]]:
        _ = context  # context currently unused but kept for interface parity
        item_texts = collect_item_texts(item, field_mapping=self._field_mapping)

        embedder = self._embedder
        if embedder is None:
            raise RuntimeError(
                "TreePruner was constructed without an embedder; "
                "precomputed embeddings are required."
            )

        item_embs = encode_item_texts(
            item,
            embedder,
            field_mapping=self._field_mapping,
            texts=item_texts,
        )

        similarity_map = self._build_similarity_map(item_embs)

        return item_texts, item_embs, similarity_map

    def _normalise_precomputed_inputs(
        self, precomputed: Tuple[Sequence[ItemTextChunk], np.ndarray]
    ) -> Tuple[Sequence[ItemTextChunk], np.ndarray]:
        item_texts, item_embs = precomputed
        if not isinstance(item_texts, Sequence):
            item_texts = tuple(item_texts)  # type: ignore[assignment]
        item_embs = np.asarray(item_embs, dtype=np.float32)
        if item_embs.ndim != 2:
            raise ValueError(
                "Precomputed embeddings must be a 2D array of shape (n_texts, dim)."
            )
        return item_texts, item_embs

    def _build_similarity_map(self, item_embs: np.ndarray) -> Dict[str, float]:
        similarity_scores = taxonomy_similarity_scores(item_embs, self._tax_embs_unit)
        similarity_map: Dict[str, float] = {
            name: float(score)
            for name, score in zip(self._tax_names, similarity_scores)
        }

        if len(similarity_map) < len(self._tax_names):
            for name in self._tax_names[len(similarity_map) :]:
                similarity_map.setdefault(name, float("-inf"))

        return similarity_map

    def _prepare_anchors(
        self,
        context: PruningContext,
        *,
        item_embs: np.ndarray,
        item_texts: Sequence[ItemTextChunk],
    ) -> Tuple[
        List[str],
        Dict[str, float],
        Dict[str, float],
        Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]],
    ]:
        distance_map: Dict[str, float] = {}
        pagerank_map: Dict[str, float] = {}
        pagerank_data: Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]] = (
            None
        )

        if not context.enable_taxonomy_pruning:
            return [], distance_map, pagerank_map, pagerank_data

        anchors = self._select_anchors(context, item_embs, item_texts)

        if context.requires_proximity and anchors:
            undirected = get_undirected_taxonomy(self._graph)
            anchor_nodes = [a for a in anchors if undirected.has_node(a)]
            if anchor_nodes:
                distances = nx.multi_source_dijkstra_path_length(
                    undirected, sources=anchor_nodes
                )
                distance_map = {node: float(dist) for node, dist in distances.items()}
                for anchor in anchor_nodes:
                    distance_map.setdefault(anchor, 0.0)

        if context.needs_pagerank and anchors:
            pagerank_data = prepare_pagerank_scores(
                self._graph,
                anchors,
                max_descendant_depth=context.max_descendant_depth,
                community_clique_size=context.community_clique_size,
                max_community_size=context.max_community_size,
                pagerank_damping=context.pagerank_damping,
                pagerank_score_floor=context.pagerank_score_floor,
                pagerank_candidate_limit=context.pagerank_candidate_limit,
            )
            pagerank_map = pagerank_data[1]

        return anchors, distance_map, pagerank_map, pagerank_data

    def _compute_ranking_inputs(
        self,
        context: PruningContext,
        *,
        anchors: Sequence[str],
        similarity_map: Dict[str, float],
        distance_map: Dict[str, float],
        pagerank_map: Dict[str, float],
        pagerank_data: Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]],
    ) -> Tuple[set[str], List[str]]:
        if context.enable_taxonomy_pruning:
            allowed = self._select_allowed_nodes(
                context,
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
            sort_mode=context.suggestion_sort_mode,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

        return allowed, allowed_ranked

    def _render_output(
        self,
        context: PruningContext,
        *,
        allowed: set[str],
        allowed_ranked: Sequence[str],
        similarity_map: Dict[str, float],
        distance_map: Dict[str, float],
        pagerank_map: Dict[str, float],
    ) -> PrunedTreeResult:
        tree_markdown = render_tree_markdown(
            self._graph,
            allowed,
            self._frame,
            name_col=self._name_col,
            order_col=self._order_col,
            gloss_map=self._gloss_map,
            similarity_map=similarity_map,
            order_map=self._order_map,
            tree_sort_mode=context.tree_sort_mode,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
            surrogate_root_label=context.surrogate_root_label,
        )

        suggestion_limit = context.suggestion_limit
        top_k = len(allowed_ranked)
        if suggestion_limit > 0:
            top_k = min(top_k, suggestion_limit)

        suggestion_lines = [
            f"- {make_label_display(label, self._gloss_map, use_definition=False)}"
            for label in allowed_ranked[:top_k]
        ]

        sections: List[str] = ["\n# TREE", tree_markdown, "\n# SUGGESTIONS"]
        sections.extend(suggestion_lines)
        markdown = "\n".join(sections)

        allowed_lookup = set(allowed_ranked)
        depth_limit = max(1, int(context.max_descendant_depth))
        allowed_children: Dict[str, List[List[str]]] = {}
        for node in allowed_lookup:
            if not self._graph.has_node(node):
                continue

            visited = {node}
            frontier = [
                child
                for child in self._graph.successors(node)
                if child in allowed_lookup and child not in visited
            ]
            if not frontier:
                continue

            layers: List[List[str]] = []
            depth = 1
            while frontier and depth <= depth_limit:
                layers.append(list(frontier))
                visited.update(frontier)
                if depth == depth_limit:
                    break

                next_frontier: List[str] = []
                for parent in frontier:
                    if not self._graph.has_node(parent):
                        continue
                    for child in self._graph.successors(parent):
                        if child in allowed_lookup and child not in visited:
                            next_frontier.append(child)

                frontier = next_frontier
                depth += 1

            if layers:
                allowed_children[node] = layers

        return PrunedTreeResult(
            markdown=markdown,
            allowed_labels=list(allowed_ranked),
            allowed_children=allowed_children,
        )

    def _select_anchors(
        self,
        context: PruningContext,
        item_embs: np.ndarray,
        item_texts: Sequence[str],
    ) -> List[str]:
        cfg = context.config
        if item_embs.size == 0:
            anchor_idxs = list(range(min(context.anchor_top_k, len(self._tax_names))))
        else:
            anchor_idxs = hnsw_anchor_indices(
                item_embs,
                self._hnsw_index,
                N=len(self._tax_names),
                anchor_top_k=context.anchor_top_k,
                overfetch_mult=context.anchor_overfetch_multiplier,
                min_overfetch=context.anchor_min_overfetch,
            )

        lexical_idxs = lexical_anchor_indices(
            [chunk.text for chunk in item_texts],
            self._tax_names,
            tax_names_normalized=self._tax_names_normalized,
            existing=anchor_idxs,
            max_anchors=context.lexical_anchor_limit,
            extractor=self._lexical_extractor,
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
        context: PruningContext,
        *,
        anchors: Sequence[str],
        similarity_map: Dict[str, float],
        distance_map: Dict[str, float],
        pagerank_map: Dict[str, float],
        pagerank_data: Optional[Tuple[set[str], Dict[str, float], Dict[str, float]]],
    ) -> set[str]:
        sort_key = tree_sort_key_factory(
            context.tree_sort_mode,
            similarity_map=similarity_map,
            order_map=self._order_map,
            distance_map=distance_map,
            pagerank_map=pagerank_map,
        )

        if context.pruning_mode == "anchor_hull":
            return anchor_hull_subtree(
                self._graph,
                anchors,
                max_descendant_depth=context.max_descendant_depth,
                community_clique_size=context.community_clique_size,
                max_community_size=context.max_community_size,
                node_budget=context.node_budget,
                sort_key=sort_key,
            )

        if context.pruning_mode == "similarity_threshold":
            return similarity_threshold_subtree(
                self._graph,
                anchors=anchors,
                similarity_map=similarity_map,
                threshold=context.similarity_threshold,
                node_budget=context.node_budget,
                sort_key=sort_key,
            )

        if context.pruning_mode == "radius":
            return radius_limited_subtree(
                self._graph,
                anchors,
                radius=context.pruning_radius,
                node_budget=context.node_budget,
                sort_key=sort_key,
            )

        if context.pruning_mode == "community_pagerank":
            return community_pagerank_subtree(
                self._graph,
                anchors,
                max_descendant_depth=context.max_descendant_depth,
                node_budget=context.node_budget,
                community_clique_size=context.community_clique_size,
                max_community_size=context.max_community_size,
                pagerank_damping=context.pagerank_damping,
                pagerank_score_floor=context.pagerank_score_floor,
                pagerank_candidate_limit=context.pagerank_candidate_limit,
                sort_key=sort_key,
                precomputed=pagerank_data,
            )

        if context.pruning_mode == "steiner_similarity":
            return steiner_similarity_subtree(
                self._graph,
                anchors=anchors,
                similarity_map=similarity_map,
                node_budget=context.node_budget,
                sort_key=sort_key,
            )

        allowed, pagerank_map = dominant_anchor_forest(
            self._graph,
            anchors,
            max_descendant_depth=context.max_descendant_depth,
            node_budget=context.node_budget,
            community_clique_size=context.community_clique_size,
            max_community_size=context.max_community_size,
            pagerank_damping=context.pagerank_damping,
            pagerank_score_floor=context.pagerank_score_floor,
            pagerank_candidate_limit=context.pagerank_candidate_limit,
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
    field_mapping: Optional[FieldMappingConfig] = None,
    name_col: str = "name",
    order_col: str = "order",
    gloss_map: Optional[Dict[str, str]] = None,
    snake_case_to_title: bool = False,
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
        field_mapping=field_mapping,
        name_col=name_col,
        order_col=order_col,
        gloss_map=gloss_map,
        snake_case_to_title=snake_case_to_title,
    )
    result = pruner.prune(item)
    return result.markdown, result.allowed_labels
