"""Optimize pruning configuration parameters using Optuna."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import networkx as nx
import optuna
import pandas as pd

from check_pruned_tree import _compute_effective_subset
from config import AppConfig, PruningConfig, TaxonomyEmbeddingConfig, load_config
from main import _prepare_keywords
from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.pruning.tree import TreePruner
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)


@dataclass
class EvaluationRow:
    """Lightweight container for evaluation inputs."""

    item: Dict[str, Optional[str]]
    gold_labels: List[str]


@dataclass
class EvaluationContext:
    """Precomputed assets reused across Optuna trials."""

    graph: nx.DiGraph
    frame: pd.DataFrame
    embedder: Embedder
    tax_names: Sequence[str]
    tax_embs: object
    hnsw_index: object
    gloss_map: Mapping[str, str]
    rows: Sequence[EvaluationRow]
    total_nodes: int
    encode_lock: threading.Lock
    index_lock: threading.Lock
    ancestor_cache: Dict[str, set[str]]
    taxonomy_cache: Dict[Tuple[float, float], Tuple[object, object]]
    summary_frame: Optional[pd.DataFrame]
    hnsw_kwargs: Dict[str, object]


@dataclass
class EvaluationMetrics:
    possible_correct_under_allowed_rate: float
    allowed_subtree_contains_gold_or_parent_rate: float
    mean_pruned_percentage: float
    mean_nodes_pruned: float
    mean_nodes_allowed: float
    n_evaluated: int
    n_possible_correct_under_allowed: int
    n_allowed_subtree_contains_gold_or_parent: int


class CachingEmbedder:
    """Wrapper around ``Embedder`` adding simple memoisation."""

    def __init__(self, base: Embedder) -> None:
        self._base = base
        self._cache: Dict[tuple[str, ...], object] = {}
        # Mirror key attributes used by downstream utilities.
        for attr in ("device", "tok", "model", "max_length", "batch_size", "mean_pool"):
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def encode(self, texts: Sequence[str]):
        key = tuple(texts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        encoded = self._base.encode(texts)
        self._cache[key] = encoded
        return encoded


@dataclass
class ObjectiveState:
    base_config: AppConfig
    context: EvaluationContext
    min_coverage: float
    total_trials: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune pruning parameters to maximise coverage and pruning.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    parser.add_argument(
        "--variables",
        type=Path,
        default=None,
        help="Optional override for the variables CSV file.",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=None,
        help="Optional override for the taxonomy keywords CSV file.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Number of Optuna trials to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for Optuna's sampler.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL for persisting studies.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name when using persistent storage.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N eligible rows after shuffling.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.97,
        help="Minimum allowed_subtree_contains_gold_or_parent_rate to target.",
    )
    return parser.parse_args(argv)


def _resolve_path(base_path: Path, override: Optional[Path], default: Path) -> Path:
    if override is None:
        return default
    return override if override.is_absolute() else (base_path / override)


def prepare_context(
    config: AppConfig,
    base_path: Path,
    *,
    variables: Optional[Path],
    keywords: Optional[Path],
    row_limit: Optional[int],
) -> EvaluationContext:
    variables_default, keywords_default = config.data.to_paths(base_path)
    variables_path = _resolve_path(base_path, variables, variables_default)
    keywords_path = _resolve_path(base_path, keywords, keywords_default)

    variables_df = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)
    keywords_df, summary_df = _prepare_keywords(keywords_raw)

    graph = build_taxonomy_graph(
        keywords_df,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    build_name_maps_from_graph(graph)  # validation side-effect

    embedder = Embedder(**config.embedder.to_kwargs())
    cached_embedder = CachingEmbedder(embedder)

    taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
    hnsw_kwargs = config.hnsw.to_kwargs()
    summary_source = summary_df if summary_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        graph,
        cached_embedder,
        summaries=summary_source,
        **taxonomy_kwargs,
    )
    hnsw_index = build_hnsw_index(tax_embs, **hnsw_kwargs)
    gloss_map = build_gloss_map(summary_source)

    work_df, token_sets, _meta = _compute_effective_subset(
        variables_df,
        tax_names,
        cfg=config.evaluation,
        row_limit_override=row_limit,
    )

    tax_name_set = set(tax_names)
    rows: List[EvaluationRow] = []
    for idx in range(len(work_df)):
        row = work_df.iloc[idx]
        token_set = token_sets.iloc[idx]
        item = {
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "name": row.get("name"),
            "description": row.get("description"),
        }
        gold_labels = sorted(set(token_set) & tax_name_set)
        rows.append(EvaluationRow(item=item, gold_labels=gold_labels))

    return EvaluationContext(
        graph=graph,
        frame=keywords_df,
        embedder=cached_embedder,
        tax_names=tax_names,
        tax_embs=tax_embs,
        hnsw_index=hnsw_index,
        gloss_map=gloss_map,
        rows=rows,
        total_nodes=int(graph.number_of_nodes()),
        encode_lock=threading.Lock(),
        index_lock=threading.Lock(),
        ancestor_cache={},
        taxonomy_cache={
            (taxonomy_kwargs["gamma"], taxonomy_kwargs["summary_weight"]): (
                tax_embs,
                hnsw_index,
            )
        },
        summary_frame=summary_source,
        hnsw_kwargs=hnsw_kwargs,
    )


def build_pruning_config(trial: optuna.Trial, base_cfg: PruningConfig) -> PruningConfig:
    params = {
        "enable_taxonomy_pruning": True,
        "pruning_mode": trial.suggest_categorical(
            "pruning_mode", ["anchor_hull", "dominant_forest"]
        ),
        "anchor_top_k": trial.suggest_int("anchor_top_k", 8, 32),
        "max_descendant_depth": trial.suggest_int("max_descendant_depth", 1, 4),
        "lexical_anchor_limit": trial.suggest_int("lexical_anchor_limit", 0, 4),
        "community_clique_size": trial.suggest_int("community_clique_size", 0, 4),
        "max_community_size": trial.suggest_categorical(
            "max_community_size", [None, 64, 96, 128, 160, 192, 224, 256]
        ),
        "anchor_overfetch_multiplier": trial.suggest_int(
            "anchor_overfetch_multiplier", 1, 6
        ),
        "anchor_min_overfetch": trial.suggest_categorical(
            "anchor_min_overfetch", [64, 96, 128, 160, 192, 224, 256]
        ),
        "pagerank_damping": trial.suggest_float("pagerank_damping", 0.75, 0.95),
        "node_budget": trial.suggest_int("node_budget", 80, 200),
        "tree_sort_mode": trial.suggest_categorical(
            "tree_sort_mode",
            ["relevance", "proximity", "pagerank"],
        ),
    }
    params["pagerank_candidate_limit"] = trial.suggest_categorical(
        "pagerank_candidate_limit", [None, 0, 128, 256, 384, 512]
    )
    clique = params["community_clique_size"]
    if clique and clique < 2:
        params["community_clique_size"] = 2
    return replace(base_cfg, **params)


def build_taxonomy_config(
    trial: optuna.Trial, base_cfg: TaxonomyEmbeddingConfig
) -> TaxonomyEmbeddingConfig:
    gamma = trial.suggest_float("gamma", 0.05, 0.9)
    summary_weight = trial.suggest_float("summary_weight", 0.05, 0.9)
    return replace(base_cfg, gamma=gamma, summary_weight=summary_weight)


def resolve_taxonomy_artifacts(
    context: EvaluationContext,
    taxonomy_cfg: TaxonomyEmbeddingConfig,
) -> Tuple[object, object]:
    key = (round(taxonomy_cfg.gamma, 6), round(taxonomy_cfg.summary_weight, 6))
    cached = context.taxonomy_cache.get(key)
    if cached is not None:
        return cached

    _, tax_embs = build_taxonomy_embeddings_composed(
        context.graph,
        context.embedder,
        summaries=context.summary_frame,
        **taxonomy_cfg.to_kwargs(),
    )
    hnsw_index = build_hnsw_index(tax_embs, **context.hnsw_kwargs)
    context.taxonomy_cache[key] = (tax_embs, hnsw_index)
    return tax_embs, hnsw_index


def evaluate_pruning(
    context: EvaluationContext,
    pruning_cfg: PruningConfig,
    tax_embs: object,
    hnsw_index: object,
) -> EvaluationMetrics:
    pruner = TreePruner(
        graph=context.graph,
        frame=context.frame,
        embedder=context.embedder,
        tax_names=context.tax_names,
        tax_embs_unit=tax_embs,
        hnsw_index=hnsw_index,
        pruning_cfg=pruning_cfg,
        name_col="name",
        order_col="order",
        gloss_map=context.gloss_map,
        encode_lock=context.encode_lock,
        index_lock=context.index_lock,
    )

    total_nodes = max(context.total_nodes, 1)
    n_evaluated = 0
    n_possible_hits = 0
    n_gold_parent_hits = 0
    sum_pruned_pct = 0.0
    sum_allowed = 0.0
    sum_pruned = 0.0

    ancestor_cache = context.ancestor_cache

    for row in context.rows:
        result = pruner.prune(row.item)
        allowed = set(result.allowed_labels)
        allowed_count = len(allowed)
        pruned_count = max(total_nodes - allowed_count, 0)
        pct_pruned = float(pruned_count) / float(total_nodes)

        gold_labels = row.gold_labels
        gold_set = set(gold_labels)

        has_possible = bool(allowed & gold_set)
        if has_possible:
            n_possible_hits += 1

        has_gold_or_parent = has_possible
        for gold in gold_labels:
            if gold in allowed:
                has_gold_or_parent = True
                break
            ancestors = ancestor_cache.get(gold)
            if ancestors is None:
                ancestors = (
                    nx.ancestors(context.graph, gold)
                    if context.graph.has_node(gold)
                    else set()
                )
                ancestor_cache[gold] = ancestors
            if allowed & ancestors:
                has_gold_or_parent = True
                break

        if has_gold_or_parent:
            n_gold_parent_hits += 1

        n_evaluated += 1
        sum_allowed += allowed_count
        sum_pruned += pruned_count
        sum_pruned_pct += pct_pruned

    possible_rate = float(n_possible_hits) / float(n_evaluated) if n_evaluated else 0.0
    coverage = float(n_gold_parent_hits) / float(n_evaluated) if n_evaluated else 0.0
    mean_pruned_pct = sum_pruned_pct / float(n_evaluated) if n_evaluated else 0.0
    mean_allowed = sum_allowed / float(n_evaluated) if n_evaluated else 0.0
    mean_pruned = sum_pruned / float(n_evaluated) if n_evaluated else 0.0

    return EvaluationMetrics(
        possible_correct_under_allowed_rate=possible_rate,
        allowed_subtree_contains_gold_or_parent_rate=coverage,
        mean_pruned_percentage=mean_pruned_pct,
        mean_nodes_pruned=mean_pruned,
        mean_nodes_allowed=mean_allowed,
        n_evaluated=n_evaluated,
        n_possible_correct_under_allowed=n_possible_hits,
        n_allowed_subtree_contains_gold_or_parent=n_gold_parent_hits,
    )


def create_objective(state: ObjectiveState):
    def objective(trial: optuna.Trial) -> float:
        pruning_cfg = build_pruning_config(trial, state.base_config.pruning)
        taxonomy_cfg = build_taxonomy_config(
            trial, state.base_config.taxonomy_embeddings
        )
        tax_embs, hnsw_index = resolve_taxonomy_artifacts(state.context, taxonomy_cfg)
        metrics = evaluate_pruning(state.context, pruning_cfg, tax_embs, hnsw_index)
        possible_rate = metrics.possible_correct_under_allowed_rate
        coverage = metrics.allowed_subtree_contains_gold_or_parent_rate
        mean_pruned_pct = metrics.mean_pruned_percentage
        mean_nodes_pruned = metrics.mean_nodes_pruned

        trial.set_user_attr("possible_correct_under_allowed_rate", possible_rate)
        trial.set_user_attr(
            "n_possible_correct_under_allowed", metrics.n_possible_correct_under_allowed
        )
        trial.set_user_attr(
            "n_allowed_subtree_contains_gold_or_parent",
            metrics.n_allowed_subtree_contains_gold_or_parent,
        )
        trial.set_user_attr("coverage", coverage)
        trial.set_user_attr("mean_pruned_percentage", mean_pruned_pct)
        trial.set_user_attr("mean_nodes_pruned", mean_nodes_pruned)
        trial.set_user_attr("mean_nodes_allowed", metrics.mean_nodes_allowed)
        trial.set_user_attr("tree_sort_mode", pruning_cfg.tree_sort_mode)
        trial.set_user_attr("gamma", taxonomy_cfg.gamma)
        trial.set_user_attr("summary_weight", taxonomy_cfg.summary_weight)

        print(
            f"trial {trial.number + 1}/{state.total_trials}: "
            f"possible={possible_rate:.2%}, coverage={coverage:.2%}, pruned={mean_pruned_pct:.2%}",
            flush=True,
        )

        if possible_rate < state.min_coverage:
            penalty = (
                possible_rate * 50.0 - (state.min_coverage - possible_rate) * 400.0
            )
            return penalty

        if coverage < state.min_coverage:
            penalty = (possible_rate + coverage) * 50.0 - (
                state.min_coverage - coverage
            ) * 200.0
            return penalty

        pruning_bonus = mean_nodes_pruned if mean_nodes_pruned else 0.0
        return possible_rate * 1200.0 + coverage * 400.0 + pruning_bonus

    return objective


def select_best_trial(
    study: optuna.Study, *, min_coverage: float
) -> Optional[optuna.trial.FrozenTrial]:
    qualified = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("possible_correct_under_allowed_rate", 0.0) >= min_coverage
        and t.user_attrs.get("coverage", 0.0) >= min_coverage
    ]
    if qualified:
        return max(
            qualified,
            key=lambda t: (
                t.user_attrs.get("possible_correct_under_allowed_rate", 0.0),
                t.user_attrs.get("coverage", 0.0),
                t.user_attrs.get("mean_pruned_percentage", 0.0),
            ),
        )
    return study.best_trial if study.best_trial is not None else None


def dump_trial_parameters(trial: optuna.trial.FrozenTrial) -> Dict[str, object]:
    params = dict(trial.params)
    params.update(
        {
            "possible_correct_under_allowed_rate": trial.user_attrs.get(
                "possible_correct_under_allowed_rate"
            ),
            "coverage": trial.user_attrs.get("coverage"),
            "mean_pruned_percentage": trial.user_attrs.get("mean_pruned_percentage"),
            "mean_nodes_pruned": trial.user_attrs.get("mean_nodes_pruned"),
            "mean_nodes_allowed": trial.user_attrs.get("mean_nodes_allowed"),
            "n_possible_correct_under_allowed": trial.user_attrs.get(
                "n_possible_correct_under_allowed"
            ),
            "n_allowed_subtree_contains_gold_or_parent": trial.user_attrs.get(
                "n_allowed_subtree_contains_gold_or_parent"
            ),
        }
    )
    return params


def print_config_section(
    name: str,
    cfg: object,
    *,
    comments: Optional[Mapping[str, Sequence[str]]] = None,
    trailing_comments: Sequence[str] = (),
) -> None:
    print(f"\nSuggested [{name}] configuration:")
    items = cfg.to_kwargs()
    for key, value in items.items():
        if comments:
            for line in comments.get(key, cast(Sequence[str], ())):
                print(line)
        print(f"{key} = {json.dumps(value)}")
    for line in trailing_comments:
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    app_config = load_config(config_path)

    context = prepare_context(
        app_config,
        base_path,
        variables=args.variables,
        keywords=args.keywords,
        row_limit=args.row_limit,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage and args.study_name),
    )

    state = ObjectiveState(
        base_config=app_config,
        context=context,
        min_coverage=args.min_coverage,
        total_trials=args.trials,
    )
    study.optimize(create_objective(state), n_trials=args.trials)

    best = select_best_trial(study, min_coverage=args.min_coverage)
    if best is None:
        print("No successful trials were completed.")
        return

    params = dump_trial_parameters(best)
    print("Best trial parameters:")
    print(json.dumps(params, indent=2, sort_keys=True))

    pruning_keys = {
        "suggestion_list_limit",
        "anchor_top_k",
        "max_descendant_depth",
        "lexical_anchor_limit",
        "community_clique_size",
        "max_community_size",
        "anchor_overfetch_multiplier",
        "anchor_min_overfetch",
        "pagerank_damping",
        "pagerank_score_floor",
        "node_budget",
        "pagerank_candidate_limit",
        "tree_sort_mode",
    }
    pruning_updates = {k: v for k, v in best.params.items() if k in pruning_keys}
    best_cfg = replace(app_config.pruning, **pruning_updates)
    pruning_comment_map = {
        "pagerank_candidate_limit": [
            "# Cap candidate nodes before PageRank; the smaller of this and node_budget wins."
        ]
    }
    trailing_pruning_comments = [
        '# surrogate_root_label = "Study variables" # optional: introduce a surrogate root label for the taxonomy tree'
    ]
    print_config_section(
        "pruning",
        best_cfg,
        comments=pruning_comment_map,
        trailing_comments=trailing_pruning_comments,
    )

    taxonomy_updates = {
        k: v for k, v in best.params.items() if k in {"gamma", "summary_weight"}
    }
    if taxonomy_updates:
        best_taxonomy_cfg = replace(app_config.taxonomy_embeddings, **taxonomy_updates)
        print_config_section("taxonomy_embeddings", best_taxonomy_cfg)


if __name__ == "__main__":
    main()
