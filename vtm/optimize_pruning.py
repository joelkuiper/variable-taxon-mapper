"""Optimize pruning configuration parameters using Optuna."""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, cast

import networkx as nx
import numpy as np
import optuna
import pandas as pd

from vtm.cli.prune_check import _compute_effective_subset
from vtm.config import AppConfig, HNSWConfig, PruningConfig, TaxonomyEmbeddingConfig
from vtm.pipeline.service import prepare_keywords_dataframe
from vtm.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from vtm.pruning.tree import TreePruner
from vtm.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from vtm.utils import ensure_file_exists, resolve_path

try:
    from typer import Exit as _TyperExit
    from typer.main import get_command as _get_typer_command
except ImportError:  # pragma: no cover - Typer ships with the project
    _TyperExit = None  # type: ignore[assignment]
    _get_typer_command = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ======================================================================================
# Utilities
# ======================================================================================


def _q(x: float, nd: int = 3) -> float:
    """Pretty-print as float with n decimals (for logs/user attrs only)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return x
    return float(f"{x:.{nd}f}")


# ======================================================================================
# Data containers
# ======================================================================================


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
    tax_embs: np.ndarray
    hnsw_index: object
    gloss_map: Dict[str, str]
    rows: Sequence[EvaluationRow]
    total_nodes: int
    ancestor_cache: Dict[str, set[str]]
    taxonomy_cache: Dict[Tuple[float, float], Tuple[np.ndarray, object]]
    definition_frame: Optional[pd.DataFrame]
    hnsw_config: HNSWConfig


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


class CachingEmbedder(Embedder):
    """``Embedder`` variant that memoises ``encode`` calls by input tuple."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cache: Dict[tuple[str, ...], np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        key = tuple(texts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        encoded = super().encode(texts)
        self._cache[key] = encoded
        return encoded


class SupportsToKwargs(Protocol):
    """Objects exposing a ``to_kwargs`` helper for logging."""

    def to_kwargs(self) -> Dict[str, Any]:
        ...


@dataclass
class ObjectiveState:
    base_config: AppConfig
    context: EvaluationContext
    min_coverage: float
    total_trials: int
    min_possible: Optional[float] = None
    objective_weights: Optional[Dict[str, float]] = None


# ======================================================================================
# Setup helpers
# ======================================================================================
def prepare_context(
    config: AppConfig,
    base_path: Path,
    *,
    variables: Optional[Path],
    keywords: Optional[Path],
    row_limit: Optional[int],
) -> EvaluationContext:
    variables_default, keywords_default = config.data.to_paths(base_path)
    variables_path = resolve_path(base_path, variables_default, variables)
    keywords_path = resolve_path(base_path, keywords_default, keywords)

    ensure_file_exists(variables_path, "variables CSV")
    ensure_file_exists(keywords_path, "keywords CSV")

    variables_df = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)
    keywords_df, definition_df, multi_parents = prepare_keywords_dataframe(
        keywords_raw, config.taxonomy_fields
    )

    graph = build_taxonomy_graph(
        keywords_df,
        name_col="name",
        parent_col="parent",
        order_col="order",
        multi_parents=multi_parents,
    )
    build_name_maps_from_graph(graph)  # validation side-effect

    embedder = CachingEmbedder(**config.embedder.to_kwargs())

    taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
    hnsw_kwargs = config.hnsw.to_kwargs()
    definition_source = definition_df if definition_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        graph,
        embedder,
        definitions=definition_source,
        **taxonomy_kwargs,
    )
    hnsw_index = build_hnsw_index(tax_embs, **hnsw_kwargs)
    gloss_map = build_gloss_map(definition_source)

    field_cfg = config.fields
    dataset_col = field_cfg.resolve_column("dataset")
    label_col = field_cfg.resolve_column("label")
    name_col = field_cfg.resolve_column("name")
    desc_col = field_cfg.resolve_column("description")
    text_keys = field_cfg.item_text_keys()

    work_df, token_sets, _meta = _compute_effective_subset(
        variables_df,
        tax_names,
        cfg=config.evaluation,
        field_mapping=field_cfg,
        row_limit_override=row_limit,
    )

    tax_name_set = set(tax_names)
    rows: List[EvaluationRow] = []
    for idx in range(len(work_df)):
        row = work_df.iloc[idx]
        token_set = token_sets.iloc[idx]
        item = {
            "dataset": row.get(dataset_col) if dataset_col else None,
            "label": row.get(label_col) if label_col else None,
            "name": row.get(name_col) if name_col else None,
            "description": row.get(desc_col) if desc_col else None,
        }
        if text_keys:
            item["_text_fields"] = tuple(text_keys)
        gold_labels = sorted(set(token_set) & tax_name_set)
        rows.append(EvaluationRow(item=item, gold_labels=gold_labels))

    return EvaluationContext(
        graph=graph,
        frame=keywords_df,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs=tax_embs,
        hnsw_index=hnsw_index,
        gloss_map=gloss_map,
        rows=rows,
        total_nodes=int(graph.number_of_nodes()),
        ancestor_cache={},
        taxonomy_cache={
            (
                round(float(taxonomy_kwargs["gamma"]), 6),
                round(float(taxonomy_kwargs["summary_weight"]), 6),
            ): (tax_embs, hnsw_index)
        },
        definition_frame=definition_source,
        hnsw_config=config.hnsw,
    )


# ======================================================================================
# Trial evaluation
# ======================================================================================


def build_pruning_config(trial: optuna.Trial, base_cfg: PruningConfig) -> PruningConfig:
    params = {
        "enable_taxonomy_pruning": True,
        "pruning_mode": trial.suggest_categorical(
            "pruning_mode",
            [
                "community_pagerank",
                "steiner_similarity",
                "anchor_hull",
                "dominant_forest",
            ],
        ),
        "anchor_top_k": trial.suggest_int("anchor_top_k", 4, 64),
        "max_descendant_depth": trial.suggest_int("max_descendant_depth", 1, 4),
        # Allow 0; if >0 and <2 for clique, we coerce to 2 after sampling.
        "lexical_anchor_limit": trial.suggest_int("lexical_anchor_limit", 0, 6),
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
        # Quantized floats (2–3 decimals)
        "pagerank_damping": trial.suggest_float(
            "pagerank_damping", 0.75, 0.95, step=0.01
        ),
        "pagerank_score_floor": trial.suggest_float(
            "pagerank_score_floor", 0.0, 0.05, step=0.001
        ),
        "node_budget": trial.suggest_int("node_budget", 60, 250),
        "tree_sort_mode": trial.suggest_categorical(
            "tree_sort_mode", ["relevance", "proximity", "pagerank"]
        ),
    }
    params["pagerank_candidate_limit"] = trial.suggest_categorical(
        "pagerank_candidate_limit", [None, 64, 128, 256, 384, 512]
    )

    # Coerce clique size to ≥2 if nonzero (avoid degenerate cliques)
    clique = params["community_clique_size"]
    if clique and clique < 2:
        params["community_clique_size"] = 2

    return replace(base_cfg, **params)


def build_taxonomy_config(
    trial: optuna.Trial, base_cfg: TaxonomyEmbeddingConfig
) -> TaxonomyEmbeddingConfig:
    # Quantized to 3 decimals
    gamma = trial.suggest_float("gamma", 0.05, 0.90, step=0.001)
    summary_weight = trial.suggest_float("summary_weight", 0.05, 0.90, step=0.001)
    return replace(base_cfg, gamma=gamma, summary_weight=summary_weight)


def resolve_taxonomy_artifacts(
    context: EvaluationContext,
    taxonomy_cfg: TaxonomyEmbeddingConfig,
) -> Tuple[np.ndarray, object]:
    key = (round(taxonomy_cfg.gamma, 6), round(taxonomy_cfg.summary_weight, 6))
    cached = context.taxonomy_cache.get(key)
    if cached is not None:
        return cached

    _, tax_embs = build_taxonomy_embeddings_composed(
        context.graph,
        context.embedder,
        definitions=context.definition_frame,
        **taxonomy_cfg.to_kwargs(),
    )
    hnsw_index = build_hnsw_index(tax_embs, **context.hnsw_config.to_kwargs())
    context.taxonomy_cache[key] = (tax_embs, hnsw_index)
    return tax_embs, hnsw_index


def evaluate_pruning(
    context: EvaluationContext,
    pruning_cfg: PruningConfig,
    tax_embs: np.ndarray,
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


# ======================================================================================
# Objective (gentle, smooth-gated)
# ======================================================================================


def create_objective(state: ObjectiveState):
    """
    Gentler objective:
      1) Prioritize correctness (possible_rate & coverage) with softer penalties.
      2) Gradually unlock pruning rewards as feasibility improves (smooth gating).
      3) Keep small shaping rewards so Optuna can "see" the boundary.
    """

    weights = state.objective_weights or {
        "miss_linear": 200_000.0,
        "miss_quadratic": 400_000.0,
        "w_possible": 10_000.0,
        "w_coverage": 10_000.0,
        "w_pruned_pct": 8_000.0,
        "w_nodes_pruned": 0.25,
    }

    min_possible = (
        state.min_possible if state.min_possible is not None else state.min_coverage
    )
    min_coverage = state.min_coverage
    gate_lo, gate_hi = 0.70, 1.00

    def _smoothstep(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
        return t * t * (3.0 - 2.0 * t)

    def _huberish_penalty(
        shortfall: float, w_lin: float, w_quad: float, delta: float = 0.05
    ) -> float:
        if shortfall <= 0.0:
            return 0.0
        if shortfall <= delta:
            return w_quad * shortfall * shortfall / (delta if delta > 0 else 1.0)
        quad_at_delta = w_quad * delta
        return quad_at_delta + w_lin * (shortfall - delta)

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

        possible_shortfall = max(0.0, min_possible - possible_rate)
        coverage_shortfall = max(0.0, min_coverage - coverage)

        miss_penalty = _huberish_penalty(
            possible_shortfall, weights["miss_linear"], weights["miss_quadratic"]
        ) + _huberish_penalty(
            coverage_shortfall, weights["miss_linear"], weights["miss_quadratic"]
        )

        shaping_reward = (
            weights["w_possible"] * possible_rate + weights["w_coverage"] * coverage
        )

        norm_possible = possible_rate / max(min_possible, 1e-6)
        norm_coverage = coverage / max(min_coverage, 1e-6)
        feasibility_raw = max(0.0, min(1.0, min(norm_possible, norm_coverage)))
        feas_gate = _smoothstep(feasibility_raw, gate_lo, gate_hi)

        pruning_reward = feas_gate * (
            weights["w_pruned_pct"] * mean_pruned_pct
            + weights["w_nodes_pruned"] * mean_nodes_pruned
        )

        score = shaping_reward + pruning_reward - miss_penalty

        # attrs
        trial.set_user_attr("objective_score", _q(score, 2))
        trial.set_user_attr("possible_correct_under_allowed_rate", _q(possible_rate, 4))
        trial.set_user_attr("coverage", _q(coverage, 4))
        trial.set_user_attr("mean_pruned_percentage", _q(mean_pruned_pct, 4))
        trial.set_user_attr("mean_nodes_pruned", _q(mean_nodes_pruned, 2))
        trial.set_user_attr("mean_nodes_allowed", _q(metrics.mean_nodes_allowed, 2))
        trial.set_user_attr(
            "tree_sort_mode", getattr(pruning_cfg, "tree_sort_mode", None)
        )
        trial.set_user_attr("pruning_mode", getattr(pruning_cfg, "pruning_mode", None))
        trial.set_user_attr(
            "gamma", _q(getattr(taxonomy_cfg, "gamma", float("nan")), 3)
        )
        trial.set_user_attr(
            "summary_weight",
            _q(getattr(taxonomy_cfg, "summary_weight", float("nan")), 3),
        )
        trial.set_user_attr(
            "pagerank_damping",
            _q(getattr(pruning_cfg, "pagerank_damping", float("nan")), 2),
        )
        trial.set_user_attr(
            "pagerank_score_floor",
            _q(getattr(pruning_cfg, "pagerank_score_floor", float("nan")), 3),
        )
        trial.set_user_attr(
            "n_possible_correct_under_allowed",
            int(metrics.n_possible_correct_under_allowed),
        )
        trial.set_user_attr(
            "n_allowed_subtree_contains_gold_or_parent",
            int(metrics.n_allowed_subtree_contains_gold_or_parent),
        )
        trial.set_user_attr("possible_shortfall", _q(possible_shortfall, 4))
        trial.set_user_attr("coverage_shortfall", _q(coverage_shortfall, 4))
        trial.set_user_attr("miss_penalty", _q(miss_penalty, 2))
        trial.set_user_attr("shaping_reward", _q(shaping_reward, 2))
        trial.set_user_attr("pruning_reward", _q(pruning_reward, 2))
        trial.set_user_attr("feasibility_raw", _q(feasibility_raw, 3))
        trial.set_user_attr("feas_gate", _q(feas_gate, 3))

        logger.info(
            "\n".join(
                [
                    f"Trial #{trial.number} (run {trial.number + 1}/{state.total_trials})",
                    f"  objective={score:.2f}",
                    f"  possible_rate={possible_rate:.2%}  (min={min_possible:.2%}, shortfall={possible_shortfall:.4f})",
                    f"  coverage={coverage:.2%}       (min={min_coverage:.2%}, shortfall={coverage_shortfall:.4f})",
                    f"  mean_pruned_pct={mean_pruned_pct:.2%}",
                    f"  mean_nodes_pruned={mean_nodes_pruned:.2f}",
                    f"  mean_nodes_allowed={metrics.mean_nodes_allowed:.2f}",
                    f"  feas_raw={feasibility_raw:.3f}  gate={feas_gate:.3f}",
                    f"  pruning_reward={pruning_reward:.2f}",
                    f"  shaping_reward={shaping_reward:.2f}",
                    f"  miss_penalty={miss_penalty:.2f}",
                ]
            )
        )
        logger.info("-" * 60)

        return score

    return objective


# ======================================================================================
# Study utilities
# ======================================================================================


def _make_pruner(name: str) -> Optional[optuna.pruners.BasePruner]:
    if name == "none":
        return None
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=10)
    if name == "halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.MedianPruner(n_warmup_steps=10)


def select_best_trial(
    study: optuna.Study, *, min_possible: float, min_coverage: float
) -> Optional[optuna.trial.FrozenTrial]:
    feasible = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("possible_correct_under_allowed_rate", 0.0) >= min_possible
        and t.user_attrs.get("coverage", 0.0) >= min_coverage
    ]
    if feasible:
        return max(
            feasible,
            key=lambda t: (
                t.user_attrs.get("mean_pruned_percentage", 0.0),
                t.user_attrs.get("mean_nodes_pruned", 0.0),
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
            "objective_score": trial.user_attrs.get("objective_score"),
            "feasible": trial.user_attrs.get("feasible"),
            "pruning_mode": trial.user_attrs.get("pruning_mode"),
        }
    )
    return params


# --------------------------------------------------------------------------------------
# Printing config with inline comments
# --------------------------------------------------------------------------------------


def print_config_with_inline_comments(
    name: str,
    cfg: SupportsToKwargs,
    inline_comments: Mapping[str, str],
    *,
    inline_placeholders_if_missing: Optional[Mapping[str, str]] = None,
) -> None:
    logger.info("\nSuggested [%s] configuration:", name)
    items = cfg.to_kwargs()
    for key, value in items.items():
        comment = inline_comments.get(key)
        if comment:
            logger.info("%s = %s  # %s", key, json.dumps(value), comment)
        else:
            logger.info("%s = %s", key, json.dumps(value))
    if inline_placeholders_if_missing:
        for key, placeholder in inline_placeholders_if_missing.items():
            if key not in items:
                comment = inline_comments.get(key, "")
                tail = f"  # {comment}" if comment else ""
                logger.info("# %s = %s%s", key, placeholder, tail)


# ======================================================================================
# Optimization runner
# ======================================================================================


def run_optimization(
    app_config: AppConfig,
    *,
    base_path: Path,
    variables: Optional[Path] = None,
    keywords: Optional[Path] = None,
    trials: int = 60,
    seed: Optional[int] = None,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    row_limit: Optional[int] = None,
    min_coverage: float = 0.97,
    min_possible: Optional[float] = None,
    timeout: Optional[int] = None,
    pruner: str = "median",
    save_trials_csv: Optional[Path] = None,
    ensure_mode_repeats: int = 2,
    tpe_startup: Optional[int] = None,
    tpe_multivariate: bool = False,
    tpe_constant_liar: bool = False,
    suppress_experimental_warnings: bool = False,
) -> None:
    """Run the Optuna-based pruning configuration optimisation."""

    if suppress_experimental_warnings:
        try:
            from optuna.exceptions import ExperimentalWarning as _ImportedExperimentalWarning
        except Exception:  # pragma: no cover - optuna too old
            class _FallbackExperimentalWarning(Warning):
                """Fallback warning when Optuna lacks ``ExperimentalWarning``."""

                pass

            experimental_warning_type: type[Warning] = _FallbackExperimentalWarning
        else:
            experimental_warning_type = cast(
                type[Warning], _ImportedExperimentalWarning
            )

        warnings.filterwarnings("ignore", category=experimental_warning_type)

    context = prepare_context(
        app_config,
        base_path,
        variables=variables,
        keywords=keywords,
        row_limit=row_limit,
    )

    logger.info(
        "Prepared optimization context with %d rows across %d taxonomy nodes",
        len(context.rows),
        context.total_nodes,
    )

    if len(context.rows) == 0:
        raise RuntimeError("No eligible rows found for evaluation (after filtering).")

    pruning_modes = [
        "community_pagerank",
        "steiner_similarity",
        "anchor_hull",
        "dominant_forest",
    ]
    repeats = max(0, int(ensure_mode_repeats))
    default_startup = max(24, 3 * repeats * len(pruning_modes))
    tpe_startup_trials = (
        int(tpe_startup) if tpe_startup is not None else default_startup
    )

    tpe_kwargs = dict(seed=seed, n_startup_trials=tpe_startup_trials)
    if tpe_multivariate:
        tpe_kwargs["multivariate"] = True
    if tpe_constant_liar:
        tpe_kwargs["constant_liar"] = True

    sampler = optuna.samplers.TPESampler(**tpe_kwargs)
    pruner_impl = _make_pruner(pruner)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner_impl,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    if repeats > 0:
        for _ in range(repeats):
            for mode in pruning_modes:
                study.enqueue_trial({"pruning_mode": mode})

    state = ObjectiveState(
        base_config=app_config,
        context=context,
        min_coverage=min_coverage,
        total_trials=trials,
        min_possible=min_possible,
    )

    logger.info(
        "Starting optimization with %d trials (timeout=%s)",
        trials,
        timeout,
    )

    try:
        study.optimize(
            create_objective(state),
            n_trials=trials,
            timeout=timeout,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user. Proceeding to results...")

    resolved_min_possible = (
        min_possible if min_possible is not None else min_coverage
    )
    best = select_best_trial(
        study, min_possible=resolved_min_possible, min_coverage=min_coverage
    )
    if best is None:
        logger.warning("No successful trials were completed.")
        return

    params = dump_trial_parameters(best)
    logger.info("Best trial parameters:\n%s", json.dumps(params, indent=2, sort_keys=True))

    INLINE_PRUNING_COMMENTS: Dict[str, str] = {
        "enable_taxonomy_pruning": "",
        "tree_sort_mode": "options: relevance, topological, alphabetical, proximity, pagerank",
        "suggestion_sort_mode": "independent ranking for the suggestion list; options mirror tree_sort_mode",
        "suggestion_list_limit": "number of candidates surfaced alongside the tree",
        "pruning_mode": "options: dominant_forest, anchor_hull, similarity_threshold, radius",
        "similarity_threshold": 'cosine threshold used when pruning_mode="similarity_threshold"',
        "pruning_radius": 'undirected hop limit when pruning_mode="radius"',
        "anchor_top_k": "ANN anchors to fetch per item before pruning",
        "max_descendant_depth": "limit descendants pulled under each anchor",
        "lexical_anchor_limit": "additional anchors sourced from lexical overlap",
        "community_clique_size": "k for k-clique community expansion",
        "max_community_size": "max nodes pulled from any single community, if set",
        "anchor_overfetch_multiplier": "ANN search overfetch multiplier (before pruning)",
        "anchor_min_overfetch": "",
        "pagerank_damping": "",
        "pagerank_score_floor": "",
        "pagerank_candidate_limit": "Cap candidate nodes before PageRank; the smaller of this and node_budget wins.",
        "node_budget": "hard cap on nodes retained in the final allowed set",
    }
    PLACEHOLDERS_IF_MISSING: Dict[str, str] = {
        "surrogate_root_label": json.dumps("Study variables"),
    }

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
        "pruning_mode",
    }
    pruning_updates = {k: v for k, v in best.params.items() if k in pruning_keys}
    best_cfg = replace(app_config.pruning, **pruning_updates)

    print_config_with_inline_comments(
        "pruning",
        best_cfg,
        INLINE_PRUNING_COMMENTS,
        inline_placeholders_if_missing=PLACEHOLDERS_IF_MISSING,
    )

    taxonomy_updates = {
        k: v for k, v in best.params.items() if k in {"gamma", "summary_weight"}
    }
    if taxonomy_updates:
        best_taxonomy_cfg = replace(app_config.taxonomy_embeddings, **taxonomy_updates)
        logger.info("\nSuggested [taxonomy_embeddings] configuration:")
        items = best_taxonomy_cfg.to_kwargs()
        taxonomy_inline = {
            "gamma": "blend between name embedding and structural context (higher → more structure)",
            "summary_weight": "weight for summary/gloss text when available",
        }
        for key, value in items.items():
            comment = taxonomy_inline.get(key)
            if comment:
                logger.info("%s = %s  # %s", key, json.dumps(value), comment)
            else:
                logger.info("%s = %s", key, json.dumps(value))

    if save_trials_csv:
        rows = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {"number": t.number, "value": t.value, "state": str(t.state)}
            row.update({f"param.{k}": v for k, v in t.params.items()})
            row.update({f"user.{k}": v for k, v in t.user_attrs.items()})
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(save_trials_csv, index=False)
            logger.info("Saved %d completed trials → %s", len(rows), save_trials_csv)
        else:
            logger.info("No completed trials to save at %s", save_trials_csv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Delegate to the unified Typer CLI when executed as a script."""

    if _get_typer_command is None or _TyperExit is None:  # pragma: no cover - safety
        raise RuntimeError("Typer is required to invoke the unified CLI")

    if argv is None:
        import sys

        args = list(sys.argv[1:])
    else:
        args = list(argv)

    from vtm.cli import app as cli_app

    command = _get_typer_command(cli_app)
    try:
        command.main(
            args=["optimize-pruning", *args],
            prog_name="vtm",
            standalone_mode=False,
        )
    except _TyperExit as exc:  # pragma: no cover - propagate exit code
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()


