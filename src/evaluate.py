"""Evaluation helpers for running llama.cpp benchmarks."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Set

import aiohttp
import numpy as np
import pandas as pd

from config import EvaluationConfig

from .embedding import Embedder
from .matching import match_item_to_tree
from .taxonomy_pruning import pruned_tree_markdown_for_item
from .taxonomy import is_ancestor_of
from .utils import clean_str_or_none, split_keywords_comma


def is_correct_prediction(
    pred_label: Optional[str], gold_labels: List[str], *, G
) -> bool:
    if not isinstance(pred_label, str):
        return False
    for g in gold_labels:
        if not isinstance(g, str):
            continue
        if pred_label == g:
            return True
        if is_ancestor_of(G, pred_label, g):
            return True
        if is_ancestor_of(G, g, pred_label):
            return True
    return False


def _coerce_eval_config(
    config: EvaluationConfig | Mapping[str, Any] | None,
) -> EvaluationConfig:
    if config is None:
        return EvaluationConfig()
    if isinstance(config, EvaluationConfig):
        return config
    if isinstance(config, Mapping):
        return EvaluationConfig(**config)
    raise TypeError(
        "eval_config must be an EvaluationConfig or a mapping of keyword arguments"
    )


def run_label_benchmark(
    variables: pd.DataFrame,
    keywords: pd.DataFrame,
    *,
    G,
    embedder: Embedder,
    tax_names: List[str],
    tax_embs_unit: np.ndarray,
    hnsw_index,
    name_to_id: Dict[str, str],
    name_to_path: Dict[str, str],
    gloss_map: Dict[str, str],
    eval_config: EvaluationConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = _coerce_eval_config(eval_config)

    dedupe_on = [c for c in (cfg.dedupe_on or []) if c]
    if "keywords" not in variables.columns:
        raise KeyError("variables must have a 'keywords' column")

    work_df = variables.copy()
    if dedupe_on:
        missing = [c for c in dedupe_on if c not in work_df.columns]
        if missing:
            raise KeyError(f"dedupe_on columns missing: {missing}")
        work_df = work_df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(
            drop=True
        )

    known_labels: Set[str] = set(tax_names)

    cleaned_kw = work_df["keywords"].map(clean_str_or_none)
    token_lists = cleaned_kw.map(split_keywords_comma)
    token_sets = token_lists.map(lambda lst: set(t for t in lst if t))
    eligible_mask = token_sets.map(lambda st: len(st & known_labels) > 0)

    total_rows = len(work_df)
    total_with_any_keyword = int(cleaned_kw.notna().sum())
    n_eligible = int(eligible_mask.sum())
    n_excluded_not_in_taxonomy = total_with_any_keyword - n_eligible

    if n_eligible == 0:
        raise ValueError(
            "No eligible rows: no comma-split keywords present in the taxonomy."
        )

    eligible_idxs = list(work_df.index[eligible_mask])
    rnd = random.Random(cfg.seed)
    rnd.shuffle(eligible_idxs)
    idxs = eligible_idxs[: min(cfg.n, len(eligible_idxs))]

    async def _evaluate_all() -> List[Dict[str, Any]]:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None,
            sock_connect=float(cfg.http_sock_connect),
            sock_read=max(float(cfg.http_sock_read_floor), float(cfg.n_predict)),
        )
        connector = aiohttp.TCPConnector(
            limit=max(1, cfg.pool_maxsize), limit_per_host=max(1, cfg.pool_maxsize)
        )

        rows: List[Dict[str, Any]] = []
        start = time.time()
        correct_sum = 0
        total = len(idxs)

        async with aiohttp.ClientSession(
            timeout=timeout_cfg, connector=connector
        ) as session:
            for j, i in enumerate(idxs):
                r = work_df.loc[i]
                gold_tokens = token_sets.loc[i]
                gold_labels = sorted(gold_tokens & known_labels)

                item = {
                    "dataset": r.get("dataset"),
                    "label": r.get("label"),
                    "name": r.get("name"),
                    "description": r.get("description"),
                }

                slot_id = j % max(1, cfg.num_slots)

                tree_markdown, allowed_labels = pruned_tree_markdown_for_item(
                    item,
                    G=G,
                    df=keywords,
                    embedder=embedder,
                    tax_names=tax_names,
                    tax_embs_unit=tax_embs_unit,
                    hnsw_index=hnsw_index,
                    name_col="name",
                    order_col="order",
                    top_k_nodes=cfg.top_k_nodes,
                    desc_max_depth=cfg.desc_max_depth,
                    max_total_nodes=cfg.max_total_nodes,
                    gloss_map=gloss_map,
                    anchor_overfetch_mult=cfg.anchor_overfetch_mult,
                    anchor_min_overfetch=cfg.anchor_min_overfetch,
                    candidate_list_max_items=cfg.candidate_list_max_items,
                )

                match_kwargs = {}
                if cfg.llm_grammar is not None:
                    match_kwargs["grammar"] = cfg.llm_grammar

                pred = await match_item_to_tree(
                    item,
                    tree_markdown=tree_markdown,
                    allowed_labels=allowed_labels,
                    name_to_id=name_to_id,
                    name_to_path=name_to_path,
                    tax_names=tax_names,
                    tax_embs=tax_embs_unit,
                    embedder=embedder,
                    hnsw_index=hnsw_index,
                    endpoint=cfg.endpoint,
                    n_predict=cfg.n_predict,
                    temperature=cfg.temperature,
                    slot_id=slot_id,
                    cache_prompt=cfg.llm_cache_prompt,
                    n_keep=cfg.llm_n_keep,
                    top_k=cfg.llm_top_k,
                    top_p=cfg.llm_top_p,
                    min_p=cfg.llm_min_p,
                    session=session,
                    **match_kwargs,
                )

                resolved_label = pred.get("resolved_label")
                correct = is_correct_prediction(resolved_label, gold_labels, G=G)

                out: Dict[str, Any] = {
                    "dataset": item.get("dataset"),
                    "label": item.get("label"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "gold_labels": gold_labels,
                    "pred_label_raw": pred.get("pred_label_raw"),
                    "resolved_label": resolved_label,
                    "resolved_id": pred.get("resolved_id"),
                    "resolved_path": pred.get("resolved_path"),
                    "correct": bool(correct),
                    "_idx": i,
                    "_j": j,
                    "_slot": slot_id,
                    "_error": None,
                }
                if "raw" in pred:
                    out["raw"] = pred["raw"]

                rows.append(out)
                correct_sum += 1 if out.get("correct") else 0

                done = len(rows)
                interval = max(1, int(cfg.progress_log_interval))
                if done % interval == 0 or done == total:
                    elapsed = max(time.time() - start, 1e-6)
                    rps = done / elapsed if elapsed > 0 else 0.0
                    acc = (correct_sum / done) if done else 0.0
                    sys.stderr.write(
                        f"\rEvaluating: {done}/{total} "
                        f"(acc≈{acc:.3f}, {rps:.1f} rows/s)"
                    )
                    sys.stderr.flush()

        if total:
            sys.stderr.write("\n")

        return rows

    try:
        rows = asyncio.run(_evaluate_all())
    except KeyboardInterrupt:
        sys.stderr.write("\nEvaluation cancelled.\n")
        raise
    except RuntimeError as exc:
        if "asyncio.run()" in str(exc) and "running event loop" in str(exc):
            raise RuntimeError(
                "run_label_benchmark must be called from a synchronous context "
                "without an active event loop."
            ) from exc
        raise

    rows.sort(key=lambda r: r["_j"])
    for r in rows:
        r.pop("_j", None)
        r.pop("_idx", None)
        r.pop("_slot", None)

    df = pd.DataFrame(rows)

    metrics: Dict[str, Any] = {
        "n_total_rows_after_dedupe": int(total_rows),
        "n_with_any_keyword": int(total_with_any_keyword),
        "n_eligible": int(n_eligible),
        "n_excluded_not_in_taxonomy": int(n_excluded_not_in_taxonomy),
        "n_evaluated": int(len(df)),
        "label_accuracy_any_match": float(df["correct"].mean()) if len(df) else 0.0,
        "n_errors": int(df["_error"].notna().sum()) if "_error" in df.columns else 0,
    }

    return df, metrics
