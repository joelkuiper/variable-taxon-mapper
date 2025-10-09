"""Evaluation helpers for running llama.cpp benchmarks."""

from __future__ import annotations

import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from embedding import Embedder
from infer import match_item_to_tree, pruned_tree_markdown_for_item
from taxonomy import is_ancestor_of


def clean_str_or_none(v) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def split_keywords_comma(s: Optional[str]) -> List[str]:
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


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
    endpoint: str = "http://127.0.0.1:8080/completions",
    n: int = 50,
    seed: int = 0,
    n_predict: int = 256,
    temperature: float = 0.0,
    dedupe_on: Optional[List[str]] = None,
    top_k_nodes: int = 32,
    desc_max_depth: int = 3,
    max_total_nodes: int = 800,
    max_workers: int = 4,
    num_slots: int = 4,
    pool_maxsize: int = 64,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
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
    rnd = random.Random(seed)
    rnd.shuffle(eligible_idxs)
    idxs = eligible_idxs[: min(n, len(eligible_idxs))]

    _tls = threading.local()

    def _make_adapter(pool_maxsize: int) -> HTTPAdapter:
        retry = Retry(
            total=2,
            backoff_factor=0.2,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        return HTTPAdapter(
            pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry
        )

    def _get_thread_session(pool_maxsize: int = 64) -> requests.Session:
        s = getattr(_tls, "session", None)
        if s is None:
            s = requests.Session()
            adapter = _make_adapter(pool_maxsize)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            _tls.session = s
        return s

    def _worker(j: int, i: int) -> Dict[str, Any]:
        r = work_df.loc[i]
        gold_tokens = token_sets.loc[i]
        gold_labels = sorted(gold_tokens & known_labels)

        item = {
            "dataset": r.get("dataset"),
            "label": r.get("label"),
            "name": r.get("name"),
            "description": r.get("description"),
        }

        slot_id = j % max(1, num_slots)

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
            top_k_nodes=top_k_nodes,
            desc_max_depth=desc_max_depth,
            max_total_nodes=max_total_nodes,
            gloss_map=gloss_map,
        )
        try:
            pred = match_item_to_tree(
                item,
                tree_markdown=tree_markdown,
                allowed_labels=allowed_labels,
                name_to_id=name_to_id,
                name_to_path=name_to_path,
                tax_names=tax_names,
                tax_embs=tax_embs_unit,
                embedder=embedder,
                hnsw_index=hnsw_index,
                endpoint=endpoint,
                n_predict=n_predict,
                temperature=temperature,
                slot_id=slot_id,
                cache_prompt=True,
                n_keep=-1,
                session=_get_thread_session(pool_maxsize=pool_maxsize),
            )
            resolved_label = pred.get("resolved_label")
            correct = is_correct_prediction(resolved_label, gold_labels, G=G)
            out = {
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
                "rationale": pred.get("rationale"),
                "_idx": i,
                "_j": j,
                "_slot": slot_id,
                "_error": None,
            }
            if "raw" in pred:
                out["raw"] = pred["raw"]
            return out
        except Exception as e:
            return {
                "dataset": item.get("dataset"),
                "label": item.get("label"),
                "name": item.get("name"),
                "description": item.get("description"),
                "gold_labels": gold_labels,
                "pred_label_raw": None,
                "resolved_label": None,
                "resolved_id": None,
                "resolved_path": None,
                "correct": False,
                "rationale": None,
                "_idx": i,
                "_j": j,
                "_slot": slot_id,
                "_error": f"{type(e).__name__}: {e}",
            }

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, j, i): j for j, i in enumerate(idxs)}
        start = time.time()
        done = 0
        total = len(futures)
        err = 0
        correct_sum = 0

        for fut in as_completed(futures):
            res = fut.result()
            rows.append(res)
            done += 1
            err += 1 if res.get("_error") else 0
            correct_sum += 1 if res.get("correct") else 0

            if done % 10 == 0 or done == total:
                elapsed = max(time.time() - start, 1e-6)
                rps = done / elapsed
                acc = (correct_sum / done) if done else 0.0
                sys.stderr.write(
                    f"\rEvaluating: {done}/{total} "
                    f"(errors={err}, acc≈{acc:.3f}, {rps:.1f} rows/s)"
                )
                sys.stderr.flush()

        sys.stderr.write("\n")

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
        "dedupe_on": dedupe_on or [],
        "max_workers": int(max_workers),
        "num_slots": int(num_slots),
        "pool_maxsize": int(pool_maxsize),
        "n_predict": int(n_predict),
        "temperature": float(temperature),
        "endpoint": endpoint,
        "n_errors": int(df["_error"].notna().sum()) if "_error" in df.columns else 0,
    }

    return df, metrics
