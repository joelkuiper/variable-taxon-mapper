#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import re
from textwrap import dedent
from typing import Any, Dict, Iterable, Optional, Tuple, Union, Set, List

import pandas as pd
import networkx as nx
import requests
import random
from tqdm.auto import tqdm


# =============================================================================
# Utilities
# =============================================================================


def unique_dataset_label_name_desc(
    variables: pd.DataFrame,
    cols: tuple[str, str, str, str] = ("dataset", "label", "name", "description"),
    *,
    strip: bool = True,
    drop_all_na: bool = True,  # drop rows where all four are NA/empty
) -> list[dict]:
    """
    Return unique (dataset, label, name, description) rows as JSON-like dicts.
    """
    if not set(cols).issubset(variables.columns):
        missing = [c for c in cols if c not in variables.columns]
        raise KeyError(f"Missing expected columns: {missing}")

    df = variables.loc[:, cols].copy()

    # Normalize strings (strip whitespace)
    if strip:
        for c in cols:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Optionally drop rows where all 4 fields are NA / empty string
    if drop_all_na:
        df = df.replace("", pd.NA)
        df = df.dropna(how="all")

    # Deduplicate on the 4-tuple
    df = df.drop_duplicates(subset=list(cols), keep="first")

    # Replace pandas NA with None for clean JSON-like dicts
    df = df.where(pd.notna(df), None)

    return df.to_dict(orient="records")


# =============================================================================
# Tree building with stable IDs
# =============================================================================
def _path_id_hex8(path_parts: list[str]) -> str:
    """Deterministic 8-hex id from the full path (joined by ' / ')."""
    s = " / ".join(path_parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _sort_key_factory(order_map: dict[str, float]):
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
    """
    Build a directed acyclic graph (forest of trees) with edges (parent -> child).

    - Nodes created for all `name` values and any `parent` not present in `name`.
    - NaN parent => node is a root (no incoming edges).
    - Validates: acyclic and ≤1 parent per node.
    """
    G = nx.DiGraph()

    # Add all names
    for n in df[name_col].dropna().astype(str):
        G.add_node(n, label=n)

    # Add placeholder parents (if referenced but not present in names)
    for p in df[parent_col].dropna().astype(str):
        if not G.has_node(p):
            G.add_node(p, label=p, placeholder_root=True)

    # Add edges parent -> child
    for _, row in df.iterrows():
        child = row[name_col]
        if pd.isna(child):
            continue
        c = str(child)
        parent = row[parent_col]
        if not pd.isna(parent):
            G.add_edge(str(parent), c)

    # Validate DAG and single-parent constraint
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


def _roots_in_order(G: nx.DiGraph, sort_key) -> list[str]:
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    roots.sort(key=sort_key)
    return roots


def _path_to_root(G: nx.DiGraph, node: str) -> list[str]:
    """Unique path from root to `node` (since ≤1 parent per node)."""
    path = [node]
    cur = node
    while True:
        preds = list(G.predecessors(cur))
        if not preds:
            break
        cur = preds[0]  # unique parent
        path.append(cur)
    path.reverse()
    return path


def tree_to_markdown_with_ids_from_graph(
    G: nx.DiGraph,
    df: pd.DataFrame,
    name_col: str = "name",
    order_col: str = "order",
) -> str:
    """
    Render the graph as nested markdown lists with [hex8] IDs computed from full paths.
    """
    order_map = df.groupby(name_col)[order_col].min().to_dict()
    sort_key = _sort_key_factory(order_map)

    lines: list[str] = []

    def _walk(node: str, depth: int):
        path = _path_to_root(G, node)
        nid = _path_id_hex8(path)
        lines.append("  " * depth + f"- [{nid}] {node}")
        for c in sorted(G.successors(node), key=sort_key):
            _walk(c, depth + 1)

    for r in _roots_in_order(G, sort_key):
        _walk(r, 0)

    return "\n".join(lines)


def build_id_maps_from_graph(G: nx.DiGraph) -> tuple[dict, dict]:
    """
    Reverse maps:
      - id_to_label: hex8 -> node label
      - id_to_path:  hex8 -> "Root / Child / Leaf"
    """
    id_to_label: dict[str, str] = {}
    id_to_path: dict[str, str] = {}

    for n in G.nodes:
        path = _path_to_root(G, n)
        nid = _path_id_hex8(path)
        id_to_label[nid] = n
        id_to_path[nid] = " / ".join(path)

    return id_to_label, id_to_path


# =============================================================================
# Prompt builder
# =============================================================================


def _clean_text(v) -> str:
    """Return a clean, safe string for prompt fields. NaN/None -> '(empty)'."""
    if v is None:
        return "(empty)"
    if isinstance(v, str):
        s = v.strip()
        return s if s else "(empty)"
    # handle pandas NaN / floats
    try:
        import math

        if isinstance(v, float) and math.isnan(v):
            return "(empty)"
    except Exception:
        pass
    s = str(v).strip()
    return s if s else "(empty)"


def make_tree_match_prompt(
    tree_markdown_with_ids: str,
    item: Dict[str, Optional[str]],
) -> str:
    ds = _clean_text(item.get("dataset"))
    lab = _clean_text(item.get("label"))
    nam = _clean_text(item.get("name"))
    desc = _clean_text(item.get("description"))
    tree_md = (tree_markdown_with_ids or "").strip()

    template = """\
        <|im_start|>system
        You are a biomedical taxonomy matcher.

        ## TASK
        • You must match the ITEM to the TREE. Choose the best-matching node for the ITEM.
        • If NO node is a reasonably confident match, return a NO-MATCH result.

        The TREE is a nested (indented) Markdown list where each line is:
        - [<node_id>] <label>

        ## OUTPUT
        EITHER a match:

        {{"node_id":"[xxxxxxxx]","node_label":"...","rationale":"..."}}

        OR a no-match:
        {{"node_id":null,"node_label":null,"rationale":"Why no suitable node fits"}}

        ## TREE
        {TREE}.<|im_end|>

        <|im_start|>user
        ## ITEM:
        - label: {LAB}
        - name: {NAM}
        - description: {DESC}<|im_end|>
        <|im_start|>assistant
    """

    return (
        dedent(template)
        .format(TREE=tree_md, DS=ds, LAB=lab, NAM=nam, DESC=desc)
        .strip()
    )


# =============================================================================
# GBNF: allow node_id/node_label to be string OR null
# Keep obj on ONE physical line for llama.cpp compatibility
# =============================================================================

GRAMMAR_RESPONSE = r"""
root      ::= obj
quote     ::= "\""

string ::=
  quote (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )+ quote

obj ::= ("{" quote "node_id" quote ": " string ","
             quote "node_label" quote ": " string ","
             quote "rationale" quote ": " string "}")
"""


class LlamaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


# Reuse one session for the whole process (keeps TCP alive, faster)
_HTTP_SESSION = requests.Session()


def llama_completion(
    prompt: Union[str, Iterable[Union[str, int]]],
    endpoint: str,  # e.g. "http://127.0.0.1:8080/completions"
    *,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> str:
    """
    POST /completions, expects JSON with 'content'.
    Pass KV-cache hints via kwargs (cache_prompt=True, n_keep=-1, slot_id=0).
    On non-2xx, raise with server's error text.
    """
    s = session or _HTTP_SESSION
    payload: Dict[str, Any] = {"prompt": prompt, "stream": False}
    payload.update(kwargs)

    resp = s.post(endpoint, json=payload, timeout=(10, timeout))  # (connect, read)
    if not resp.ok:
        raise LlamaHTTPError(resp.status_code, resp.text)
    return resp.json()["content"]


# =============================================================================
# Set-up and testing
# =============================================================================


variables = pd.read_csv("data/Variables.csv", low_memory=False)
keywords = pd.read_csv("data/Keywords.csv")

# Build taxonomy graph
G = build_taxonomy_graph(
    keywords, name_col="name", parent_col="parent", order_col="order"
)

# Render markdown (for prompting) with [node_id] tags
tree_md = tree_to_markdown_with_ids_from_graph(
    G, keywords, name_col="name", order_col="order"
)

# Reverse maps: id -> label, id -> path
id_to_label, id_to_path = build_id_maps_from_graph(G)

items = unique_dataset_label_name_desc(variables)

_ID_RX = re.compile(r"^\[?([0-9a-fA-F]{8,40})\]?$")  # accept with/without brackets


def _norm_node_id(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = _ID_RX.match(s)
    return m.group(1).lower() if m else None


def match_item_to_tree(
    item: Dict[str, Optional[str]],
    *,
    tree_markdown_with_ids: str,
    id_to_label: Dict[str, str],
    id_to_path: Dict[str, str],
    endpoint: str = "http://127.0.0.1:8080/completions",
    grammar: str = GRAMMAR_RESPONSE,
    temperature: float = 0.0,
    n_predict: int = 128,
    slot_id: int = 0,
    cache_prompt: bool = True,
    n_keep: int = -1,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    prompt = make_tree_match_prompt(tree_markdown_with_ids, item)

    raw = llama_completion(
        prompt,
        endpoint,
        session=session,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        min_p=0,
        n_predict=n_predict,
        grammar=grammar,
        stop=["\n"],
        cache_prompt=cache_prompt,
        n_keep=n_keep,
        slot_id=slot_id,
    )

    def _pack(matched: bool, node_id, node_label, path, rationale, *, keep_raw=False):
        out = {
            "input_item": item,
            "matched": matched,
            "no_match": not matched,
            "matched_node_id": node_id,
            "matched_node_label": node_label,
            "matched_node_path": path,
            "rationale": rationale,
        }
        if keep_raw:
            out["raw"] = raw
        return out

    try:
        p = json.loads(raw)
    except Exception:
        # parsing failed → treat as no-match but keep raw for debugging
        return _pack(False, None, None, None, None, keep_raw=True)

    node_label = p.get("node_label")
    node_id = _norm_node_id(p.get("node_id")) if p.get("node_id") is not None else None
    rationale = p.get("rationale")

    # Explicit no-match from model
    if node_id is None and node_label is None:
        return _pack(False, None, None, None, rationale)

    # Resolve id → path; fall back to unique label→id mapping
    resolved_id = (
        node_id if isinstance(node_id, str) and node_id in id_to_path else None
    )
    if resolved_id is None and isinstance(node_label, str):
        ids = [k for k, v in id_to_label.items() if v == node_label]
        resolved_id = ids[0] if len(ids) == 1 else None

    path = id_to_path.get(resolved_id)
    return _pack(bool(path), node_id, node_label, path, rationale)


def _clean_str_or_none(v) -> Optional[str]:
    """Return a stripped string or None for NaN/None/empty."""
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    try:
        import math

        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def _split_keywords_comma(s: Optional[str]) -> List[str]:
    """Split a comma-delimited keywords field into clean tokens."""
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def run_label_benchmark(
    variables: pd.DataFrame,
    *,
    tree_markdown_with_ids: str,
    id_to_label: Dict[str, str],
    id_to_path: Dict[str, str],
    endpoint: str = "http://127.0.0.1:8080/completions",
    grammar: str = GRAMMAR_RESPONSE,
    n: int = 50,
    seed: int = 0,
    slot_id: int = 0,
    n_predict: int = 256,
    temperature: float = 0.0,
    dedupe_on: Optional[
        List[str]
    ] = None,  # e.g. ["dataset","label","name","description","keywords"]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comma-aware benchmark on UNIQUE rows:
      - Optionally drop duplicate rows by `dedupe_on` before evaluating.
      - Parse variables['keywords'] as comma-separated labels.
      - A row is ELIGIBLE if ANY token is in taxonomy labels.
      - Correct if predicted node_label is in the token set.
    """
    if "keywords" not in variables.columns:
        raise KeyError("variables must have a 'keywords' column")

    # 0) Work on a de-duplicated copy (if requested)
    work_df = variables.copy()
    if dedupe_on:
        missing = [c for c in dedupe_on if c not in work_df.columns]
        if missing:
            raise KeyError(f"dedupe_on columns missing: {missing}")
        work_df = work_df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(
            drop=True
        )

    known_labels: Set[str] = set(id_to_label.values())

    # 1) Build token sets per row and eligibility mask up front
    cleaned_kw = work_df["keywords"].map(_clean_str_or_none)
    token_lists = cleaned_kw.map(_split_keywords_comma)
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

    # 2) Sample eligible rows
    eligible_idxs = list(work_df.index[eligible_mask])
    random.Random(seed).shuffle(eligible_idxs)
    idxs = eligible_idxs[: min(n, len(eligible_idxs))]

    # 3) Evaluate
    rows = []
    for i in tqdm(idxs, total=len(idxs), desc="Evaluating", unit="row"):
        r = work_df.loc[i]
        gold_tokens = token_sets.loc[i]
        gold_labels = sorted(gold_tokens & known_labels)

        item = {
            "dataset": r.get("dataset"),
            "label": r.get("label"),
            "name": r.get("name"),
            "description": r.get("description"),
        }

        pred = match_item_to_tree(
            item,
            tree_markdown_with_ids=tree_markdown_with_ids,
            id_to_label=id_to_label,
            id_to_path=id_to_path,
            endpoint=endpoint,
            grammar=grammar,
            n_predict=n_predict,
            temperature=temperature,
            slot_id=slot_id,
            cache_prompt=True,
            n_keep=-1,
        )

        pred_label = pred.get("matched_node_label")
        correct = isinstance(pred_label, str) and (pred_label in gold_labels)

        rows.append(
            {
                "dataset": item.get("dataset"),
                "label": item.get("label"),
                "name": item.get("name"),
                "description": item.get("description"),
                "gold_labels": gold_labels,  # list (filtered to known labels)
                "pred_label": pred_label,
                "pred_id": pred.get("matched_node_id"),
                "correct": bool(correct),
                "rationale": pred.get("rationale"),
            }
        )

    df = pd.DataFrame(rows)

    # 4) Metrics
    metrics: Dict[str, Any] = {
        "n_total_rows_after_dedupe": int(total_rows),
        "n_with_any_keyword": int(total_with_any_keyword),
        "n_eligible": int(n_eligible),
        "n_excluded_not_in_taxonomy": int(n_excluded_not_in_taxonomy),
        "n_evaluated": int(len(df)),
        "label_accuracy_any_match": float(df["correct"].mean()) if len(df) else 0.0,
        "dedupe_on": dedupe_on or [],
    }

    return df, metrics


#########
# Example
#########
# match_item_to_tree(
#     items[225],
#     tree_markdown_with_ids=tree_md,
#     id_to_label=id_to_label,
#     id_to_path=id_to_path,
#     grammar=GRAMMAR_RESPONSE,
#     n_predict=512,
#     slot_id=0,  # reuse slot across calls
#     cache_prompt=True,  # keep KV cache for the shared TREE section
#     n_keep=-1,  # keep entire prompt cached
# )

# df, metrics = run_label_benchmark(
#     variables,
#     tree_markdown_with_ids=tree_md,
#     id_to_label=id_to_label,
#     id_to_path=id_to_path,
#     n=100,
#     dedupe_on=["label"],
#     seed=100,
# )
