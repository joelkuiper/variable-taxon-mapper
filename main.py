#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from textwrap import dedent
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd
import requests


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


def build_tree_with_ids(
    df: pd.DataFrame,
    name_col: str = "name",
    parent_col: str = "parent",
    order_col: str = "order",
) -> dict:
    """
    Build a nested dict:
      {
        root_label: {
          "id": <hex8>,
          "children": {
            child_label: { "id": <hex8>, "children": {...} },
            ...
          }
        },
        ...
      }

    - Siblings sorted by (order, name).
    - Node IDs are SHA-1 of the full path parts (stable & unique per path).
    """
    order_map = df.groupby(name_col)[order_col].min().to_dict()
    children_map = defaultdict(list)
    all_names = set(df[name_col].dropna().astype(str))
    parent_vals = df[parent_col]

    for _, row in df.iterrows():
        child = row[name_col]
        parent = row[parent_col]
        if pd.isna(child):
            continue
        child = str(child)
        parent_key = None if pd.isna(parent) else str(parent)
        children_map[parent_key].append(child)

    # Roots: NaN parents, parents not present as names, and names with no parent
    roots = set()
    for p in parent_vals.unique():
        if pd.isna(p):
            roots.add(None)
        else:
            ps = str(p)
            if ps not in all_names:
                roots.add(ps)

    all_children = {c for lst in children_map.values() for c in lst}
    for n in all_names:
        if n not in all_children:
            roots.add(n)

    def sort_key(node_name: str):
        o = order_map.get(node_name, math.inf)
        return (o if pd.notna(o) else math.inf, node_name.lower())

    def build_subtree(parent_key: Optional[str], path_prefix: list[str]) -> dict:
        seen, uniq = set(), []
        for k in children_map.get(parent_key, []):
            if k not in seen:
                uniq.append(k)
                seen.add(k)
        uniq.sort(key=sort_key)

        out = {}
        for k in uniq:
            path = path_prefix + [k]
            out[k] = {
                "id": _path_id_hex8(path),
                "children": build_subtree(k, path),
            }
        return out

    tree = {}
    if None in roots:
        for name, meta in build_subtree(None, []).items():
            tree[name] = meta

    named_roots = sorted([r for r in roots if r is not None], key=sort_key)
    for r in named_roots:
        tree[r] = {
            "id": _path_id_hex8([r]),
            "children": build_subtree(r, [r]),
        }
    return tree


def tree_to_markdown_with_ids(tree: dict, indent: int = 0) -> str:
    """
    Render {label: {"id":..,"children":{...}}} as nested markdown lists:
      - [abcd1234] Label
        - [ef567890] Child
    """
    lines = []
    for label, meta in tree.items():
        nid = meta.get("id", "????????")
        lines.append("  " * indent + f"- [{nid}] {label}")
        kids = meta.get("children") or {}
        if kids:
            lines.append(tree_to_markdown_with_ids(kids, indent + 1))
    return "\n".join(lines)


def build_id_maps(tree: dict) -> Tuple[dict, dict]:
    """
    Build reverse maps:
      - id_to_label: hex8 -> node label
      - id_to_path:  hex8 -> "Root / Child / Leaf"
    """
    id_to_label: Dict[str, str] = {}
    id_to_path: Dict[str, str] = {}

    def _walk(subtree: dict, path: list[str]):
        for label, meta in subtree.items():
            nid = meta.get("id")
            curr_path = path + [label]
            if nid:
                id_to_label[nid] = label
                id_to_path[nid] = " / ".join(curr_path)
            kids = meta.get("children") or {}
            if kids:
                _walk(kids, curr_path)

    _walk(tree, [])
    return id_to_label, id_to_path


# =============================================================================
# Prompt builder
# =============================================================================


def make_tree_match_prompt(
    tree_markdown_with_ids: str,
    item: Dict[str, Optional[str]],
) -> str:
    ds = (item.get("dataset") or "").strip() or "(empty)"
    lab = (item.get("label") or "").strip() or "(empty)"
    nam = (item.get("name") or "").strip() or "(empty)"
    desc = (item.get("description") or "").strip() or "(empty)"
    tree_md = (tree_markdown_with_ids or "").strip()

    template = """\
        <|im_start|>system
        You are a precise taxonomy matcher.

        The TREE is a nested Markdown list where each node line is:
        - [<node_id>] <label>.

        Select the SINGLE best-matching node for the ITEM and copy:
        - node_id: the bracketed id from the chosen line,
        - node_label: the exact node label text as it appears in TREE.

        Return JSON ONLY on one line with this exact schema:
        {{"node_id":"[xxxxxxxx]","node_label":"...","rationale":"..."}}

        CRITICAL SELECTION RULES (follow in order):
        1) **Prefer the deepest / most specific node (leaf)** that fits the ITEM.
           Do NOT select a parent if any descendant matches better.
        2) If multiple leaves fit, prefer the one whose label appears verbatim in the ITEM
           (dataset/label/name/description). Otherwise prefer the longer path (more segments).
        3) Only choose a top-level node if no suitable child exists anywhere below it.

        Be concise but decisive. Copy the id EXACTLY as shown (including brackets).
        TREE
        {TREE}.<|im_end|>

        <|im_start|>user
        ITEM:
        - dataset: {DS}
        - label: {LAB}
        - name: {NAM}
        - description: {DESC} <|im_end|>
        <|im_start|>assistant
    """

    return (
        dedent(template)
        .format(
            TREE=tree_md,
            DS=ds,
            LAB=lab,
            NAM=nam,
            DESC=desc,
        )
        .strip()
    )


GRAMMAR_RESPONSE = r"""
root ::= obj
obj ::= "{" ws "\"" "node_id" "\"" ws ":" ws string ws "," ws "\"" "node_label" "\"" ws ":" ws string ws "," ws "\"" "rationale" "\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" .
ws ::= [ \t\n\r]*
"""


# =============================================================================
# HTTP client (sticky session; KV cache options)
# =============================================================================


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
    POST /completions (no streaming), expects JSON with 'content'.
    Pass KV-cache hints via kwargs (e.g., cache_prompt=True, n_keep=-1, slot_id=0).
    On non-2xx, raise with server's error text for easy diagnosis.
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

# Build tree WITH IDS and render as markdown including the [node_id]
tree = build_tree_with_ids(
    keywords, name_col="name", parent_col="parent", order_col="order"
)
tree_md = tree_to_markdown_with_ids(tree)

# Build reverse maps: id -> label, id -> path
id_to_label, id_to_path = build_id_maps(tree)

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
    slot_id: int = 0,  # <— pin to a slot so KV cache can be reused
    cache_prompt: bool = True,  # <— reuse prompt KV across calls
    n_keep: int = -1,  # <— keep entire prompt in cache
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    # 1) Build prompt
    prompt = make_tree_match_prompt(tree_markdown_with_ids, item)

    # 2) Call model — deterministic & single-line; enable KV cache reuse
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
        stop=["\n"],  # encourage exactly one JSON line
        cache_prompt=cache_prompt,
        n_keep=n_keep,
        slot_id=slot_id,
    )

    # 3) Parse (no robust fallback)
    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "input_item": item,
            "raw": raw,
            "parsed": None,
            "matched_node_id": None,
            "matched_node_label": None,
            "matched_node_path": None,
            "matched": False,
            "rationale": None,
        }

    # 4) Extract fields
    node_id_raw = parsed.get("node_id")
    node_label = parsed.get("node_label")
    rationale = parsed.get("rationale")
    node_id = _norm_node_id(node_id_raw)

    # 5) Resolve id → path (simple, correct order)
    #    A) If node_id exists in the map, accept it immediately.
    resolved_id = None
    if isinstance(node_id, str) and node_id in id_to_path:
        resolved_id = node_id
    else:
        #    B) Else, if node_label uniquely maps to one id, use that.
        label_to_ids: Dict[str, list[str]] = {}
        for k, v in id_to_label.items():
            label_to_ids.setdefault(v, []).append(k)
        ids = label_to_ids.get(node_label or "", [])
        if len(ids) == 1:
            resolved_id = ids[0]

    resolved_path = id_to_path.get(resolved_id) if resolved_id else None
    matched = resolved_path is not None

    return {
        "input_item": item,
        # "raw": raw,          # uncomment if you need debugging
        # "parsed": parsed,    # uncomment if you need debugging
        "matched_node_id": node_id,
        "matched_node_label": node_label,
        "matched_node_path": resolved_path,
        "matched": matched,
        "rationale": rationale,
    }


#########
# Example
#########
match_item_to_tree(
    items[222],
    tree_markdown_with_ids=tree_md,
    id_to_label=id_to_label,
    id_to_path=id_to_path,
    grammar=GRAMMAR_RESPONSE,
    n_predict=512,
)
