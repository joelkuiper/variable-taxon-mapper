#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import sys
import time
import json
import math
from textwrap import dedent
from typing import Any, Dict, Iterable, Optional, Tuple, Union, Set, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import networkx as nx
import requests
import random
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


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
# Tree building with stable IDs (networkx)
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


def build_name_maps_from_graph(G: nx.DiGraph) -> tuple[dict, dict]:
    """Return name -> id (hex8) and name -> full path string."""
    name_to_id: dict[str, str] = {}
    name_to_path: dict[str, str] = {}
    for n in G.nodes:
        path = _path_to_root(G, n)
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
    sort_key = _sort_key_factory(order_map)

    lines: list[str] = []

    def _walk(node: str, depth: int):
        lines.append("  " * depth + f"- {node}")
        for c in sorted(G.successors(node), key=sort_key):
            _walk(c, depth + 1)

    for r in _roots_in_order(G, sort_key):
        _walk(r, 0)

    return "\n".join(lines)


# (Optional external consumers)
def build_id_maps_from_graph(G: nx.DiGraph) -> tuple[dict, dict]:
    id_to_label: dict[str, str] = {}
    id_to_path: dict[str, str] = {}
    for n in G.nodes:
        path = _path_to_root(G, n)
        nid = _path_id_hex8(path)
        id_to_label[nid] = n
        id_to_path[nid] = " / ".join(path)
    return id_to_label, id_to_path


# =============================================================================
# SapBERT embedder + NN helpers
# =============================================================================


def _l2_normalize(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, eps)


class SapBERTEmbedder:
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: Optional[str] = None,
        max_length: int = 32,
        batch_size: int = 128,
        fp16: bool = True,
        mean_pool: bool = False,  # ClS
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.mean_pool = mean_pool

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        bs = self.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            toks = self.tok.batch_encode_plus(
                batch,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            last_hidden = self.model(**toks)[0]  # (B, T, H)
            if self.mean_pool:
                # mean over non-padding tokens
                attn = (toks["attention_mask"].unsqueeze(-1)).float()  # (B,T,1)
                summed = (last_hidden * attn).sum(dim=1)
                denom = attn.sum(dim=1).clamp(min=1e-6)
                rep = summed / denom
            else:
                rep = last_hidden[:, 0, :]  # CLS
            out.append(rep.float().cpu().numpy())
        embs = (
            np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)
        )
        return _l2_normalize(embs).astype(np.float32)


def cosine_topk(
    matrix_unit: np.ndarray, query_unit: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    sims = matrix_unit @ query_unit  # (N,)
    idxs = np.argpartition(-sims, kth=min(k, sims.size - 1))[:k]
    order = np.argsort(-sims[idxs])
    idxs = idxs[order]
    return sims[idxs], idxs


# =============================================================================
# Taxonomy embedding & pruning
# =============================================================================


def taxonomy_node_texts(G: nx.DiGraph) -> List[str]:
    nodes = list(G.nodes)
    nodes_sorted = sorted(nodes, key=lambda s: s.lower())
    return nodes_sorted


def build_taxonomy_embeddings(
    G: nx.DiGraph, embedder: SapBERTEmbedder
) -> Tuple[List[str], np.ndarray]:
    names = taxonomy_node_texts(G)
    embs = embedder.encode(names)
    return names, embs


def _ancestors_to_root(G: nx.DiGraph, node: str) -> List[str]:
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


def _collect_descendants(G: nx.DiGraph, node: str, max_depth: int) -> Set[str]:
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


def _maybe_keywords_str(row_like: Dict[str, Optional[str]]) -> str:
    kv = row_like.get("keywords")
    if isinstance(kv, str) and kv.strip():
        # keep it short; commas already split
        return kv.strip()
    return ""


def pruned_tree_markdown_for_item(
    item: Dict[str, Optional[str]],
    *,
    G: nx.DiGraph,
    df: pd.DataFrame,
    embedder: SapBERTEmbedder,
    tax_names: List[str],
    tax_embs_unit: np.ndarray,
    name_col: str = "name",
    order_col: str = "order",
    top_k_nodes: int = 128,  # a bit larger to avoid missing obvious ones
    desc_max_depth: int = 2,
    max_total_nodes: int = 1200,
) -> Tuple[str, List[str]]:
    # 1) Query text (with keywords)
    parts = [
        _clean_text(item.get("label")),
        _clean_text(item.get("name")),
        _clean_text(item.get("description")),
    ]
    kw = item.get("keywords")
    if isinstance(kw, str) and kw.strip():
        parts.append(kw.strip())
    query = " | ".join(p for p in parts if p and p != "(empty)")
    q_emb = embedder.encode([query])[0]

    # 2) Top-k nearest nodes (anchors)
    sims, idxs = cosine_topk(tax_embs_unit, q_emb, k=min(top_k_nodes, len(tax_names)))
    anchors = [tax_names[i] for i in idxs]

    # 3) Expand: ancestors + descendants
    allowed: Set[str] = set()
    for a in anchors:
        for n in _ancestors_to_root(G, a):
            allowed.add(n)
        for n in _collect_descendants(G, a, max_depth=desc_max_depth):
            allowed.add(n)

    # Size cap
    if len(allowed) > max_total_nodes:
        core = set(anchors[: max_total_nodes // 4])
        allowed = core | {n for a in core for n in _ancestors_to_root(G, a)}

    # 4) ORDER the allowed labels: first by similarity rank, then fill the rest
    #    (LLMs are position-biased; put best candidates first)
    allowed_ranked = []
    seen = set()
    for a in anchors:
        if a in allowed and a not in seen:
            allowed_ranked.append(a)
            seen.add(a)
    for n in sorted(allowed, key=lambda s: s.lower()):
        if n not in seen:
            allowed_ranked.append(n)
            seen.add(n)

    # 5) Build a small ranked "Candidates" section + full pruned tree
    order_map = df.groupby(name_col)[order_col].min().to_dict()
    sort_key = _sort_key_factory(order_map)

    lines_tree: List[str] = []

    def _walk(node: str, depth: int):
        if node not in allowed:
            return
        lines_tree.append("  " * depth + f"- {node}")
        for c in sorted(G.successors(node), key=sort_key):
            if c in allowed:
                _walk(c, depth + 1)

    for r in _roots_in_order(G, sort_key):
        if r in allowed:
            _walk(r, 0)

    tree_md = "\n".join(lines_tree)
    if not tree_md.strip():
        tree_md = tree_to_markdown(G, df, name_col=name_col, order_col=order_col)
        allowed_ranked = sorted(list(G.nodes), key=lambda s: s.lower())

    top_show = min(40, len(allowed_ranked))  # keep prompt light
    md = []
    md.append("### Candidates (ranked)\n")
    for lbl in allowed_ranked[:top_show]:
        md.append(f"- {lbl}")
    md.append("\n### Taxonomy (pruned view)\n")
    md.append(tree_md)
    return "\n".join(md), allowed_ranked


# =============================================================================
# Prompt builder
# =============================================================================


def _clean_text(v) -> str:
    if v is None:
        return "(empty)"
    if isinstance(v, str):
        s = v.strip()
        return s if s else "(empty)"
    try:
        if isinstance(v, float) and math.isnan(v):
            return "(empty)"
    except Exception:
        pass
    s = str(v).strip()
    return s if s else "(empty)"


def make_tree_match_prompt(
    tree_markdown_labels_only: str,
    item: Dict[str, Optional[str]],
) -> str:
    lab = _clean_text(item.get("label"))
    nam = _clean_text(item.get("name"))
    desc = _clean_text(item.get("description"))
    tree_md = (tree_markdown_labels_only or "").strip()

    template = """\
        <|im_start|>system
        You are a biomedical taxonomy matcher.

        ## TASK
        • From the TREE, choose **exactly one** label that best matches the ITEM.
        • If uncertain, choose the closest label — do **not** abstain.

        The TREE is a nested (indented) Markdown list where each line is:
        - <label>  (exact text; choose from these)

        ## OUTPUT (single-line JSON)
        {{"node_label":"...","rationale":"..."}}

        ## TREE
        {TREE}.<|im_end|>

        <|im_start|>user
        ## ITEM:
        - label: {LAB}
        - name: {NAM}
        - description: {DESC}<|im_end|>
        <|im_start|>assistant
    """
    return dedent(template).format(TREE=tree_md, LAB=lab, NAM=nam, DESC=desc).strip()


# =============================================================================
# Per-item constrained grammar
# =============================================================================


def _json_escape(s: str) -> str:
    return json.dumps(s)  # JSON string literal with quotes/escapes


def build_label_constrained_grammar(allowed_labels: List[str]) -> str:
    """
    GBNF grammar that forces node_label to be one of allowed_labels.
    """
    if not allowed_labels:
        # shouldn't happen (we already fallback to full tree), but guard anyway
        allowed_labels = ["(no-candidates)"]

    alts = " | ".join(json.dumps(lbl) for lbl in allowed_labels)
    return rf"""
root      ::= obj
quote     ::= "\""

string ::=
  quote (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{{4}})
  )+ quote

label ::= {alts}

obj ::= ("{{" quote "node_label" quote ": " label ","
             quote "rationale" quote ": " string "}}")
""".strip()


# Legacy
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

# =============================================================================
# HTTP to llama.cpp
# =============================================================================


class LlamaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


_HTTP_SESSION = requests.Session()  # shared fallback


def llama_completion(
    prompt: Union[str, Iterable[Union[str, int]]],
    endpoint: str,
    *,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> str:
    """
    POST /completions, expects JSON with 'content'.
    When using a grammar, DO NOT pass newline stops (can truncate valid JSON).
    """
    s = session or _HTTP_SESSION
    payload: Dict[str, Any] = {"prompt": prompt, "stream": False}
    payload.update(kwargs)

    resp = s.post(endpoint, json=payload, timeout=(10, timeout))
    if not resp.ok:
        raise LlamaHTTPError(resp.status_code, resp.text)
    return resp.json().get("content", "")


# =============================================================================
# Set-up and testing (globals initialized once)
# =============================================================================

variables = pd.read_csv("data/Variables.csv", low_memory=False)
keywords = pd.read_csv("data/Keywords.csv")

# Build taxonomy graph
G = build_taxonomy_graph(
    keywords, name_col="name", parent_col="parent", order_col="order"
)

# Embedder & taxonomy embeddings
EMBEDDER = SapBERTEmbedder(
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    batch_size=128,
    fp16=True,
)
TAX_NAMES, TAX_EMBS = build_taxonomy_embeddings(G, EMBEDDER)  # (N,), (N,D) L2-normed

# Name maps (label -> hex id/path)
NAME_TO_ID, NAME_TO_PATH = build_name_maps_from_graph(G)

# (Optional external compatibility)
id_to_label, id_to_path = build_id_maps_from_graph(G)

items = unique_dataset_label_name_desc(variables)


# =============================================================================
# Matching: label-only + constrained grammar + robust fallback
# =============================================================================


def _nn_among_labels(query_text: str, candidate_labels: List[str]) -> Optional[str]:
    """k=1 SapBERT among candidate_labels (returns the chosen label or None)."""
    if not candidate_labels:
        return None
    q_emb = EMBEDDER.encode([query_text])[0]
    # map candidate labels to indices in TAX_NAMES
    try:
        idxs = [TAX_NAMES.index(lbl) for lbl in candidate_labels if lbl in TAX_NAMES]
    except Exception:
        idxs = []
    if not idxs:
        return None
    sub = TAX_EMBS[idxs]
    _, top = cosine_topk(sub, q_emb, k=1)
    return candidate_labels[int(top[0])] if top.size else None


def match_item_to_tree(
    item: Dict[str, Optional[str]],
    *,
    tree_markdown: str,
    allowed_labels: List[str],
    endpoint: str = "http://127.0.0.1:8080/completions",
    temperature: float = 0.0,
    n_predict: int = 128,
    slot_id: int = 0,
    cache_prompt: bool = True,
    n_keep: int = -1,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    grammar = GRAMMAR_RESPONSE  # build_label_constrained_grammar(allowed_labels)
    prompt = make_tree_match_prompt(tree_markdown, item)

    raw = llama_completion(
        prompt,
        endpoint,
        session=session,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        min_p=0,
        n_predict=max(n_predict, 64),
        grammar=grammar,  # constrained; no newline stop
        cache_prompt=cache_prompt,
        n_keep=n_keep,
        slot_id=slot_id,
    )

    def _pack(
        pred_label_raw,
        resolved_label,
        resolved_id,
        resolved_path,
        rationale,
        *,
        keep_raw=False,
        raw_text=None,
    ):
        out = {
            "input_item": item,
            "pred_label_raw": pred_label_raw,
            "resolved_label": resolved_label,
            "resolved_id": resolved_id,
            "resolved_path": resolved_path,
            "rationale": rationale,
            "matched": bool(resolved_label),
            "no_match": not bool(resolved_label),
        }
        if keep_raw:
            out["raw"] = raw_text
        return out

    try:
        p = json.loads(raw)
        node_label_raw = p.get("node_label")
        rationale = p.get("rationale")
    except Exception:
        # Parse failed: pick k=1 among the same allowed labels
        parts = [
            _clean_text(item.get("label")),
            _clean_text(item.get("name")),
            _clean_text(item.get("description")),
        ]
        q = " | ".join(t for t in parts if t and t != "(empty)")
        # constrained NN among allowed_labels
        mask = [TAX_NAMES.index(lbl) for lbl in allowed_labels if lbl in TAX_NAMES]
        if mask:
            q_emb = EMBEDDER.encode([q])[0]
            sub = TAX_EMBS[mask]
            _, top = cosine_topk(sub, q_emb, k=1)
            chosen = allowed_labels[int(top[0])] if top.size else None
        else:
            chosen = None
        if chosen:
            return _pack(
                pred_label_raw=None,
                resolved_label=chosen,
                resolved_id=NAME_TO_ID.get(chosen),
                resolved_path=NAME_TO_PATH.get(chosen),
                rationale="Fallback: embedding k=1 among allowed labels (parse fail).",
                keep_raw=True,
                raw_text=raw,
            )
        return _pack(
            None,
            None,
            None,
            None,
            "Parse failed; no NN fallback candidate.",
            keep_raw=True,
            raw_text=raw,
        )

    # Normal path: grammar forces `node_label_raw` to be in allowed_labels
    resolved_label = node_label_raw
    resolved_id = NAME_TO_ID.get(resolved_label)
    resolved_path = NAME_TO_PATH.get(resolved_label)
    return _pack(node_label_raw, resolved_label, resolved_id, resolved_path, rationale)


# =============================================================================
# Helpers for variables['keywords']
# =============================================================================


def _clean_str_or_none(v) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def _split_keywords_comma(s: Optional[str]) -> List[str]:
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


# =============================================================================
# Parallel benchmark (HTTP to llama.cpp; thread-pooled)
# =============================================================================

_tls = threading.local()  # per-thread HTTP session


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


def _ancestors_inclusive(G: nx.DiGraph, node: str) -> List[str]:
    """node plus all ancestors up to the root (order root..node)."""
    out = []
    cur = node
    seen = set()
    while True:
        if cur in seen:  # safety in case of dirty graphs
            break
        seen.add(cur)
        out.append(cur)
        preds = list(G.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
    return out[::-1]  # root..node


def is_correct_prediction(
    pred_label: Optional[str],
    gold_labels: List[str],
    *,
    G: nx.DiGraph,
    mode: str = "path_any",  # "exact" | "ancestor" | "path_any"
) -> bool:
    """
    - exact: pred_label in gold_labels
    - ancestor: any ancestor-of-pred equals a gold label (inclusive of pred)
    - path_any (recommended): exact OR ancestor
    """
    if not isinstance(pred_label, str):
        return False
    if mode == "exact":
        return pred_label in gold_labels
    path_nodes = (
        set(_ancestors_inclusive(G, pred_label))
        if pred_label in G.nodes
        else {pred_label}
    )
    if mode == "ancestor":
        return any(g in path_nodes for g in gold_labels)
    # default "path_any"
    return (pred_label in gold_labels) or any(g in path_nodes for g in gold_labels)


def run_label_benchmark(
    variables: pd.DataFrame,
    keywords: pd.DataFrame,
    *,
    endpoint: str = "http://127.0.0.1:8080/completions",
    n: int = 50,
    seed: int = 0,
    n_predict: int = 256,
    temperature: float = 0.0,
    dedupe_on: Optional[List[str]] = None,
    top_k_nodes: int = 64,
    desc_max_depth: int = 2,
    max_total_nodes: int = 800,
    # ---- parallel controls ----
    max_workers: int = 4,
    num_slots: int = 4,
    pool_maxsize: int = 64,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parallel, comma-aware benchmark on UNIQUE rows.
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

    known_labels: Set[str] = set(TAX_NAMES)

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

    # 2) Sample eligible rows deterministically
    eligible_idxs = list(work_df.index[eligible_mask])
    rnd = random.Random(seed)
    rnd.shuffle(eligible_idxs)
    idxs = eligible_idxs[: min(n, len(eligible_idxs))]

    # 3) Evaluate in parallel
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

        # Rotate slot_id across available llama.cpp slots
        slot_id = j % max(1, num_slots)

        # Per-item pruned tree + allowed label set
        tree_markdown, allowed_labels = pruned_tree_markdown_for_item(
            item,
            G=G,
            df=keywords,
            embedder=EMBEDDER,
            tax_names=TAX_NAMES,
            tax_embs_unit=TAX_EMBS,
            name_col="name",
            order_col="order",
            top_k_nodes=top_k_nodes,
            desc_max_depth=desc_max_depth,
            max_total_nodes=max_total_nodes,
        )

        try:
            pred = match_item_to_tree(
                item,
                tree_markdown=tree_markdown,
                allowed_labels=allowed_labels,
                endpoint=endpoint,
                n_predict=n_predict,
                temperature=temperature,
                slot_id=slot_id,
                cache_prompt=True,
                n_keep=-1,
                session=_get_thread_session(pool_maxsize=pool_maxsize),
            )
            resolved_label = pred.get("resolved_label")
            correct = is_correct_prediction(
                resolved_label, gold_labels, G=G, mode="path_any"
            )
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
                out["raw"] = pred["raw"]  # capture raw text if parse failed earlier
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

    rows: list[Dict[str, Any]] = []
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

    # Sort rows by our sampled order j to keep deterministic outputs
    rows.sort(key=lambda r: r["_j"])
    for r in rows:
        r.pop("_j", None)
        r.pop("_idx", None)
        r.pop("_slot", None)

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
        "max_workers": int(max_workers),
        "num_slots": int(num_slots),
        "pool_maxsize": int(pool_maxsize),
        "n_predict": int(n_predict),
        "temperature": float(temperature),
        "endpoint": endpoint,
        "n_errors": int(df["_error"].notna().sum()) if "_error" in df.columns else 0,
    }

    return df, metrics


#########
# Example
#########

df, metrics = run_label_benchmark(
    variables,
    keywords,
    n=100,
    dedupe_on=["label"],
    seed=100,
)
print(metrics)
