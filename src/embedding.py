"""Embedding utilities including SapBERT wrappers and HNSW helpers."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


import pandas as pd

from .taxonomy import taxonomy_node_texts
from .utils import clean_text


def l2_normalize(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, eps)


def _topological_order(G) -> List[str]:
    indegree = {node: G.in_degree(node) for node in G.nodes}
    stack = [node for node, deg in indegree.items() if deg == 0]
    order: List[str] = []
    while stack:
        node = stack.pop()
        order.append(node)
        for succ in G.successors(node):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                stack.append(succ)
    if len(order) != len(indegree):
        raise ValueError("Taxonomy graph contains a cycle; expected a DAG.")
    return order


class Embedder:
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 128,
        fp16: bool = True,
        mean_pool: bool = False,  # False = [CLS]
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
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        n_texts = len(texts)
        if n_texts == 0:
            hidden_size = getattr(self.model.config, "hidden_size", 768)
            return np.zeros((0, hidden_size), dtype=np.float32)

        hidden_size = getattr(self.model.config, "hidden_size", 768)
        out = np.empty((n_texts, hidden_size), dtype=np.float32)
        bs = self.batch_size
        total_tokens = 0
        offset = 0

        for i in range(0, n_texts, bs):
            batch = texts[i : i + bs]
            toks = self.tok.batch_encode_plus(
                batch,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            total_tokens += int(toks["attention_mask"].sum().item())
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                last_hidden = self.model(**toks)[0]
                if self.mean_pool:
                    attn = (toks["attention_mask"].unsqueeze(-1)).float()
                    summed = (last_hidden * attn).sum(dim=1)
                    denom = attn.sum(dim=1).clamp(min=1e-6)
                    rep = summed / denom
                else:
                    rep = last_hidden[:, 0, :]
            batch_size = rep.shape[0]
            out[offset : offset + batch_size] = rep.float().cpu().numpy()
            offset += batch_size

        return l2_normalize(out)


def collect_item_texts(
    item: Mapping[str, Optional[str]],
    *,
    fields: Sequence[str] = ("label", "name", "description"),
    clean: bool = True,
    max_length: int = 256,
) -> List[str]:
    """Extract candidate text fields from an item for embedding."""

    texts: List[str] = []
    for key in fields:
        raw = item.get(key)
        if clean:
            text = clean_text(raw)
            if text and text != "(empty)":
                texts.append(text[:max_length])
        else:
            if isinstance(raw, str):
                text = raw.strip()
            elif raw is None:
                text = ""
            else:
                text = str(raw).strip()
            if text:
                texts.append(text[:max_length])
    return texts


def encode_item_texts(
    item: Mapping[str, Optional[str]],
    embedder: Embedder,
    *,
    fields: Sequence[str] = ("label", "name", "description"),
    clean: bool = True,
    max_length: int = 256,
    texts: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Encode selected text fields from ``item`` using ``embedder``."""

    if texts is None:
        texts = collect_item_texts(
            item, fields=fields, clean=clean, max_length=max_length
        )
    else:
        texts = list(texts)

    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    return embedder.encode(texts)


def cosine_match_maxpool(
    node_vecs: np.ndarray, item_embs: np.ndarray
) -> Tuple[int, float]:
    sims = node_vecs @ item_embs.T
    best = sims.max(axis=1)
    j = int(np.argmax(best))
    return j, float(best[j])


def build_hnsw_index(
    embs_unit: np.ndarray,
    *,
    space: str = "cosine",
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    num_threads: int = 0,
):
    if embs_unit.dtype != np.float32:
        embs_unit = embs_unit.astype(np.float32, copy=False)
    N, D = embs_unit.shape
    import hnswlib

    index = hnswlib.Index(space=space, dim=D)
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    index.add_items(embs_unit, np.arange(N), num_threads=num_threads)
    index.set_ef(ef_search)
    return index


def _extract_summary_map(
    summaries: Optional[object],
) -> Dict[str, str]:
    if summaries is None:
        return {}

    summary_map: Dict[str, str] = {}

    if isinstance(summaries, Mapping):
        items: Iterable[Tuple[object, object]] = summaries.items()
    elif isinstance(summaries, pd.Series):
        items = summaries.items()
    elif isinstance(summaries, pd.DataFrame):
        items = summaries[["name", "definition_summary"]].itertuples(
            index=False, name=None
        )
    else:
        raise TypeError("summaries must be a mapping, pandas Series/DataFrame, or None")

    for key, value in items:
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.strip()
        summary = value.strip()
        if name and summary and name not in summary_map:
            summary_map[name] = summary

    return summary_map


def build_taxonomy_embeddings_composed(
    G,
    embedder: Embedder,
    gamma: float = 0.3,
    *,
    summaries: Optional[object] = None,
    summary_weight: float = 1.0,
    taxonomy_text_transform: Optional[Callable[[str], str]] = None,
) -> Tuple[List[str], np.ndarray]:
    names = taxonomy_node_texts(G)
    label2idx = {n: i for i, n in enumerate(names)}

    topo = _topological_order(G)
    parent_idx = np.full(len(names), -1, dtype=np.int32)
    for node in topo:
        preds = list(G.predecessors(node))
        if preds:
            parent_idx[label2idx[node]] = label2idx[preds[0]]

    if taxonomy_text_transform is not None:
        texts = [taxonomy_text_transform(n) for n in names]
    else:
        texts = names

    name_vecs = embedder.encode(texts)

    summary_map = _extract_summary_map(summaries)
    summary_vecs = np.zeros_like(name_vecs)
    if summary_map:
        texts: List[str] = []
        idxs: List[int] = []
        for i, n in enumerate(names):
            s = summary_map.get(n)
            if s:
                texts.append(s)
                idxs.append(i)
        if texts:
            encoded = embedder.encode(texts)
            for j, idx in enumerate(idxs):
                summary_vecs[idx] = encoded[j]

    composed_base = l2_normalize(name_vecs + summary_weight * summary_vecs)

    D = composed_base.shape[1] if composed_base.size else 0
    out = np.zeros((len(names), D), dtype=np.float32)
    for node in topo:
        idx = label2idx[node]
        pidx = parent_idx[idx]
        if pidx >= 0:
            out[idx] = composed_base[idx] + gamma * out[pidx]
        else:
            out[idx] = composed_base[idx]

    out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    return names, out
