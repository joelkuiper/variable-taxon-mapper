"""Embedding utilities including SapBERT wrappers and HNSW helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import pandas as pd

from .taxonomy import ancestors_to_root, taxonomy_node_texts
from .utils import clean_text


def l2_normalize(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, eps)


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
            last_hidden = self.model(**toks)[0]
            if self.mean_pool:
                attn = (toks["attention_mask"].unsqueeze(-1)).float()
                summed = (last_hidden * attn).sum(dim=1)
                denom = attn.sum(dim=1).clamp(min=1e-6)
                rep = summed / denom
            else:
                rep = last_hidden[:, 0, :]
            out.append(rep.float().cpu().numpy())
        embs = (
            np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)
        )
        return l2_normalize(embs).astype(np.float32)


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
) -> np.ndarray:
    """Encode selected text fields from ``item`` using ``embedder``."""

    texts = collect_item_texts(
        item, fields=fields, clean=clean, max_length=max_length
    )
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
) -> Tuple[List[str], np.ndarray]:
    names = taxonomy_node_texts(G)
    label2idx = {n: i for i, n in enumerate(names)}

    name_vecs = embedder.encode(names)

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
    for n in names:
        idx = label2idx[n]
        v = composed_base[idx].copy()
        anc = ancestors_to_root(G, n)[:-1]
        for k, a in enumerate(reversed(anc), start=1):
            v += (gamma**k) * composed_base[label2idx[a]]
        out[idx] = v

    out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    return names, out
