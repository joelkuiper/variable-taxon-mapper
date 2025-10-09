"""Embedding utilities including SapBERT wrappers and HNSW helpers."""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from taxonomy import ancestors_to_root, taxonomy_node_texts


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
        mean_pool: bool = True,
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


def build_taxonomy_embeddings_composed(
    G,
    embedder: Embedder,
    gamma: float = 0.3,
) -> Tuple[List[str], np.ndarray]:
    names = taxonomy_node_texts(G)
    label2idx = {n: i for i, n in enumerate(names)}
    name_vecs = embedder.encode(names)

    D = name_vecs.shape[1] if name_vecs.size else 0
    out = np.zeros((len(names), D), dtype=np.float32)
    for n in names:
        idx = label2idx[n]
        v = name_vecs[idx].copy()
        anc = ancestors_to_root(G, n)[:-1]
        for k, a in enumerate(reversed(anc), start=1):
            v += (gamma**k) * name_vecs[label2idx[a]]
        out[idx] = v

    out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    return names, out


def maxpool_scores(
    item_embs: np.ndarray,
    *,
    index,
    pool_k: int,
) -> Dict[int, float]:
    """
    Query HNSW per item part; return dict of {tax_idx: max_similarity}.
    Assumes index space='cosine' (distance = 1 - cosine_sim).
    """
    scores: Dict[int, float] = {}
    if item_embs.size == 0:
        return scores
    for q in item_embs:
        labels, dists = index.knn_query(q[np.newaxis, :].astype(np.float32), k=pool_k)
        labels, dists = labels[0], dists[0]
        for idx, dist in zip(labels, dists):
            sim = 1.0 - float(dist)
            if sim > scores.get(int(idx), -1.0):
                scores[int(idx)] = sim
    return scores
