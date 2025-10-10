from __future__ import annotations

import pandas as pd

from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)

from src.evaluate import run_label_benchmark

variables = pd.read_csv("data/Variables.csv", low_memory=False)
keywords = pd.read_csv("data/Keywords_summarized.csv")

G = build_taxonomy_graph(
    keywords, name_col="name", parent_col="parent", order_col="order"
)
NAME_TO_ID, NAME_TO_PATH = build_name_maps_from_graph(G)

EMBEDDER = Embedder(
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    batch_size=128,
    fp16=True,
    max_length=256,
)

TAX_NAMES, TAX_EMBS = build_taxonomy_embeddings_composed(G, EMBEDDER, gamma=0.6)
HNSW_INDEX = build_hnsw_index(TAX_EMBS, ef_search=128)

GLOSS_MAP = build_gloss_map(keywords)


df, metrics = run_label_benchmark(
    variables,
    keywords,
    G=G,
    embedder=EMBEDDER,
    tax_names=TAX_NAMES,
    tax_embs_unit=TAX_EMBS,
    hnsw_index=HNSW_INDEX,
    name_to_id=NAME_TO_ID,
    name_to_path=NAME_TO_PATH,
    gloss_map=GLOSS_MAP,
    dedupe_on=["name"],
    n=1000,
    n_predict=512,
    seed=37,
)
print(df[["label", "name", "description", "gold_labels", "resolved_label", "correct"]])
print(metrics)
