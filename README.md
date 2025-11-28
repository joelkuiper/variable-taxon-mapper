# Variable Taxon Mapper

Variable Taxon Mapper maps free text variable metadata to a curated biomedical taxonomy.

It works in two main phases:

1. Use embeddings and the taxonomy graph to prune the taxonomy to a small, relevant sub tree.
2. Ask an LLM to pick the single best label from that sub tree.

This supports dataset harmonization and consistent variable annotation across studies.

---

## High level pipeline

Inputs:

- A variables table (CSV, Parquet, or Feather).
- A taxonomy table with one row per taxonomy node.

For each variable, the system:

1. Embeds the taxonomy and builds a nearest neighbor index.
2. Gets anchors by comparing the variable text to the taxonomy.
3. Expands anchors into a local neighborhood in the taxonomy graph.
4. Trims the tree back to a small, budgeted sub tree.
5. Asks the LLM for the final label and records it.

### ASCII flow

```text
Variables.csv + Taxonomy.csv
        │
        ▼
Build taxonomy graph
Embed taxonomy nodes
Build HNSW index
        │
For each variable:
  ├─ Build variable text
  ├─ Embed variable
  ├─ Get anchors
  │    ├─ ANN similarity search
  │    └─ Lexical similarity
  ├─ Expand anchors
  ├─ Trim candidates to a sub tree
  ├─ Render prompt with sub tree
  ├─ Call LLM for final label
  └─ Record prediction
````

---

## Conceptual overview

Short summary:

* Taxonomy nodes are turned into vectors that reflect both meaning and hierarchy.
* Variables are embedded the same way.
* Nearest neighbors plus lexical matches give a small set of anchor nodes.
* Anchors are expanded, then pruned down to a compact sub tree.
* The variable and pruned tree are shown to an LLM, which selects the best label.
* The result is the final mapping.

Defaults:

* Embeddings use SapBERT or similar biomedical models.
* Nearest neighbor search uses HNSW.
* The LLM is Qwen3 4B via llama.cpp.
* Any OpenAI compatible endpoint can be used instead.

Outputs and summaries live in `doc/`, e.g. `doc/results`.

---

## Why it is built this way

This design is driven by two observations:

### 1. Embeddings alone do not solve the task

Variable names and descriptions are often noisy, domain specific, or ambiguous. Pure cosine similarity to taxonomy labels fails in predictable ways:

* Many taxonomy labels are abstract, short, or broad.
* Variables often contain idiosyncratic wording, abbreviations, or measurement details.
* Even a top-10 retrieval list may miss the correct node entirely if it is located in a distant branch of the hierarchy.

We need embeddings for fast retrieval, but we cannot rely on them for the final decision.

### 2. LLMs cannot read large taxonomies

Even small biomedical taxonomies contain thousands of nodes. LLMs:

* Cannot handle prompts that show large trees.
* Lose accuracy when given long lists of similar items.
* Become slow or unstable with large context windows.

Therefore the model must be shown only the part of the taxonomy that is relevant for the current variable.

### 3. Combining both gives the best of both

So the system splits the problem:

* Use embedding search + hierarchical pruning to find **where to look**.
* Use the LLM to decide **what fits best** in that small region.

This gives:

* The speed and robustness of vector search.
* The discrimination and contextual understanding of the LLM.
* A reproducible, configurable pipeline controlled by `config.example.toml` (see that file for a complete reference).

A full configuration file lives at:
**[`config.example.toml`](./config.example.toml)**.

---

## Pipeline stages

### Embedding the taxonomy

Goal: represent each taxonomy node as a vector encoding meaning and hierarchy.

Steps:

1. Read the taxonomy file (`[data].keywords_csv`).
2. Map columns using `[taxonomy_fields]` (`name`, `parent`, `parents`, `label`, `definition`).
3. Build a directed acyclic graph using `networkx`.
4. Compute base embeddings with `[embedder]` (e.g. using name and definition text).
5. Inject hierarchy context (parent → child mixing, child → parent mixing).
6. Normalize all vectors and build the HNSW index (`[hnsw]`).

Key config: `[data]`, `[taxonomy_fields]`, `[embedder]`, `[taxonomy_embeddings]`, `[hnsw]`.

### Getting anchors

Goal: identify a small set of relevant taxonomy nodes for each variable.

Steps:

1. Build variable text from `[fields].embedding_columns`.
2. Embed variable text.
3. Retrieve nearest neighbors: `[pruning].anchor_top_k`.
4. Add lexical matches: `[pruning].lexical_anchor_limit`.
5. Combine and dedupe.

Key config: `[fields]`, `[pruning].anchor_top_k`, `[pruning].lexical_anchor_limit`.

### Expanding anchors

Goal: expand around anchors to form a candidate region.

Steps:

* Add ancestors up to the root (or mode-specific limit).
* Add descendants up to `max_descendant_depth`.
* Optionally detect communities using `community_clique_size` and `max_community_size`.
* Optionally compute PageRank scores.

Key config: `[pruning].pruning_mode`, `max_descendant_depth`, `community_clique_size`.

### Trimming the tree back

Goal: reduce the expanded region to a small, sorted sub tree for the LLM.

Steps:

* Score nodes based on the pruning mode (similarity, radius, Steiner, community-pagerank).
* Apply `node_budget` as a hard limit.
* Sort nodes for LLM presentation using `tree_sort_mode` and `suggestion_sort_mode`.

Key config: `[pruning].node_budget`, `tree_sort_mode`, `suggestion_list_limit`.

### Asking the LLM for the final label

Goal: pick one taxonomy node from the pruned sub tree.

Steps:

1. Render the system and user prompts using `[prompts]` templates.
2. Call an OpenAI compatible chat endpoint (`[llm]`).
3. Parse the constrained JSON output, e.g. `{ "label": "Body Height" }`.
4. Postprocess:

   * Remap aliases and spelling variations.
   * Optionally snap broad predictions to more specific children.

Key config: `[llm]`, `[prompts]`, `alias_similarity_threshold`, `snap_to_child`.

---

## Installation and running

### Install dependencies

```bash
uv sync
source .venv/bin/activate
```

### Start an LLM backend

Local llama.cpp:

```bash
llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

Or use OpenAI:

```toml
[llm]
endpoint = "https://api.openai.com/v1"
model = "gpt-4"
api_key = "sk-..."
```

### Core commands

Evaluate:

```bash
vtm evaluate config.example.toml
```

Predict:

```bash
vtm predict config.example.toml --output data/predictions.parquet
```

Pruning coverage:

```bash
vtm prune-check config.example.toml --limit 10000
```

Optimize pruning:

```bash
vtm optimize-pruning config.example.toml --trials 80
```

---

## Extra tools

### Error review CLI

```bash
python -m vtm.error_review_cli \
  --predictions path/to/predictions.csv \
  --keywords path/to/Keywords.csv \
  --output data/error_review_decisions.csv \
  --config config.toml
```

### Programmatic use

```python
from vtm import VariableTaxonMapper
mapper = VariableTaxonMapper(config_path="config.toml")
preds = mapper.predict()
```

### HPC usage

For running on an HPC cluster with Slurm and multiple GPUs, see: [doc/nibbler_cluster.md](doc/nibbler_cluster.md).
This document describes cluster specific launch scripts and recommended settings.
