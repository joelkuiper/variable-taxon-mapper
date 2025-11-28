# Variable Taxon Mapper

Variable Taxon Mapper maps free text variable metadata to a curated biomedical taxonomy.

It solves a common harmonization problem: datasets often contain thousands of variables with inconsistent names, vague descriptions, and no standard annotation. This tool links each variable to a taxonomy term, making datasets comparable and interoperable.

The system works in two phases:

1. Use embeddings and the taxonomy graph to prune the taxonomy to a small, relevant sub tree.
2. Ask an LLM to pick the best term within that tree.

Together these produce mostly accurate mappings while keeping the LLM workload small.

---

## High level pipeline

Inputs:

- A variables table (CSV, Parquet, or Feather)
- A taxonomy table with one row per taxonomy node

For each variable:

1. Embeds taxonomy nodes and builds an approximate nearest neighbor index.
2. Finds anchors in the taxonomy using semantic and lexical similarity.
3. Expands those anchors to a local neighborhood in the taxonomy graph.
4. Trims that region down to a small, well structured tree.
5. Sends the pruned tree and variable metadata to an LLM for final selection.

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

A short summary of how it works:

* Each taxonomy node becomes a vector that encodes meaning and some hierarchical structure.
* The variable text is also embedded.
* The system retrieves a small set of nearest taxonomy nodes (anchors).
* Those anchors define a region of interest in the taxonomy graph.
* That region is expanded, cleaned, and trimmed to a compact candidate sub tree.
* The sub tree and variable metadata are shown to an LLM.
* The LLM selects the best fitting term.
* The term becomes the final taxonomy assignment.

Default components:

* Embeddings: SapBERT or similar biomedical models
* ANN search: HNSW
* LLM: Qwen3 4B via llama.cpp
* Alternative LLM backends: any OpenAI-compatible endpoint

Outputs and summaries live in `doc/`, for example `doc/results`.

---

## Why it is built this way

This architecture comes from practical constraints when mapping real biomedical variables.

### 1. Embeddings alone do not solve the task

Variables are messy:

* They use abbreviations, shorthand, and lab-specific conventions.
* Many contain measurement units or custom formatting.
* Some taxonomy labels are very abstract or appear in unexpected branches.

Cosine similarity alone often retrieves superficially similar but incorrect labels.
Even top-10 or top-20 neighbors may not contain the right term if the taxonomy is large or deeply hierarchical.

### 2. LLMs cannot handle large taxonomies

A typical taxonomy might contain thousands of terms.
LLMs:

* cannot read full taxonomies in a single prompt,
* degrade when given long lists of similar items, and
* become slow or unstable with large contexts.

The LLM must be shown a tiny, relevant subset.

### 3. Combining both methods solves both problems

The system separates:

* **Where to look** → handled by embeddings + graph pruning
* **What fits best** → handled by the LLM

This gives:

* Fast, deterministic candidate retrieval
* A small and meaningful context for the LLM
* A consistent JSON output schema
* A fully configurable pipeline in `config.example.toml`

A complete configuration reference is available at:
[`config.example.toml`](./config.example.toml)

---

## Pipeline stages

Below is a deeper, technical walk-through of the internal stages.

---

### 1. Embedding the taxonomy

Goal: represent each taxonomy node as a vector that reflects both its meaning and its position in the hierarchy.

Steps:

1. Read taxonomy from `[data].keywords_csv`.
2. Map column names using `[taxonomy_fields]` (name, parent, parents, label, definition).
3. Build a directed acyclic graph using `networkx`.

   * One row → one node
   * Parent relationships define the hierarchy
   * Multi-parent taxonomies are supported via a delimited `parents` column
4. Compute base embeddings with the model from `[embedder]`.

   * Usually embedding the node’s name and optionally its definition
5. Inject hierarchy:

   * parent → child mixing via `taxonomy_embeddings.gamma`
   * optional child → parent mixing via `child_aggregation_weight`
6. Normalize vectors and build an HNSW index (`[hnsw]`).

Why this stage exists:

* Taxonomy labels are often short or ambiguous.
* Definitions clarify meaning.
* Hierarchy mixing ensures that local embeddings reflect broader context.

---

### 2. Getting anchors

Goal: identify a small set of promising taxonomy nodes for each variable.

Steps:

1. Build variable text from `[fields].embedding_columns`.
2. Embed it using the same encoder.
3. Semantic anchors:

   * Query the HNSW index for the top `anchor_top_k` neighbors
4. Lexical anchors:

   * Fuzzy matches between variable text and taxonomy labels
   * Up to `lexical_anchor_limit` matches
5. Merge and dedupe.

Why this stage exists:

* Semantic anchors capture meaning.
* Lexical anchors catch obvious direct matches (e.g. “BMI” → “Body mass index”).
* Combining both improves recall before pruning.

---

### 3. Expanding anchors

Goal: build a local candidate region around the anchors.

Steps:

* Add ancestors up to the root (unless the pruning mode limits this).
* Add descendants up to `max_descendant_depth`.
* Optionally perform community detection:

  * controlled by `community_clique_size` and `max_community_size`
* Some modes compute PageRank scores inside this region.

Why this stage exists:

* A variable often belongs *near* a semantic anchor, but not exactly at it.
* Expanding up/down the graph increases recall.
* Community detection helps in dense areas like symptoms or phenotypes.

Key config: `[pruning].pruning_mode`, `max_descendant_depth`, community settings.

---

### 4. Trimming the tree back

Goal: shrink the expanded region to a small, well-organized sub tree to show the LLM.

Steps:

1. Score or filter nodes depending on pruning mode:

   * similarity-based
   * radius-based
   * Steiner-like
   * community-pagerank
2. Apply `node_budget` as a hard cap on the maximum number of nodes.
3. Sort nodes for LLM display:

   * hierarchical sorting: `tree_sort_mode`
   * top suggestions: `suggestion_sort_mode`, `suggestion_list_limit`

Why this stage exists:

* LLM prompts must stay small.
* A compact, well ordered tree improves comprehension.
* The budgeted sub tree usually contains the correct term in practice.

---

### 5. Asking the LLM for the final label

Goal: pick the single best taxonomy node from the pruned sub tree.

Steps:

1. Render system and user prompts using:

   * `system_template_path`
   * `user_template_path`

2. Call an OpenAI-compatible endpoint (`[llm]`).

3. Receive a constrained JSON object, e.g.:

   ```json
   {"label": "Body height"}
   ```

4. Postprocess the result:

   * remap aliases or misspellings
   * optionally snap broad labels to child nodes (`snap_to_child`)

Why this stage exists:

* LLMs are good at resolving borderline cases:

  * similar siblings
  * multiword synonyms
  * vague variable names
* The pruned sub tree gives the LLM exactly the context it needs.

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

Using OpenAI instead:

```toml
[llm]
endpoint = "https://api.openai.com/v1"
model = "gpt-4"
api_key = "sk-..."
```

---

## Core commands

Evaluate mappings using gold labels:

```bash
vtm evaluate config.example.toml
```

Predict without evaluation:

```bash
vtm predict config.example.toml --output data/predictions.parquet
```

Check pruning coverage:

```bash
vtm prune-check config.example.toml --limit 10000
```

Optimize pruning parameters:

```bash
vtm optimize-pruning config.example.toml --trials 80
```

---

## Extra tools

### Error review CLI

Interactive review:

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
pred_df = mapper.predict()
```

### HPC usage

For Slurm jobs and GPU cluster instructions see:
[`doc/nibbler_cluster.md`](doc/nibbler_cluster.md)

---

## Full configuration reference

The full configuration is in:
[`config.example.toml`](./config.example.toml)

It is self-documenting: every option includes comments describing what it controls and why it exists.
