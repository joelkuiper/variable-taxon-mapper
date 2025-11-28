# Variable Taxon Mapper

Variable Taxon Mapper maps free text **variable metadata** to a curated **biomedical taxonomy**.

It works in two main phases:

1. Use **embeddings** and the taxonomy graph to prune the taxonomy to a small, relevant sub tree.
2. Ask an **LLM** to pick the single best label from that sub tree.

This helps with dataset harmonization and consistent variable annotation across studies.

---

## High level pipeline

Inputs:

- A **variables table** (CSV, Parquet, or Feather).
- A **taxonomy table** with one row per taxonomy node.

For each variable, the system:

1. **Embeds the taxonomy** and builds a nearest neighbor index.
2. **Gets anchors** by comparing the variable text to the taxonomy.
3. **Expands anchors** into a local neighborhood in the taxonomy graph.
4. **Trims the tree back** to a small, budgeted sub tree.
5. **Asks the LLM for the final label** and records that label as the mapping.

### ASCII flow

```text
Variables.csv + Taxonomy.csv
        │
        ▼
Build taxonomy graph (tree or forest)
Embed all taxonomy terms
Build HNSW index over taxonomy embeddings
        │
For each variable:
  ├─ Build variable text from configured columns
  ├─ Embed variable text
  ├─ Get anchors
  │    ├─ ANN similarity search (semantic anchors)
  │    └─ Lexical similarity (lexical anchors)
  ├─ Expand anchors into a candidate subgraph
  ├─ Trim the tree back to a node budget
  ├─ Render subgraph + variable into an LLM prompt
  ├─ Call LLM to pick best label
  └─ Post process and record final taxonomy label
        │
        ▼
Predictions table (+ optional evaluation report in doc/results)
````

---

## Conceptual overview

Non technical summary:

* The taxonomy is converted into vectors that encode meaning plus some hierarchy context.
* Each variable description is converted into a vector with the same embedder.
* The system finds a small set of anchors in the taxonomy that are close to the variable.
* It expands around those anchors, then trims back to a compact candidate sub tree.
* It shows that sub tree, plus the variable context, to an LLM and asks for the best fitting label.
* The chosen label is the final taxonomy mapping for that variable.

By default:

* Embeddings use a biomedical model such as **SapBERT**.
* Semantic search uses an **HNSW** index.
* The LLM is **Qwen3 4B Instruct** served through **llama.cpp**.
* The LLM endpoint is OpenAI compatible and can be pointed at **OpenAI** or any other compatible service.

---

## Pipeline stages

### Embedding the taxonomy

**Goal:** represent each taxonomy node as a vector that reflects both its local meaning and its position in the hierarchy.

Steps:

1. Read the taxonomy table from `[data].keywords_csv`.
2. Map raw columns to logical fields using `[taxonomy_fields]`:

   * `name`, `parent`, optional `parents`, `label`, `definition`.
3. Build a directed acyclic graph using `networkx`:

   * One node per taxonomy term.
   * Each node has one primary parent by default.
   * Optional multi parent support via a delimited `parents` column.
4. Compute base embeddings with the model from `[embedder]`:

   * Base text is usually `name`.
   * Optionally mix in `definition` controlled by `taxonomy_embeddings.summary_weight`.
5. Inject hierarchy structure:

   * Add a fraction `gamma` of each parent embedding into each child.
   * Optionally add a fraction of each child into its parent using `child_aggregation_weight` and `child_aggregation_depth`.
6. Normalize and index:

   * L2 normalize all embeddings.
   * Build an **HNSW** index with settings from `[hnsw]` for fast nearest neighbor search.

Important config sections:

* `[data]` for paths.
* `[taxonomy_fields]` for column mapping.
* `[embedder]` for the embedding model.
* `[taxonomy_embeddings]` for hierarchy aware tweaks.
* `[hnsw]` for approximate nearest neighbor settings.

### Getting anchors

**Goal:** find a small set of clearly relevant taxonomy nodes for each variable.

For each variable row:

1. Build variable text by concatenating columns in `[fields].embedding_columns`.
2. Embed this text with the same embedder as the taxonomy.
3. **Semantic anchors**:

   * Query the HNSW index for top `anchor_top_k` nearest neighbors.
4. **Lexical anchors**:

   * Compare variable text to taxonomy `name` or `label` via lexical similarity.
   * Keep up to `lexical_anchor_limit` matching nodes.
5. Combine anchors and deduplicate.

Key config:

* `[fields].embedding_columns`.
* `[pruning].anchor_top_k`.
* `[pruning].lexical_anchor_limit`.
* `[pruning].anchor_overfetch_multiplier`.
* `[pruning].anchor_min_overfetch`.

### Expanding anchors

**Goal:** grow a local region around the anchors to form a candidate subgraph.

From the union of all anchors:

1. Add ancestors up to the root or until limited by the pruning strategy.
2. Add descendants up to `max_descendant_depth`.
3. Optionally group anchors into communities and expand within these communities:

   * Use parameters like `community_clique_size` and `max_community_size` to control community size.
4. Optionally compute PageRank scores for nodes inside each region.

Key config:

* `[pruning].pruning_mode`.
* `[pruning].max_descendant_depth`.
* `[pruning].community_clique_size`.
* `[pruning].max_community_size`.

### Trimming the tree back

**Goal:** reduce the expanded candidate set to a compact subgraph that fits within a node budget and is well sorted for the LLM prompt.

Steps:

1. Score candidate nodes based on `pruning_mode`:

   * Similarity based modes use embedding similarity.
   * Graph based modes use degrees, distances, or PageRank.
   * Steiner like modes favor nodes that connect multiple anchors.
2. Apply global limits:

   * Enforce `[pruning].node_budget` as an absolute cap on nodes in the pruned subgraph.
   * For PageRank based modes, use `pagerank_damping`, `pagerank_score_floor`, and optionally `pagerank_candidate_limit`.
3. Sort nodes for display:

   * `tree_sort_mode` controls order of siblings in the hierarchical view.
   * `suggestion_sort_mode` and `suggestion_list_limit` control an optional top suggestions list.

Supported `pruning_mode` values include for example:

* `anchor_hull` or `dominant_forest` for anchor centered forests.
* `similarity_threshold` for similarity cutoff based pruning.
* `radius` for fixed graph distance from anchors.
* `steiner_similarity` for Steiner like subgraphs.
* `community_pagerank` for community plus PageRank based pruning.

Key config:

* `[pruning].pruning_mode`.
* `[pruning].tree_sort_mode`.
* `[pruning].suggestion_sort_mode`.
* `[pruning].suggestion_list_limit`.
* `[pruning].node_budget`.
* `[pruning].pagerank_damping`.
* `[pruning].pagerank_score_floor`.
* Optional thresholds such as `similarity_threshold` or `pruning_radius`.
* Optional `surrogate_root_label` for a synthetic root label in the prompt tree.

### Asking the LLM for the final label

**Goal:** select a single taxonomy node from the pruned subgraph as the final mapping.

For each variable:

1. Render the **system** and **user** messages using templates from `[prompts]`:

   * `system_template_path` for instructions and JSON schema.
   * `user_template_path` for variable context plus the pruned taxonomy tree and optional suggestions list.
   * Include metadata configured in `[fields].metadata_columns`.
2. Call the LLM endpoint from `[llm]`:

   * Default: local **llama.cpp** server hosting **Qwen3 4B Instruct**.
   * Alternative: OpenAI or any OpenAI compatible endpoint, using `llm.endpoint`, `llm.model`, `llm.api_key`.
3. Constrain the output to a small JSON snippet, for example:

   * `{"label": "Body Height"}`.
4. Post process the answer:

   * Align the returned label to actual taxonomy nodes.
   * Use embedding and alias similarity thresholds:

     * `embedding_remap_threshold`.
     * `alias_similarity_threshold`.
   * Optional child snapping if `snap_to_child` is enabled:

     * Use `snap_margin`, `snap_similarity`, and `snap_descendant_depth` to refine broad parent choices.
5. Record the final label id, label name, and path.

Key config:

* `[llm]` for endpoint, model, key, and sampling parameters.
* `[prompts]` for template locations.
* `[fields].metadata_columns` for what goes into the prompt.
* `[llm]` post processing options:

  * `embedding_remap_threshold`.
  * `alias_similarity_threshold`.
  * `snap_to_child` and related parameters.

---

## Installation and running

### Install dependencies

Install [uv](https://docs.astral.sh/uv/get-started/installation/), then:

```bash
uv sync
source .venv/bin/activate
```

This creates a virtual environment and installs the `vtm` console script.

### Start an LLM backend

To use a local model with llama.cpp (for example Qwen3 4B GGUF):

```bash
llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

Or run it explicitly with `-m` pointing to a `.gguf` file. Adjust port, context length, and GPU flags as needed.

Alternatively, to use OpenAI:

* Set `llm.endpoint = "https://api.openai.com/v1"`.
* Set `llm.model` to the desired model.
* Provide `llm.api_key` or set `OPENAI_API_KEY`.

### Core commands

Evaluate with gold labels:

```bash
vtm evaluate config.example.toml
```

Predict without evaluation:

```bash
vtm predict config.example.toml \
  --output data/predictions.parquet \
  --output-format parquet
```

Check pruning coverage:

```bash
vtm prune-check config.example.toml \
  --limit 10000 \
  --output data/Keyword_coverage.csv
```

Optimize pruning parameters with Optuna:

```bash
vtm optimize-pruning config.example.toml --trials 80
```


## Extra tools

### Error review CLI

Interactive review of misclassifications:

```bash
python -m vtm.error_review_cli \
  --predictions path/to/predictions.csv \
  --keywords path/to/Keywords.csv \
  --output data/error_review_decisions.csv \
  --config config.toml
```

Key bindings:

* `A` acceptable mistake.
* `X` reject prediction.
* `U` unclear.
* `B` back.
* `Q` quit.

Progress is saved incrementally to the output CSV.

### Programmatic usage

You can also call the mapper from Python:

```python
from vtm import VariableTaxonMapper

mapper = VariableTaxonMapper(config_path="config.toml")
predictions_df = mapper.predict()
```

Or use the lower level async evaluation helpers if you already have an event loop.

### Clusters and HPC

For running on an HPC cluster with Slurm and multiple GPUs, see: [doc/nibbler_cluster.md](doc/nibbler_cluster.md).
This document describes cluster specific launch scripts and recommended settings.
