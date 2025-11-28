# Variable Taxon Mapper

Variable Taxon Mapper maps free-text variable metadata (such as variable names and descriptions) to a curated biomedical taxonomy.

It addresses a common harmonization problem: datasets often include thousands of variables with inconsistent naming, vague descriptions, or no standard annotation. This tool links each variable to a taxonomy term, making datasets comparable and interoperable.

The system works in two phases:

* Use embeddings and the taxonomy graph to prune the taxonomy to a small, relevant subtree.
* Ask an LLM to pick the best term within that subtree.

Together these steps produce accurate mappings while keeping LLM workload small.

---

## Evaluation

On a real dataset of 1,363 variables with known ground truth, the system achieved roughly:

* **87% accuracy** when considering any correct placement in the taxonomy (exact match or correct branch).
* **63% exact matches** to the correct term.
* **An additional ~24%** mapped to a closely related parent/child within the right branch.
* **~13% incorrect branch placements**.

See `doc/results/` for detailed metrics.

---

## High-Level Pipeline

### Inputs

* A variables table (CSV, Parquet, or Feather).
* A taxonomy table (CSV or Feather) with one row per taxonomy node (including term names and parent relationships).

### Per-Variable Steps

* Embed taxonomy and build ANN index.
* Find anchors via semantic similarity and lexical matching.
* Expand these anchors in the taxonomy graph.
* Trim the expanded region into a small subtree.
* Ask an LLM to choose the best term from this subtree.

---

## ASCII Flow

```
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
  ├─ Trim candidates to a subtree
  ├─ Render prompt with subtree
  ├─ Call LLM for final label
  └─ Record prediction
```

---

## Conceptual Overview

The mapping process works as follows:

* Each taxonomy node is embedded into a vector space encoding its meaning and hierarchical context.
* The variable text is embedded into the same space.
* Nearest taxonomy nodes are retrieved as **anchors**.
* These anchors define a local region of the taxonomy.
* The region is expanded, pruned, and arranged into a small subtree.
* An LLM receives this subtree plus the variable’s metadata and selects the best term.
* The chosen term becomes the final taxonomy assignment.

By default, the mapper uses:

* **SapBERT** or similar biomedical model for embeddings.
* **HNSW** for approximate nearest neighbor search.
* **Qwen-3 4B** via `llama.cpp` for the LLM step (or any OpenAI-compatible model via API).

See `doc/` for detailed evaluation outputs.

---

## Why It Is Built This Way

### Embeddings Alone Are Not Enough

Variable names in the wild are messy. They may use acronyms, units, or local shorthand. Some taxonomy terms are abstract or located in unexpected branches. Pure embedding similarity can be misleading in such cases.

### LLMs Cannot Handle Large Taxonomies

Biomed taxonomies often contain thousands of terms. LLMs:

* cannot read them in one prompt,
* struggle with long lists of similar items,
* become slow or unreliable with huge contexts.

The LLM must only receive a small, curated subset.

### Combining Approaches Solves Both Problems

This architecture separates:

* **Where to look:** embeddings + graph pruning,
* **What fits best:** LLM selection from a small candidate set.

This provides:

* Fast, deterministic retrieval of candidates.
* Small, meaningful context for the LLM.
* Structured JSON output.
* A configurable pipeline.

See `config.example.toml` for a complete reference.

---

# Pipeline Stages

## 1. Embedding the Taxonomy

* Load taxonomy and construct a directed acyclic graph.
* Compute embeddings using a biomedical encoder.
* Optionally incorporate hierarchy by mixing parent/child vectors.
* Normalize embeddings and index them via HNSW.

This provides semantic and structural grounding for later retrieval steps.

---

## 2. Getting Anchors

For each variable:

* Embed the variable’s combined text fields.
* Retrieve top-K nearest nodes via ANN search.
* Perform lexical matching for direct or fuzzy hits.
* Merge results into a final anchor set.

Semantic and lexical recall complement one another, ensuring the correct region is included early.

---

## 3. Expanding the Anchors

The anchor set is expanded to include:

* **Ancestors** (to generalize).
* **Descendants** (to specialize).
* Optional **community detection** to isolate cohesive regions.
* Optional **graph scoring** (e.g., PageRank) to identify central nodes.

This greatly increases the chance that the true label appears in the candidate mix.

---

## 4. Trimming the Tree

The expanded candidate set is pruned:

* Apply a pruning strategy (similarity-based, radius-based, centrality-based, or connecting-tree heuristics).
* Enforce a **node budget** to ensure LLM-friendly size.
* Arrange remaining nodes into a clean subtree.

A compact and structured subtree helps the LLM make a precise choice.

---

## 5. Final LLM Selection

* Build prompt using system and user templates.
* Provide the variable description and pruned subtree.
* Call the configured LLM.
* Expect a minimal JSON output such as:

```json
{"label": "Body height"}
```

* Perform optional post-processing, including alias normalization or snapping high-level labels to appropriate children.

This step handles the nuanced semantic decision that embeddings and heuristics cannot fully resolve.

---

# Installation and Running

## Install Dependencies

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

## Start an LLM Backend

### Local LLM (via `llama.cpp`)

Example:

```bash
llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

### Remote LLM (OpenAI API)

```toml
[llm]
endpoint = "https://api.openai.com/v1"
model = "gpt-4"
api_key = "sk-...your API key..."
```

---

## Core Commands

### Evaluate With Gold Labels

```bash
vtm evaluate config.example.toml
```

### Predict Without Evaluation

```bash
vtm predict config.example.toml --output data/predictions.parquet
```

### Check Pruning Coverage

```bash
vtm prune-check config.example.toml --limit 10000
```

### Optimize Pruning Settings

```bash
vtm optimize-pruning config.example.toml --trials 80
```

---

# Additional Tools

## Error Review CLI

```bash
python -m vtm.error_review_cli \
  --predictions path/to/predictions.csv \
  --keywords path/to/Keywords.csv \
  --output data/error_review_decisions.csv \
  --config config.toml
```

This provides an interactive interface for examining and annotating misclassifications.

---

## Programmatic Use

```python
from vtm import VariableTaxonMapper

mapper = VariableTaxonMapper(config_path="config.toml")
pred_df = mapper.predict()
```

Use `mapper.evaluate()` if gold labels are available.

---

## HPC Usage

See [doc/nibbler_cluster.md](doc/nibbler_cluster.md) for Slurm and multi-GPU cluster instructions.

---

# Full Configuration Reference

The complete configuration is defined in the TOML file.
See the heavily annotated `config.example.toml` for a full option reference and explanations.
