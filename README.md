# LLM-assisted Taxonomy Matcher (embedding + llama.cpp)

This script benchmarks an LLM-assisted pipeline for mapping free-text variable metadata to a curated biomedical taxonomy. It loads `data/Variables.csv` and `data/Keywords.csv`, builds a directed acyclic taxonomy (with `networkx`). Taxonomy labels are embedded with an embedder to enable cosine similarity using [hsnwlib](https://github.com/nmslib/hnswlib). For each item (unique `(dataset, label, name, description)`), it prunes the tree to a small, relevant subgraph by nearest-neighbor and lexicographical similarity, then prompts a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `/completions` endpoint to select one node label.

Evaluation uses comma-split `variables['keywords']` as the gold set (intersected with the taxonomy) and counts a prediction as correct if it matches the gold label **or** lies on the same ancestor/descendant chain. Execution is optionally multithreaded with pooled HTTP sessions.

## General Set-Up and Idea

The **Variable Taxon Mapper** is a tool designed to map free-text variable metadata (from datasets) to a curated biomedical taxonomy. It achieves this by combining embedding-based similarity search with a Large Language Model (LLM) for final label selection. At a high level, the pipeline works as follows:

-   **Input Data**: The tool expects two CSV files as input: one containing the variables (with fields like dataset name, variable label, variable name, description, etc.), and another containing the taxonomy (a list of *keywords* or terms with their parent relationships). The taxonomy is treated as a directed acyclic graph (essentially a hierarchy/forest) where each node has at most one parent.

-   **Taxonomy Construction**: The taxonomy CSV is read and transformed into a **directed acyclic graph** (DAG) using NetworkX. Each term becomes a node, and parent-child relationships become directed edges. The code enforces that the graph remains acyclic and that no term has multiple parents, throwing an error if a cycle or multi-parent is detected. This ensures a tree (or forest) structure, which is important for defining ancestor/descendant relationships in evaluation.

-   **Embedding of Taxonomy**: All taxonomy labels (and optionally their definitions/summaries if provided) are converted into vector embeddings. By default, it uses a biomedical language model (SapBERT based on PubMedBERT) to create these embeddings. The code **enhances each taxonomy node's embedding with its ancestors\' information**: it first generates a base embedding for each term (and mixes in any definition text with a weight), then iteratively adds a fraction of the parent's vector (`gamma` factor) to the child's vector and normalizes it. This means each node's final embedding encodes both its own meaning and some context of its position in the hierarchy. All embeddings are L2-normalized (unit-length) for cosine similarity comparisons. For fast similarity search, an HNSW index (approximate nearest neighbor search) is built over the taxonomy embeddings.

-   **Processing Each Variable**: For each variable record (with fields like *label*, *name*, *description*), the pipeline generates a composite text (concatenating or considering these fields) and computes an **embedding for the variable** using the same embedder. It then uses two strategies to find candidate taxonomy nodes relevant to this variable:

-   **ANN Search (Semantic)** -- The variable's embedding is used to query the HNSW index to retrieve the top `K` most similar taxonomy nodes (by cosine similarity). These serve as *semantic anchors* in the taxonomy.

-   **Lexical Match (Textual)** -- The variable's text (name/description) is also compared lexically to taxonomy labels using a token-based similarity measure. Up to a few top terms that have high token overlap or similarity (e.g. edit distance or normalized compression distance) are taken as *lexical anchors*. This catches cases where a variable's name closely matches a taxonomy label even if embeddings might not rank it highest.

-   **Taxonomy Pruning**: Using the above anchors as starting points, the tool **prunes the taxonomy graph** to a small subgraph likely to contain the correct label. The pruning is quite sophisticated:

  - It will include the anchors, their close neighbors (children/ancestors up to a depth), and possibly other nodes that are in the same *communities* or connected components as the anchor. There are different modes for pruning (configurable via `pruning_mode`), including selecting the connected component (via "dominant forest" or "anchor hull") around anchors, or keeping everything above a similarity threshold, or within a certain radius in the graph. By default it uses a **"dominant_forest"** strategy that combines descendant expansion with a PageRank-based filtering to keep the subgraph size under a budget.
  - The result of pruning is a reduced set of allowed nodes (typically a few dozen out of potentially thousands). This subset retains the most relevant terms for the given variable.

-   **LLM Matching**: The pruned taxonomy subgraph is then turned into a **nested markdown list** (indentation representing hierarchy) and included in a prompt to a local LLM. The prompt essentially asks the LLM (running via a `llama.cpp` server) to pick the most appropriate taxonomy label for the variable, given its name/description and the pruned list of candidates. The LLM is constrained by a context grammar so that it returns a JSON with a field for the chosen concept label. In practice, it sends a request to a `POST /completions` endpoint of a running `llama.cpp` instance with the prompt, and the LLM responds with a proposed label.

-   **Post-processing & Fallbacks**: After the LLM returns a candidate label, the system checks if that label exactly matches one of the allowed taxonomy terms. If it does, great -- that's the prediction. If not (e.g. the LLM output some phrase not exactly in the taxonomy), the code will **attempt to map it back** to a valid node:

  - It first normalizes the LLM's text (trimming punctuation or quotes) and sees if it matches a taxonomy label case-insensitively.

  - If not, it embeds that LLM-proposed text and finds the closest taxonomy label embedding among the allowed set (this is a semantic **remapping** of the LLM output).

  - If that still fails to produce a match, as a final fallback it simply takes the variable's own embedding and finds the nearest label among the allowed subset (essentially defaulting to pure ANN prediction).

  - These fallbacks ensure the system always returns *some* taxonomy node rather than failing, and they improve robustness when the LLM output is slightly off. The code clearly labels the strategy used for each prediction (e.g., `"match_strategy": "llm_direct"` for direct LLM picks, versus `"embedding_remap"` or `"ann_fallback"`).

-   **Output**: For each variable, the final chosen taxonomy label (and some metadata like its ID/path in the taxonomy) is recorded. The pipeline produces an output CSV of results and prints them to console, including whether each prediction was correct and what type of match it was (exact or ancestor/descendant match). It also prints a summary of evaluation metrics (detailed below).

In summary, the tool's idea is to use **semantic embedding search to narrow the taxonomy** and then leverage an **LLM's contextual understanding to pick the best-fitting category**, with careful pruning and fallback logic to balance **accuracy** and **efficiency**. All operational parameters (model names, number of neighbors, pruning depth, LLM endpoint, etc.) are defined in a config TOML file for flexibility, making it easy to tweak the pipeline without changing code.


## Dependencies
Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

``` shell
uv sync
source .venv/bin/activate
ipython
```

Download a [Qwen3-4B-Instruct-2507-GUFF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF).
Download and compile [llama.cpp](https://github.com/ggml-org/llama.cpp) (or run via Docker).

``` shell
./llama-server -m /media/array/Models/guff/Qwen3-4B-Instruct-2507-Q6_K.gguf

# Optionally pass --ctx-size 50000 -ngl 999 --parallel 4 on CUDA devices
```

``` shell
# Configuration and parameters are set via toml

# Run evaluation
python -m main config.example.toml

# Predictions
python -m predict --config config.example.toml

# Testing the tree pruning
python -m check_pruned_tree config.example.toml --limit 10_000 --output data/Keyword_coverage.csv
```
