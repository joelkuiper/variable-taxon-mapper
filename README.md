# LLM-assisted Taxonomy Matcher (embedding + llama.cpp)

This is a tool designed to map free-text variable metadata (from datasets) to a curated biomedical taxonomy. It achieves this by combining embedding-based similarity search with a Large Language Model (LLM) for final label selection. At a high level, the pipeline works as follows:

-   **Input Data**: The tool expects two CSV files as input: one containing the variables (with fields like dataset name, variable label, variable name, description, etc.), and another containing the taxonomy (a list of *keywords* or terms with their parent relationships). The taxonomy is treated as a directed acyclic graph (essentially a hierarchy/forest) where each node has at most one parent.

-   **Taxonomy Construction**: The taxonomy CSV is read and transformed into a directed acyclic graph (DAG) using `networkx`. Each term becomes a node, and parent-child relationships become directed edges. The code enforces that the graph remains acyclic and that no term has multiple parents, throwing an error if a cycle or multi-parent is detected. This ensures a tree (or forest) structure, which is important for defining ancestor/descendant relationships in evaluation.

-   **Embedding of Taxonomy**: All taxonomy labels (and optionally their definitions/summaries if provided) are converted into vector embeddings. By default, it uses a biomedical language model ([SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)) to create these embeddings, but others like [FlagEmbedding](https://huggingface.co/BAAI/bge-large-en-v1.5) can be used. The code enhances each taxonomy node's embedding with its ancestors\' information: it first generates a base embedding for each term (and mixes in any definition text with a weight), then iteratively adds a fraction of the parent's vector (`gamma` factor) to the child's vector and normalizes it. This means each node's final embedding encodes both its own meaning and some context of its position in the hierarchy. All embeddings are L2-normalized (unit-length) for cosine similarity comparisons. For fast similarity search, an HNSW index (approximate nearest neighbor search, [hsnwlib](https://github.com/nmslib/hnswlib)) is built over the taxonomy embeddings.

-   **Processing Each Variable**: For each variable record (with fields like *label*, *name*, *description*), the pipeline generates a composite text (concatenating or considering these fields) and computes an embedding for the variable using the same embedder. It then uses two strategies to find candidate taxonomy nodes relevant to this variable:

    -  **ANN Search (Semantic)** -- The variable's embedding is used to query the index to retrieve the top `K` most similar taxonomy nodes (by cosine similarity). These serve as *semantic anchors* in the taxonomy.
    -  **Lexical Match (Textual)** -- The variable's text (name/description) is also compared lexically to taxonomy labels using a token-based similarity measure. Up to a few top terms that have high token overlap or similarity (e.g. edit distance or normalized compression distance) are taken as *lexical anchors*. This catches cases where a variable's name closely matches a taxonomy label even if embeddings might not rank it highest.

-   **Taxonomy Pruning**: Using the above anchors as starting points, the tool prunes the taxonomy graph to a small subgraph likely to contain the correct label. The pruning is quite elaborate:

    - It will include the anchors, their close neighbors (children/ancestors up to a depth), and possibly other nodes that are in the same *communities* or connected components as the anchor. There are different modes for pruning (configurable via `pruning_mode`), including selecting the connected component (via "dominant forest" or "anchor hull") around anchors, prioritising one node per anchor community with PageRank ("community_pagerank"), connecting anchors to top-similarity labels with a Steiner tree approximation ("steiner_similarity"), keeping everything above a similarity threshold, or limiting exploration to an undirected radius. By default it uses a `dominant_forest` strategy that combines descendant expansion with a PageRank-based filtering to keep the subgraph size under a budget.
    - The result of pruning is a reduced set of allowed nodes. This subset retains the most relevant terms for the given variable.

-   **LLM Matching**: The pruned taxonomy subgraph is then turned into a nested Markdown list (indentation representing hierarchy) and included in a prompt ([example prompt](./doc/example_prompt.md)) to a local LLM. The prompt essentially asks the LLM (running via a `llama.cpp` server) to pick the most appropriate taxonomy label for the variable, given its name/description and the pruned list of candidates. The LLM is constrained by a context grammar so that it returns a JSON with a field for the chosen concept label. In practice, it sends a request to a `POST /completions` endpoint of a running `llama.cpp` instance with the prompt, and the LLM responds with a proposed label.

-   **Output**: For each variable, the final chosen taxonomy label (and some metadata like its ID/path in the taxonomy) is recorded. The pipeline produces an output CSV of results and prints them to console, including whether each prediction was correct and what type of match it was (exact or ancestor/descendant match). It also prints a summary of evaluation metrics (see [example output](./doc/20251103_results.md)).

In summary, the tool's idea is to use semantic embedding search to narrow the taxonomy and then leverage an LLM's contextual understanding to pick the best-fitting category. All operational parameters (model names, number of neighbors, pruning depth, LLM endpoint, etc.) are defined in a config TOML file (see [config.example.toml](./config.example.toml)) for flexibility, making it easy to tweak the pipeline without changing code.


## Dependencies
Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

``` shell
uv sync
source .venv/bin/activate
ipython
```
Download and install [llama.cpp](https://github.com/ggml-org/llama.cpp) (or run via Docker). The system has been tested with [Qwen3-4B-Instruct-2507-GUFF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), you can either download a GUFF and pass it via the `--model` parameter, or let the server download and cache it (as below).

``` shell
llama-server -hf  unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M

# Optionally pass --ctx-size 50000 -ngl 999 --parallel 4 on CUDA devices, --flash-attn on might help
```

``` shell
# Configuration and parameters are set via TOML

# Run evaluation
python -m main config.example.toml

# Predictions
python -m predict config.example.toml

# Testing the tree pruning
python -m check_pruned_tree config.example.toml --limit 10_000 --output data/Keyword_coverage.csv
```

## Running on HPC clusters
See [Nibbler Cluster documentation](./doc/nibbler_cluster.md)
