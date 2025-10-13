# LLM-assisted Taxonomy Matcher (embedding + llama.cpp)

This script benchmarks an LLM-assisted pipeline for mapping free-text variable metadata to a curated biomedical taxonomy. It loads `data/Variables.csv` and `data/Keywords.csv`, builds a directed acyclic taxonomy (with `networkx`). Taxonomy labels are embedded with an embedder to enable cosine similarity using [hsnwlib](https://github.com/nmslib/hnswlib). For each item (unique `(dataset, label, name, description)`), it prunes the tree to a small, relevant subgraph by nearest-neighbor and lexicographical similarity, then prompts a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `/completions` endpoint to select one node label.

Evaluation uses comma-split `variables['keywords']` as the gold set (intersected with the taxonomy) and counts a prediction as correct if it matches the gold label **or** lies on the same ancestor/descendant chain. Execution is optionally multithreaded with pooled HTTP sessions.

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
