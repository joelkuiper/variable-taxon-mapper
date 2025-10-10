# LLM-assisted Taxonomy Matcher (SapBERT + llama.cpp)

This script benchmarks an LLM-assisted pipeline for mapping free-text variable metadata to a curated biomedical taxonomy. It loads `data/Variables.csv` and `data/Keywords.csv`, builds a directed acyclic taxonomy (with `networkx`), and creates deterministic node IDs and paths. Taxonomy labels are embedded with **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) to enable cosine similarity. For each item (unique `(dataset, label, name, description)`), it prunes the tree to a small, relevant subgraph by nearest-neighbor expansion (ancestors + limited-depth descendants), then prompts a local **llama.cpp** `/completions` endpoint with a constrained JSON grammar to select one node label. If the model’s output can’t be parsed, the script falls back to a k=1 SapBERT match among the same allowed labels.

Evaluation uses comma-split `variables['keywords']` as the gold set (intersected with the taxonomy) and counts a prediction as correct if it matches the gold label **or** lies on the same ancestor/descendant chain. Execution is multithreaded with pooled HTTP sessions. Running the module (e.g., `python script.py`) executes `run_label_benchmark(...)` on a sampled subset and prints a compact metrics dict (coverage, evaluated count, accuracy, errors, and llama.cpp settings).

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
./llama-server -m /media/array/Models/guff/Qwen3-4B-Instruct-2507-Q6_K.gguf --ctx-size 50000 -ngl 999 --parallel 4
```
