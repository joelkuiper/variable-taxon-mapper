# Running on the Nibbler HPC Cluster

These notes describe how to run **`variable-taxon-mapper`** and its accompanying multi-GPU llama.cpp backend on the [Nibbler HPC cluster](https://docs.gcc.rug.nl/nibbler/cluster/).
Most steps are portable to other SLURM-based clusters with similar restrictions (no root access, shared home quotas, temporary storage under `/groups/.../tmp02`).

---

## Prerequisites

* You must be able to log in to the **jumphost** (`ssh tunnel+nibbler`).
* Configure **GitHub SSH access** (add your cluster SSH public key to GitHub).
* Create a personal workspace on `tmp02` (change the group and username to reflect your own):

  ```bash
  mkdir -p /groups/<your group>/tmp02/users/<your username>
  ln -s /groups/<your group>/tmp02/users/<your username> ~/tmp02
  mkdir -p ~/tmp02/Repositories
  ```
* Clone this repository into `~/tmp02/Repositories/variable-taxon-mapper`.
* Copy your `Variables.csv` and `Keywords.csv` data into
  `~/tmp02/Repositories/variable-taxon-mapper/data/` using `scp` or `rsync`.


## Setting up `uv` and cache directories

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) with:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Since `$HOME` has limited quota, redirect heavy caches to your `tmp02` directory.
Append the following to your `~/.bashrc`:

```bash
export UV_CACHE_DIR="$HOME/tmp02/.cache/uv"
export HF_HOME="$HOME/tmp02/.cache/huggingface"
export HF_HUB_DISABLE_XET=True
export TERM=xterm-256color
```

Notes:

* `HF_HUB_DISABLE_XET=True` avoids HTTP 500 errors when Hugging Face tries to use Xet for large model downloads.
* `TERM` is set for compatibility with text-based tools and job shells.
* After editing `.bashrc`, re-source it (`source ~/.bashrc`) or log out and back in.

Then install the Python dependencies with:

```bash
cd ~/tmp02/Repositories/variable-taxon-mapper
uv sync
```

This will:

* Create the virtual environment at `~/tmp02/Repositories/variable-taxon-mapper/.venv`
* Install all dependencies listed in `pyproject.toml` or `requirements.txt`
* Prepare the environment used by `run_multi_llama_with_lb.sh`

You can verify that the venv works with:

```bash
source .venv/bin/activate
python --version
```

## Compiling `llama.cpp` (with CUDA)

Although pre-built binaries exist, the safest option on Nibbler is to **compile from source**.

```bash
cd ~/tmp02/Repositories
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

module load CUDA

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$LIBRARY_PATH"
export LDFLAGS="${LDFLAGS} -L$CUDA_HOME/lib64/stubs -Wl,-rpath,$CUDA_HOME/lib64"

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89" \
  -DLLAMA_CURL=OFF \
  -DCUDAToolkit_ROOT="$CUDA_HOME"

cmake --build build -j 2 --config Release
```

This compilation can take several hours.
Run it inside a `screen` or `tmux` session so it survives disconnections.
Do not change the number of threads (e.g. `-j 8`)! It is detrimental to other users to pin the CPU on high load, and the process will be killed.


## Downloading a model

```bash
mkdir -p ~/tmp02/Models/GGUF
wget https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf \
     -O ~/tmp02/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf
```


## Running interactively

Reserve an interactive GPU node (example: 2×A40 GPUs, 16 CPUs, 32 GB RAM):

```bash
srun --pty --gres=gpu:a40:2 --time=04:00:00 --cpus-per-task=16 --mem=32G bash -l
```

Once inside the node:

1. Check that the `config.example.toml` points to your correct CSV files.
2. Launch the pipeline with the bundled script:

   ```bash
   cd ~/tmp02/Repositories/variable-taxon-mapper
   ./run_multi_llama_with_lb.sh
   ```


## What the script does

`run_multi_llama_with_lb.sh` is a self-contained orchestrator.

### 1. Environment setup

* Loads necessary HPC modules (`cURL`, `GCCcore`, `CUDA`).
* Activates the Python venv in this repo (`.venv/` is required).

### 2. Launch one `llama-server` per GPU

* Each GPU ID in `$GPU_IDS` spawns its own `llama.cpp` instance using
  `CUDA_VISIBLE_DEVICES=<id>`.
* Default ports: starting at `$BASE_PORT` (e.g. 18080, 18081, …).
* Logs are written to:

  ```
  ~/tmp02/logs/llama-g<gpu>-p<port>.log
  ```

### 3. Health checks

* The script polls `/health`, `/healthz`, or `/v1/models` for up to 180 s.
* If any backend exits prematurely, it prints the tail of the log and aborts.

### 4. Lightweight load balancer

* Generates an inline **`tcp_lb_rr.py`** in the logs directory.
* The balancer is a ~100-line pure-Python round-robin proxy (asyncio-based).
* No dependencies, no root access, supports HTTP/SSE streaming transparently.
* Logs are written to:

  ```
  ~/tmp02/logs/lb-<LB_PORT>.log
  ```
* Listens on `$LB_PORT` (default 18000) and forwards to all healthy backends.

### 5. Run `variable-taxon-mapper`

* The venv Python executes:

  ```bash
  python -u -m main config.example.toml
  ```
* Output is streamed live to the terminal and duplicated to:

  ```
  ~/tmp02/logs/vtm-<timestamp>.log
  ```

## Submitting as a SLURM batch job

To run non-interactively, wrap the script in a simple `sbatch` file:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vtm
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/tmp02/Repositories/variable-taxon-mapper
./run_multi_llama_with_lb.sh
```

Submit it with:

```bash
sbatch run_vtm.sbatch
```

Logs from llama.cpp, the load balancer, and the main Python process all land under `~/tmp02/logs/`, while SLURM captures a combined stdout/stderr in its usual `vtm-<jobid>.out` file.
