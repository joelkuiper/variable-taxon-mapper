# Running on the Nibbler HPC Cluster

These notes describe how to run **`variable-taxon-mapper`** and its accompanying multi-GPU `llama.cpp` backend on the [Nibbler HPC cluster](https://docs.gcc.rug.nl/nibbler/cluster/) using the provided [run_pipeline_lb.sh](../run_pipeline_lb.sh) script (lb for load balancer).
Most steps are portable to other SLURM-based clusters with similar configuration and restrictions (no root access, shared home quotas, temporary storage under `/groups/.../tmp02`).

---

## Prerequisites

* You must be able to log in to the **jumphost** (`ssh tunnel+nibbler`).
* Configure **GitHub SSH access** (add your cluster SSH public key to GitHub). See [the Nibbler documentation](https://docs.gcc.rug.nl/nibbler/generate-key-pair-openssh/) for generating the key if none exists yet. 
* Create a personal workspace on `tmp02` (change the group and username to reflect your own):

  ```bash
  mkdir -p /groups/<your group>/tmp02/users/<your username>
  ```

* Define an environment variable that points to your personal temporary workspace, add this to your `~/.bashrc`:

  ```bash
  export WORKDIR=/groups/<your group>/tmp02/users/<your username>
  # or some other path where the code, llama.cpp, and the models will live
  ```

  Then reload it:

  ```bash
  source ~/.bashrc
  ```

  You can verify it works with:

  ```bash
  echo "$WORKDIR"
  ```

All paths below use the `$WORKDIR`.

* Create the Repositories directory:

  ```bash
  mkdir -p "$WORKDIR/Repositories"
  ```

* Clone this repository:

  ```bash
  git clone https://github.com/joelkuiper/variable-taxon-mapper.git \
      "$WORKDIR/Repositories/variable-taxon-mapper"
  ```

* Copy your local `Variables.csv` and `Keywords.csv` data into `$WORKDIR/Repositories/variable-taxon-mapper/data/` using `scp` or `rsync`.
  For example, if your data lives locally in `~/Repositories/variable-taxon-mapper/data/`:

  ```bash
  rsync -avhP ~/Repositories/variable-taxon-mapper/data/ \
    tunnel+nibbler:/groups/<your group>/tmp02/users/<your username>/Repositories/variable-taxon-mapper/data/
  ```

  > Note the trailing slash `/` after `data/`: it copies contents into the target folder, not the directory itself!

---

## Setting up `uv` and cache directories

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) with:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Since `$HOME` has limited quota, redirect heavy caches to `$WORKDIR`:

Append to your `~/.bashrc`:

```bash
export UV_CACHE_DIR="$WORKDIR/.cache/uv"
export HF_HOME="$WORKDIR/.cache/huggingface"
export HF_HUB_DISABLE_XET=True
export TERM=xterm-256color
```

Notes:

* `HF_HUB_DISABLE_XET=True` avoids HTTP 500 errors when Hugging Face tries to use Xet for large model downloads.
* `TERM` is set for compatibility with text-based tools and job shells.

After editing `.bashrc`, re-source it (`source ~/.bashrc`) or log out and back in. Then install the Python dependencies with:

```bash
cd "$WORKDIR/Repositories/variable-taxon-mapper"
uv sync
```

This will:

* Create the virtual environment at `$WORKDIR/Repositories/variable-taxon-mapper/.venv`
* Install all dependencies listed in `pyproject.toml`
* Prepare the environment used by [run_pipeline_lb.sh](../run_pipeline_lb.sh)

Verify that the venv works:

```bash
source .venv/bin/activate
python --version # should be Python ≥3.11
```

---

## Compiling `llama.cpp` (with CUDA)

Although pre-built binaries exist, the safest option on Nibbler is to **compile from source**.

```bash
cd "$WORKDIR/Repositories"
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

module load GCCcore/11.3.0
module load CMake/3.23.1-GCCcore-11.3.0
module load CUDA/12.2.0

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
Use `-j 2` to avoid hogging CPUs on the shared jumphost.
Alternatively, compile on a compute node (faster, safer):

```bash
srun --pty --time=00:30:00 --cpus-per-task=32 --mem=32G bash -l
# once inside the node:
cmake --build build -j 32 --config Release
```

---

## Downloading a model

```bash
mkdir -p "$WORKDIR/Models/GGUF"
wget https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf \
     -O "$WORKDIR/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf"
```

> **Note:** You can pass a custom model at runtime:
> `MODEL=$WORKDIR/Models/GGUF/SomeOtherModel.gguf ./run_pipeline_lb.sh`

---

## Running interactively

Reserve an interactive GPU node (example: 2×A40 GPUs, 16 CPUs, 32 GB RAM for 4 h):

```bash
srun --pty --gres=gpu:a40:2 --time=04:00:00 --cpus-per-task=16 --mem=32G bash -l
```

Once inside the node:

1. Check that `config.example.toml` points to your correct CSV files and adjust parallelism tunables as needed.
2. Launch the pipeline:

   ```bash
   cd "$WORKDIR/Repositories/variable-taxon-mapper"
   LB_PORT=8080 ./run_pipeline_lb.sh
   ```

---

## What the script does

`run_pipeline_lb.sh` is a self-contained orchestrator.

### 1. Environment setup

* Loads necessary HPC modules (`GCCcore`, `CUDA`).
* Activates the Python venv in this repo (`.venv/` is required).

### 2. Launch one `llama-server` per GPU

* Each GPU ID in `$GPU_IDS` spawns its own `llama.cpp` instance using `CUDA_VISIBLE_DEVICES=<id>`.
* Default ports: starting at `$BASE_PORT` (e.g. 18080, 18081, …).
* Logs written to `$WORKDIR/logs/llama-g<gpu>-p<port>.log`

### 3. Health checks

* Polls `/health`, `/healthz`, or `/v1/models` for up to 180 s.
* If any backend exits prematurely, prints the tail of the log and aborts.

### 4. Lightweight load balancer

* Generates an inline **`tcp_lb_rr.py`** in the logs directory.
* The balancer is a ~100-line pure-Python round-robin proxy (asyncio-based).
* No dependencies, no root access, supports HTTP/SSE streaming transparently.
* Logs written to `$WORKDIR/logs/lb-<LB_PORT>.log`
* Listens on `$LB_PORT` and forwards to all healthy backends.
* The client asks llama.cpp to auto-assign request slots (`slot_id=-1`),
  so round-robin dispatching does not conflict with cached contexts.

### 5. Run `variable-taxon-mapper`

* By default executes:

  ```bash
  python -u -m main config.example.toml
  ```

* You can override the final command by passing arguments, for example:

  ```bash
  ./run_pipeline_lb.sh python -u -m predict config.example.toml
  ```

* Output is streamed live to the terminal and duplicated to `$WORKDIR/logs/vtm-<timestamp>.log`.

### 6. Configuration (environment variables)

| Variable    | Description                               | Default                                                  |
| ----------- | ----------------------------------------- | -------------------------------------------------------- |
| `GPU_IDS`   | space-separated CUDA IDs                  | `"0 1"`                                                  |
| `BASE_PORT` | first backend port                        | `18080`                                                  |
| `LB_PORT`   | load balancer port                        | `18000`                                                  |
| `CTX`       | llama.cpp context size `-c`               | `120000`                                                 |
| `SLOTS`     | llama.cpp parallel slots `-np` per server | `6`                                                      |
| `LOG_DIR`   | logs output dir                           | `$WORKDIR/logs`                                          |
| `LLAMA_BIN` | path to `llama-server`                    | `$WORKDIR/Repositories/llama.cpp/build/bin/llama-server` |
| `MODEL`     | path to GGUF model                        | `$WORKDIR/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf`  |
| `VTM_DIR`   | repo dir                                  | `$WORKDIR/Repositories/variable-taxon-mapper`            |
| `VTM_CFG`   | config file passed to Python              | `config.example.toml`                                    |
| `PORT`      | exported app port                         | `$LB_PORT`                                               |

Example (custom model, single GPU):

```bash
MODEL=$WORKDIR/Models/GGUF/MyModel.gguf GPU_IDS="0" ./run_pipeline_lb.sh
```

---

## Submitting as a SLURM batch job

To run non-interactively, wrap the script in a simple `sbatch` file:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vtm
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

LB_PORT=8080 ./run_pipeline_lb.sh
```

Submit with:

```bash
mkdir -p "$WORKDIR/logs"

sbatch --chdir="$WORKDIR/Repositories/variable-taxon-mapper" \
       --output="$WORKDIR/logs/%x_%j.out" \
       --error="$WORKDIR/logs/%x_%j.err" \
       vtm.sbatch
```

Logs from `llama.cpp`, the load balancer, and the main Python process all land under `$WORKDIR/logs/`, while SLURM captures combined stdout/stderr in its usual `vtm-<jobid>.out` file.

