# Running on the Nibbler HPC Cluster

These notes describe how to run **`variable-taxon-mapper`** and its accompanying multi-GPU llama.cpp backend on the [Nibbler HPC cluster](https://docs.gcc.rug.nl/nibbler/cluster/) using the provided  [run_pipeline_lb.sh](../run_pipeline_lb.sh) script (lb for load balancer).
Most steps are portable to other SLURM-based clusters with similar configuration and restrictions (no root access, shared home quotas, temporary storage under `/groups/.../tmp02`).

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
* Clone this repository

```bash
git clone https://github.com/joelkuiper/variable-taxon-mapper.git ~/tmp02/Repositories/variable-taxon-mapper
```

* Copy your `Variables.csv` and `Keywords.csv` data into `~/tmp02/Repositories/variable-taxon-mapper/data/` using `scp` or `rsync`.
For example, if your data lives in `~/Repositories/variable-taxon-mapper/data/ (otherwise change the path):

```bash
rsync -avhP ~/Repositories/variable-taxon-mapper/data/ \
    tunnel+nibbler:~/tmp02/Repositories/variable-taxon-mapper/data/
```

> Note the trailing slash / after data/: it copies contents into the target folder, not the directory itself!

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
* Prepare the environment used by [run_pipeline_lb.sh](../run_pipeline_lb.sh)

You can verify that the venv works with:

```bash
source .venv/bin/activate
python --version
```

## Compiling `llama.cpp` (with CUDA)

Although pre-built binaries exist, the safest option on Nibbler is to **compile from source**.

```bash
cd ~/tmp02/Repositories
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
Use `-j 2` to avoid hogging CPUs on the shared jumphost. Jobs with high parallelism may be killed by admins. Alternatively, and ideally, compile on a compute node and set a higher `-j <num cores>`; the commands are the same and will finish faster, and you don't need a GPU for this. So `srun --pty --time=00:30:00 --cpus-per-task=32 --mem=32G bash -l` and run these commands from there (with `-j 32`) is strongly advisable (but you may have to wait in the queue).


## Downloading a model

```bash
mkdir -p ~/tmp02/Models/GGUF
wget https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf \
     -O ~/tmp02/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf
```
> **Note:** You can pass a custom model at runtime:
> `MODEL=~/tmp02/Models/GGUF/SomeOtherModel.gguf ./run_pipeline_lb.sh`

## Running interactively

Reserve an interactive GPU node (example: 2×A40 GPUs, 16 CPUs, 32 GB RAM for 4 hours):

```bash
srun --pty --gres=gpu:a40:2 --time=04:00:00 --cpus-per-task=16 --mem=32G bash -l
```

Once inside the node:

1. Check that the `config.example.toml` points to your correct CSV files and adjust parallelism tunables as needed.
2. Launch the pipeline with the bundled script:

   ```bash
   cd ~/tmp02/Repositories/variable-taxon-mapper
   LB_PORT=8080 ./run_pipeline_lb.sh
   ```


## What the script does

`run_pipeline_lb.sh` is a self-contained orchestrator.

### 1. Environment setup

* Loads necessary HPC modules (`GCCcore`, `CUDA`).
* Activates the Python venv in this repo (`.venv/` is required).

### 2. Launch one `llama-server` per GPU

* Each GPU ID in `$GPU_IDS` spawns its own `llama.cpp` instance using
  `CUDA_VISIBLE_DEVICES=<id>`.
* Default ports: starting at `$BASE_PORT` (e.g. 18080, 18081, …).
* Logs are written to `~/tmp02/logs/llama-g<gpu>-p<port>.log

### 3. Health checks

* The script polls `/health`, `/healthz`, or `/v1/models` for up to 180 s.
* If any backend exits prematurely, it prints the tail of the log and aborts.

### 4. Lightweight load balancer

* Generates an inline **`tcp_lb_rr.py`** in the logs directory.
* The balancer is a ~100-line pure-Python round-robin proxy (asyncio-based).
* No dependencies, no root access, supports HTTP/SSE streaming transparently.
* Logs are written to `~/tmp02/logs/lb-<LB_PORT>.log
* Listens on `$LB_PORT` and forwards to all healthy backends.

### 5. Run `variable-taxon-mapper`

* The venv Python executes:

  ```bash
  python -u -m main config.example.toml
  ```
* Output is streamed live to the terminal and duplicated to: `~/tmp02/logs/vtm-<timestamp>.log`.

### 6. Configuration (env vars)

The launcher is configured entirely via environment variables (no file edits needed):

- `GPU_IDS` — space-separated CUDA IDs (default `"0 1"`).
- `BASE_PORT` — first backend port (default `18080`).
- `LB_PORT` — load balancer port (default `18000`).
- `CTX` — llama.cpp context size `-c` (default `120000`).
- `SLOTS` — llama.cpp parallel slots `-np` per server (default `6`).
- `LOG_DIR` — logs output dir (default `~/tmp02/logs`).
- `LLAMA_BIN` — path to `llama-server` (default `~/tmp02/Repositories/llama.cpp/build/bin/llama-server`).
- `MODEL` — path to your GGUF (default `~/tmp02/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf`).
- `VTM_DIR` — repo dir (default `~/tmp02/Repositories/variable-taxon-mapper`).
- `VTM_CFG` — config file passed to `python -m main` (default `config.example.toml`).
- `PORT` — exported to your app (defaults to `LB_PORT`).

For example, use a custom model path and single GPU 0:

```bash
MODEL=~/tmp02/Models/GGUF/MyModel.gguf GPU_IDS="0" ./run_pipeline_lb.sh
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
LB_PORT=8080 ./run_pipeline_lb.sh
```

Submit it with:

```bash
sbatch run_vtm.sbatch
```

Logs from llama.cpp, the load balancer, and the main Python process all land under `~/tmp02/logs/`, while SLURM captures a combined stdout/stderr in its usual `vtm-<jobid>.out` file.
