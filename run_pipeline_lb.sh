#!/usr/bin/env bash
# run_multi_llama_with_lb.sh
# Self-contained sbatch-friendly launcher:
#  - loads modules
#  - Activates venv (hard requirement; aborts if missing)
#  - starts one llama-server per GPU (CUDA_VISIBLE_DEVICES per process)
#  - waits for each backend to be healthy
#  - writes & launches a tiny pure-python TCP round-robin LB (no deps)
#  - runs variable-taxon-mapper against the LB with live logs (unbuffered + tee)
#  - cleans up on exit
#
# Usage (defaults):
#   GPU_IDS="0 1" ./run_multi_llama_with_lb.sh
#
# Env overrides:
#   GPU_IDS   (space separated list, default "0 1")
#   BASE_PORT (first backend port, default 18080)
#   LB_PORT   (load balancer port, default 18000)
#   CTX       (llama -c, default 120000)
#   SLOTS     (llama -np, default 6)
#   PORT      (exported to variable-taxon-mapper, default -> LB_PORT)
#   (arguments) final command override, e.g. `./run_pipeline_lb.sh python -u -m predict config.toml`
set -euo pipefail

# -------------------- Configurable defaults --------------------
GPU_IDS=${GPU_IDS:-"0 1"}            # e.g. "0" or "0 1 2"
BASE_PORT=${BASE_PORT:-18080}        # backends: BASE_PORT, BASE_PORT+1, ...
LB_PORT=${LB_PORT:-18000}            # front-door LB port
PORT="${PORT:-$LB_PORT}"             # exported to your app
CTX=${CTX:-120000}                   # llama.cpp -c
SLOTS=${SLOTS:-6}                    # llama.cpp -np

LOG_DIR="${LOG_DIR:-$HOME/tmp02/logs}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/tmp02/Repositories/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/tmp02/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf}"

VTM_DIR="${VTM_DIR:-$HOME/tmp02/Repositories/variable-taxon-mapper}"
VTM_CFG="${VTM_CFG:-config.example.toml}"

# -------------------- Determine final command --------------------
DEFAULT_CMD=(python -u -m main "$VTM_CFG")

if [[ $# -gt 0 ]]; then
  RUN_CMD=("$@")
else
  RUN_CMD=("${DEFAULT_CMD[@]}")
fi

# -------------------- Environment (HPC) --------------------
# Load modules if available (no-op on systems without module)
if command -v module >/dev/null 2>&1; then
  module load GCCcore/11.3.0 || true
  module load CUDA/12.2.0 || true
fi

mkdir -p "$LOG_DIR"
export TERM="${TERM:-xterm}"

# -------------------- Activate venv --------------------
if [[ ! -f "$VTM_DIR/.venv/bin/activate" ]]; then
  echo "ERROR: Python venv not found at: $VTM_DIR/.venv"
  exit 1
fi
pushd "$VTM_DIR" >/dev/null
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "ERROR: failed to activate venv at $VTM_DIR/.venv"; exit 1; }
popd >/dev/null

# From here on, we assume `python` is the venv python. No system-python fallback.
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found after venv activation"; exit 1; }

# -------------------- Cleanup trap --------------------
PIDS=()          # llama server PIDs
LB_PID=""        # load balancer PID
LB_FILE=""       # path to inline LB script file

cleanup() {
  echo "[CLEANUP] Stopping load balancer and backend servers..."
  if [[ -n "${LB_PID:-}" ]]; then
    if kill -0 "$LB_PID" 2>/dev/null; then
      echo "  killing LB pid $LB_PID"
      kill "$LB_PID" || true
    fi
  fi

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  killing backend pid $pid"
      kill "$pid" || true
    fi
  done

  # allow processes to terminate
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  force-kill $pid"
      kill -9 "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

# -------------------- 1) Start llama servers --------------------
echo "[1/4] Launching llama.cpp servers per GPU..."
BACKENDS=()   # will contain "127.0.0.1:PORT" strings
i=0
for gpu in $GPU_IDS; do
  port=$((BASE_PORT + i))
  logfile="$LOG_DIR/llama-g${gpu}-p${port}.log"
  echo "  • GPU $gpu -> http://127.0.0.1:${port} (log: $logfile)"

  # start the server pinned to a single visible CUDA device
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_BIN" \
    -m "$MODEL" \
    -ngl 999 \
    -c "$CTX" \
    -np "$SLOTS" \
    -fa on \
    --port "$port" \
    >>"$logfile" 2>&1 &

  pid=$!
  PIDS+=("$pid")
  BACKENDS+=("127.0.0.1:$port")
  i=$((i+1))
done

# -------------------- 2) Wait for backends to be healthy --------------------
echo "[2/4] Waiting for backends to become healthy..."
for be in "${BACKENDS[@]}"; do
  echo "  • Waiting for http://$be"
  ok=0
  for t in {1..180}; do
    # check any of the common health endpoints
    if curl -fsS "http://$be/health" >/dev/null 2>&1 \
      || curl -fsS "http://$be/healthz" >/dev/null 2>&1 \
      || curl -fsS "http://$be/v1/models" >/dev/null 2>&1 ; then
      ok=1
      echo "    OK: $be"
      break
    fi

    # If all backend processes died, bail out early
    any_alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    if [[ $any_alive -eq 0 ]]; then
      echo "ERROR: all llama-server processes appear to have exited. Check logs in $LOG_DIR"
      # show last lines of each log to help debugging
      for lg in "$LOG_DIR"/llama-g*-p*.log; do
        echo "----- tail $lg -----"
        tail -n 40 "$lg" || true
      done
      exit 1
    fi

    sleep 1
  done

  if [[ $ok -ne 1 ]]; then
    echo "ERROR: backend $be not healthy after timeout."
    exit 1
  fi
done

# -------------------- 3) Write & start tiny TCP round-robin LB --------------------
echo "[3/4] Writing lightweight TCP round-robin load balancer..."
LB_FILE="$LOG_DIR/tcp_lb_rr.py"
cat > "$LB_FILE" <<'PY'
#!/usr/bin/env python
# Minimal TCP round-robin proxy (pure stdlib).
# Byte-for-byte proxy so it supports HTTP chunked/SSE streaming transparently.
import argparse, asyncio, itertools

async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass

async def handle_client(client_reader, client_writer, rr_cycle):
    # pick backend
    backend = next(rr_cycle)
    be_host, be_port = backend
    try:
        be_reader, be_writer = await asyncio.open_connection(be_host, be_port)
    except Exception:
        # try one more backend
        backend = next(rr_cycle)
        be_host, be_port = backend
        be_reader, be_writer = await asyncio.open_connection(be_host, be_port)

    # bidirectional piping
    await asyncio.gather(
        pipe(client_reader, be_writer),
        pipe(be_reader, client_writer),
    )

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=str, default="0.0.0.0:18000")
    ap.add_argument("--backends", nargs="+", required=True)
    args = ap.parse_args()
    host, port_s = args.listen.split(":", 1)
    port = int(port_s)
    backends = tuple((h.split(":")[0], int(h.split(":")[1])) for h in args.backends)
    rr_cycle = itertools.cycle(backends)
    server = await asyncio.start_server(lambda r,w: handle_client(r,w,rr_cycle), host, port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[LB] Listening on {addrs} -> backends: {backends}", flush=True)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
PY
chmod +x "$LB_FILE"

echo "  • Starting LB on :$LB_PORT (backends: ${BACKENDS[*]})"
python -u "$LB_FILE" --listen "0.0.0.0:${LB_PORT}" --backends "${BACKENDS[@]}" \
  >> "$LOG_DIR/lb-${LB_PORT}.log" 2>&1 &
LB_PID=$!

# Wait for LB to bind to port (best-effort)
for t in {1..10}; do
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | grep -q ":${LB_PORT} "; then break; fi
  else
    # fallback: small sleep to allow bind
    sleep 1
  fi
  sleep 0.5
done

echo "  • LB ready at http://127.0.0.1:${LB_PORT}"

# -------------------- 4) Run variable-taxon-mapper  --------------------
echo "[4/4] Running variable-taxon-mapper against LB :$LB_PORT"
cd "$VTM_DIR"
VTM_LOG="$LOG_DIR/vtm-$(date +%F_%H%M%S).log"
echo "  • VTM log -> $VTM_LOG"
echo "  • Command -> ${RUN_CMD[*]}"
export PORT

# Use venv python only
PYTHONUNBUFFERED=1 stdbuf -oL -eL \
  "${RUN_CMD[@]}" 2>&1 | tee -a "$VTM_LOG"
