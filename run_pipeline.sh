#!/usr/bin/env bash
set -euo pipefail

# ---------- Settings ----------
PORT="${PORT:-8080}"   # llama.cpp default port
LOG_DIR="$HOME/tmp02/logs"
LLAMA_BIN="$HOME/tmp02/Repositories/llama.cpp/build/bin/llama-server"
MODEL="$HOME/tmp02/Models/GGUF/Qwen3-4B-Instruct-2507-Q8_0.gguf"
VTM_DIR="$HOME/tmp02/Repositories/variable-taxon-mapper"
VTM_CFG="config.example.toml"

# Make sure logs dir exists
mkdir -p "$LOG_DIR"

echo "[1/4] Loading modules..."
module load cURL/8.7.1-GCCcore-13.3.0
module load GCCcore/13.3.0
module load CUDA/12.2.0

# Some clusters get fussy about $TERM when spawning background jobs; set something safe.
export TERM=xterm

echo "[2/4] Starting llama.cpp server on port ${PORT}..."
# Start llama.cpp in a background subprocess with nohup; capture logs.
# -ngl 999 uses GPU offload aggressively; -fa on enables flash-attn if available.
# set a default if not provided
: "${PORT:=8080}"

nohup \
    "$LLAMA_BIN" \
    -m "$MODEL" \
    -ngl 999 \
    -c 120000 \
    -np 6 \
    -fa on \
    --port "$PORT" \
    >> "$LOG_DIR/llama-server.out" 2>> "$LOG_DIR/llama-server.err" &

LLAMA_PID=$!
echo "  • PID: $LLAMA_PID"
echo "  • Logs: $LOG_DIR/llama-server.out (stdout), $LOG_DIR/llama-server.err (stderr)"

# Optional: use GNU screen instead. Uncomment below if you prefer a screen session.
# screen -DmS llama_srv bash -lc \
    #  '"$LLAMA_BIN" -m "'"$MODEL"'" -ngl 999 -fa on --port "'"$PORT"'" >> "'"$LOG_DIR/llama-server.out"'" 2>> "'"$LOG_DIR/llama-server.err"'"'

echo "[3/4] Waiting for llama.cpp to become ready..."
# We’ll try a few health-ish endpoints commonly exposed by llama.cpp.
# Give it up to ~3 minutes.
READY=0
for i in {1..180}; do
    # process still running?
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited early. Check logs in $LOG_DIR."
        exit 1
    fi

    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
            || curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 \
            || curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 ; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: llama-server did not become ready in time."
    echo "Tail of stderr:"
    tail -n 60 "$LOG_DIR/llama-server.err" || true                                                                                             exit 1
fi

echo "  • llama.cpp is online on http://127.0.0.1:${PORT}"

echo "[4/4] Running variable-taxon-mapper..."
cd "$VTM_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m main "$VTM_CFG"
