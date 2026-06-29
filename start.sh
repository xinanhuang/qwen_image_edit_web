#!/bin/bash
# Start Qwen-Image-Edit-2511 Web UI (ComfyUI + Flask in one go)
# Usage: ./start.sh [flask_port] [host]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FLASK_PORT=${1:-7860}
HOST=${2:-0.0.0.0}
COMFY_PORT=8188
COMFY_DIR="$HOME/ComfyUI"
COMFY_ENV="comfyui"
FLASK_ENV="qwen_webui"

LOG_DIR="/tmp"
COMFY_LOG="$LOG_DIR/comfyui.log"
FLASK_LOG="$LOG_DIR/qwen-webui.log"
PID_DIR="$SCRIPT_DIR/.pids"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Qwen-Rapid-AIO (Uncensored) Web UI — Full Stack Startup${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# --- Check GPU ---
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}[GPU]${NC} $GPU_NAME — $GPU_MEM"
fi

# --- Check for SSL certs (auto-HTTPS for mic access) ---
SSL_CERT=""
SSL_KEY=""
PEM_FILES=($(ls "$SCRIPT_DIR"/*.pem 2>/dev/null))
if [ ${#PEM_FILES[@]} -ge 2 ]; then
    SSL_KEY=$(printf '%s\n' "${PEM_FILES[@]}" | grep -i key | head -1)
    SSL_CERT=$(printf '%s\n' "${PEM_FILES[@]}" | grep -iv key | head -1)
    USE_HTTPS=true
    CURL_CMD="curl -sk"
    PROTOCOL="https"
    echo -e "${GREEN}[SSL]${NC} Auto-detected HTTPS (cert=$SSL_CERT)"
else
    USE_HTTPS=false
    CURL_CMD="curl -s"
    PROTOCOL="http"
fi

# --- Check for existing processes ---
COMFY_PID=$(lsof -t -i:$COMFY_PORT 2>/dev/null || true)
FLASK_PID=$(lsof -t -i:$FLASK_PORT 2>/dev/null || true)

if [ -n "$COMFY_PID" ]; then
    echo -e "${YELLOW}[WARN]${NC} ComfyUI already running on port $COMFY_PORT (PID $COMFY_PID)"
fi
if [ -n "$FLASK_PID" ]; then
    echo -e "${YELLOW}[WARN]${NC} Flask already running on port $FLASK_PORT (PID $FLASK_PID)"
fi

# --- Check llama-server ---
LLAMA_PID=$(pgrep -f "llama-server" 2>/dev/null || true)
if [ -n "$LLAMA_PID" ]; then
    LLAMA_MEM=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader -i 0 2>/dev/null | tail -1 || echo "?")
    echo -e "${YELLOW}[WARN]${NC} llama-server running (PID $LLAMA_PID) — expect slower image editing"
    echo -e "${YELLOW}       ${NC}Stop with: kill $LLAMA_PID"
fi

echo ""

# --- Start ComfyUI ---
echo -e "${GREEN}[1/2]${NC} Starting ComfyUI on port $COMFY_PORT..."

if [ -n "$COMFY_PID" ]; then
    echo -e "       Already running (PID $COMFY_PID)"
else
    # Find conda
    CONDA_EXE=""
    for c in "$HOME/miniconda3/bin/conda" "$HOME/miniconda/bin/conda" "$HOME/anaconda3/bin/conda" "/usr/local/miniconda3/bin/conda"; do
        [ -x "$c" ] && CONDA_EXE="$c" && break
    done

    if [ -z "$CONDA_EXE" ]; then
        echo -e "${RED}[FAIL]${NC} conda not found"
        exit 1
    fi

    # Extract python path from conda env
    COMFY_PYTHON=$(eval "$($CONDA_EXE shell.bash hook)" && conda activate $COMFY_ENV && which python 2>/dev/null)

    if [ -z "$COMFY_PYTHON" ]; then
        echo -e "${RED}[FAIL]${NC} Python not found in conda env '$COMFY_ENV'"
        exit 1
    fi

    cd "$COMFY_DIR"
    nohup "$COMFY_PYTHON" main.py --port $COMFY_PORT --auto-launch > "$COMFY_LOG" 2>&1 &
    COMFY_PID=$!
    echo -e "${GREEN}       Started${NC} (PID $COMFY_PID) → $COMFY_LOG"
    cd "$SCRIPT_DIR"
fi

# Wait for ComfyUI to be ready
echo -e "       Waiting for ComfyUI to start..."
for i in $(seq 1 30); do
    if curl -s --connect-timeout 2 http://127.0.0.1:$COMFY_PORT/api/system_stats &>/dev/null; then
        echo -e "${GREEN}       Ready${NC} ✓"
        break
    fi
    sleep 1
done

# --- Start Flask ---
echo ""
echo -e "${GREEN}[2/2]${NC} Starting Flask Web UI on port $FLASK_PORT..."

if [ -n "$FLASK_PID" ]; then
    echo -e "       Already running (PID $FLASK_PID)"
else
    FLASK_PYTHON=$(eval "$($CONDA_EXE shell.bash hook)" && conda activate $FLASK_ENV && which python 2>/dev/null)

    if [ -z "$FLASK_PYTHON" ]; then
        echo -e "${RED}[FAIL]${NC} Python not found in conda env '$FLASK_ENV'"
        exit 1
    fi

    export PORT=$FLASK_PORT
    export HOST=$HOST
    [ -n "$SSL_CERT" ] && export SSL_CERT=$SSL_CERT
    [ -n "$SSL_KEY" ] && export SSL_KEY=$SSL_KEY
    nohup "$FLASK_PYTHON" server_comfy.py > "$FLASK_LOG" 2>&1 &
    FLASK_PID=$!
    echo -e "${GREEN}       Started${NC} (PID $FLASK_PID) → $FLASK_LOG"
fi

# Wait for Flask to be ready
echo -e "       Waiting for Flask to start..."
for i in $(seq 1 15); do
    if $CURL_CMD --connect-timeout 2 $PROTOCOL://127.0.0.1:$FLASK_PORT/api/status &>/dev/null; then
        echo -e "${GREEN}       Ready${NC} ✓"
        break
    fi
    sleep 1
done

# --- Save PIDs ---
mkdir -p "$PID_DIR"
echo "$COMFY_PID" > "$PID_DIR/comfyui.pid"
echo "$FLASK_PID" > "$PID_DIR/flask.pid"

# --- Summary ---
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}  All services started${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "  ComfyUI:    http://127.0.0.1:$COMFY_PORT  (PID $COMFY_PID)"
echo -e "  Flask UI:   $PROTOCOL://127.0.0.1:$FLASK_PORT  (PID $FLASK_PID)"
if [ "$USE_HTTPS" = true ]; then
    echo -e "  HTTP fallback: http://127.0.0.1:$((FLASK_PORT-1))"
fi
# Detect public IP for display
PUBLIC_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")
LAN_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}' || echo "unknown")
echo -e "  LAN:        $PROTOCOL://$LAN_IP:$FLASK_PORT"
echo -e "  Public:     $PROTOCOL://$PUBLIC_IP:$FLASK_PORT"
echo ""
echo -e "  Logs:"
echo -e "    ComfyUI:  $COMFY_LOG"
echo -e "    Flask:    $FLASK_LOG"
echo ""
echo -e "  Stop all:   $SCRIPT_DIR/kill.sh"
echo ""
