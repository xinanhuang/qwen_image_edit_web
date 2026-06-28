#!/bin/bash
# Stop Qwen-Image-Edit-2511 Web UI (ComfyUI + Flask)
# Usage: ./kill.sh [all|comfyui|flask|llama]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

COMFY_PORT=8188
FLASK_PORT=7860

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

kill_by_port() {
    local port=$1
    local name=$2
    local pid=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        # Wait up to 5 seconds for graceful shutdown
        for i in $(seq 1 5); do
            if ! lsof -t -i:$port &>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $name stopped (PID $pid)"
                return
            fi
            sleep 1
        done
        # Force kill if still running
        kill -9 "$pid" 2>/dev/null || true
        echo -e "  ${YELLOW}⚠${NC} $name force-killed (PID $pid)"
    else
        echo -e "  ${GREEN}✓${NC} $name not running"
    fi
}

kill_by_name() {
    local name=$1
    local pids=$(pgrep -f "$name" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 2
        # Check if still running
        local remaining=$(pgrep -f "$name" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            for pid in $remaining; do
                kill -9 "$pid" 2>/dev/null || true
            done
            echo -e "  ${YELLOW}⚠${NC} $name force-killed"
        else
            echo -e "  ${GREEN}✓${NC} $name stopped"
        fi
    else
        echo -e "  ${GREEN}✓${NC} $name not running"
    fi
}

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Qwen-Rapid-AIO (Uncensored) Web UI — Shutdown${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

TARGET=${1:-all}

case "$TARGET" in
    all)
        echo -e "${GREEN}[1/3]${NC} Stopping Flask..."
        kill_by_port $FLASK_PORT "Flask"

        echo -e "${GREEN}[2/3]${NC} Stopping ComfyUI..."
        kill_by_port $COMFY_PORT "ComfyUI"

        echo -e "${GREEN}[3/3]${NC} Stopping llama-server..."
        kill_by_name "llama-server"
        ;;
    comfyui)
        kill_by_port $COMFY_PORT "ComfyUI"
        ;;
    flask)
        kill_by_port $FLASK_PORT "Flask"
        ;;
    llama)
        kill_by_name "llama-server"
        ;;
    *)
        echo -e "${RED}Usage:${NC} $0 [all|comfyui|flask|llama]"
        echo ""
        echo "  all      — Stop Flask, ComfyUI, and llama-server (default)"
        echo "  comfyui  — Stop only ComfyUI"
        echo "  flask    — Stop only Flask"
        echo "  llama    — Stop only llama-server"
        exit 1
        ;;
esac

# Cleanup PID files
rm -rf "$PID_DIR"

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}  Done${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Show GPU status after shutdown
if command -v nvidia-smi &>/dev/null; then
    sleep 2
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "Unknown")
    echo -e "  GPU Memory: used=$GPU_MEM"
    echo ""
fi
