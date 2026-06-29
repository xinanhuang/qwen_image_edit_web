# Deployment Steps — Qwen-Image-Edit-2511 Web UI (Local RTX 5090)

**Date:** 2026-06-20  
**Machine:** xh97-ml (RTX 5090, 32GB VRAM)  
**Status:** ✅ Running

---

## Steps Performed

### 1. Code Review Fixes (Phase 1)

#### SQLite Migration
- Migrated job history from `history/*.json` files to `history.db` (SQLite)
- Created `_get_db()`, `_ensure_db()`, `_row_to_dict()` helpers
- Updated `save_job_history()`, `get_all_history()`, `get_ip_history()` to use SQLite
- Added `get_job_history()` for single-record lookups
- **Result:** 44 existing records migrated successfully

#### Bug Fixes
- **B1:** Wired `guidance_scale` into `ModelSamplingAuraFlow` as `shift` parameter
- **B2:** Fixed indentation bug in `_convert_dng_to_jpeg()`
- **P2:** Created `_finalize_job()` helper to consolidate 4 duplicate `save_job_history()` calls
- **P3:** Added floor/ceiling (10s-300s) to `AVG_JOB_DURATION` EMA
- **P4:** Removed redundant `import pillow_heif` inside `convert_image()`

#### Output Cleanup
- Added 7-day cleanup of old PNG files in `outputs/` to the `cleanup_loop()`

#### Archive System
- Created `archive/` directory for permanent job backups
- Every completed job is archived with:
  - `metadata.json` — Full job details
  - `input.png` — Original input image
  - `output_0.png`, `output_1.png`, etc. — Generated output images
- Added API endpoints:
  - `GET /api/archive` — List all archived jobs
  - `GET /archive/<job_id>/<file>` — Download archived files

#### Project Structure
- Created `.gitignore` (excludes `venv/`, `outputs/`, `history/*.json`, `archive/`, `history.db`)
- Created `requirements.txt` (flask, websocket-client, Pillow, pillow-heif, torch)
- Updated `start.sh` to reference `server_comfy.py` instead of legacy `server.py`

### 2. Environment Setup

#### Conda Environment
```bash
conda create -n qwen_web python=3.12
conda activate qwen_web
pip install -r requirements.txt
```

**Installed packages:**
- flask 3.1.2
- websocket-client 1.9.0
- Pillow 12.2.0
- pillow-heif (latest)
- torch 2.12.1+cu130 (CUDA 13.0)
- sqlite3 (built-in)

#### ComfyUI Setup
- Found existing ComfyUI installation at `~/comfy-blackwell`
- Used existing `comfy-bw` conda environment
- Created symlinks for missing models:
  - `qwen_image_edit_2511_fp8_e4m3fn.safetensors` → `qwen_image_edit_fp8_e4m3fn.safetensors`
  - `Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors` → `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`

### 3. Service Startup

#### ComfyUI (Port 8188)
```bash
eval "$(/home/xh97/miniconda3/bin/conda shell.bash hook)"
conda activate comfy-bw
cd ~/comfy-blackwell
nohup python main.py --port 8188 --auto-launch > /tmp/comfyui.log 2>&1 &
```

#### Flask Web UI (Port 7860)
```bash
eval "$(/home/xh97/miniconda3/bin/conda shell.bash hook)"
conda activate qwen_web
cd ~/qwen_image_edit_web
nohup python3 server_comfy.py > /tmp/qwen-webui.log 2>&1 &
```

### 4. Verification

```bash
# Server status
curl http://127.0.0.1:7860/api/status
# → GPU: NVIDIA GeForce RTX 5090 (33.7GB), ComfyUI: connected

# Queue status
curl http://127.0.0.1:7860/api/queue
# → Empty queue, ready for jobs

# History
curl http://127.0.0.1:7860/api/history
# → 44 records migrated from JSON to SQLite

# Archive
curl http://127.0.0.1:7860/api/archive
# → Empty (new jobs will be archived automatically)
```

---

## Current State

| Service | PID | Port | Status |
|---------|-----|------|--------|
| ComfyUI | 1154158 | 8188 | ✅ Running |
| Flask Web UI | 1154724 | 7860 | ✅ Running |
| llama-server | 1152709 | 8080 | ⚠️ Running (97% GPU) |

**Note:** llama-server is using ~97% GPU. ComfyUI will load models on first job request, which may cause temporary GPU contention.

---

## Access URLs

| Interface | URL |
|-----------|-----|
| Web UI (Local) | http://127.0.0.1:7860 |
| Web UI (LAN) | http://192.168.1.223:7860 |
| ComfyUI API | http://127.0.0.1:8188 |
| llama-server | http://127.0.0.1:8080 |

---

## Logs

| Service | Log File |
|---------|----------|
| ComfyUI | `/tmp/comfyui.log` |
| Flask Web UI | `/tmp/qwen-webui.log` |
| ComfyUI (internal) | `~/comfy-blackwell/user/comfyui.log` |

---

## Directory Structure

```
qwen_image_edit_web/
├── server_comfy.py          # Main Flask server (updated)
├── templates/
│   └── index.html           # Frontend UI
├── outputs/                 # Generated images (cleaned after 7 days)
├── history/                 # Input thumbnails (kept for images)
├── archive/                 # Permanent backup of all completed jobs
│   └── <job_id>/
│       ├── metadata.json    # Full job metadata
│       ├── input.png        # Original input image
│       └── output_*.png     # Generated output images
├── history.db               # SQLite database (44 records)
├── requirements.txt         # Python dependencies
├── .gitignore
├── start.sh                 # Startup script
├── DEPLOY.md                # Deployment guide
└── DEPLOY_STEPS.md          # This file
```

---

## Restart Commands

### Quick Start (Everything in one command)
```bash
~/qwen_image_edit_web/start.sh
```

### Quick Stop (Everything in one command)
```bash
~/qwen_image_edit_web/kill.sh          # Stop Flask + ComfyUI + llama-server
~/qwen_image_edit_web/kill.sh flask    # Stop only Flask
~/qwen_image_edit_web/kill.sh comfyui  # Stop only ComfyUI
~/qwen_image_edit_web/kill.sh llama    # Stop only llama-server
```

### Manual Restart (if needed)
```bash
# Restart Flask Web UI
kill $(lsof -t -i:7860)
eval "$(/home/xh97/miniconda3/bin/conda shell.bash hook)" && conda activate qwen_web
cd ~/qwen_image_edit_web && nohup python3 server_comfy.py > /tmp/qwen-webui.log 2>&1 &

# Restart ComfyUI
kill $(lsof -t -i:8188)
eval "$(/home/xh97/miniconda3/bin/conda shell.bash hook)" && conda activate comfy-bw
cd ~/comfy-blackwell && nohup python main.py --port 8188 --auto-launch > /tmp/comfyui.log 2>&1 &
```

---

## Known Issues

1. **GPU Contention:** llama-server uses ~30.4GB of 32.6GB VRAM. ComfyUI must share the remaining ~2.2GB, causing heavy CPU offloading and slower inference. **Stop llama-server** for optimal image editing performance.
2. **Development Server:** Flask is running in development mode. For production, use gunicorn or waitress.
3. **No systemd:** Services are running with `nohup`. Consider creating systemd units for auto-restart.
