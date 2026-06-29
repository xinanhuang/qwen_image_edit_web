# Qwen Image Edit Web UI

A web-based image editing interface powered by ComfyUI, supporting multiple Qwen image models.

| Branch | Model | Description |
|--------|-------|-------------|
| `main` | Qwen-Image-Edit-2511 (FP8) | Standard image edit with Lightning 4-step LoRA |
| `uncensored` | Qwen-Rapid-AIO-NSFW-v19 + Qwen-Image-Edit-2511 (FP8) | Uncensored AIO checkpoint (Lightning) + base model (20-step) with 2-image multi-conditioning |

## Features

- **Image Upload** — Drag & drop or tap to upload (PNG, JPG, WEBP, HEIC, HEIF, DNG, TIFF, BMP)
- **Voice Input (STT)** — Tap 🎤 to speak your edit prompt (faster-whisper INT8, auto-evicting)
- **Real-time Progress** — Step-by-step sampling progress bar via WebSocket
- **Queue System** — Global FIFO queue with per-IP ETA and position tracking
- **Session History** — Persistent SQLite history; per-IP job records
- **Memory Engine** — Self-learning preferences, trending prompts, prompt enhancement
- **Advanced Settings** — Steps, guidance scale, CFG, seed, batch size, Lightning toggle
- **HTTPS Support** — Auto-detected SSL certificates for microphone access
- **Multi-language** — English / 中文 (automatic browser detection)

## Quick Start

### Prerequisites

- **Python 3.12+** with `miniconda3`
- **ComfyUI** installed and running on port 8188
- **NVIDIA GPU** with 24GB+ VRAM (RTX 3090 / RTX 5090 recommended)
- **CUDA 12.x** (for RTX 5090 Blackwell)

### Setup

```bash
cd ~/qwen_image_edit_web

# Create conda env (one-time)
conda create -n qwen_webui python=3.12 -y
conda activate qwen_webui

# Install dependencies
pip install -r requirements.txt

# Start ComfyUI (separate terminal)
cd ~/ComfyUI
conda activate comfyui
python main.py --port 8188 --auto-launch

# Start the web server
cd ~/qwen_image_edit_web
bash start.sh
```

Server will be available at `http://localhost:7860` (or `https://` if SSL certs are present).

### Stop All Services

```bash
bash kill.sh          # Stop Flask + ComfyUI + llama-server
bash kill.sh flask    # Stop only Flask
bash kill.sh comfyui  # Stop only ComfyUI
```

## Branch Details

### `main` — Qwen-Image-Edit-2511 (FP8)

Standard image editing with separate model components:

| Component | File | Size |
|-----------|------|------|
| UNET | `qwen_image_edit_2511_fp8_e4m3fn.safetensors` | ~12 GB |
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~8.8 GB |
| VAE | `qwen_image_vae.safetensors` | ~243 MB |
| LoRA | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors` | ~811 MB |

**VRAM Usage:** ~21.5 GB peak (fits on RTX 3090 24GB)

**Workflow:** UNETLoader → CLIPLoader → VAELoader → LoadImage → ImageScaleToTotalPixels → VAEEncode → TextEncodeQwenImageEditPlus × 2 → LoraLoaderModelOnly → ModelSamplingAuraFlow → CFGNorm → KSampler (euler/simple) → VAEDecode → SaveImage

### `uncensored` — Qwen-Rapid-AIO-NSFW-v19 + Qwen-Image-Edit-2511 (FP8)

Dual-mode workflow with Lightning (AIO) and Base (separate models) support:

**Lightning Mode (⚡ ON):**

| Component | File | Size |
|-----------|------|------|
| AIO Checkpoint | `Qwen-Rapid-AIO-NSFW-v19.safetensors` | ~27 GB |

**Workflow:** CheckpointLoaderSimple → LoadImage × 2 → ResizeAndPadImage (1024×1024) → TextEncodeQwenImageEditPlus × 2 (with image1/image2 conditioning) → EmptyLatentImage (1024×1024) → KSampler (sa_solver/beta, 4 steps, cfg=1.0) → VAEDecode → ImageScale (crop to original aspect ratio) → SaveImage

**Base Mode (⚡ OFF):**

| Component | File | Size |
|-----------|------|------|
| UNET | `qwen_image_edit_2511_fp8_e4m3fn.safetensors` | ~12 GB |
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~8.8 GB |
| VAE | `qwen_image_vae.safetensors` | ~243 MB |

**Workflow:** UNETLoader → CLIPLoader → VAELoader → LoadImage → ImageScaleToTotalPixels (1.0 MP) → VAEEncode → TextEncodeQwenImageEditPlus × 2 (with image1/image2 conditioning) → ModelSamplingAuraFlow (shift=3) → CFGNorm (strength=1) → KSampler (euler/simple, 20 steps, cfg=4.0) → VAEDecode → SaveImage

**Key Differences from `main`:**
- **Dual-mode**: Lightning (AIO checkpoint, 4-step, uncensored) or Base (separate models, 20-step, standard)
- Supports **up to 2 input images** for multi-conditioning
- **Input padding**: Images padded to 1024×1024 square (Lightning mode) for consistent generation
- **Output alignment**: Output cropped to match original input aspect ratio (Lightning mode)
- **Prompt Guide**: Collapsible panel with categorized prompt keywords (consistency, quality, identity, text, objects, lighting, negatives)
- **Face Swap**: BFS Head V5 integration with 2-image input (Image 1 = Body, Image 2 = Face)
- **i18n**: Full English/Chinese support with localized prompt guide
- **HTTPS**: Auto-detected SSL certificates for microphone/STT access

## Architecture

```
┌─────────────┐     HTTP/WS     ┌──────────────┐
│   Browser    │ ◄─────────────► │  Flask API    │
│  (index.html) │                │  (server_     │
│              │                │   comfy.py)   │
└─────────────┘                 └──────┬───────┘
                                       │
                    ┌──────────────────┼────────────┐
                    │                  │            │
              ┌─────▼─────┐   ┌───────▼────┐  ┌────▼────┐
              │  Queue     │   │  SQLite    │  │  Memory │
              │  (FIFO)    │   │  History   │  │  Engine │
              └─────┬─────┘   └────────────┘  └─────────┘
                    │
              ┌─────▼────────────┐
              │   ComfyUI API     │
              │   (port 8188)     │
              └──────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/status` | GET | Server + GPU status |
| `/api/edit` | POST | Submit image edit job (JSON) |
| `/api/progress/<client_id>` | GET | Real-time job progress |
| `/api/queue` | GET | Queue snapshot + ETA |
| `/api/my-job` | GET | Current job for this IP |
| `/api/cancel` | POST | Cancel current job |
| `/api/transcribe` | POST | Speech-to-text (audio file) |
| `/api/convert-image` | POST | Convert HEIC/DNG to PNG |
| `/api/memory/*` | GET | Memory engine endpoints |
| `/api/history` | GET | Job history |
| `/outputs/<filename>` | GET | Output images |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFYUI_PORT` | `8188` | ComfyUI port |
| `PORT` | `7860` | Flask server port |
| `HOST` | `0.0.0.0` | Flask bind address |
| `SSL_CERT` | *(auto-detect)* | PEM certificate path |
| `SSL_KEY` | *(auto-detect)* | PEM key path |
| `STT_COMPUTE` | `int8_float16` | Whisper compute type |
| `STT_DEVICE` | `cuda` | Whisper device |
| `STT_MODEL` | `large-v3-turbo` | Whisper model |

### SSL / HTTPS

Place `*.pem` files (cert + key) in the project directory. The server auto-detects them and enables HTTPS — required for microphone access on most browsers.

## STT (Speech-to-Text)

Uses **faster-whisper** with INT8 quantization via CTranslate2:

- **Compute:** `int8_float16` (default) — ~1.5-1.8 GB VRAM
- **Auto-eviction:** Model unloaded after 120s idle
- **Thread-safe:** Lock prevents race conditions
- **Audio formats:** WebM (Chrome/Firefox), MP4 (Safari/iOS)

### Pre-download Whisper Model

```bash
conda activate qwen_webui
python -c "from stt_engine import get_stt_model; get_stt_model()"
```

## Hardware Requirements

| GPU | VRAM | `main` Branch | `uncensored` Branch |
|-----|------|---------------|---------------------|
| RTX 3090 | 24 GB | ✅ Comfortable | ⚠️ Tight (27 GB model) |
| RTX 4090 | 24 GB | ✅ Comfortable | ⚠️ Tight |
| RTX 5090 | 32 GB | ✅ Comfortable | ✅ Comfortable |

## Project Structure

```
qwen_image_edit_web/
├── server_comfy.py      # Flask API + ComfyUI integration
├── stt_engine.py        # Speech-to-text (faster-whisper)
├── memory_engine.py     # Persistent intelligence / self-learning
├── templates/
│   └── index.html       # Single-page web UI
├── start.sh             # Startup script (ComfyUI + Flask)
├── kill.sh              # Shutdown script
├── requirements.txt     # Python dependencies
├── outputs/             # Generated images (auto-cleaned after 7 days)
├── history/             # Job history files
├── archive/             # Permanent job archive
└── *.db                 # SQLite databases
```

## Troubleshooting

### "ComfyUI image upload failed"
- Ensure ComfyUI is running on port 8188
- Check `/tmp/comfyui.log` for errors
- Verify model files exist in `~/ComfyUI/models/`

### "No progress data"
- Check that node IDs in `server_comfy.py` match your workflow
- `main` branch: KSampler = node "3", SaveImage = node "60"
- `uncensored` branch: KSampler = node "2", SaveImage = node "6"

### Microphone not working
- HTTPS required (place SSL certs in project directory)
- Check browser console for `getUserMedia` errors
- Grant microphone permission in browser settings

### Model OOM (Out of Memory)
- For `main` branch on 24 GB GPU: reduce batch size (`num_images = 1`)
- For `uncensored` branch: RTX 5090 (32 GB) recommended
- Close other GPU processes (llama-server, other ComfyUI instances)

### Browser shows stale UI after changes
- Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Flask dev server sends `Cache-Control: no-cache` headers, but some browsers still cache aggressively

## License

MIT
