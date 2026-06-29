# Deployment Guide — Qwen-Image-Edit-2511 Web UI

## Quick Start (Local Machine)

```bash
cd ~/qwen_image_edit_web

# 1. Activate virtual environment (or create if first time)
source venv/bin/activate
# or: python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure ComfyUI is running on port 8188
#    (start separately: cd ~/ComfyUI && python main.py --port 8188)

# 4. Start the web server
./start.sh
# or: python3 server_comfy.py
```

Server will be available at: `http://localhost:7860`

## Remote Deployment (RTX 3090 — 100.89.22.74)

### Initial Setup (one-time)

```bash
# SSH into remote machine
ssh xh97-ml@100.89.22.74

# Clone or update the project
cd ~
git clone <repo-url> qwen_image_edit_web
# or: cd ~/qwen_image_edit_web && git pull

# Create virtual environment
python3 -m venv ~/qwen_image_edit_web/venv
source ~/qwen_image_edit_web/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure ComfyUI is installed and models are downloaded
# (ComfyUI should already be running on port 8188)
```

### Start Services

```bash
# Option A: Manual start (foreground)
cd ~/qwen_image_edit_web
source venv/bin/activate
python3 server_comfy.py

# Option B: Background with nohup
cd ~/qwen_image_edit_web
nohup venv/bin/python3 server_comfy.py > server.log 2>&1 &

# Option C: systemd (recommended for production)
# See "Production with systemd" section below
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Flask server port |
| `HOST` | `0.0.0.0` | Bind address |
| `COMFYUI_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFYUI_PORT` | `8188` | ComfyUI port |

```bash
# Example: custom port
PORT=8080 python3 server_comfy.py
```

## Production with systemd

```bash
# Create service file
sudo tee /etc/systemd/system/qwen-webui.service <<EOF
[Unit]
Description=Qwen-Image-Edit-2511 Web UI
After=network.target
Requires=comfyui.service

[Service]
Type=simple
User=xh97-ml
WorkingDirectory=/home/xh97-ml/qwen_image_edit_web
Environment=PORT=7860
Environment=HOST=0.0.0.0
ExecStart=/home/xh97-ml/qwen_image_edit_web/venv/bin/python3 server_comfy.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable qwen-webui
sudo systemctl start qwen-webui

# Check status
sudo systemctl status qwen-webui
journalctl -u qwen-webui -f
```

## Directory Structure

```
qwen_image_edit_web/
├── server_comfy.py          # Main Flask server
├── templates/
│   └── index.html           # Frontend UI
├── outputs/                 # Generated images (cleaned after 7 days)
├── history/                 # Input thumbnails (JSON removed, kept for images)
├── archive/                 # Permanent backup of all completed jobs
│   └── <job_id>/
│       ├── metadata.json    # Full job metadata
│       ├── input.png        # Original input image
│       └── output_*.png     # Generated output images
├── history.db               # SQLite database (job records)
├── requirements.txt         # Python dependencies
├── .gitignore
└── start.sh                 # Startup script
```

## Archive System

Every completed job is automatically archived to `archive/<job_id>/` with:
- `metadata.json` — Full job details (prompt, settings, timing)
- `input.png` — Original input image
- `output_0.png`, `output_1.png`, etc. — All generated output images

**Archive is permanent** — never cleaned up by the cleanup loop.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/archive` | GET | List all archived jobs (summary) |
| `/archive/<job_id>/<file>` | GET | Download archived file |

Example:
```bash
# List all archived jobs
curl http://localhost:7860/api/archive

# Download a specific job's output
curl http://localhost:7860/archive/<job_id>/output_0.png -o result.png

# Download metadata
curl http://localhost:7860/archive/<job_id>/metadata.json
```

## Health Check

```bash
# Check server status
curl http://localhost:7860/api/status

# Check ComfyUI connectivity
curl http://localhost:8188/api/system_stats

# Check queue
curl http://localhost:7860/api/queue

# Check history
curl http://localhost:7860/api/history
```

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :7860

# Check if ComfyUI is running
lsof -i :8188

# Check Python version
python3 --version  # Should be 3.10+

# Check dependencies
pip list | grep -E "flask|websocket|pillow"
```

### ComfyUI connection issues
```bash
# Verify ComfyUI is accessible
curl http://127.0.0.1:8188/api/system_stats

# Check ComfyUI logs
tail -f ~/ComfyUI/comfyui.log
```

### Database issues
```bash
# Check SQLite database
sqlite3 history.db "SELECT count(*) FROM jobs;"

# Verify database integrity
sqlite3 history.db "PRAGMA integrity_check;"
```

### Disk space
```bash
# Check disk usage
du -sh outputs/ history/ archive/

# Clean old outputs (older than 7 days — done automatically)
find outputs/ -name "*.png" -mtime +7 -delete
```

## Backup Strategy

```bash
# Backup archive and database
tar czf qwen-webui-backup-$(date +%Y%m%d).tar.gz \
    history.db archive/ history/

# Restore
tar xzf qwen-webui-backup-YYYYMMDD.tar.gz
```
