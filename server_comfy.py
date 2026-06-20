#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 Web Server (ComfyUI Backend)
Uses ComfyUI with fp8 models that fit on 24GB GPUs (RTX 3090).

Queue system: single global FIFO queue with dispatcher thread.
Every user sees their queue position and ETA regardless of IP.
Session history: all jobs persisted; each IP can view their own history.
"""

import io
import os
import sys
import json
import time
import uuid
import base64
import socket
import shutil
import threading
import urllib.request
import urllib.parse
import websocket  # websocket-client
import torch
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# --- Config ---
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Average job duration (seconds) — used for ETA estimation.
AVG_JOB_DURATION = 45.0
avg_duration_lock = threading.Lock()

# ===================================================================
# Global FIFO Queue
# ===================================================================
job_queue = []
running_job = None
ip_jobs = {}
job_progress = {}
queue_lock = threading.Lock()
queue_event = threading.Event()


def get_client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()


# ------------------------------------------------------------------
# History helpers
# ------------------------------------------------------------------

def save_job_history(job_id, client_id, ip, prompt, negative_prompt, num_inference_steps,
                     guidance_scale, true_cfg_scale, seed, num_images, use_lightning,
                     image_b64_thumb, status, result_info, queued_at, completed_at):
    """Persist a job record to disk as JSON."""
    record = {
        'job_id': job_id,
        'client_id': client_id,
        'ip': ip,
        'prompt': prompt[:500],
        'negative_prompt': (negative_prompt or '')[:200],
        'settings': {
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'true_cfg_scale': true_cfg_scale,
            'seed': seed,
            'num_images': num_images,
            'use_lightning': use_lightning,
        },
        'status': status,
        'result_info': result_info,
        'queued_at': queued_at,
        'completed_at': completed_at,
        'elapsed': result_info.get('elapsed') if result_info else None,
    }
    path = os.path.join(HISTORY_DIR, f"{job_id}.json")
    try:
        with open(path, 'w') as f:
            json.dump(record, f, default=str)
    except Exception as e:
        print(f"[HISTORY] Save error for {job_id}: {e}")

    # Save thumbnail of input image (first 30KB of base64)
    if image_b64_thumb:
        thumb_path = os.path.join(HISTORY_DIR, f"{job_id}_input.png")
        try:
            raw = base64.b64decode(image_b64_thumb[:40000])
            with open(thumb_path, 'wb') as f:
                f.write(raw)
        except Exception:
            pass


def get_all_history(limit=100):
    """Return list of all job history records, newest first."""
    records = []
    if not os.path.isdir(HISTORY_DIR):
        return records
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith('.json'):
            path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(path) as f:
                    rec = json.load(f)
                records.append(rec)
            except Exception:
                pass
    records.sort(key=lambda r: r.get('queued_at', 0), reverse=True)
    return records[:limit]


def get_ip_history(ip, limit=100):
    """Return history records for a specific IP."""
    all_recs = get_all_history(limit * 2)
    return [r for r in all_recs if r.get('ip') == ip][:limit]


# ------------------------------------------------------------------
# Queue helpers  (all hold queue_lock)
# ------------------------------------------------------------------

def enqueue_job(ip, client_id, job_id, prompt, workflow, image_b64):
    with queue_lock:
        entry = {
            'job_id': job_id,
            'client_id': client_id,
            'ip': ip,
            'prompt': prompt[:100],
            'workflow': workflow,
            'image_b64': image_b64,
            'queued_at': time.time(),
        }
        job_queue.append(entry)
        pos = len(job_queue)
        ip_jobs[ip] = {
            'job_id': job_id,
            'client_id': client_id,
            'prompt': prompt[:100],
            'status': 'queued',
            'queue_pos': pos,
        }
        for other_ip, other_job in ip_jobs.items():
            if other_ip != ip and other_job['status'] == 'queued':
                other_jid = other_job['job_id']
                for i, e in enumerate(job_queue):
                    if e['job_id'] == other_jid:
                        other_job['queue_pos'] = i + 1
                        break
        queue_event.set()
        return pos


def dequeue_job():
    with queue_lock:
        if not job_queue:
            return None
        entry = job_queue.pop(0)
        for i, e in enumerate(job_queue):
            eip = e['ip']
            if eip in ip_jobs and ip_jobs[eip]['status'] == 'queued':
                ip_jobs[eip]['queue_pos'] = i + 1
        return entry


def get_queue_snapshot(ip=None):
    with queue_lock:
        snapshot = {
            'running': None,
            'queue_length': len(job_queue),
            'ahead': 0,
            'eta_seconds': 0,
            'entries': [],
        }
        if running_job:
            cid = running_job.get('client_id')
            prog = job_progress.get(cid, {})
            snapshot['running'] = {
                'job_id': running_job['job_id'],
                'ip': running_job['ip'],
                'prompt': running_job['prompt'],
                'status': prog.get('status', 'running'),
                'percent': prog.get('percent', 0),
            }
        for i, e in enumerate(job_queue):
            prog = job_progress.get(e.get('client_id'), {})
            snapshot['entries'].append({
                'position': i + 1,
                'job_id': e['job_id'],
                'ip': e['ip'],
                'prompt': e['prompt'],
                'status': prog.get('status', 'queued'),
                'percent': prog.get('percent', 0),
            })
        if ip:
            ahead = 0
            if running_job and running_job.get('ip') != ip:
                ahead += 1
            for e in job_queue:
                if e['ip'] != ip:
                    ahead += 1
                else:
                    break
            snapshot['ahead'] = ahead
            snapshot['eta_seconds'] = round(ahead * AVG_JOB_DURATION, 0)
        return snapshot


def get_my_job(ip):
    with queue_lock:
        return dict(ip_jobs.get(ip)) if ip in ip_jobs else None


def update_ip_job_status(ip, status, **kw):
    with queue_lock:
        if ip in ip_jobs:
            ip_jobs[ip]['status'] = status
            ip_jobs[ip].update(kw)


# ------------------------------------------------------------------
# Dispatcher thread
# ------------------------------------------------------------------

def _run_single_job(entry):
    global running_job
    client_id = entry['client_id']
    job_id = entry['job_id']
    ip = entry['ip']
    workflow = entry['workflow']
    queued_at = entry['queued_at']

    # Extract settings from workflow for history
    neg_prompt = workflow.get("102", {}).get("inputs", {}).get("prompt", "")
    num_steps = workflow.get("3", {}).get("inputs", {}).get("steps", 20)
    seed = workflow.get("3", {}).get("inputs", {}).get("seed", 42)
    cfg = workflow.get("3", {}).get("inputs", {}).get("cfg", 1.0)
    num_images = 1
    if "94" in workflow:
        num_images = workflow["94"].get("inputs", {}).get("amount", 1)
    use_lightning = "89" in workflow

    with queue_lock:
        running_job = entry
        ip_jobs[ip]['status'] = 'running'
        ip_jobs[ip]['queue_pos'] = 0

    print(f"\n[DISPATCH] [{job_id}] [{ip}] Running: {entry['prompt'][:80]}...")

    input_filename = upload_image_to_comfyui(entry['image_b64'])
    if input_filename is None:
        job_progress[client_id] = {
            'status': 'error', 'message': 'ComfyUI image upload failed',
            'result': {'success': False, 'error': 'ComfyUI image upload failed'},
            '_ts': time.time(),
        }
        update_ip_job_status(ip, 'error', completed_at=time.time())
        save_job_history(job_id, client_id, ip, entry['prompt'], neg_prompt,
                        num_steps, cfg, cfg, seed, num_images, use_lightning,
                        entry['image_b64'][:3000], 'error',
                        {'error': 'ComfyUI image upload failed'}, queued_at, time.time())
        with queue_lock:
            running_job = None
        return

    workflow["78"]["inputs"]["image"] = input_filename

    try:
        output_images = run_comfyui_workflow(workflow, client_id)
        elapsed = round(time.time() - queued_at, 1)

        with avg_duration_lock:
            global AVG_JOB_DURATION
            AVG_JOB_DURATION = AVG_JOB_DURATION * 0.85 + elapsed * 0.15

        completed_at = time.time()

        if not output_images:
            job_progress[client_id] = {
                'status': 'error', 'message': 'No output images', 'elapsed': elapsed,
                'result': {'success': False, 'error': 'No output images', 'elapsed': elapsed},
                '_ts': time.time(),
            }
            update_ip_job_status(ip, 'error', completed_at=completed_at)
            save_job_history(job_id, client_id, ip, entry['prompt'], neg_prompt,
                            num_steps, cfg, cfg, seed, num_images, use_lightning,
                            entry['image_b64'][:3000], 'error',
                            {'error': 'No output images', 'elapsed': elapsed}, queued_at, completed_at)
        else:
            results = []
            output_paths = []
            for i, img_data in enumerate(output_images):
                output_path = os.path.join(OUTPUT_DIR, f"{job_id}_{i}.png")
                with open(output_path, "wb") as f:
                    f.write(img_data)
                output_paths.append(f"/outputs/{job_id}_{i}.png")
                b64 = base64.b64encode(img_data).decode()
                results.append(f"data:image/png;base64,{b64}")

            print(f"[{job_id}] Done in {elapsed}s")
            result_info = {
                'success': True, 'job_id': job_id, 'elapsed': elapsed,
                'images': results,
                'output_url': f"/outputs/{job_id}_0.png",
                'output_paths': output_paths,
            }
            job_progress[client_id] = {
                'status': 'complete', 'message': f"Done in {elapsed}s",
                'elapsed': elapsed, 'percent': 100,
                'result': result_info,
                '_ts': time.time(),
            }
            update_ip_job_status(ip, 'complete', completed_at=completed_at)
            save_job_history(job_id, client_id, ip, entry['prompt'], neg_prompt,
                            num_steps, cfg, cfg, seed, num_images, use_lightning,
                            entry['image_b64'][:3000], 'complete',
                            {'success': True, 'elapsed': elapsed, 'output_paths': output_paths},
                            queued_at, completed_at)

    except Exception as e:
        elapsed = round(time.time() - queued_at, 1)
        completed_at = time.time()
        print(f"[{job_id}] Error after {elapsed}s: {e}")
        import traceback; traceback.print_exc()
        job_progress[client_id] = {
            'status': 'error', 'message': str(e), 'elapsed': elapsed,
            'result': {'success': False, 'error': str(e), 'elapsed': elapsed},
            '_ts': time.time(),
        }
        update_ip_job_status(ip, 'error', completed_at=completed_at)
        save_job_history(job_id, client_id, ip, entry['prompt'], neg_prompt,
                        num_steps, cfg, cfg, seed, num_images, use_lightning,
                        entry['image_b64'][:3000], 'error',
                        {'error': str(e), 'elapsed': elapsed}, queued_at, completed_at)
    finally:
        with queue_lock:
            running_job = None


def dispatcher_loop():
    while True:
        queue_event.wait(timeout=1.0)
        entry = dequeue_job()
        if entry:
            queue_event.clear()
            _run_single_job(entry)


# ------------------------------------------------------------------
# Cleanup thread
# ------------------------------------------------------------------

def cleanup_loop():
    while True:
        time.sleep(300)
        cutoff = time.time() - 300
        with queue_lock:
            stale = [ip for ip, j in ip_jobs.items()
                     if j['status'] in ('complete', 'error', 'cancelled')
                     and j.get('completed_at', cutoff) < cutoff]
            for ip in stale:
                del ip_jobs[ip]
            stale_cid = [cid for cid, p in job_progress.items()
                         if p.get('status') in ('complete', 'error')
                         and p.get('_ts') and (time.time() - p['_ts']) > 300]
            for cid in stale_cid:
                del job_progress[cid]


# ===================================================================
# ComfyUI helpers
# ===================================================================

def build_workflow(input_image_name, prompt, negative_prompt, num_steps, guidance,
                   cfg_scale, seed, num_images, use_lightning=True):
    if use_lightning:
        num_steps = 4
        cfg_scale = 1.0

    workflow = {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "qwen_image_edit_2511_fp8_e4m3fn.safetensors",
                        "weight_dtype": "fp8_e4m3fn"}
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "type": "qwen_image", "device": "default"}
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        "78": {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name}
        },
        "93": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"upscale_method": "lanczos", "megapixels": 1.0,
                        "resolution_steps": 1, "image": ["78", 0]}
        },
        "88": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["93", 0], "vae": ["39", 0]}
        },
        "101": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": prompt, "clip": ["38", 0], "vae": ["39", 0],
                        "image1": ["93", 0]}
        },
        "102": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": negative_prompt, "clip": ["38", 0], "vae": ["39", 0],
                        "image1": ["93", 0]}
        },
    }

    if use_lightning:
        workflow["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["37", 0],
                        "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors",
                        "strength_model": 1.0}
        }
        workflow["66"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"shift": 3, "model": ["89", 0]}
        }
    else:
        workflow["66"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"shift": 3, "model": ["37", 0]}
        }

    workflow["75"] = {
        "class_type": "CFGNorm",
        "inputs": {"strength": 1, "model": ["66", 0]}
    }

    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed, "steps": num_steps, "cfg": cfg_scale,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["75", 0], "positive": ["101", 0], "negative": ["102", 0],
            "latent_image": ["88", 0],
        }
    }

    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["39", 0]}
    }

    workflow["60"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "qwen_edit"}
    }

    if num_images > 1:
        workflow["94"] = {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["88", 0], "amount": num_images}
        }
        workflow["3"]["inputs"]["latent_image"] = ["94", 0]

    return workflow


def upload_image_to_comfyui(image_b64):
    img_data = base64.b64decode(image_b64)
    filename = f"input_{uuid.uuid4().hex[:8]}.png"

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + img_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="type"\r\n\r\n'
        f"input\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    upload_req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upload_req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"[UPLOAD] {filename} -> {result}")
            return filename
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        return None


def run_comfyui_workflow(workflow, client_id):
    prompt_data = {"prompt": workflow, "client_id": client_id}
    req = urllib.request.Request(
        f"{COMFYUI_URL}/api/prompt",
        data=json.dumps(prompt_data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        prompt_id = json.loads(resp.read())["prompt_id"]
    print(f"[PROMPT] ID: {prompt_id}")

    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws?clientId={client_id}")
    ws.settimeout(5.0)

    output_images = []
    total_steps = workflow.get("3", {}).get("inputs", {}).get("steps", 20)
    sampling_done = False

    job_progress[client_id] = {
        "step": 0, "total": total_steps, "status": "running",
        "message": "Starting...", "_ts": time.time(),
    }

    try:
        while True:
            try:
                out = ws.recv()
            except (websocket._exceptions.WebSocketTimeoutException, TimeoutError, socket.timeout):
                if sampling_done:
                    history_req = urllib.request.Request(f"{COMFYUI_URL}/api/history/{prompt_id}")
                    with urllib.request.urlopen(history_req, timeout=10) as resp:
                        history = json.loads(resp.read())
                    if prompt_id in history and "outputs" in history[prompt_id]:
                        break
                out = None

            if out is None:
                time.sleep(0.5)
                continue

            if isinstance(out, str):
                msg = json.loads(out)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "progress_state":
                    nodes = data.get("nodes", {})
                    for nid, nstate in nodes.items():
                        if nid == "3":
                            value = int(nstate.get("value", 0))
                            max_val = int(nstate.get("max", total_steps))
                            pct = round(value / max_val * 100, 1) if max_val > 0 else 0
                            job_progress[client_id].update({
                                "step": value, "total": max_val, "status": "sampling",
                                "message": f"Step {value}/{max_val} ({pct}%)",
                                "percent": pct,
                            })
                            print(f"[PROGRESS] Step {value}/{max_val} ({pct}%)")
                            if value >= max_val:
                                sampling_done = True
                            break
                        elif nstate.get("state") == "running":
                            job_progress[client_id].update({
                                "status": "running", "message": f"Running node {nid}..."
                            })
                        elif nstate.get("state") == "finished" and nid == "60":
                            job_progress[client_id].update({
                                "status": "saving", "message": "Saving result...", "percent": 100
                            })

                elif msg_type == "executing":
                    node_id = data.get("node")
                    stage = data.get("stage")
                    if node_id is None and stage == "complete":
                        time.sleep(2)
                        break
                    elif stage == "executed" and node_id == "60":
                        time.sleep(2)
                        break
                    elif node_id == "3":
                        job_progress[client_id].update({
                            "status": "sampling", "message": "Sampling started...",
                            "step": 0, "total": total_steps,
                        })
            elif isinstance(out, bytes):
                pass

        time.sleep(2)
        for attempt in range(5):
            try:
                history_req = urllib.request.Request(f"{COMFYUI_URL}/api/history/{prompt_id}")
                with urllib.request.urlopen(history_req, timeout=30) as resp:
                    history = json.loads(resp.read())
                if prompt_id in history and "outputs" in history[prompt_id]:
                    outputs = history[prompt_id]["outputs"]
                    for node_output in outputs.values():
                        if "images" in node_output:
                            for img_info in node_output["images"]:
                                img_url = (f"{COMFYUI_URL}/view?filename={img_info['filename']}"
                                           f"&subfolder={img_info.get('subfolder', '')}"
                                           f"&type={img_info.get('type', 'output')}")
                                with urllib.request.urlopen(img_url) as img_resp:
                                    output_images.append(img_resp.read())
                    break
                print(f"[WS] History not ready, attempt {attempt+1}/5")
                time.sleep(2)
            except Exception as e:
                print(f"[WS] History fetch error: {e}")
                time.sleep(2)
    finally:
        ws.close()

    return output_images


# ===================================================================
# Flask Routes
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    comfyui_ready = False
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/api/system_stats")
        with urllib.request.urlopen(req, timeout=5) as resp:
            comfyui_ready = resp.status == 200
    except:
        pass
    return jsonify({
        "model_loaded": comfyui_ready,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
        "comfyui_url": COMFYUI_URL,
    })



# ------------------------------------------------------------------
# Image format conversion endpoint (HEIC, DNG, TIFF, BMP -> JPEG)
# ------------------------------------------------------------------

def _convert_dng_to_jpeg(raw_bytes, max_edge=2048):
    """Convert DNG to JPEG by extracting embedded preview or using Pillow."""
    # Try Pillow first (some DNG files are TIFF-based)
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        w, h = img.size
        if w > max_edge or h > max_edge:
            ratio = min(max_edge / w, max_edge / h)
            w, h = int(w * ratio), int(h * ratio)
            img = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        return buf.getvalue()
    except Exception:
        pass

    # Try to extract embedded JPEG preview from DNG
    # DNG files often contain a JPEG preview in the IFD
    import struct
    try:
        # Find JPEG markers in the file
        # JPEG SOI marker: FF D8 FF
        idx = raw_bytes.find(b'\xff\xd8\xff')
        if idx >= 0:
            # Find the end of the JPEG (FF D9)
            end_idx = raw_bytes.find(b'\xff\xd9', idx + 3)
            if end_idx >= 0:
                    jpeg_data = raw_bytes[idx:end_idx + 2]
                    if len(jpeg_data) > 1000:  # Valid JPEG preview
                        img = Image.open(io.BytesIO(jpeg_data))
                        img.load()
                        if img.mode not in ('RGB', 'RGBA'):
                            img = img.convert('RGB')
                        w, h = img.size
                        if w > max_edge or h > max_edge:
                            ratio = min(max_edge / w, max_edge / h)
                            w, h = int(w * ratio), int(h * ratio)
                            img = img.resize((w, h), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format='JPEG', quality=95)
                        return buf.getvalue()
    except Exception:
        pass

    return None


@app.route("/api/convert-image", methods=["POST"])
def convert_image():
    """Convert HEIC/HEIF/DNG/TIFF/BMP images to JPEG data URL.
    Accepts raw file bytes, returns JSON with base64-encoded JPEG."""
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image file provided"}), 400

    raw_bytes = file.read()
    filename = file.filename or ''
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    max_edge = 2048
    img = None

    try:
        if ext in ('heic', 'heif', 'heics', 'heifs'):
            # Use pillow-heif for HEIC/HEIF
            import pillow_heif
            heif_file = pillow_heif.read_heif(io.BytesIO(raw_bytes))
            img = heif_file.to_pillow()
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

        elif ext in ('dng',):
            # DNG: try Pillow first, then embedded JPEG preview
            jpeg_bytes = _convert_dng_to_jpeg(raw_bytes, max_edge)
            if jpeg_bytes:
                b64 = base64.b64encode(jpeg_bytes).decode()
                return jsonify({"success": True, "image": f"data:image/jpeg;base64,{b64}"})
            # If all fails, return error
            return jsonify({"error": "DNG conversion failed (no embedded preview)"}), 400

        elif ext in ('tif', 'tiff'):
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')

        elif ext in ('bmp',):
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')

        elif ext in ('webp',):
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

        else:
            # Try generic Pillow open
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

    except Exception as e:
        print(f"[CONVERT] Error converting {filename}: {e}")
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 400

    if img is None:
        return jsonify({"error": "Unknown format"}), 400

    # Resize if needed
    w, h = img.size
    if w > max_edge or h > max_edge:
        ratio = min(max_edge / w, max_edge / h)
        w, h = int(w * ratio), int(h * ratio)
        img = img.resize((w, h), Image.LANCZOS)

    # Convert to JPEG
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode()

    print(f"[CONVERT] {filename} -> JPEG ({w}x{h}, {len(jpeg_bytes)} bytes)")
    return jsonify({"success": True, "image": f"data:image/jpeg;base64,{b64}"})


@app.route("/api/progress/<client_id>", methods=["GET"])
def get_progress(client_id):
    with queue_lock:
        prog = dict(job_progress.get(client_id, {"status": "idle", "message": "No progress data"}))
    prog.pop('_ts', None)
    return jsonify(prog)


@app.route("/api/my-job", methods=["GET"])
def my_job():
    client_ip = get_client_ip()
    job = get_my_job(client_ip)
    if not job:
        return jsonify({"status": "none", "ip": client_ip})

    client_id = job.get("client_id")
    with queue_lock:
        progress = dict(job_progress.get(client_id, {}))
    progress.pop('_ts', None)

    result = {
        "ip": client_ip,
        "status": job["status"],
        "client_id": client_id,
        "job_id": job.get("job_id"),
        "prompt": job.get("prompt"),
    }

    if job["status"] in ("running", "queued"):
        result["status"] = progress.get("status", job["status"])
        result["message"] = progress.get("message", "Waiting...")
        for key in ("percent", "step", "total"):
            if key in progress:
                result[key] = progress[key]
    elif job["status"] == "complete":
        r = progress.get("result")
        if r:
            result["result"] = r
            result["message"] = progress.get("message", "")
    elif job["status"] == "error":
        result["message"] = progress.get("message", "Job failed")
        r = progress.get("result")
        if r:
            result["result"] = r

    return jsonify(result)


@app.route("/api/queue", methods=["GET"])
def queue_endpoint():
    client_ip = get_client_ip()
    snap = get_queue_snapshot(client_ip)
    return jsonify(snap)


@app.route("/api/history", methods=["GET"])
def history_endpoint():
    """Get all job history (summary view for everyone)."""
    limit = int(request.args.get('limit', 100))
    records = get_all_history(limit)
    # Strip heavy image data from summary
    for r in records:
        r.pop('settings', None)
        r.pop('negative_prompt', None)
        ri = r.get('result_info', {})
        ri.pop('output_paths', None)
        r['result_info'] = ri
    return jsonify({"count": len(records), "records": records})


@app.route("/api/my-history", methods=["GET"])
def my_history_endpoint():
    """Get job history for this IP (full detail including settings and results)."""
    client_ip = get_client_ip()
    limit = int(request.args.get('limit', 100))
    records = get_ip_history(client_ip, limit)
    return jsonify({"ip": client_ip, "count": len(records), "records": records})


@app.route("/api/history/<job_id>", methods=["GET"])
def history_job_endpoint(job_id):
    """Get a single job's history record. Full detail only for the originating IP."""
    client_ip = get_client_ip()
    path = os.path.join(HISTORY_DIR, f"{job_id}.json")
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        record = json.load(f)
    # If IP matches, return full record; otherwise strip sensitive fields
    if record.get('ip') != client_ip:
        record.pop('settings', None)
        record.pop('negative_prompt', None)
        result_info = record.get('result_info', {})
        result_info.pop('images', None)
        result_info.pop('output_paths', None)
        record['result_info'] = result_info
    return jsonify(record)


@app.route("/api/cancel-job", methods=["POST"])
def cancel_job():
    client_ip = get_client_ip()
    job = get_my_job(client_ip)
    if not job or job["status"] not in ("running", "queued"):
        return jsonify({"status": "none", "ip": client_ip})

    job_id = job["job_id"]
    client_id = job["client_id"]

    # Remove from queue if still queued
    with queue_lock:
        job_queue[:] = [e for e in job_queue if e["client_id"] != client_id]
        # Update positions for remaining queued jobs
        for i, e in enumerate(job_queue):
            eip = e["ip"]
            if eip in ip_jobs and ip_jobs[eip]["status"] == "queued":
                ip_jobs[eip]["queue_pos"] = i + 1
        if client_id in job_progress:
            job_progress[client_id].update({
                "status": "cancelled", "message": "Cancelled by user",
                "_ts": time.time(),
            })

    update_ip_job_status(client_ip, "cancelled", completed_at=time.time())
    print(f"[CANCEL] [{job_id}] [{client_ip}] Cancelled by user")
    return jsonify({"status": "cancelled", "job_id": job_id})


@app.route("/api/edit", methods=["POST"])
def edit_image():
    # Accept both JSON and FormData (FormData breaks on iOS Safari with large base64)
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()
    prompt = data.get("prompt", "").strip()
    negative_prompt = data.get("negative_prompt", " ").strip() or " "

    try:
        num_inference_steps = int(data.get("num_inference_steps", 20))
        guidance_scale = float(data.get("guidance_scale", 1.0))
        true_cfg_scale = float(data.get("true_cfg_scale", 4.0))
        seed = int(data.get("seed", 42))
        num_images = int(data.get("num_images", 1))
        ul = data.get("use_lightning", "true")
        use_lightning = ul if isinstance(ul, bool) else str(ul).lower() == "true"
    except (ValueError, TypeError):
        num_inference_steps = 20
        guidance_scale = 1.0
        true_cfg_scale = 4.0
        seed = 42
        num_images = 1
        use_lightning = True

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    image_b64 = data.get("image_base64", "")
    if image_b64 and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    if not image_b64:
        return jsonify({"error": "Image is required"}), 400

    client_ip = get_client_ip()
    fn = data.get("force_new", "false"); force_new = fn if isinstance(fn, bool) else str(fn).lower() == "true"

    existing = get_my_job(client_ip)
    if existing and existing["status"] in ("running", "queued"):
        existing_cid = existing.get("client_id")
        with queue_lock:
            ep = job_progress.get(existing_cid, {})
        if ep.get("status") not in ("complete", "error", "cancelled"):
            if force_new:
                # Cancel old job atomically and proceed with new one
                old_jid = existing["job_id"]
                # Remove old job from queue
                job_queue[:] = [e for e in job_queue if e["client_id"] != existing_cid]
                # Update positions for remaining queued jobs
                for i, e in enumerate(job_queue):
                    eip = e["ip"]
                    if eip in ip_jobs and ip_jobs[eip]["status"] == "queued":
                        ip_jobs[eip]["queue_pos"] = i + 1
                if existing_cid in job_progress:
                    job_progress[existing_cid].update({
                        "status": "cancelled", "message": "Cancelled (replaced by new job)",
                        "_ts": time.time(),
                    })
                update_ip_job_status(client_ip, "cancelled", completed_at=time.time())
                print(f"[CANCEL] [{old_jid}] [{client_ip}] Cancelled (replaced by new job)")
                # Fall through to enqueue the new job
            else:
                return jsonify({
                    "client_id": existing["client_id"],
                    "job_id": existing["job_id"],
                    "status": "resumed",
                    "message": "Job already in queue for this IP",
                })

    job_id = str(uuid.uuid4())[:8]
    client_id = str(uuid.uuid4())

    workflow = build_workflow(
        "placeholder.png", prompt, negative_prompt,
        num_inference_steps, guidance_scale, true_cfg_scale, seed, num_images,
        use_lightning,
    )

    pos = enqueue_job(client_ip, client_id, job_id, prompt, workflow, image_b64)
    snap = get_queue_snapshot(client_ip)

    print(f"[{job_id}] [{client_ip}] Queued (pos={pos}): {prompt[:80]}...")

    return jsonify({
        "client_id": client_id,
        "job_id": job_id,
        "status": "queued",
        "queue_position": pos,
        "queue_length": snap['queue_length'],
        "ahead": snap['ahead'],
        "eta_seconds": int(snap['eta_seconds']),
    })


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/history/<filename>")
def serve_history_file(filename):
    """Serve history assets (input thumbnails, etc.)."""
    return send_from_directory(HISTORY_DIR, filename)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n{'='*60}")
    print(f"  Qwen-Image-Edit-2511 Web Server (ComfyUI Backend)")
    print(f"  ComfyUI: {COMFYUI_URL}")
    print(f"{'='*60}")
    print(f"\n  Global FIFO queue enabled — one job at a time")
    print(f"  Session history: {HISTORY_DIR}")
    print(f"  ETA estimation: /api/queue endpoint\n")

    disp_thread = threading.Thread(target=dispatcher_loop, daemon=True, name="dispatcher")
    disp_thread.start()
    print("[OK] Dispatcher thread started")

    clean_thread = threading.Thread(target=cleanup_loop, daemon=True, name="cleanup")
    clean_thread.start()
    print("[OK] Cleanup thread started")

    print(f"\nStarting server on http://{host}:{port}")
    print(f"Open this URL on your phone or browser to begin.\n")
    app.run(host=host, port=port, threaded=True, debug=False)
