#!/usr/bin/env python3
"""
Qwen-Rapid-AIO Web Server (ComfyUI Backend)
Uses ComfyUI with Qwen-Rapid-AIO-NSFW-v19 single-checkpoint model.
Supports up to 2 input images for image editing.

Queue system: single global FIFO queue with dispatcher thread.
Every user sees their queue position and ETA regardless of IP.
Session history: all jobs persisted; each IP can view their own history.
"""

import glob
import io
import os
import sys
import json
import time
import uuid
import base64
import socket
import shutil
import sqlite3
import tempfile
import threading
import urllib.request
import urllib.parse
import websocket  # websocket-client
import torch
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
from flask import Flask, request, jsonify, render_template, send_from_directory

# --- Persistent Intelligence / Memory Engine ---
from memory_engine import (
    record_job_outcome,
    get_suggestions_for_ip,
    get_user_preferences,
    get_trending_prompts,
    get_memory_stats,
    get_recent_insights,
    get_smart_default_settings,
    search_memory,
    enhance_prompt,
    categorize_prompt,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# --- Config ---
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "archive")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Average job duration (seconds) — used for ETA estimation.
AVG_JOB_DURATION = 45.0
avg_duration_lock = threading.Lock()

# --- SQLite History Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def _get_db():
    """Get a thread-local SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _ensure_db():
    """Create the jobs table if it doesn't exist."""
    conn = _get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            client_id TEXT,
            ip TEXT,
            prompt TEXT,
            negative_prompt TEXT,
            num_inference_steps INTEGER,
            guidance_scale REAL,
            true_cfg_scale REAL,
            seed INTEGER,
            num_images INTEGER,
            use_lightning BOOLEAN,
            status TEXT,
            result_info TEXT,
            queued_at REAL,
            completed_at REAL,
            elapsed REAL
        )
    ''')
    conn.commit()
    conn.close()

_ensure_db()

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
    """Persist a job record to SQLite."""
    elapsed = result_info.get('elapsed') if isinstance(result_info, dict) else None
    conn = _get_db()
    try:
        conn.execute('''
            INSERT OR REPLACE INTO jobs
            (job_id, client_id, ip, prompt, negative_prompt,
             num_inference_steps, guidance_scale, true_cfg_scale,
             seed, num_images, use_lightning, status, result_info,
             queued_at, completed_at, elapsed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id, client_id, ip,
            prompt[:500], (negative_prompt or '')[:200],
            num_inference_steps, guidance_scale, true_cfg_scale,
            seed, num_images, use_lightning,
            status, json.dumps(result_info, default=str),
            queued_at, completed_at, elapsed
        ))
        conn.commit()
    except Exception as e:
        print(f"[HISTORY] Save error for {job_id}: {e}")
    finally:
        conn.close()

    # Save thumbnail of input image (first 30KB of base64)
    if image_b64_thumb:
        thumb_path = os.path.join(HISTORY_DIR, f"{job_id}_input.png")
        try:
            raw = base64.b64decode(image_b64_thumb[:40000])
            with open(thumb_path, 'wb') as f:
                f.write(raw)
        except Exception:
            pass


def archive_job(job_id, client_id, ip, prompt, negative_prompt, num_inference_steps,
                guidance_scale, true_cfg_scale, seed, num_images, use_lightning,
                image_b64_thumb, status, result_info, queued_at, completed_at, output_paths=None):
    """Archive completed job: input image, all output PNGs, and metadata JSON.
    Stored in archive/<job_id>/ — never cleaned up by the cleanup loop."""
    job_dir = os.path.join(ARCHIVE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save input image
    if image_b64_thumb:
        try:
            raw = base64.b64decode(image_b64_thumb[:40000])
            with open(os.path.join(job_dir, "input.png"), 'wb') as f:
                f.write(raw)
        except Exception:
            pass

    # Save output images (copy from outputs/)
    if output_paths:
        for i, rel_path in enumerate(output_paths):
            src = os.path.join(os.path.dirname(__file__), rel_path.lstrip('/'))
            dst = os.path.join(job_dir, f"output_{i}.png")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[ARCHIVE] Copy error {src} -> {dst}: {e}")

    # Save metadata JSON
    meta = {
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
        'elapsed': result_info.get('elapsed') if isinstance(result_info, dict) else None,
        'archived_at': time.time(),
    }
    try:
        with open(os.path.join(job_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as e:
        print(f"[ARCHIVE] Metadata save error for {job_id}: {e}")


def _row_to_dict(row):
    """Convert a sqlite3.Row to a dict with nested settings."""
    d = dict(row)
    # Flatten into the same structure the frontend expects
    d['settings'] = {
        'num_inference_steps': d.pop('num_inference_steps'),
        'guidance_scale': d.pop('guidance_scale'),
        'true_cfg_scale': d.pop('true_cfg_scale'),
        'seed': d.pop('seed'),
        'num_images': d.pop('num_images'),
        'use_lightning': d.pop('use_lightning'),
    }
    # Parse result_info JSON
    ri = d.get('result_info')
    if isinstance(ri, str):
        try:
            d['result_info'] = json.loads(ri)
        except (json.JSONDecodeError, TypeError):
            d['result_info'] = {}
    return d


def get_all_history(limit=100):
    """Return list of all job history records, newest first."""
    conn = _get_db()
    try:
        rows = conn.execute(
            'SELECT * FROM jobs ORDER BY queued_at DESC LIMIT ?', (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[HISTORY] Read error: {e}")
        return []
    finally:
        conn.close()


def get_ip_history(ip, limit=100):
    """Return history records for a specific IP."""
    conn = _get_db()
    try:
        rows = conn.execute(
            'SELECT * FROM jobs WHERE ip = ? ORDER BY queued_at DESC LIMIT ?',
            (ip, limit)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[HISTORY] Read error: {e}")
        return []
    finally:
        conn.close()


def get_job_history(job_id):
    """Return a single job record by ID."""
    conn = _get_db()
    try:
        row = conn.execute(
            'SELECT * FROM jobs WHERE job_id = ?', (job_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    except Exception as e:
        print(f"[HISTORY] Read error: {e}")
        return None
    finally:
        conn.close()


# ------------------------------------------------------------------
# Queue helpers  (all hold queue_lock)
# ------------------------------------------------------------------

def enqueue_job(ip, client_id, job_id, prompt, workflow, image_b64_1, image_b64_2=""):
    with queue_lock:
        entry = {
            'job_id': job_id,
            'client_id': client_id,
            'ip': ip,
            'prompt': prompt[:100],
            'workflow': workflow,
            'image_b64_1': image_b64_1,
            'image_b64_2': image_b64_2,
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

def _finalize_job(job_id, client_id, ip, prompt, neg_prompt,
                  num_steps, guidance, cfg, seed, num_images, use_lightning,
                  image_b64_thumb, status, result_info, queued_at, completed_at):
    """Consolidated helper: update progress, IP status, persist history, and archive."""
    elapsed = result_info.get('elapsed') if isinstance(result_info, dict) else None

    # --- Persistent Intelligence: Record job outcome for learning ---
    settings_for_memory = {
        'num_inference_steps': num_steps,
        'guidance_scale': guidance,
        'true_cfg_scale': cfg,
        'seed': seed,
        'num_images': num_images,
        'use_lightning': use_lightning,
    }
    try:
        record_job_outcome(job_id, ip, prompt, settings_for_memory,
                           status, elapsed, completed_at)
    except Exception as e:
        print(f"[MEMORY] Record outcome error for {job_id}: {e}")
    if status == 'complete':
        job_progress[client_id] = {
            'status': 'complete', 'message': f"Done in {elapsed}s" if elapsed else 'Done',
            'elapsed': elapsed, 'percent': 100,
            'result': result_info,
            '_ts': time.time(),
        }
        update_ip_job_status(ip, 'complete', completed_at=completed_at)
    else:
        job_progress[client_id] = {
            'status': 'error', 'message': result_info.get('error', 'Job failed'),
            'elapsed': elapsed,
            'result': {'success': False, 'error': result_info.get('error', 'Job failed'), 'elapsed': elapsed},
            '_ts': time.time(),
        }
        update_ip_job_status(ip, 'error', completed_at=completed_at)

    save_job_history(job_id, client_id, ip, prompt, neg_prompt,
                    num_steps, guidance, cfg, seed, num_images, use_lightning,
                    image_b64_thumb, status, result_info, queued_at, completed_at)

    # Archive completed jobs: input + output PNGs + metadata JSON
    if status == 'complete':
        output_paths = result_info.get('output_paths', []) if isinstance(result_info, dict) else []
        archive_job(job_id, client_id, ip, prompt, neg_prompt,
                   num_steps, guidance, cfg, seed, num_images, use_lightning,
                   image_b64_thumb, status, result_info, queued_at, completed_at,
                   output_paths=output_paths)

    with queue_lock:
        global running_job
        running_job = None


def _run_single_job(entry):
    global running_job
    client_id = entry['client_id']
    job_id = entry['job_id']
    ip = entry['ip']
    workflow = entry['workflow']
    queued_at = entry['queued_at']

    # Extract settings from workflow for history (new node IDs for Qwen-Rapid-AIO)
    neg_prompt = workflow.get("4", {}).get("inputs", {}).get("prompt", "")
    num_steps = workflow.get("2", {}).get("inputs", {}).get("steps", 4)
    seed = workflow.get("2", {}).get("inputs", {}).get("seed", 42)
    cfg = workflow.get("2", {}).get("inputs", {}).get("cfg", 1.0)
    num_images = workflow.get("9", {}).get("inputs", {}).get("batch_size", 1)
    use_lightning = True  # Qwen-Rapid-AIO always uses lightning-like fast sampling

    with queue_lock:
        running_job = entry
        ip_jobs[ip]['status'] = 'running'
        ip_jobs[ip]['queue_pos'] = 0

    print(f"\n[DISPATCH] [{job_id}] [{ip}] Running: {entry['prompt'][:80]}...")

    # Upload up to 2 images
    image_b64_1 = entry.get('image_b64_1', '')
    image_b64_2 = entry.get('image_b64_2', '')
    input_filenames = []

    if image_b64_1:
        fname1 = upload_image_to_comfyui(image_b64_1)
        if fname1:
            input_filenames.append(fname1)
            # Create LoadImage node if not already in workflow
            if "7" not in workflow:
                workflow["7"] = {"class_type": "LoadImage", "inputs": {"image": fname1}}
            else:
                workflow["7"]["inputs"]["image"] = fname1
            # Wire into TextEncodeQwenImageEditPlus nodes (3=positive, 4=negative)
            for node_id in ("3", "4"):
                if node_id in workflow:
                    workflow[node_id]["inputs"]["image1"] = ["7", 0]

    if image_b64_2:
        fname2 = upload_image_to_comfyui(image_b64_2)
        if fname2:
            input_filenames.append(fname2)
            if "8" not in workflow:
                workflow["8"] = {"class_type": "LoadImage", "inputs": {"image": fname2}}
            else:
                workflow["8"]["inputs"]["image"] = fname2
            for node_id in ("3", "4"):
                if node_id in workflow:
                    workflow[node_id]["inputs"]["image2"] = ["8", 0]

    if not input_filenames:
        _finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
                      num_steps, cfg, cfg, seed, num_images, use_lightning,
                      image_b64_1[:3000], 'error',
                      {'error': 'ComfyUI image upload failed'}, queued_at, time.time())
        return

    try:
        output_images = run_comfyui_workflow(workflow, client_id)
        elapsed = round(time.time() - queued_at, 1)

        with avg_duration_lock:
            global AVG_JOB_DURATION
            # EMA with floor/ceiling to prevent outlier jobs from skewing the average
            raw_avg = AVG_JOB_DURATION * 0.85 + elapsed * 0.15
            AVG_JOB_DURATION = max(10.0, min(raw_avg, 300.0))

        completed_at = time.time()

        if not output_images:
            _finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
                          num_steps, cfg, cfg, seed, num_images, use_lightning,
                          image_b64_1[:3000], 'error',
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
            # Pass the full result_info (with images) to _finalize_job
            # so that job_progress contains the images for the frontend
            _finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
                          num_steps, cfg, cfg, seed, num_images, use_lightning,
                          image_b64_1[:3000], 'complete',
                          result_info,
                          queued_at, completed_at)

    except Exception as e:
        elapsed = round(time.time() - queued_at, 1)
        completed_at = time.time()
        print(f"[{job_id}] Error after {elapsed}s: {e}")
        import traceback; traceback.print_exc()
        _finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
                      num_steps, cfg, cfg, seed, num_images, use_lightning,
                      image_b64_1[:3000], 'error',
                      {'error': str(e), 'elapsed': elapsed}, queued_at, completed_at)


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
    """Periodic cleanup of stale in-memory state and old output files."""
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

        # Clean up old output PNG files (older than 7 days)
        try:
            cutoff_ts = time.time() - 7 * 86400  # 7 days
            for fname in os.listdir(OUTPUT_DIR):
                fpath = os.path.join(OUTPUT_DIR, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff_ts:
                    os.remove(fpath)
        except Exception as e:
            print(f"[CLEANUP] Output dir cleanup error: {e}")


# ===================================================================
# ComfyUI helpers
# ===================================================================

def build_workflow(input_image_names, prompt, negative_prompt, num_steps, guidance,
                   cfg_scale, seed, num_images, use_lightning=True):
    """Build ComfyUI workflow for Qwen-Rapid-AIO (single checkpoint, 2 image inputs).
    
    input_image_names: list of uploaded image filenames (1 or 2 images)
    """
    if use_lightning:
        num_steps = 4
        cfg_scale = 1.0

    # CheckpointLoaderSimple loads the AIO model (MODEL + CLIP + VAE in one file)
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "Qwen-Rapid-AIO-NSFW-v19.safetensors"}
        },
    }

    # Load up to 2 input images
    img1_name = input_image_names[0] if len(input_image_names) > 0 else ""
    img2_name = input_image_names[1] if len(input_image_names) > 1 else ""

    if img1_name:
        workflow["7"] = {
            "class_type": "LoadImage",
            "inputs": {"image": img1_name}
        }
    if img2_name:
        workflow["8"] = {
            "class_type": "LoadImage",
            "inputs": {"image": img2_name}
        }

    # Build image references for TextEncodeQwenImageEditPlus
    # image1 and image2 are optional — only include if images were uploaded
    pos_inputs = {"prompt": prompt, "clip": ["1", 1], "vae": ["1", 2]}
    neg_inputs = {"prompt": negative_prompt, "clip": ["1", 1], "vae": ["1", 2]}

    if img1_name:
        pos_inputs["image1"] = ["7", 0]
        neg_inputs["image1"] = ["7", 0]
    if img2_name:
        pos_inputs["image2"] = ["8", 0]
        neg_inputs["image2"] = ["8", 0]

    # Positive conditioning (with prompt + images)
    workflow["3"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": pos_inputs
    }
    # Negative conditioning (blank prompt + same images)
    workflow["4"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": neg_inputs
    }

    # Empty latent (768x768) — output resolution
    workflow["9"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 768, "height": 768, "batch_size": num_images}
    }

    # KSampler with sa_solver + beta scheduler (Qwen-Rapid-AIO defaults)
    workflow["2"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed, "steps": num_steps, "cfg": cfg_scale,
            "sampler_name": "sa_solver", "scheduler": "beta", "denoise": 1.0,
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["9", 0],
        }
    }

    # Decode latent → image
    workflow["5"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["2", 0], "vae": ["1", 2]}
    }

    # Save output
    workflow["6"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["5", 0], "filename_prefix": "qwen_rapid_edit"}
    }

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
    total_steps = workflow.get("2", {}).get("inputs", {}).get("steps", 4)
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
                        if nid == "2":  # KSampler
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
                        elif nstate.get("state") == "finished" and nid == "6":  # SaveImage
                            job_progress[client_id].update({
                                "status": "saving", "message": "Saving result...", "percent": 100
                            })

                elif msg_type == "executing":
                    node_id = data.get("node")
                    stage = data.get("stage")
                    if node_id is None and stage == "complete":
                        time.sleep(2)
                        break
                    elif stage == "executed" and node_id == "6":  # SaveImage
                        time.sleep(2)
                        break
                    elif node_id == "2":  # KSampler
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
# Cache Control
# ===================================================================

@app.after_request
def add_cache_headers(response):
    # No-cache for HTML pages (prevents stale JS issues)
    if response.content_type and response.content_type.startswith('text/html'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


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
            # Use pillow-heif for HEIC/HEIF (already imported at module level)
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
    record = get_job_history(job_id)
    if not record:
        return jsonify({"error": "Not found"}), 404
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

    # Accept up to 2 images
    image_b64_1 = data.get("image_base64", "")
    image_b64_2 = data.get("image_base64_2", "")
    if image_b64_1 and "," in image_b64_1:
        image_b64_1 = image_b64_1.split(",", 1)[1]
    if image_b64_2 and "," in image_b64_2:
        image_b64_2 = image_b64_2.split(",", 1)[1]
    if not image_b64_1:
        return jsonify({"error": "At least one image is required"}), 400

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
        [], prompt, negative_prompt,
        num_inference_steps, guidance_scale, true_cfg_scale, seed, num_images,
        use_lightning,
    )

    pos = enqueue_job(client_ip, client_id, job_id, prompt, workflow, image_b64_1, image_b64_2)
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


@app.route("/archive/<path:filepath>")
def serve_archive(filepath):
    """Serve archived job files (input.png, output_*.png, metadata.json)."""
    # filepath is like "<job_id>/input.png" or "<job_id>/metadata.json"
    parts = filepath.rsplit('/', 1)
    if len(parts) == 2:
        job_id, filename = parts
        job_dir = os.path.join(ARCHIVE_DIR, job_id)
        return send_from_directory(job_dir, filename)
    return jsonify({"error": "Not found"}), 404


@app.route("/api/archive", methods=["GET"])
def archive_list():
    """List all archived jobs (summary only)."""
    records = []
    if not os.path.isdir(ARCHIVE_DIR):
        return jsonify({"count": 0, "records": records})
    for job_id in sorted(os.listdir(ARCHIVE_DIR), reverse=True):
        meta_path = os.path.join(ARCHIVE_DIR, job_id, 'metadata.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                records.append({
                    'job_id': meta.get('job_id'),
                    'prompt': meta.get('prompt', '')[:100],
                    'status': meta.get('status'),
                    'elapsed': meta.get('elapsed'),
                    'completed_at': meta.get('completed_at'),
                })
            except Exception:
                pass
    return jsonify({"count": len(records), "records": records})


# ===================================================================
# Persistent Intelligence / Memory API Endpoints
# ===================================================================

@app.route("/api/memory/suggestions", methods=["GET"])
def memory_suggestions():
    """Get personalized prompt suggestions based on user history + trending."""
    client_ip = get_client_ip()
    limit = int(request.args.get('limit', 8))
    suggestions = get_suggestions_for_ip(client_ip, limit)
    return jsonify({"ip": client_ip, "count": len(suggestions), "suggestions": suggestions})


@app.route("/api/memory/preferences", methods=["GET"])
def memory_preferences():
    """Get learned user preferences."""
    client_ip = get_client_ip()
    prefs = get_user_preferences(client_ip)
    return jsonify({"ip": client_ip, "preferences": prefs})


@app.route("/api/memory/trending", methods=["GET"])
def memory_trending():
    """Get trending prompts across all users."""
    limit = int(request.args.get('limit', 10))
    trending = get_trending_prompts(limit)
    return jsonify({"count": len(trending), "trending": trending})


@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    """Get overall memory system statistics."""
    stats = get_memory_stats()
    return jsonify(stats)


@app.route("/api/memory/insights", methods=["GET"])
def memory_insights():
    """Get recent system-generated insights."""
    limit = int(request.args.get('limit', 5))
    insights = get_recent_insights(limit)
    return jsonify({"count": len(insights), "insights": insights})


@app.route("/api/memory/settings", methods=["GET"])
def memory_smart_settings():
    """Get smart default settings based on learned preferences."""
    client_ip = get_client_ip()
    settings = get_smart_default_settings(client_ip)
    return jsonify({"ip": client_ip, "settings": settings})


@app.route("/api/memory/search", methods=["GET"])
def memory_search():
    """Search memory for similar prompts."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    results = search_memory(query, limit)
    return jsonify({"query": query, "count": len(results), "results": results})


@app.route("/api/memory/enhance", methods=["POST"])
def memory_enhance():
    """Enhance a prompt based on learned knowledge."""
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    client_ip = get_client_ip()
    enhanced = enhance_prompt(prompt, client_ip)
    categories = categorize_prompt(prompt)
    return jsonify({
        'original': prompt,
        'enhanced': enhanced,
        'categories': categories,
        'was_enhanced': enhanced != prompt,
    })


# ===================================================================
# STT (Speech-to-Text) Endpoint
# ===================================================================

def _cleanup_stale_temp_files():
    """Remove orphaned temp audio files from crashed sessions (runs at startup)."""
    temp_dir = tempfile.gettempdir()  # Cross-platform: /tmp on Linux, %TEMP% on Windows
    patterns = [
        os.path.join(temp_dir, "recording.*.webm"),
        os.path.join(temp_dir, "recording.*.mp4"),
        os.path.join(temp_dir, "tmp*.webm"),
        os.path.join(temp_dir, "tmp*.mp4"),
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                age = time.time() - os.path.getmtime(f)
                if age > 3600:  # Only delete files older than 1 hour
                    os.unlink(f)
            except OSError:
                pass


@app.before_request
def _stt_startup_cleanup():
    """Run temp file cleanup once on first request."""
    if not hasattr(_stt_startup_cleanup, "_done"):
        _stt_startup_cleanup._done = True
        _cleanup_stale_temp_files()


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    """Accept an audio file, return transcription text.
    Accepts: audio file via FormData ('audio' key) or base64 in JSON.
    Optional: 'language' (e.g. 'en', 'zh'). None = auto-detect.
    """
    from stt_engine import transcribe as stt_transcribe
    tmp_path = None  # Pre-declare to prevent UnboundLocalError in finally block
    data = None      # Pre-declare to prevent UnboundLocalError in language parsing

    try:
        audio_file = request.files.get("audio")
        if audio_file:
            filename = audio_file.filename or "recording.audio"
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            tmp.close()  # Release OS file descriptor lock; path string remains valid
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        elif request.is_json:
            data = request.get_json() or {}
            audio_b64 = data.get("audio_base64", "")
            if audio_b64 and "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]
            if not audio_b64:
                return jsonify({"error": "No audio provided"}), 400
            raw = base64.b64decode(audio_b64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            tmp.write(raw)
            tmp.flush()  # Ensure buffer is written to disk
            tmp.close()  # Release OS file descriptor lock
            tmp_path = tmp.name
        else:
            return jsonify({"error": "No audio file provided"}), 400

        language = request.form.get("language") if not request.is_json else data.get("language")
        if language and language in ("auto", ""):
            language = None

        text = stt_transcribe(tmp_path, language=language)
        return jsonify({"success": True, "text": text})

    except Exception as e:
        print(f"[STT] Transcription error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    cert_file = os.environ.get("SSL_CERT", "")
    key_file = os.environ.get("SSL_KEY", "")

    # Auto-detect SSL certs if not specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not cert_file:
        # Look for *.pem files in the script directory
        pem_files = sorted(glob.glob(os.path.join(script_dir, "*.pem")))
        if len(pem_files) >= 2:
            # Usually the +3-key.pem is the key, the other is the cert
            key_files = [f for f in pem_files if "key" in os.path.basename(f).lower()]
            cert_files = [f for f in pem_files if "key" not in os.path.basename(f).lower()]
            if key_files and cert_files:
                key_file = key_files[0]
                cert_file = cert_files[0]

    use_https = bool(cert_file) and os.path.exists(cert_file)

    # Always start dispatcher and cleanup threads
    disp_thread = threading.Thread(target=dispatcher_loop, daemon=True, name="dispatcher")
    disp_thread.start()
    print("[OK] Dispatcher thread started")

    clean_thread = threading.Thread(target=cleanup_loop, daemon=True, name="cleanup")
    clean_thread.start()
    print("[OK] Cleanup thread started")

    if use_https:
        # Also try without key if only cert exists (combined cert)
        if not key_file or not os.path.exists(key_file):
            key_file = cert_file  # Use same file
        ssl_context = (cert_file, key_file) if key_file != cert_file else cert_file

        # Also listen on HTTP (port-1) for convenience
        http_port = port - 1
        http_thread = threading.Thread(target=lambda: app.run(host=host, port=http_port, threaded=True, debug=False), daemon=True)
        http_thread.start()

        print(f"\n{'='*60}")
        print(f"  Qwen-Image-Edit-2511 Web Server (ComfyUI Backend)")
        print(f"  ComfyUI: {COMFYUI_URL}")
        print(f"  HTTPS: Enabled (cert={cert_file})")
        print(f"{'='*60}")
        print(f"\n  Global FIFO queue enabled — one job at a time")
        print(f"  Session history: {HISTORY_DIR}")
        print(f"  Archive (permanent): {ARCHIVE_DIR}")
        print(f"  Persistent Intelligence: memory.db (self-learning)")
        print(f"  ETA estimation: /api/queue endpoint")
        print(f"  Memory APIs: /api/memory/{{suggestions,preferences,trending,stats,insights,settings,search,enhance}}\n")

        print(f"\nStarting server on https://{host}:{port}")
        print(f"Also available on http://{host}:{http_port}")
        print(f"Open this URL on your phone or browser to begin.\n")
        app.run(host=host, port=port, threaded=True, debug=False, ssl_context=ssl_context)
    else:
        print(f"\n{'='*60}")
        print(f"  Qwen-Image-Edit-2511 Web Server (ComfyUI Backend)")
        print(f"  ComfyUI: {COMFYUI_URL}")
        print(f"{'='*60}")
        print(f"\n  Global FIFO queue enabled — one job at a time")
        print(f"  Session history: {HISTORY_DIR}")
        print(f"  Archive (permanent): {ARCHIVE_DIR}")
        print(f"  Persistent Intelligence: memory.db (self-learning)")
        print(f"  ETA estimation: /api/queue endpoint")
        print(f"  Memory APIs: /api/memory/{{suggestions,preferences,trending,stats,insights,settings,search,enhance}}\n")

        print(f"\nStarting server on http://{host}:{port}")
        print(f"Open this URL on your phone or browser to begin.\n")
        app.run(host=host, port=port, threaded=True, debug=False)
