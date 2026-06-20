#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 Web Server
Mobile-friendly web UI for image editing with Qwen-Image-Edit-2511.
Uses device_map="auto" for GPU+CPU offloading (fits on 24GB GPUs like RTX 3090).
"""

import io
import os
import sys
import time
import uuid
import base64
import threading
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# --- Model Loading (lazy) ---
pipeline = None
pipeline_lock = threading.Lock()  # Fix S6: use threading.Lock instead of boolean flag
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fix S1: Job tracking for progress polling (matches server_comfy.py pattern)
job_progress = {}

# Fix S8: IP-based job tracking for auto-resume on page refresh
ip_jobs = {}
ip_jobs_lock = threading.Lock()


def get_client_ip():
    '''Get the real client IP, handling proxy headers.'''
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()


def get_my_job(ip):
    '''Get the current or recent job for an IP address.'''
    with ip_jobs_lock:
        return ip_jobs.get(ip)


def update_ip_job(ip, status, **kwargs):
    '''Update the status of a job for an IP.'''
    with ip_jobs_lock:
        if ip in ip_jobs:
            ip_jobs[ip]['status'] = status
            ip_jobs[ip].update(kwargs)


def register_ip_job(ip, client_id, job_id, prompt):
    '''Register a new job for an IP. Returns True if successful.'''
    with ip_jobs_lock:
        existing = ip_jobs.get(ip)
        if existing and existing['status'] in ('running', 'queued'):
            return False
        ip_jobs[ip] = {
            'job_id': job_id,
            'client_id': client_id,
            'prompt': prompt[:100],
            'status': 'running',
        }
        return True


def cleanup_old_ip_jobs():
    '''Run periodically to remove stale job entries.'''
    while True:
        time.sleep(300)  # every 5 minutes
        cutoff = time.time() - 300
        with ip_jobs_lock:
            to_remove = [
                ip for ip, job in ip_jobs.items()
                if job['status'] in ('complete', 'error', 'cancelled', 'done')
                and job.get('completed_at', cutoff) < cutoff
            ]
            for ip in to_remove:
                del ip_jobs[ip]


def get_pipeline():
    """Load the pipeline on first request (lazy loading).

    Uses device_map='auto' to automatically partition the model:
    - text_encoder + VAE stay on GPU (fast)
    - transformer blocks are offloaded to CPU during inference
    This allows the model to run on 24GB GPUs (RTX 3090) without OOM.
    """
    global pipeline
    # Fix S6: proper lock-based synchronization
    with pipeline_lock:
        if pipeline is None:
            print(f"[INFO] Loading model {MODEL_ID}...")
            print(f"[INFO] Using device_map='auto' for GPU+CPU offloading")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}GB)")
            try:
                from diffusers import QwenImageEditPlusPipeline
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory={0: "22GB", "cpu": "64GB"},
                )
                pipeline.set_progress_bar_config(disable=False)
                print(f"[INFO] Model loaded with device_map='auto'.")
                # Fix S7: print only top-level module placement, not every sub-module
                for name, module in pipeline.named_children():
                    if hasattr(module, 'weight') and module.weight is not None:
                        dev = str(module.weight.device)
                        size_mb = round(module.weight.element_size() * module.weight.numel() / 1e6, 1)
                        print(f"  {name}: {dev} ({size_mb}MB)")
                    elif hasattr(module, 'named_children'):
                        total_params = sum(p.numel() for p in module.parameters())
                        if total_params > 0:
                            dev = str(next(module.parameters()).device)
                            size_mb = round(total_params * 2 / 1e6, 1)  # bfloat16 = 2 bytes
                            print(f"  {name}: {dev} ({size_mb}MB)")
            except Exception as e:
                print(f"[WARN] device_map='auto' failed: {e}")
                print(f"[INFO] Falling back to manual device placement...")
                from diffusers import QwenImageEditPlusPipeline
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                )
                # Place text_encoder and VAE on GPU, transformer on CPU
                pipeline.text_encoder = pipeline.text_encoder.to("cuda")
                pipeline.vae = pipeline.vae.to("cuda")
                pipeline.transformer = pipeline.transformer.to("cpu")
                pipeline.scheduler = pipeline.scheduler.to("cuda")
                pipeline.set_progress_bar_config(disable=False)
                print(f"[INFO] Model loaded with manual offloading (text_encoder+VAE on GPU, transformer on CPU).")
    return pipeline


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    """Return server status including model loading state."""
    return jsonify({
        "model_loaded": pipeline is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
    })


# Fix S1: Add /api/progress/<client_id> endpoint (was missing, causing 10-min timeout)
@app.route("/api/progress/<client_id>", methods=["GET"])
def get_progress(client_id):
    """Get progress for a running job."""
    if client_id in job_progress:
        return jsonify(job_progress[client_id])
    return jsonify({"status": "idle", "message": "No progress data"})


# Fix S9: Add /api/my-job endpoint for auto-resume on page refresh
@app.route("/api/my-job", methods=["GET"])
def my_job():
    """Get the current or recent job for this IP address."""
    client_ip = get_client_ip()
    job = get_my_job(client_ip)
    if not job:
        return jsonify({"status": "none"})

    client_id = job.get("client_id")
    progress = job_progress.get(client_id, {})

    result = {
        "status": job["status"],
        "client_id": client_id,
        "job_id": job.get("job_id"),
        "prompt": job.get("prompt"),
    }

    if job["status"] in ("running", "queued"):
        result.update(progress)
        result["status"] = progress.get("status", "running")
        result["message"] = progress.get("message", "Job running...")
        if "percent" in progress:
            result["percent"] = progress["percent"]
        if "step" in progress:
            result["step"] = progress["step"]
        if "total" in progress:
            result["total"] = progress["total"]
    elif "result" in progress:
        result["result"] = progress["result"]
    elif "message" in progress:
        result["message"] = progress["message"]

    return jsonify(result)


# Fix S10: Add /api/cancel-job endpoint
@app.route("/api/cancel-job", methods=["POST"])
def cancel_job():
    """Cancel the current job for this IP."""
    client_ip = get_client_ip()
    job = get_my_job(client_ip)
    if not job or job["status"] not in ("running", "queued"):
        return jsonify({"status": "none"})

    update_ip_job(client_ip, "cancelled")
    client_id = job["client_id"]
    if client_id in job_progress:
        job_progress[client_id].update({
            "status": "cancelled",
            "message": "Cancelled by user",
        })

    return jsonify({"status": "cancelled", "job_id": job["job_id"]})


def run_edit_job(client_id, images, prompt, negative_prompt, num_inference_steps,
                 guidance_scale, true_cfg_scale, seed, num_images, job_id, client_ip):
    """Background thread to run the diffusers pipeline."""
    start_time = time.time()
    try:
        pipe = get_pipeline()

        # Update progress: model loaded, starting inference
        job_progress[client_id].update({
            "status": "sampling",
            "message": f"Sampling ({num_inference_steps} steps)...",
            "step": 0,
            "total": num_inference_steps,
            "percent": 0
        })

        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
        }

        with torch.inference_mode():
            output = pipe(**inputs)

        elapsed = round(time.time() - start_time, 1)

        # Save and return results
        results = []
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
        for i, img in enumerate(output.images):
            if i == 0:
                img.save(output_path)
            else:
                extra_path = os.path.join(OUTPUT_DIR, f"{job_id}_{i}.png")
                img.save(extra_path)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            results.append(f"data:image/png;base64,{b64}")

        print(f"[{job_id}] Done in {elapsed}s")
        job_progress[client_id].update({
            "status": "complete",
            "message": f"Done in {elapsed}s",
            "elapsed": elapsed,
            "percent": 100,
            "result": {
                "success": True,
                "job_id": job_id,
                "elapsed": elapsed,
                "images": results,
                "output_url": f"/outputs/{job_id}.png",
            }
        })
        # Mark IP job as complete with timestamp for cleanup
        update_ip_job(client_ip, "complete", completed_at=time.time())

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        print(f"[{job_id}] Error after {elapsed}s: {e}")
        import traceback
        traceback.print_exc()
        job_progress[client_id].update({
            "status": "error",
            "message": str(e),
            "elapsed": elapsed,
            "result": {
                "success": False,
                "error": str(e),
                "elapsed": elapsed,
            }
        })


@app.route("/api/edit", methods=["POST"])
def edit_image():
    """
    Edit an image using Qwen-Image-Edit-2511.

    Expected form data:
    - image_base64: base64 encoded image (required)
    - prompt: Edit description (required)
    - negative_prompt: Optional negative prompt
    - num_inference_steps: Default 40
    - guidance_scale: Default 1.0
    - true_cfg_scale: Default 4.0
    - seed: Default 42
    - num_images: Default 1
    """
    global pipeline

    data = request.form.to_dict()
    prompt = data.get("prompt", "").strip()
    negative_prompt = data.get("negative_prompt", " ").strip() or " "

    try:
        num_inference_steps = int(data.get("num_inference_steps", 40))
        guidance_scale = float(data.get("guidance_scale", 1.0))
        true_cfg_scale = float(data.get("true_cfg_scale", 4.0))
        seed = int(data.get("seed", 42))
        num_images = int(data.get("num_images", 1))
    except (ValueError, TypeError):
        num_inference_steps = 40
        guidance_scale = 1.0
        true_cfg_scale = 4.0
        seed = 42
        num_images = 1

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Get uploaded images
    images = []
    uploaded_files = request.files.getlist("image")
    if not uploaded_files or not uploaded_files[0].filename:
        image_b64 = data.get("image_base64", "")
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            img_data = base64.b64decode(image_b64)
            images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
        else:
            return jsonify({"error": "Image is required"}), 400
    else:
        for f in uploaded_files:
            if f and f.filename:
                img = Image.open(f).convert("RGB")
                images.append(img)

    if not images:
        return jsonify({"error": "No valid image provided"}), 400

    # Fix S2: Use client_id + background thread pattern (matches frontend polling)
    job_id = str(uuid.uuid4())[:8]
    client_id = str(uuid.uuid4())
    client_ip = get_client_ip()

    # Check if this IP already has a running job
    existing = get_my_job(client_ip)
    if existing and existing["status"] in ("running", "queued"):
        existing_cid = existing.get("client_id")
        existing_progress = job_progress.get(existing_cid, {})
        if existing_progress.get("status") not in ("complete", "error", "cancelled"):
            return jsonify({
                "client_id": existing["client_id"],
                "job_id": existing["job_id"],
                "status": "resumed",
                "message": "Job already running for this IP",
            })
        # Job is done but ip_jobs not updated yet — clear it
        update_ip_job(client_ip, "done")

    # Register this job for the IP
    if not register_ip_job(client_ip, client_id, job_id, prompt):
        return jsonify({"error": "Another job is already running for this IP"}), 409

    print(f"\n[{job_id}] [{client_ip}] Editing: {prompt[:80]}... ({len(images)} image(s))")

    # Initialize progress tracking
    job_progress[client_id] = {
        "step": 0,
        "total": num_inference_steps,
        "status": "loading",
        "message": "Loading model..." if pipeline is None else "Starting inference...",
    }

    # Start background thread
    thread = threading.Thread(
        target=run_edit_job,
        args=(client_id, images, prompt, negative_prompt, num_inference_steps,
              guidance_scale, true_cfg_scale, seed, num_images, job_id, client_ip),
        daemon=True
    )
    thread.start()

    return jsonify({
        "client_id": client_id,
        "job_id": job_id,
        "status": "queued",
    })


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n{'='*60}")
    print(f"  Qwen-Image-Edit-2511 Web Server")
    print(f"  Model: {MODEL_ID}")
    print(f"  Mode: device_map='auto' (GPU+CPU offloading)")
    print(f"{'='*60}")
    # Start periodic cleanup of old job entries
    cleanup_thread = threading.Thread(target=cleanup_old_ip_jobs, daemon=True)
    cleanup_thread.start()

    print(f"\nStarting server on http://{host}:{port}")
    print(f"Open this URL on your phone or browser to begin.\n")
    print(f"Model will load on first edit request (~60-120s).\n")
    app.run(host=host, port=port, threaded=True, debug=False)
