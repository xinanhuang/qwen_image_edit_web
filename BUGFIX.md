# Bug Fix — "data.images is undefined"

**Date:** 2026-06-20
**Issue:** After image processing completes, frontend shows "Network error: can't access property 0, data.images is undefined"

---

## Root Cause

In `_run_single_job()`, the success path:
1. Builds `result_info` with `'images': results` (base64-encoded images)
2. Sets `job_progress[client_id]` with this full `result_info`
3. Calls `_finalize_job()` with a **simplified** `result_info` that only has `{'success': True, 'elapsed': elapsed, 'output_paths': output_paths}` — **no `'images'` key**
4. `_finalize_job()` **overwrites** `job_progress[client_id]` with the simplified version
5. Frontend polls `/api/progress/<client_id>` and gets the simplified version without `'images'`
6. Frontend tries `data.images[0]` → `data.images` is `undefined` → error

## Fix

Pass the full `result_info` (with images) to `_finalize_job()` in the success path.

**Before:**
```python
_finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
              num_steps, cfg, cfg, seed, num_images, use_lightning,
              entry['image_b64'][:3000], 'complete',
              {'success': True, 'elapsed': elapsed, 'output_paths': output_paths},  # ← no images!
              queued_at, completed_at)
```

**After:**
```python
_finalize_job(job_id, client_id, ip, entry['prompt'], neg_prompt,
              num_steps, cfg, cfg, seed, num_images, use_lightning,
              entry['image_b64'][:3000], 'complete',
              result_info,  # ← full result_info with images
              queued_at, completed_at)
```

---

## GPU Memory Issue

**Current State:**
| Process | GPU Memory |
|---------|-----------|
| llama-server (Qwen3.6-27B) | ~30.4 GB |
| ComfyUI (idle) | ~0.2 GB |
| **Total** | **~30.6 / 32.6 GB** |

**Problem:** llama-server is using ~93% of VRAM. When ComfyUI loads the Qwen-Image-Edit model (~29GB), it must share the remaining ~2.2GB, causing:
- Heavy CPU offloading (slower inference)
- Potential OOM errors
- High memory usage (up to 29GB as reported)

**On the 3090 server:** llama-server is not running, so ComfyUI has the full 24GB available. The model fits with CPU offloading for the overflow, resulting in lower reported VRAM usage (~21.8GB).

**Recommendations:**
1. **Stop llama-server** when running image editing jobs
2. **Use a smaller LLM** (e.g., 7B instead of 27B) if both services need to run simultaneously
3. **Configure ComfyUI** with `--normal-vram` or `--low-vram` flags for better memory management
