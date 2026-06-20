# Code Review — Qwen-Image-Edit-2511 Web UI

**Date:** 2026-06-18
**Scope:** `server_comfy.py` (1049 lines), `templates/index.html` (1964 lines), project structure

---

## Table of Contents

1. [Backend — Bugs](#1-backend---bugs)
2. [Backend — Inefficiencies](#2-backend---inefficiencies)
3. [Backend — Bad Practices](#3-backend---bad-practices)
4. [Frontend — Bugs](#4-frontend---bugs)
5. [Frontend — Inefficiencies](#5-frontend---inefficiencies)
6. [Frontend — Bad Practices](#6-frontend---bad-practices)
7. [UI / UX Improvements](#7-ui--ux-improvements)
8. [Feature Suggestions](#8-feature-suggestions)
9. [Project Structure](#9-project-structure)

---

## 1. Backend — Bugs

### B1: `guidance_scale` never used in workflow
**Severity:** Medium
**Location:** `server_comfy.py` lines 267-272, 387

`build_workflow()` accepts a `guidance` parameter but only uses `cfg_scale`. The `guidance` value is passed from the frontend but never wired into any ComfyUI node. The KSampler only uses `cfg`, and `ModelSamplingAuraFlow` uses `shift`. The guidance scale field in the UI is effectively dead.

```python
def build_workflow(input_image_name, prompt, negative_prompt, num_steps, guidance,
                   cfg_scale, seed, num_images, use_lightning=True):
    # ... guidance is never referenced after this line
```

**Fix:** Either remove the `guidance` parameter and UI field, or wire it into an appropriate node (e.g., as a `shift` override or a model parameter).

### B2: `_convert_dng_to_jpeg` indentation bug
**Severity:** Low (works by accident)
**Location:** `server_comfy.py` lines 688-694

The `if idx >= 0:` block has inconsistent indentation — the inner `if end_idx >= 0:` is indented 16 spaces (4 extra), making it visually misleading but functionally correct since Python only cares about consistency within the same block.

```python
        idx = raw_bytes.find(b'\xff\xd8\xff')
        if idx >= 0:
            end_idx = raw_bytes.find(b'\xff\xd9', idx + 3)
            if end_idx >= 0:        # ← should be 12 spaces, not 16
                    jpeg_data = raw_bytes[idx:end_idx + 2]  # ← 20 spaces
```

### B3: History file `elapsed` can be stale
**Severity:** Low
**Location:** `server_comfy.py` line 76

```python
'elapsed': result_info.get('elapsed') if result_info else None,
```

If `result_info` is `{}` (empty dict but truthy), `elapsed` becomes `None`. Later, the frontend tries to display `r.elapsed || '?'` which shows `?` instead of the actual elapsed time. Not a crash, but a data inconsistency.

### B4: `cleanup_loop` race condition
**Severity:** Low
**Location:** `server_comfy.py` lines 240-251

The cleanup thread deletes entries from `ip_jobs` and `job_progress` while holding `queue_lock`, but the dispatcher thread reads `ip_jobs[ip]` inside `_run_single_job` at line 213 without holding `queue_lock` after releasing it at line 211. A cleanup could delete the IP entry while the dispatcher is mid-job.

### B5: `get_all_history` reads all files every time
**Severity:** Medium (performance)
**Location:** `server_comfy.py` lines 89-102

Every call to `/api/history` or `/api/my-history` reads **all** JSON files from disk, parses them, sorts them, then returns the top `limit`. With 50+ history files, this is O(n) disk reads per request. `get_ip_history` doubles this by calling `get_all_history(limit * 2)` first.

---

## 2. Backend — Inefficiencies

### I1: History stored as individual JSON files on disk
**Severity:** Medium
**Location:** `server_comfy.py` lines 60-78

Each job creates a JSON file + a PNG thumbnail in `history/`. With 50 jobs, that's 50 JSON files + 50 PNGs. `get_all_history()` reads all of them on every request. A SQLite database or in-memory dict with periodic flush would be much faster.

### I2: Output PNGs never cleaned up
**Severity:** Medium
**Location:** `server_comfy.py` lines 220-224

```python
output_path = os.path.join(OUTPUT_DIR, f"{job_id}_{i}.png")
with open(output_path, "wb") as f:
    f.write(img_data)
```

Currently 86 output files consuming 98MB. The cleanup thread only cleans `ip_jobs` and `job_progress` in memory — never the actual files. Over time, `outputs/` will grow unbounded.

### I3: Image base64 stored in queue entry
**Severity:** Low
**Location:** `server_comfy.py` line 130

```python
'image_b64': image_b64,
```

The full base64-encoded image (potentially 200KB+) is stored in the queue entry dict. For a single-job queue this is fine, but if the queue grows, memory usage scales linearly with image size.

### I4: `upload_image_to_comfyui` always uploads as PNG
**Severity:** Low
**Location:** `server_comfy.py` lines 347-370

The function decodes base64 and uploads as `input_{uuid}.png`. If the original was already JPEG, this adds a PNG encode/decode cycle. Could detect format and preserve it.

### I5: WebSocket reconnection is fragile
**Severity:** Medium
**Location:** `server_comfy.py` lines 395-465

The WebSocket connection has a 5-second timeout and a fallback to the history API. If the WebSocket drops mid-sampling, the code waits for `sampling_done` before checking history. If `sampling_done` is never set (e.g., the KSampler node ID changes), the loop hangs until the 600-second frontend timeout.

---

## 3. Backend — Bad Practices

### P1: Global mutable state without type hints
**Severity:** Low
**Location:** `server_comfy.py` lines 41-45

```python
job_queue = []
running_job = None
ip_jobs = {}
job_progress = {}
queue_lock = threading.Lock()
queue_event = threading.Event()
```

All global, no type hints, no documentation of expected structure. Makes maintenance harder.

### P2: `save_job_history` called 4 times in `_run_single_job`
**Severity:** Low
**Location:** `server_comfy.py` lines 217, 239, 260, 275

The same function is called in success, error, no-output, and exception paths with nearly identical arguments. Could be extracted into a single `finalize_job()` helper.

### P3: `AVG_JOB_DURATION` updated without bounds check
**Severity:** Low
**Location:** `server_comfy.py` line 229

```python
AVG_JOB_DURATION = AVG_JOB_DURATION * 0.85 + elapsed * 0.15
```

If a job takes 300s (model loading on first run), the average jumps to ~45s. Subsequent jobs will be underestimated. Should cap or use exponential moving average with a floor/ceiling.

### P4: `pillow_heif` imported twice
**Severity:** Trivial
**Location:** `server_comfy.py` lines 26 and 734

Imported at module level (line 26) and again inside the `convert_image` function (line 734). The local import is redundant.

### P5: No request validation for image_base64 size
**Severity:** Low
**Location:** `server_comfy.py` lines 970-975

The `image_b64` field is accepted without size validation. A malicious client could send a 50MB base64 string, consuming memory and disk. Flask's `MAX_CONTENT_LENGTH` is 100MB, but the JSON body isn't counted against it the same way.

---

## 4. Frontend — Bugs

### F1: `isMine` parameter is stale in `historyItemHtml`
**Severity:** Medium
**Location:** `index.html` lines 1752-1754

The `isMine` boolean passed to `historyItemHtml()` is `true` for My History and `false` for All Jobs. But the function also checks `r.ip === myIP` separately. In My History, `isMine=true` but `r.ip === myIP` may fail if `myIP` hasn't been set yet (race condition). The dual check is inconsistent.

### F2: `p.blurred` CSS class defined but never used
**Severity:** Trivial
**Location:** `index.html` line 658

```css
.hm-section p.blurred { filter: blur(4px); user-select: none; }
```

The detail modal now uses `isOwner` to show/hide content, not blur. This CSS class is dead code.

### F3: `loadMyHistory` and `loadAllHistory` reload on every tab click
**Severity:** Low
**Location:** `index.html` lines 1703-1705

```javascript
if (tab.dataset.panel === 'panel-my-history') loadMyHistory();
if (tab.dataset.panel === 'panel-all-history') loadAllHistory();
```

No caching. Every time the user clicks the tab, it re-fetches from the server. Should cache the result and only reload on explicit action.

### F4: `checkActiveJob` runs every 10s for 2 minutes then stops
**Severity:** Low
**Location:** `index.html` lines 1466-1468

```javascript
setInterval(checkActiveJob, 10000);
setTimeout(() => { activeJobStopped = true; }, 120000);
```

After 2 minutes, `activeJobStopped = true` and the interval stops checking. If the user leaves the tab open and comes back after 3 minutes, auto-resume is dead. Should use `requestIdleCallback` or keep checking with lower frequency.

### F5: `sessionSeed` set but `seedEdited` not initialized from it
**Severity:** Low
**Location:** `index.html` lines 1193-1195

```javascript
const sessionSeed = Math.floor(Math.random() * 999999999);
document.getElementById('seed').value = sessionSeed;
```

But `seedEdited` is `false` by default. So when the user clicks Generate, the auto-random seed logic generates a **new** random seed, overwriting `sessionSeed`. The `sessionSeed` variable is effectively unused.

### F6: `showHistoryDetail` shows duplicate status in non-owner modal
**Severity:** Low
**Location:** `index.html` lines 1824-1845

For non-owner jobs, the modal shows:
1. A "Status" section with status + elapsed
2. A "Time" section with status + queued_at + completed_at

The status badge appears twice. Should merge into one section.

---

## 5. Frontend — Inefficiencies

### FI1: All CSS in a single `<style>` block (800+ lines)
**Severity:** Low
**Location:** `index.html` lines 7-800

No CSS framework, no external stylesheet. Makes the HTML file large and hard to maintain. For a single-page app this is acceptable, but extracting to `static/style.css` would enable browser caching.

### FI2: `i18n` dictionary is a massive inline object
**Severity:** Low
**Location:** `index.html` lines 970-1160

~200 lines of i18n strings inline in the HTML. Could be a separate JSON file loaded on demand.

### FI3: `escHtml` creates a DOM element per call
**Severity:** Trivial
**Location:** `index.html` line 1663

```javascript
function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
```

Works fine but creates garbage. A regex-based escape (`s.replace(/&/g, '&amp;').replace(/</g, '&lt;')...`) would be faster for high-frequency calls.

### FI4: Queue status polled every 3 seconds
**Severity:** Low
**Location:** `index.html` line 1668

```javascript
setInterval(updateQueueStatus, 3000);
```

With a single-job queue, 3s polling is reasonable. But the endpoint returns the full queue state including all entries. Could use WebSocket or Server-Sent Events for real-time updates.

---

## 6. Frontend — Bad Practices

### FP1: Inline styles everywhere
**Severity:** Low
**Location:** Multiple

```html
<button ... style="background:var(--accent);color:#fff;border:none;border-radius:6px;padding:4px 8px;font-size:0.9rem;cursor:pointer;flex-shrink:0">
<div style="display:flex;align-items:center;gap:10px;">
```

Should use CSS classes for consistency and maintainability.

### FP2: `onclick` handlers in innerHTML strings
**Severity:** Low
**Location:** `index.html` line 1818

```javascript
onclick="downloadUrl('${outUrl}', 'qwen-edit-${jobId}.png')"
```

Mixes event delegation with inline handlers. Could use `addEventListener` on the modal container.

### FP3: No error boundary for `generate()`
**Severity:** Low
**Location:** `index.html` lines 1520-1620

The `generate()` function has a try/catch, but if `pollProgress` throws an unhandled exception mid-poll, the UI gets stuck with `isGenerating = true` and the button stays disabled.

### FP4: `localStorage` and `sessionStorage` used without try/catch everywhere
**Severity:** Trivial
**Location:** `index.html` lines 1163, 1286

```javascript
localStorage.setItem('qwen_lang', currentLang);
sessionStorage.setItem('qwen_uploaded_image', selectedImageData);
```

Only the restore function has try/catch. The setItem calls can throw `QuotaExceededError` on mobile browsers with strict storage limits.

---

## 7. UI / UX Improvements

### UI1: Layout — Settings should be collapsible by default
**Priority:** High

The Advanced Settings section is collapsed by default, which is good. But the Negative Prompt toggle is also collapsed. For power users, the negative prompt should be more prominent since it's a core feature, not an "advanced" one.

**Suggestion:** Move Negative Prompt above Advanced Settings, or make it always visible with a smaller textarea.

### UI2: Layout — Result section should show input/output side by side
**Priority:** Medium

Currently only the result image is shown. Users want to compare input vs output.

**Suggestion:** Show a before/after comparison (side-by-side or slider) in the result section.

### UI3: Layout — Queue panel and Queue status are duplicated
**Priority:** Medium

There's a `queue-status` div (lines 865-900) and a `panel-queue` div (lines 950-970) that show the same data. The `queue-status` div is hidden when there are no queued jobs, but the `panel-queue` tab is always accessible. This is confusing.

**Suggestion:** Merge into a single queue view in the tabs panel.

### UI4: Layout — Language toggle is at the bottom
**Priority:** Low

The language toggle is below all content. Users switching languages have to scroll to the bottom.

**Suggestion:** Move to the header area next to the title.

### UI5: Layout — No loading state for image upload
**Priority:** Medium

When uploading a large HEIC file, `handleFileConvert` sends it to the server but shows no loading indicator. The user sees nothing happen for 2-5 seconds.

**Suggestion:** Show a spinner or "Converting..." message in the drop zone during conversion.

### UI6: Layout — History modal close button is hard to tap on mobile
**Priority:** Low

The `✕` button is small (1.2rem text). On mobile, the tap target is small.

**Suggestion:** Increase tap target to at least 44×44px, or add a "Close" text label.

### UI7: Layout — Prompt examples don't indicate language
**Priority:** Low

The example chips show English text but use `data-prompt-zh` for Chinese. Users in Chinese mode see English chip labels with Chinese translations.

**Suggestion:** The chip labels should be i18n'd (they already use `data-i18n`), but the emoji + text layout could be clearer about which language is active.

### UI8: Layout — No image size/format info shown after upload
**Priority:** Low

After uploading an image, the user sees the preview but doesn't know the dimensions or file size.

**Suggestion:** Show image dimensions below the preview (e.g., "2048×1536 • 1.2MB").

### UI9: Layout — Progress bar disappears too quickly
**Priority:** Low
**Location:** `index.html` line 1619

```javascript
setTimeout(() => progressWrap.classList.remove('visible'), 2000);
```

2 seconds may not be enough time to read the final message on mobile.

**Suggestion:** Extend to 4-5 seconds, or add a "Dismiss" button.

---

## 8. Feature Suggestions

### FEAT1: Image comparison slider (before/after)
**Effort:** Medium

Add a before/after slider in the result section so users can compare the original image with the edited result. This is a standard feature for image editing tools.

### FEAT2: Job notification
**Effort:** Low

When a job completes, use the Web Notifications API to notify the user. Useful when the browser tab is in the background.

### FEAT3: Prompt history / favorites
**Effort:** Medium

Save recent prompts in `localStorage` so users can quickly re-use them. Add a "favorite" button on completed jobs.

### FEAT4: Batch edit
**Effort:** High

Allow uploading multiple images and applying the same prompt to all of them. Queue them sequentially.

### FEAT5: Image crop/rotate before edit
**Effort:** Medium

Add basic image editing tools (crop, rotate, flip) in the upload area so users can prepare their image before sending to the model.

### FEAT6: Auto-detect dominant language in prompt
**Effort:** Low

Detect if the prompt is in Chinese or English and auto-switch the UI language, or at least show a hint.

### FEAT7: Download all results as ZIP
**Effort:** Low

For jobs with `num_images > 1`, allow downloading all results as a ZIP file.

### FEAT8: Rate limiting / API key
**Effort:** Medium

If the server is publicly accessible, add rate limiting per IP to prevent abuse. Simple token bucket: 5 jobs per minute.

### FEAT9: Model warmup indicator
**Effort:** Low

The first job takes ~118s (model loading). Show a "Model warming up — first job may take 60-120s" message prominently.

### FEAT10: Share result (public link)
**Effort:** Medium

Generate a shareable link to a completed job's result image. Useful for social media sharing.

---

## 9. Project Structure

### PS1: `venv/` inside project directory
**Severity:** Medium

The virtual environment (`venv/`) is inside the project directory, taking up significant space and cluttering `find` results. Should be at `/home/xh97-ml/qwen-image-edit-web/.venv` or `/home/xh97-ml/venvs/qwen-webui`.

### PS2: `start.sh` references `server.py` (legacy)
**Severity:** Medium

`start.sh` runs `python3 server.py` but the active server is `server_comfy.py`. The old `server.py` is the diffusers-based version that's no longer used.

### PS3: `outputs/` and `history/` not version-controlled
**Severity:** Low

No `.gitignore` file visible. `outputs/` and `history/` should be in `.gitignore` along with `venv/`.

### PS4: No `requirements.txt` or `pyproject.toml`
**Severity:** Medium

Dependencies (Flask, websocket-client, Pillow, pillow-heif, torch) are installed manually. A `requirements.txt` would make deployment reproducible.

### PS5: No systemd service or process manager
**Severity:** Medium

The server is started with `nohup` and a background `&`. If the machine reboots, the server is down. A systemd unit file would provide auto-restart and logging.

---

## Summary by Priority

| Priority | Count | Key Items |
|----------|-------|-----------|
| **High** | 1 | UI1: Negative prompt placement |
| **Medium** | 8 | B5: History file reads, I1/I2: Storage cleanup, F1: isMine race, UI2: Before/after, UI3: Duplicate queue, FEAT1: Comparison slider, PS2: start.sh, PS5: systemd |
| **Low** | 15+ | Various code quality, UX polish, and feature suggestions |
