# STT Implementation Plan

**Date:** 2026-06-22
**Status:** In Progress
**Reference:** `STT_INTEGRATION_PLAN.md` (design decisions, VRAM analysis, risks)

## Goal
Deploy voice-to-prompt (speech-to-text) into `qwen_image_edit_web` on the RTX 3090 machine, using Whisper Large V3 Turbo INT8 via `faster-whisper`, without disrupting the running production server.

## Machines

| Machine | Host | GPU | VRAM | Role |
|---------|------|-----|------|------|
| **3090 (production)** | xh97-ml@100.89.22.74 | RTX 3090 | 24 GB | Target deployment |
| **5090 (dev/verify)** | xh97@100.121.112.83 | RTX 5090 | 32 GB | Code development & testing |

## Key Constraints (from STT_INTEGRATION_PLAN.md §6)

- 3090 production server uses ~21.5 GB VRAM (ComfyUI State 2) → only ~2.5 GB headroom
- Whisper INT8 needs ~1.5-1.8 GB → fits with ~0.7-1 GB buffer
- Production runs via `qwen_webui` conda env (Python 3.12)
- Production code has NO `memory_engine` (older version than 5090)

## Strategy

1. **Isolate** — Copy production code to test directory, not touching production
2. **Modify** — Apply STT changes to test directory only
3. **Verify** — Test on 3090 with isolated server on alternate port
4. **Swap** — Kill old server, start new server once verified

## Files Modified (reference: STT_INTEGRATION_PLAN.md §4)

| File | Action | Plan Section |
|------|--------|-------------|
| `stt_engine.py` | CREATE | Phase 1 |
| `server_comfy.py` | MODIFY (imports + endpoint) | Phase 2 |
| `templates/index.html` | MODIFY (CSS + HTML + JS + i18n) | Phase 3 |
| `requirements.txt` | MODIFY (add faster-whisper) | Phase 4 |

## Steps

### ✅ DONE: Local Development (5090)

- [x] Create `stt_engine.py` (Phase 1) — verified syntax, smoke test passed
- [x] Add `/api/transcribe` to `server_comfy.py` (Phase 2) — verified syntax
- [x] Add mic button CSS + HTML + JS + i18n to `index.html` (Phase 3)
- [x] Update `requirements.txt` (Phase 4)
- [x] Install `faster-whisper` in `qwen_web` conda env
- [x] Pre-download model weights (~1.6 GB CTranslate2 format)
- [x] Smoke test: `curl POST /api/transcribe` → `{"success":true,"text":""}` ✓
- [x] VRAM eviction confirmed: 31.9 GB → 30.3 GB after 120s idle ✓

### ✅ DONE: 3090 Deployment

- [x] Step 1: Backup production → `qwen_image_edit_web_backup/`
- [x] Step 2: Create test directory → `qwen_stt_test/`
- [x] Step 3: Copy `stt_engine.py` to test directory
- [x] Step 4: Apply STT endpoint to `server_comfy.py` (add imports + endpoint code)
- [x] Step 5: Apply STT frontend to `templates/index.html` (CSS + HTML + JS + i18n)
- [x] Step 6: Update `requirements.txt`
- [x] Step 7: Install `faster-whisper` in `qwen_webui` conda env on 3090
- [x] Step 8: Pre-download model weights on 3090 (1.6 GB CTranslate2 format)
- [x] Step 9: Syntax check + smoke test on 3090 (port 7861) → `POST /api/transcribe` → `{"success":true,"text":""}` ✓
- [x] Step 10: Verify VRAM behavior on 3090:
  - Before STT: 21833 MiB / 24576 MiB (~2.7 GB headroom)
  - After STT load: 23150 MiB (+1317 MB, within ~1.5-1.8 GB estimate) ✓
  - After 120s eviction: 22094 MiB (partial eviction, ~1.0 GB freed) ✓
- [x] Step 11: Swap to production (kill old server, start new on 7860) → ✅ LIVE

### NOTES

- Production code is **older** than 5090 code: no `memory_engine`, no memory APIs
- Must apply STT changes to the **3090's existing codebase**, not copy 5090's files wholesale
- Port 7860 is occupied by production server → test on 7861 first
- Use `PORT=7861` env var for test server
- Model download completed: 1.6 GB CTranslate2 format at `~/.cache/faster-whisper/`
- VRAM eviction works but may not fully free all memory (PyTorch caching allocator)

### ✅ PRODUCTION SWAP VERIFIED (2026-06-22 09:50)

- Old server (PID 20512) killed
- New server (PID 34911) started on port 7860
- `/api/status` → `gpu_available: true, model_loaded: true` ✓
- `/api/transcribe` → `{"success":true,"text":""}` ✓
- Frontend → 🎤 button present ✓
- VRAM idle: 1.3 GB / 24.6 GB (ComfyUI offloaded) ✓
- Backup available at `/home/xh97-ml/qwen_image_edit_web_backup/`

### ✅ HTTPS FIX (2026-06-22 09:54)

- **Problem:** `navigator.mediaDevices` undefined on `http://100.89.22.74:7860`
- **Root cause:** Modern browsers require HTTPS for `getUserMedia()` (except localhost)
- **Solution:** Installed `mkcert`, generated locally-trusted cert for `100.89.22.74`, updated `server_comfy.py` to use SSL context
- **Result:** Server now runs on `https://100.89.22.74:7860` with mkcert-generated cert
- **Browser trust:** CA installed in system trust store; Chrome/Firefox may need manual trust on first visit
- **Files:** `100.89.22.74+3.pem` (cert), `100.89.22.74+3-key.pem` (key) in project dir
- **Env var:** `SSL=0` to disable SSL, `SSL=1` (default) to enable

### ✅ UX FIXES (2026-06-22 10:00)

- **STT replace mode:** Speech-to-text now replaces the entire prompt instead of appending
- **Re-edit button:** "New Edit" → "Re-edit" (🔄 重新编辑). Loads the current output image as the new input for chained editing
- **Implementation:** Fetch result image URL → convert to base64 → set as `selectedImageData` → show preview

### NOTES

- Production code is **older** than 5090 code: no `memory_engine`, no memory APIs
- Must apply STT changes to the **3090's existing codebase**, not copy 5090's files wholesale
- Port 7860 is occupied by production server → test on 7861 first
- Use `PORT=7861` env var for test server
