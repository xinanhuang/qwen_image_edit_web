# Speech-to-Text Integration Plan

**Date:** 2026-06-21
**Goal:** Integrate speech-to-text (voice-to-prompt) into `qwen_image_edit_web` using the STT models from `whisper_local_web`.

---

## 1. Problem Statement

Currently, `qwen_image_edit_web` requires users to **type** edit prompts. For mobile users, quick iterations, or non-technical users, voice input would be significantly more convenient. `whisper_local_web` already has working inference for 6 STT models on the same machine.

---

## 2. Architecture Overview

### Current State

```
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  qwen_image_edit_web            │     │  whisper_local_web           │
│                                 │     │                              │
│  Flask (port 7860)              │     │  Gradio (port 7861)          │
│  templates/index.html (SPA)     │     │  app.py + models.py          │
│  server_comfy.py                │     │                              │
│  memory_engine.py               │     │  6 STT models (lazy-load)    │
│                                 │     │  Conda env: stt-demo         │
│  Conda env: qwen_web            │     │                              │
└─────────────────────────────────┘     └──────────────────────────────┘
         │                                    │
         ▼                                    ▼
   ComfyUI (8188)                     GPU (RTX 5090 / 3090)
```

### Target State

```
┌──────────────────────────────────────────────────────────────────┐
│  qwen_image_edit_web (unified)                                   │
│                                                                  │
│  Flask (port 7860)                                               │
│  ├── templates/index.html  ← adds 🎤 mic button + recording UI  │
│  ├── server_comfy.py        ← adds /api/transcribe endpoint      │
│  ├── stt_engine.py          ← NEW: lightweight STT wrapper       │
│  └── memory_engine.py       ← unchanged                          │
│                                                                  │
│  Conda env: qwen_web  ← adds whisper transformers deps          │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
   ComfyUI (8188)  +  GPU for STT inference
```

---

## 3. Key Design Decisions

### 3.1 Which model(s) to integrate?

| Model | Params | FP16 VRAM | INT8 VRAM | Speed | Languages | Verdict |
|---|---|---|---|---|---|---|
| **Whisper Large V3 Turbo** | 809 M | **~2.5-3 GB** | **~1.5-2 GB** | Fast | 99 langs | ✅ **Primary** |
| Distil-Whisper Large V3 | ~400 M | ~1.5-2 GB | ~1-1.5 GB | 5× faster | 99 langs | ✅ Fallback |
| Parakeet TDT 0.6B v3 | 600 M | ~8 GB | — | Very fast | English | Optional |
| Qwen3-ASR 1.7B | 1.7 B | ~6 GB | — | Medium | 30 langs | Optional |
| Canary Qwen 2.5B | 2.5 B | ~6 GB | — | Fast | English | Optional |
| VibeVoice-ASR 7B | 7 B | ~20 GB | — | Slow | 50+ langs | Too heavy |

**Decision:** Use **Whisper Large V3 Turbo via `faster-whisper` with INT8 quantization** (CTranslate2 runtime). The full ComfyUI workflow (text encoder + diffusion model + LoRA + VAE) uses ~21.5 GB of the 3090's 24 GB during State 2 (image generation), leaving only ~2.5 GB headroom. INT8 (~1.5-1.8 GB peak) fits safely; FP16 (~2.8-3.2 GB) is borderline/risky.

### 3.1.1 Quantization VRAM Analysis (corrected for full workflow)

**Whisper v3-turbo** is a distilled model with only **809M parameters** (vs 1.55B for full large-v3). However, **model weight size ≠ peak operational VRAM** — activations, KV cache, and audio feature tensors add significant overhead.

| Quantization | Weight Size | Total Peak VRAM (weights + activations + KV cache) |
|---|---|---|
| FP32 (unquantized) | ~3.2 GB | ~4.5-5.0 GB |
| **FP16 (standard GPU)** | **~1.6 GB** | **~2.8-3.2 GB** |
| **INT8 (8-bit, CTranslate2)** | **~809 MB** | **~1.5-1.8 GB** |
| INT4 (4-bit / GGML Q4) | ~400 MB | ~1.0-1.2 GB |

**Sources:**
- [Spheron GPU Recommender](https://www.spheron.network/tools/gpu-recommender/openai/whisper-large-v3-turbo/) — 809M params, FP16/INT8/INT4 VRAM estimates
- [Spheron Blog](https://www.spheron.network/blog/whisper-v4-asr-gpu-cloud-production-guide/): "Whisper Large v3 at float16 uses roughly 3GB"
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper): "up to 4× faster than openai/whisper for the same accuracy while using less memory"
- [ArXiv 2503.09905](https://arxiv.org/abs/2503.09905) — INT8 WER on LibriSpeech: FP16 = 2.01%, INT8 = ~2.10% (negligible 0.09% diff)

**Full ComfyUI workflow VRAM breakdown (per [ComfyUI Qwen docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511)):**

| Component | VRAM |
|---|---|
| Qwen 2.5 VL 7B (FP8 Scaled Text Encoder) | ~9.4 GB |
| Qwen Image Edit 2511 (FP8 e4m3fn Diffusion Model) | ~20.4 GB |
| Lightning 4-Steps LoRA & VAE | ~1.0 GB |
| **Total Qwen Weights** | **~30.8 GB** (exceeds 24 GB → ComfyUI offloads to system RAM) |

**ComfyUI runs in two states (aggressive offloading):**
- **State 1 (Text Processing):** Loads 9.4 GB Qwen VL text encoder → processes prompts → flushes to system RAM
- **State 2 (Image Generation — The Bottleneck):** Loads 20.4 GB diffusion model + LoRA + VAE = **~21.4 GB active VRAM**

**On our 3090 (24 GB, headless, ~4 MB display overhead):**
- Actual measured usage: **~21.5 GB** during State 2
- Remaining headroom: **~2.5 GB** (but this is fragmented — not a single contiguous block)

**Precision comparison for our use case (short voice prompts, ~5-30s):**

| Precision | Peak VRAM | Status on 3090 (21.5 GB used) | WER | Verdict |
|---|---|---|---|---|
| FP16 | ~2.8-3.2 GB | ❌ **Likely OOM** (exceeds 2.5 GB headroom; fragmentation risk) | 2.01% | Risky |
| **INT8 (CTranslate2)** | **~1.5-1.8 GB** | ✅ **Safe** (fits in 2.5 GB headroom, ~0.7-1 GB buffer) | ~2.10% | **Sweet spot** |
| INT4 (GGML Q4) | ~1.0-1.2 GB | 🚀 **Very Safe** (leaves ~1.3 GB buffer) | ~2.5-3% | Max stability |

**Verdict for our case on the RTX 3090 (24 GB, headless):**
- **INT8 via `faster-whisper` is the default** — ~1.5-1.8 GB peak VRAM, fits comfortably in the 2.5 GB headroom with ~0.7-1 GB buffer, virtually identical WER to FP16 (0.09% increase).
- **FP16 is borderline/risky** — ~2.8-3.2 GB peak exceeds the 2.5 GB headroom. May trigger OOM due to VRAM fragmentation from Qwen's large contiguous allocations.
- **INT4** only if INT8 still OOMs (e.g., driving multiple high-res monitors adds overhead).
- **CPU fallback** (`device="cpu"`) if GPU INT8 still conflicts — adds ~1s latency but eliminates VRAM conflict entirely.

### 3.1.2 Model Weight Download Links

| Format | Repo | Link |
|---|---|---|
| **Faster-Whisper FP16 (CTranslate2)** | h2oai/faster-whisper-large-v3-turbo | https://huggingface.co/h2oai/faster-whisper-large-v3-turbo |
| **Faster-Whisper INT8 (CTranslate2)** | Zoont/faster-whisper-large-v3-turbo-int8-ct2 | https://huggingface.co/Zoont/faster-whisper-large-v3-turbo-int8-ct2 |
| **Transformers FP16/FP32 (official)** | openai/whisper-large-v3-turbo | https://huggingface.co/openai/whisper-large-v3-turbo |
| **GGUF/GGML all quants (whisper.cpp)** | Pomni/whisper-large-v3-turbo-ggml-allquants | https://huggingface.co/Pomni/whisper-large-v3-turbo-ggml-allquants |
| **ONNX** | onnx-community/whisper-large-v3-turbo | https://huggingface.co/onnx-community/whisper-large-v3-turbo |

> **Pro-tip:** You don't need to download the pre-quantized INT8 repo. If you point `faster-whisper` to the FP16 repo and set `compute_type="int8_float16"`, CTranslate2 will quantize to INT8 on-the-fly during load.

### 3.2 Integration approach: Embedded vs. Microservice

| Approach | Pros | Cons |
|---|---|---|
| **A. Embedded** (import models.py into Flask) | Single process, simple API, no IPC | Heavy deps in `qwen_web` env; VRAM contention with image model |
| **B. Microservice** (whisper runs on port 7861, Flask calls HTTP) | Isolated deps, separate GPU memory mgmt | Two processes, IPC latency, startup complexity |
| **C. `faster-whisper` wrapper** (new `stt_engine.py` in Flask) | Best of both — single pip dep, lazy-load, same process, faster than transformers |

**Decision:** **Approach C** — create a new `stt_engine.py` inside `qwen_image_edit_web/` that uses `faster-whisper` with INT8 quantization. Single `pip install faster-whisper` dep, ~1.5-1.8 GB VRAM, up to 4× faster than `transformers` baseline. Avoids pulling in NeMo, vibevoice, qwen-asr, gradio, etc.

**Default: INT8** (safe fit in 2.5 GB headroom, negligible WER increase). **FP16** available via env var `STT_COMPUTE=float16` when running on 5090 or when ComfyUI is idle.

### 3.3 VRAM contention strategy

The full ComfyUI workflow (Qwen 2.5 VL 7B text encoder + Qwen Image Edit 2511 diffusion model + Lightning LoRA + VAE) has ~30.8 GB total weights but uses aggressive offloading:
- **State 1 (Text Processing):** ~9.4 GB (text encoder only) → Whisper STT can run comfortably
- **State 2 (Image Generation):** ~21.4 GB active VRAM (diffusion + LoRA + VAE) → only ~2.5 GB headroom on 3090

**Strategy:**
- STT model loads **lazily** on first `/api/transcribe` call
- Model stays **cached for 120s** so repeated transcriptions are instant
- After cache expiry, CTranslate2 model is GC'd and VRAM is freed
- **On 3090: INT8 is default** (~1.5-1.8 GB peak) — fits in 2.5 GB headroom during State 2
- **On 5090 or when ComfyUI is idle:** set `STT_COMPUTE=float16` for max accuracy
- **If INT8 still OOMs:** set `STT_DEVICE=cpu` + `STT_COMPUTE=int8` — adds ~1s latency but eliminates VRAM conflict

---

## 4. Implementation Plan

### Phase 1: Backend — STT Engine (`stt_engine.py`)

**New file:** `qwen_image_edit_web/stt_engine.py`

Uses **`faster-whisper`** (CTranslate2 runtime). **INT8 by default** (~1.5-1.8 GB VRAM, 4× faster, WER ~2.10%). **FP16 available** via `STT_COMPUTE=float16` (~2.8-3.2 GB VRAM, WER 2.01%) when running on 5090 or when ComfyUI is idle.

```python
"""
Lightweight Speech-to-Text engine for Qwen Image Edit.
Uses faster-whisper (CTranslate2) with configurable precision.
- INT8 (default): ~1.5-1.8 GB VRAM, 4x faster, WER ~2.10% (negligible 0.09% diff vs FP16)
- FP16: ~2.8-3.2 GB VRAM, max accuracy (WER 2.01%)
Model is lazy-loaded, cached for CACHE_TTL seconds, then actively evicted.

Environment variables:
    STT_MODEL   - Model size (default: "large-v3-turbo")
    STT_COMPUTE - Precision: "int8_float16" (default), "float16", or "int8"
    STT_DEVICE  - Device: "cuda" (default) or "cpu" (fallback if GPU OOM)

Thread-safe: uses threading.RLock to prevent concurrent model loads (reentrant for nested scopes).
Active eviction: uses threading.Timer to free VRAM after CACHE_TTL seconds.
"""

import os
import gc
import time
import threading
from faster_whisper import WhisperModel

# Config
STT_MODEL_SIZE = os.environ.get("STT_MODEL", "large-v3-turbo")
STT_COMPUTE = os.environ.get("STT_COMPUTE", "int8_float16")  # int8_float16, float16, int8
DEVICE = os.environ.get("STT_DEVICE", "cuda")
CACHE_TTL = 120  # seconds to keep model loaded in memory
COMFY_API_URL = os.environ.get("COMFY_API_URL", "http://127.0.0.1:8188")  # ComfyUI API base URL

# Cache + thread safety
_model = None
_last_active_time = 0  # Rolling window: updated on every transcription use
_active_requests = 0   # Tracks concurrent inference streams (prevents mid-inference eviction)
_state_lock = threading.RLock()  # Only locks model loading/eviction, NOT inference execution
_eviction_timer = None


def _active_evict():
    """Actively evict model from VRAM only if no users are actively transcribing."""
    global _model, _eviction_timer, _last_active_time, _active_requests
    with _state_lock:
        # Triple-check: model exists, no active requests, TTL expired
        if _model is not None and _active_requests == 0 and (time.time() - _last_active_time) >= (CACHE_TTL - 1):
            _model = None
            gc.collect()  # Force instant destruction of CTranslate2 CUDA handles
            print("[STT] Active Eviction: Whisper dropped from VRAM.")
            _eviction_timer = None  # Only clear reference if eviction actually occurred


def _load_model():
    """Load the Whisper model into GPU memory, with ComfyUI cache-clear fallback."""
    global _model
    download_dir = os.path.join(os.path.expanduser("~"), ".cache", "faster-whisper")
    try:
        print(f"[STT] Loading faster-whisper {STT_MODEL_SIZE} (compute={STT_COMPUTE}, device={DEVICE})...")
        _model = WhisperModel(
            STT_MODEL_SIZE,
            device=DEVICE,
            compute_type=STT_COMPUTE,
            download_root=download_dir,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "cublas" in err_msg:
            # ComfyUI's PyTorch caching allocator may be holding VRAM
            print("[STT] VRAM blocked by PyTorch. Forcing ComfyUI cache clear...")
            try:
                import requests
                requests.post(
                    f"{COMFY_API_URL}/free",
                    json={"unload_models": False, "free_memory": True},
                    timeout=2,
                )
                _model = WhisperModel(
                    STT_MODEL_SIZE,
                    device=DEVICE,
                    compute_type=STT_COMPUTE,
                    download_root=download_dir,
                )
            except Exception as retry_e:
                print(f"[STT] Retry failed. CPU fallback. Error: {retry_e}")
                _model = WhisperModel(
                    STT_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                    download_root=download_dir,
                )
                print(f"[STT] ✓ {STT_MODEL_SIZE} loaded on cpu (int8)")
                return
        else:
            raise e
    print(f"[STT] ✓ {STT_MODEL_SIZE} loaded on {DEVICE} ({STT_COMPUTE})")


def _ensure_loaded():
    """Load model if not cached or if cache expired while fully idle. Must be called under _state_lock."""
    global _model, _last_active_time, _active_requests
    # ONLY evict/reload due to TTL if no other users are actively using the model
    if _model is None or (_active_requests == 0 and (time.time() - _last_active_time) > CACHE_TTL):
        if _model is not None:
            _model = None
            gc.collect()  # Force instant destruction of old CUDA pointers
            print("[STT] Previous idle model evicted (cache expired)")
        _load_model()
    _last_active_time = time.time()


def transcribe(audio_path: str, language: str = None) -> str:
    """
    Transcribe audio file to text using faster-whisper.
    
    Concurrent-safe: state lock protects loading/eviction, but inference runs
    in parallel. CTranslate2 shares model weights across threads — only activation
    memory (~150-200 MB per user) is allocated per inference.
    4 concurrent users = ~1.5 GB (model) + ~800 MB (activations) = ~2.3 GB total.
    
    Exception-safe: try/finally guarantees _active_requests counter and eviction timer.

    Args:
        audio_path: Path to audio file (WAV, MP3, WEBM, OGG, M4A, FLAC, etc.)
        language: Language code (e.g. 'en', 'zh', 'ja'). None = auto-detect.

    Returns:
        Transcription text string.
    """
    global _active_requests, _last_active_time, _eviction_timer

    # Lock: load model + increment active counter + cancel timer
    with _state_lock:
        _ensure_loaded()
        _active_requests += 1
        if _eviction_timer is not None:
            _eviction_timer.cancel()
            _eviction_timer = None

    try:
        # UNLOCKED: multiple users transcribe concurrently
        try:
            segments, info = _model.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},  # Tighter VAD for lower VRAM
            )
            text = "".join(seg.text for seg in segments).strip()
        except Exception as inference_error:
            err_msg = str(inference_error).lower()
            if "out of memory" in err_msg or "cublas" in err_msg:
                # ComfyUI may have spiked VRAM mid-inference
                print("[STT] VRAM spiked mid-inference. Clearing ComfyUI memory and retrying...")
                import requests
                requests.post(
                    f"{COMFY_API_URL}/free",
                    json={"unload_models": False, "free_memory": True},
                    timeout=2,
                )
                # Retry inference pass
                segments, info = _model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 250},
                )
                text = "".join(seg.text for seg in segments).strip()
            else:
                raise inference_error
        print(f"[STT] Detected lang: {info.language}, prob: {info.language_probability:.2f}")
        return text

    finally:
        # Lock: decrement counter + reset TTL + schedule eviction (if last user)
        with _state_lock:
            _active_requests -= 1
            _last_active_time = time.time()
            if _active_requests == 0:
                if _eviction_timer is not None:
                    _eviction_timer.cancel()
                _eviction_timer = threading.Timer(CACHE_TTL, _active_evict)
                _eviction_timer.start()
```

**Dependencies to add to `qwen_webui` conda env:**
```
# faster-whisper pulls in: ctranslate2, onnxruntime-gpu, sentencepiece, av (PyAV)
# No soundfile, torchaudio, or transformers needed for STT
```

### Phase 2: Backend — Flask Endpoint

**New endpoint in `server_comfy.py`:**

```python
import base64
import glob
import os
import tempfile
import time


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
```

### Phase 3: Frontend — UI Changes

**Location:** `templates/index.html`

Add a **microphone button** next to the prompt textarea:

```html
<!-- Replace existing prompt section with enhanced version -->
<div class="prompt-section">
    <label class="prompt-label">Edit Prompt</label>
    <div class="prompt-input-wrap">
        <textarea class="prompt-area" id="prompt" placeholder="..."></textarea>
        <button type="button" class="btn-mic" id="btn-mic" title="Voice input">🎤</button>
    </div>
</div>
```

**CSS additions:**
```css
.prompt-input-wrap {
    position: relative;
}
.btn-mic {
    position: absolute;
    bottom: 10px;
    right: 10px;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 1.1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}
.btn-mic:hover {
    background: var(--accent);
    border-color: var(--accent);
}
.btn-mic.recording {
    background: var(--error);
    border-color: var(--error);
    animation: pulse-mic 1s infinite;
}
@keyframes pulse-mic {
    0%, 100% { box-shadow: 0 0 0 0 rgba(248,113,113,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(248,113,113,0); }
}
```

**JavaScript additions** (appended to existing script):

```javascript
// ==================== Voice Input ====================
const btnMic = document.getElementById('btn-mic');
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingTimeout = null;

btnMic.addEventListener('click', async () => {
    // Instantly disable to prevent double-click while awaiting getUserMedia
    btnMic.disabled = true;

    if (isRecording) {
        // Stop recording — let onstop handle all UI state resets asynchronously
        clearTimeout(recordingTimeout);
        mediaRecorder.stop();
        return;
    }

    // Start recording
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1 }
        });

        // Dynamic mimeType: WebM for Chrome/Firefox, MP4 for Safari/iOS
        let recorderOptions = {};
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            recorderOptions = { mimeType: 'audio/webm;codecs=opus' };
        } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
            recorderOptions = { mimeType: 'audio/mp4' };  // iOS Safari fallback
        }
        // If neither supported, let browser use its default

        mediaRecorder = new MediaRecorder(stream, recorderOptions);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            // Centralize all recording state resets here (prevents UI flicker)
            isRecording = false;
            btnMic.classList.remove('recording');

            if (audioChunks.length === 0) {
                console.warn('[STT] No audio data recorded. Discarding.');
                stream.getTracks().forEach(t => t.stop());
                btnMic.textContent = '🎤';
                btnMic.disabled = false;
                return;
            }

            const actualMimeType = mediaRecorder.mimeType || 'audio/webm';
            const audioBlob = new Blob(audioChunks, { type: actualMimeType });

            if (audioBlob.size === 0) {
                console.warn('[STT] Empty audio blob. Discarding.');
                stream.getTracks().forEach(t => t.stop());
                btnMic.textContent = '🎤';
                btnMic.disabled = false;
                return;
            }

            // Pass verified local extension to sendTranscription (decouples from global mediaRecorder)
            const fileExt = actualMimeType.includes('mp4') ? 'mp4' : 'webm';
            await sendTranscription(audioBlob, fileExt);
            stream.getTracks().forEach(t => t.stop());
        };
        mediaRecorder.start();
        btnMic.classList.add('recording');
        btnMic.textContent = '⏹';
        btnMic.disabled = false;
        isRecording = true;

        // Auto-stop after 30 seconds to prevent runaway recording
        recordingTimeout = setTimeout(() => {
            if (isRecording) {
                btnMic.disabled = true;  // Lock button; let onstop clear states
                mediaRecorder.stop();

                const lang = (typeof currentLang !== 'undefined') ? currentLang : 'en';
                const msg = lang === 'zh' ? '最长录音时间 (30s) 已到' : 'Max recording time (30s) reached';
                showError(msg);
            }
        }, 30000);
    } catch (err) {
        clearTimeout(recordingTimeout);
        btnMic.disabled = false;  // Re-enable so user can retry after fixing permission/hardware
        showError('Microphone access denied: ' + err.message);
    }
});

async function sendTranscription(audioBlob, fileExt) {
    btnMic.disabled = true;
    btnMic.textContent = '⏳';
    try {
        const formData = new FormData();
        // fileExt passed from onstop scope — decoupled from global mediaRecorder
        formData.append('audio', audioBlob, `recording.${fileExt || 'webm'}`);
        // Safe scope evaluation: currentLang may be scoped inside an IIFE
        const safeLang = (typeof currentLang !== 'undefined' && currentLang === 'zh') ? 'zh' : 'en';
        formData.append('language', safeLang);

        const resp = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();
        if (data.success) {
            // Only update textarea if actual speech tokens were returned
            if (data.text && data.text.trim()) {
                const promptEl = document.getElementById('prompt');
                // If prompt already has text, append; otherwise replace
                if (promptEl.value.trim()) {
                    promptEl.value += ' ' + data.text;
                } else {
                    promptEl.value = data.text;
                }
                // Force SPA frameworks (React/Vue/Svelte) to register the value change
                promptEl.dispatchEvent(new Event('input', { bubbles: true }));
                promptEl.dispatchEvent(new Event('change', { bubbles: true }));
                promptEl.focus();
            }
            // Successful but empty transcriptions (silence) gracefully do nothing
        } else {
            // Strictly catches true backend engine errors
            showError('Transcription error: ' + (data.error || 'Unknown'));
        }
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        btnMic.disabled = false;
        btnMic.textContent = '🎤';
    }
}
```

### Phase 4: i18n Updates

Add to the `i18n` object in `index.html`:

```javascript
// In both en and zh sections:
mic_title: "Voice input",          // en
mic_title: "语音输入",              // zh
mic_recording: "Recording...",     // en
mic_recording: "录音中...",         // zh
mic_hint: "Tap 🎤 to speak your edit prompt",  // en
mic_hint: "点击 🎤 语音输入编辑提示词",       // zh
mic_max_time: "Max recording time (30s) reached",  // en
mic_max_time: "最长录音时间 (30s) 已到",       // zh
```

---

## 5. Dependency Management

### Current `qwen_webui` env dependencies (remote 3090):
```
Flask 3.1.3
transformers 5.12.1
```

### Additional dependencies needed:
```
faster-whisper>=1.0.0   # CTranslate2-based Whisper (INT8 quantization, 4x faster)
# faster-whisper pulls in: ctranslate2, onnxruntime-gpu (or cpu), sentencepiece
```

### Installation:
```bash
conda activate qwen_webui
pip install faster-whisper
```

> **Note:** `faster-whisper` auto-selects GPU if CUDA is available. No need for
> separate `transformers`, `soundfile`, or `torchaudio` — CTranslate2 handles
> audio decoding internally (via ffmpeg/libav).
> 
> **Current `qwen_webui` env:** Flask 3.1.3, transformers 5.12.1. Only `faster-whisper` needs to be added.

### Model download (one-time, ~1.6 GB CTranslate2 format):
```bash
# Pre-download Whisper Large V3 Turbo (downloads CTranslate2 format from mobiuslabsgmbh/faster-whisper-large-v3-turbo)
# Uses same download_root as stt_engine.py to ensure runtime finds cached model
conda activate qwen_webui
python -c "
from faster_whisper import WhisperModel
import os
download_root = os.path.join(os.path.expanduser('~'), '.cache', 'faster-whisper')
model = WhisperModel('large-v3-turbo', device='cuda', compute_type='int8_float16', download_root=download_root)
print('Model downloaded and cached at:', download_root)
"
```

> **Note:** `faster-whisper` maps `"large-v3-turbo"` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (CTranslate2 format).
> The existing transformers-format weights at `~/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/` will NOT be reused —
> faster-whisper downloads its own CTranslate2-converted `model.bin` files. The pre-download script ensures the first
> `/api/transcribe` call doesn't stall for 30-60s on network download.
>
> **Version note:** The shorthand `"large-v3-turbo"` requires `faster-whisper>=1.0.0`. If your pip installs an older
> version, use the explicit HF repo ID instead: `WhisperModel("Systran/faster-whisper-large-v3-turbo", ...)`.

> **Pro-tip:** If you later want INT8, no need to re-download. Just set
> `STT_COMPUTE=int8_float16` — CTranslate2 quantizes on-the-fly during load.

### Why `faster-whisper` over `transformers` for this use case?

| Metric | `transformers` FP16 | `faster-whisper` INT8 |
|---|---|---|
| **Peak VRAM** | ~2.8-3.2 GB | **~1.5-1.8 GB** |
| **Disk (model)** | ~1.6 GB | **~0.8 GB (on-the-fly conversion)** |
| **Speed (30s audio)** | ~5-8s | **~1-2s** |
| **Audio format support** | Needs manual preprocessing | **Native (WAV/MP3/WEBM/OGG/M4A/FLAC)** |
| **VAD (silence removal)** | Manual | **Built-in** |
| **Language detection** | Manual | **Built-in** |
| **Python deps** | transformers + soundfile + torchaudio | **faster-whisper (single pip)** |
| **Fits in 2.5 GB headroom?** | ❌ Borderline/risky | ✅ **Safe (~0.7-1 GB buffer)** |

---

## 6. VRAM Budget Analysis

### Full ComfyUI Workflow Components (per [ComfyUI Qwen docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511))

| Component | Disk Size | VRAM (loaded) |
|---|---|---|
| Qwen 2.5 VL 7B (FP8 Scaled Text Encoder) | 8.8 GB | ~9.4 GB |
| Qwen Image Edit 2511 (FP8 e4m3fn Diffusion Model) | 20 GB | ~20.4 GB |
| Lightning 4-Steps LoRA | 811 MB | ~811 MB |
| VAE | 243 MB | ~243 MB |
| **Total Weights** | **~30 GB** | **~30.8 GB** (exceeds 24 GB → ComfyUI offloads) |

### On RTX 3090 (24 GB) — primary target (headless, ~4 MB display overhead):

ComfyUI runs in two states with aggressive offloading:

| State | Active Components | VRAM Used | Headroom |
|---|---|---|---|
| **State 1: Text Processing** | Qwen 2.5 VL 7B text encoder | ~9.4 GB | ~14.6 GB |
| **State 2: Image Generation** | Diffusion model + LoRA + VAE | **~21.4 GB** | **~2.5 GB** |
| **Measured actual usage** | (above) | **~21.5 GB** | **~2.5 GB** |

**VRAM budget during State 2 (the bottleneck):**

| Component | VRAM Usage | Notes |
|---|---|---|
| Qwen-Image-Edit-2511 workflow (State 2) | ~21.5 GB | Measured actual usage |
| Whisper Large V3 Turbo **INT8** | **~1.5-1.8 GB** | Loaded lazily, cached 120s |
| **Total peak** | **~23-23.3 GB** | **Safe (24 GB available, ~0.7-1 GB buffer)** |

**Strategy:** INT8 is the default — ~1.5-1.8 GB VRAM fits safely in the 2.5 GB headroom with ~0.7-1 GB buffer for activations/KV cache. The model stays cached for 120s so repeated transcriptions are instant.

### On RTX 5090 (32 GB):

| Component | VRAM Usage | Notes |
|---|---|---|
| Qwen-Image-Edit-2511 fp8 | ~29 GB | Loaded by ComfyUI (with CPU offloading for overflow) |
| Whisper Large V3 Turbo **FP16** | **~2.8-3.2 GB** | Loaded lazily |
| **Peak (if both loaded)** | ~31.8-32.2 GB | Tight — use INT8 if OOM |

**Strategy:** On 5090, INT8 is default. If llama-server or other processes consume VRAM, FP16 may still fit due to larger headroom.

### When to use FP16 instead of INT8

| Scenario | Recommended Precision |
|---|---|
| Normal use on 3090 (State 2 active) | **INT8** (default) |
| State 1 only (text processing, no image gen) | **FP16** (`STT_COMPUTE=float16`) |
| 5090 with headroom | **FP16** (`STT_COMPUTE=float16`) |
| CPU-only fallback | **INT8** (`STT_COMPUTE=int8`, `STT_DEVICE=cpu`) |

### Comparison: FP16 vs INT8 for our integration

| | FP16 (faster-whisper) | INT8 (faster-whisper) |
|---|---|---|
| Peak VRAM | ~2.8-3.2 GB | **~1.5-1.8 GB** |
| 3090 headroom after STT (State 2) | ~-0.3 to -0.7 GB (OOM) | **~0.7-1.0 GB (safe)** |
| Transcription time (10s audio) | ~2-4s | **~1-2s** |
| WER impact | Baseline (2.01%) | +0.09% (2.10%) — **negligible** |

---

## 7. Implementation Order

### Step 1: Backend STT Engine (1-2 hours)
- [ ] Create `stt_engine.py` with `faster-whisper` (INT8 default, thread-safe, active eviction)
- [ ] Add Flask `/api/transcribe` endpoint in `server_comfy.py` (tmp.close(), tmp.flush())
- [ ] Test with `curl` or `test_transcribe.py`-style script
- [ ] Install `faster-whisper` in `qwen_webui` conda env (remote 3090)
- [ ] Pre-download model weights (CTranslate2 format, matches runtime download_root)

### Step 2: Frontend UI (1-2 hours)
- [ ] Add microphone button to prompt section in `index.html`
- [ ] Add CSS styles for recording state
- [ ] Add JavaScript: MediaRecorder API → `/api/transcribe` → populate prompt
- [ ] Dynamic mimeType detection (WebM → MP4 → default fallback for Safari)
- [ ] 0-byte payload guard + 30s auto-stop timeout
- [ ] Add i18n strings for EN/ZH (including mic_max_time)
- [ ] Test on desktop Chrome + mobile Safari

### Step 3: Testing & Polish (30 min)
- [ ] Test voice → prompt → generate full flow
- [ ] Test with various audio lengths (5s-30s)
- [ ] Test VRAM behavior (nvidia-smi monitoring)
- [ ] Test error handling (no mic permission, network failure)
- [ ] Test on both 5090 and 3090 GPUs

### Step 4: Optional Enhancements
- [ ] Add audio file upload as alternative to microphone
- [ ] Add language selection dropdown for STT
- [ ] Add "transcribing..." progress indicator
- [ ] Add confidence score display
- [ ] Support Distil-Whisper as configurable fallback via env var

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| VRAM OOM on 3090 during State 2 (peak ~23-23.3 GB / 24 GB) | STT or image job fails | INT8 (~1.5-1.8 GB) fits in 2.5 GB headroom; active eviction frees VRAM after 120s; `STT_DEVICE=cpu` fallback |
| VRAM fragmentation on 3090 | STT fails to allocate despite sufficient free VRAM | Qwen's large contiguous blocks may fragment remaining 2.5 GB; INT8's smaller footprint reduces risk |
| Multi-process CUDA cache contention | PyTorch (ComfyUI) holds cached VRAM, blocking faster-whisper | `gc.collect()` forces instant CTranslate2 destructor; active eviction via `threading.Timer` |
| Flask multi-threaded race condition | Two threads load model simultaneously → 3 GB spike → OOM | `_state_lock` (RLock) serializes `_ensure_loaded()`; inference runs concurrent (CTranslate2 shares weights) |
| WSGI multi-process OOM | Gunicorn `-w 4` → 4 workers × 1.5 GB = 6 GB VRAM → instant OOM | Use `-w 1 --threads N`; thread locks don't span OS processes |
| VRAM OOM on 5090 (peak ~31-32 GB / 32 GB) | STT or image job fails | Use INT8 on 5090 when llama-server or other GPU processes are active |
| `faster-whisper` / CTranslate2 CUDA conflict | Import or runtime errors | `faster-whisper` auto-detects CUDA; test on target env; fallback: `STT_DEVICE=cpu` + `STT_COMPUTE=int8` |
| INT8 accuracy on non-English | Slightly higher WER for rare langs | INT8 WER increase is ~0.1% across all langs per arXiv 2503.09905; negligible for image-edit prompts |
| Mobile Safari WebM incompatibility | `MediaRecorder` throws `TypeError` on iOS | Dynamic mimeType detection: WebM → MP4 → browser default fallback |
| 0-byte audio payload | FFmpeg throws fatal decode error | `audioChunks.length === 0` and `audioBlob.size === 0` checks before POST |
| Silence → false "Transcription error" | Empty speech returns `{"success": true, "text": ""}` → falls into else → shows error | Separate `data.success` check from `data.text` validation; silent success does nothing |
| Zombie temp files on hard kill | `SIGKILL` skips `finally: os.unlink()` → orphaned `.webm`/`.mp4` in `/tmp` | Startup cleanup function deletes stale files >1h old on first request |
| Mixed content block on HTTPS | Browser blocks `fetch('http://...')` from HTTPS page | All frontend calls use relative URLs; new ComfyUI calls must proxy through Flask |
| `faster-whisper` version mismatch | Shorthand `"large-v3-turbo"` requires `>=1.0.0`; older version throws `ValueError` | Use explicit HF repo ID `"Systran/faster-whisper-large-v3-turbo"` as fallback |
| Multi-user eviction race | `_ensure_loaded()` evicts model while User 1 is mid-inference → segfault | Check `_active_requests == 0` before TTL-based eviction/reload |
| Mid-inference OOM | ComfyUI spikes VRAM during transcription → CUDA runtime error | Nested try/except: catch OOM → `POST /free` → retry inference |
| Client-side mediaRecorder race | Rapid click overwrites global `mediaRecorder` before `sendTranscription` reads it | Pass `fileExt` as parameter from `onstop` scope (decoupled from global) |
| ComfyUI offloading latency | Delay between State 1 and State 2 | Qwen weights swap between VRAM and system RAM via PCIe; ~1-2s overhead per generation (unavoidable) |
| Mobile Safari MediaRecorder quirks | WebM not supported | Accept `audio/ogg` or `audio/mp4` as fallback MIME types |
| Long audio (>1 min) transcription timeout | Poor UX | VAD filter limits to 30s max speech; show "recording timer" in UI |
| STT model download on first use | Slow first transcription (~30s for 1.6 GB) | Pre-download via script; show "downloading model..." message in UI |

---

## 9. Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `stt_engine.py` | **CREATE** | STT engine using `faster-whisper` INT8 |
| `server_comfy.py` | **MODIFY** | Add `/api/transcribe` endpoint + import stt_engine |
| `templates/index.html` | **MODIFY** | Add 🎤 button, CSS, JS for voice input |
| `requirements.txt` | **MODIFY** | Add `faster-whisper>=1.0.0` |
| `start.sh` | **MODIFY** (optional) | Add STT model pre-download step |

---

## 10. Deployment Notes

### WSGI Multi-Process Trap (Instant OOM)

`threading.RLock` only protects within a **single OS process**. If you wrap Flask in a WSGI server like Gunicorn or uWSGI with multiple workers, each worker loads its own Whisper model independently:

| Config | Workers | Threads | Whisper Loads (peak) | VRAM Impact |
|---|---|---|---|---|
| `gunicorn -w 4` (default) | 4 | 1 | 4 × 1.5 GB = **6 GB** | ❌ OOM with Qwen State 2 |
| `gunicorn -w 1 --threads 10 --timeout 120` | 1 | 10 | 1 × 1.5 GB = **1.5 GB** | ✅ Safe (concurrent STT) |
| `python server_comfy.py` (dev) | 1 | 4 | 1 × 1.5 GB = **1.5 GB** | ✅ Safe (Flask default) |

**Rule:** Always use `-w 1 --threads 10 --timeout 120` for production WSGI deployment.

**Why 10 threads?** With concurrent STT (CTranslate2 shares model weights), 4 simultaneous users need ~2.3 GB VRAM total (1.5 GB shared model + 0.8 GB activations). The lock only protects state (loading/eviction), not inference. With `--threads 4`, all threads could block on `/api/transcribe`, starving other endpoints. `--threads 10` ensures the UI stays responsive even during peak STT load.

**Why `--timeout 120`?** If ComfyUI image generation blocks a thread for >30s, gunicorn's default timeout kills it. `--timeout 120` accommodates queued image jobs.

### Zombie Temp File Cleanup

If the Python process receives a hard kill (`SIGKILL`, power failure, `Ctrl+\\`), the `finally: os.unlink(tmp_path)` block may not execute, leaving orphaned `.webm`/`.mp4` files in `/tmp`. The plan includes a startup cleanup function (`_cleanup_stale_temp_files`) that runs once on the first request, deleting temp audio files older than 1 hour.

### React Framework Compatibility (Forward-Looking)

The current frontend uses vanilla JS (Flask Jinja2 templates), so `promptEl.value` + `dispatchEvent('input')` works natively. If the frontend is later migrated to React, React's synthetic event system overrides native `.value` setters. The fallback is:

```javascript
const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, "value"
).set;
nativeInputValueSetter.call(promptEl, promptEl.value + ' ' + data.text);
promptEl.dispatchEvent(new Event('input', { bubbles: true }));
```

### Mixed Content Security (HTTPS Deployment)

When serving Flask over HTTPS (e.g., via mkcert or ngrok), modern browsers enforce **Mixed Content Policies**. If the frontend makes direct client-side calls to ComfyUI's HTTP endpoint (e.g., `fetch('http://127.0.0.1:8188/...')` or `ws://127.0.0.1:8188/...`), the browser will block them.

**Current state:** The STT frontend uses relative URLs (`/api/transcribe`), so all requests stay within the Flask HTTPS origin — no mixed content issue.

**Rule for future additions:** Any new frontend-to-ComfyUI communication should be reverse-proxied through Flask (e.g., `/api/comfy/queue` → Flask → `http://127.0.0.1:8188/queue`). Backend-to-backend Python `requests` calls are immune to browser mixed-content restrictions.

---

## 11. What NOT to integrate from whisper_local_web

The following from `whisper_local_web` are **out of scope** for this integration:

- **Gradio UI** — Flask SPA is the frontend
- **Parakeet, Canary, Qwen3-ASR, VibeVoice** — too many heavy deps; can be added later if needed
- **Model comparison table** — single model is sufficient for voice-to-prompt
- **Diarization** — not needed for short edit prompts
- **JSON export** — not needed for inline transcription

These can be added in future phases if users demand multi-model comparison or specialized ASR features.

---

## 12. Safari HTTPS & Microphone Access

### The Problem

**`navigator.mediaDevices.getUserMedia()` requires a Secure Context** (HTTPS or `localhost`).

| Access Method | Secure? | Microphone Works? |
|---|---|---|
| `http://localhost:7860` | ✅ (loopback exception) | ✅ |
| `http://192.168.1.223:7860` | ❌ | ❌ (Safari) / ✅ (Chrome, with flag) |
| `http://100.89.22.74:7860` (Tailscale) | ❌ | ❌ (Safari) / ✅ (Chrome, with flag) |
| `https://...` (any) | ✅ | ✅ |

**Desktop Safari (Mac):** Can bypass via `Develop > WebRTC > Allow Media Capture on Insecure Sites`.
**Mobile Safari (iOS):** No bypass — strictly requires HTTPS.

### Solutions (choose one)

#### Option A: mkcert (Free, Local HTTPS, Offline)

```bash
# Install on remote 3090 machine
sudo apt install -y libnss3-tools
sudo snap install mkcert   # or: curl -sS https://install.mkcert.org/install.sh | bash

# Generate cert for the Tailscale IP
mkcert -install
mkcert 100.89.22.74
# → creates 100.89.22.74.pem and 100.89.22.74-key.pem

# On iOS: AirDrop rootCA.pem → Settings → General → About → Certificate Trust Settings → Enable
```

Then update `server_comfy.py`:
```python
ssl_context = None
cert_file = os.environ.get("SSL_CERT")
key_file = os.environ.get("SSL_KEY")
if cert_file and key_file and os.path.exists(cert_file):
    ssl_context = (cert_file, key_file)
    scheme = "https"
else:
    scheme = "http"

app.run(host=host, port=port, threaded=True, debug=False, ssl_context=ssl_context)
```

#### Option B: ngrok (Free, Public URL)

```bash
# Install
 curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | sudo tar xvz -C /usr/local/bin ngrok

# Run tunnel
ngrok http 7860
# → https://randomstring.ngrok-free.app (Safari trusts automatically)
```

#### Option C: Tailscale Standard Plan ($62/user/mo)

Upgrade to Standard plan → MagicDNS + HTTPS certs enabled:
```bash
sudo tailscale cert xh97-ml-3090.ts.net
# → creates .crt and .key files with valid Let's Encrypt certs
```

### Current Status (2026-06-22)

| Machine | Tailscale Domain | HTTPS Cert Support |
|---|---|---|
| Remote 3090 (`xh97-ml-3090`) | `xh97-ml-3090.tailf489c8.ts.net` | ❌ Free plan — "500 Internal Server Error: your Tailscale account does not support getting TLS certs" |
| Local 5090 (`xh97-ml`) | `xh97-ml.tailf489c8.ts.net` | ❌ Same free plan |

**Workaround for now:** Desktop Chrome/Firefox works on plain HTTP. Mobile Safari needs one of the above options.

---


## 13. Client-Side ASR (Primary) + Server-Side Fallback

### 13.1 Overview

**Primary STT engine:** [transcribe-asr](https://github.com/harisnae/transcribe-asr) approach — **Hugging Face Transformers.js** + **ONNX Runtime Web** running Whisper models directly in the browser via WebAssembly.

**Fallback STT engine:** Existing server-side `faster-whisper` (INT8, ~1.5 GB VRAM on RTX 3090).

**Goal:** Client-side ASR is the **default path** for all voice-to-prompt input. Server-side `faster-whisper` activates only when the client-side engine is too slow, unsupported, or produces unsatisfactory results.

### 13.2 Why Client-First?

| Factor | Client-Side (Primary) | Server-Side (Fallback) |
|---|---|---|
| **Latency** | Instant (no network round-trip) | Network-dependent (100ms-2s) |
| **VRAM** | Zero GPU usage on server | ~1.5 GB VRAM (contention with ComfyUI) |
| **Privacy** | Audio never leaves device | Audio uploaded to server |
| **Scalability** | Each device uses its own CPU | Single GPU bottleneck (3-4 users) |
| **Offline** | Works after model cached | Requires network |
| **Quality** | Medium (base.en WER ~3-5%) | Excellent (large-v3-turbo WER ~2%) |
| **First-load cost** | ~78 MB model download | None |

**Rationale:** For typical voice prompts (5-15 seconds), client-side inference on modern devices completes in 3-10 seconds — acceptable UX. The server GPU is freed for ComfyUI image generation, eliminating VRAM contention entirely. Server-side STT is reserved for edge cases where client-side falls short.

### 13.3 How It Works

```
┌────────────────────────────────────────────────────────────┐
│  Browser (Client) — PRIMARY PATH                           │
│                                                            │
│  1. navigator.mediaDevices.getUserMedia() → AudioStream    │
│  2. MediaRecorder → audio/webm (or mp4) Blob               │
│  3. AudioContext.decodeAudioData() → decode to PCM         │
│  4. OfflineAudioContext → resample to 16kHz mono           │
│  5. @huggingface/transformers pipeline(                    │
│       'automatic-speech-recognition', model,               │
│       { device: 'auto' } → ONNX Runtime Web (WASM)        │
│     ) → transcription text                                 │
│  6. Insert text into prompt textarea                       │
│                                                            │
│  Model files cached via Cache API (offline after 1st load) │
└────────────────────────────────────────────────────────────┘
```

**Key technical details:**
- Uses `@huggingface/transformers@3` (CDN: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js`)
- Under the hood: **ONNX Runtime Web** (WASM + SIMD + multi-threading via Web Workers)
- Device auto-detection: `device: 'auto'` → tries WebGPU first, falls back to WASM SIMD
- Audio must be **16kHz mono Float32Array** (Whisper's expected input format)
- Inference runs entirely on **CPU** (WASM) or **WebGPU** (if supported)

### 13.4 Available ONNX Whisper Models

| Model | Repo | Encoder (q4f16) | Decoder (q4f16) | Total (q4f16) | Languages | Quality |
|---|---|---|---|---|---|---|
| **tiny.en** | `harisnaeem/whisper-tiny.en-ONNX` | 6.0 MB | 43.6 MB | **~50 MB** | English | Low (WER ~5-7%) |
| **tiny** (multi) | `harisnaeem/whisper-tiny-ONNX` | 6.3 MB | 44.8 MB | **~51 MB** | 99 langs | Low |
| **base.en** | `harisnaeem/whisper-base.en-ONNX` | 13.5 MB | 64.9 MB | **~78 MB** | English | Medium (WER ~3-5%) |
| **base** (multi) | `harisnaeem/whisper-base-ONNX` | 13.7 MB | 65.8 MB | **~80 MB** | 99 langs | Medium |
| **small.en** | `onnx-community/whisper-small.en` | 63.1 MB | 222.3 MB | **~285 MB** | English | Good (WER ~2.5-3.5%) |
| **large-v3-turbo** | `onnx-community/whisper-large-v3-turbo` | 352.8 MB | 184.3 MB | **~537 MB** | 99 langs | Excellent (WER ~2%) |

**Recommended for client-side primary:**
- **Default choice:** `base.en` q4f16 (~78 MB) — good balance of speed/quality, English-only
- **Multilingual default:** `base` q4f16 (~80 MB) — 99 languages, same speed
- **Quality upgrade (desktop/flagship):** `small.en` q4f16 (~285 MB) — significantly better WER
- **Avoid:** `large-v3-turbo` (~537 MB q4f16, ~1.3 GB fp32) — too large for mobile, slow on CPU

### 13.5 Browser Compatibility

| Browser | Min Version | WebAudio API | Cache API | WASM SIMD | WebGPU | Verdict |
|---|---|---|---|---|---|---|
| **Chrome/Edge** | ≥ 124 | ✅ | ✅ | ✅ | ✅ (opt-in) | ✅ Excellent |
| **Firefox** | ≥ 125 | ✅ | ✅ | ✅ | ✅ (opt-in) | ✅ Good |
| **Safari** | ≥ 17.4 (iOS 17.4, macOS 14.4) | ✅ | ✅ | ✅ | ✅ | ✅ Good |
| **Safari** | 14-17.3 | ✅ | ✅ | ✅ | ❌ | ⚠️ WASM only (slower) |
| **Samsung Internet** | ≥ 26 | ✅ | ✅ | ✅ | ❌ | ⚠️ WASM only |
| **UC Browser / old Android WebView** | < 89 | ⚠️ | ⚠️ | ❌ | ❌ | ❌ Poor → Server fallback |

**Key constraint:** Safari < 17.4 lacks native HEIC decoding AND may have slower WASM performance. The ONNX approach still works (WASM fallback) but inference may take 10-30s for a 10s audio clip on older devices.

### 13.6 Performance Estimates (Client-Side Inference)

| Device | Model | Audio Duration | Est. Inference Time | Memory Usage |
|---|---|---|---|---|---|
| **Desktop Chrome (M1/M2)** | base.en q4f16 | 10s | ~3-5s | ~200 MB |
| **Desktop Chrome (M1/M2)** | base.en q4f16 | 30s | ~8-12s | ~200 MB |
| **iPhone 15 (Safari 17.4+)** | base.en q4f16 | 10s | ~5-8s | ~250 MB |
| **iPhone 13 (Safari 16)** | base.en q4f16 | 10s | ~10-15s | ~300 MB |
| **Android Chrome (Snapdragon 8 Gen 2)** | base.en q4f16 | 10s | ~8-12s | ~250 MB |
| **Low-end Android (Exynos 1280)** | base.en q4f16 | 10s | ~15-25s | ~350 MB |

**Note:** These are estimates based on ONNX Runtime Web benchmarks. Actual performance varies by device, browser version, and background load.

### 13.7 Fallback Triggers (Client → Server)

The system automatically falls back to server-side `faster-whisper` when **any** of these conditions are met:

| Trigger | Detection Method | Action |
|---|---|---|
| **Slow device** | Client inference > 15s for audio < 10s | Show "Switching to server mode..." → POST `/api/transcribe` |
| **OOM on client** | `pipe()` throws RangeError / MemoryError | Auto-fallback to server with toast message |
| **WASM unsupported** | `onnxruntime-web` init fails | Auto-fallback to server on first recording |
| **Repeated re-recordings** | User records ≥ 3 times for same prompt | Auto-fallback: "Server mode activated for better accuracy" |
| **User explicit choice** | Settings toggle or long-press mic button | Switch to server mode until toggled back |
| **CDN/model download fail** | Model fetch timeout or Cache API error | Auto-fallback to server |
| **Network offline + no cache** | `navigator.onLine === false` + Cache miss | Show "Offline — model not cached yet" |

### 13.8 Proposed Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (index.html)                                      │
│                                                             │
│  Recording flow:                                            │
│  1. getUserMedia() → MediaRecorder → Blob                   │
│  2. Attempt client-side transcription (PRIMARY):            │
│     ├─ Check if model loaded → if not, load from CDN/cache  │
│     ├─ Resample audio to 16kHz mono                         │
│     ├─ Run ONNX pipeline → return text                      │
│     └─ Measure inference time                               │
│  3. Evaluate result:                                        │
│     ├─ Success < 15s → Insert text into prompt ✅           │
│     ├─ Success > 15s → Flag for server fallback next time   │
│     ├─ OOM / error → POST /api/transcribe (server fallback) │
│     └─ User re-records ≥ 3× → POST /api/transcribe          │
│  4. Insert text into prompt textarea                        │
│                                                             │
│  Client-side pipeline (lazy-init on first mic press):       │
│  - Load @huggingface/transformers@3 (CDN, ~120 KB gzipped)  │
│  - Load ONNX model (base.en q4f16, ~78 MB, cached after 1st)│
│  - Resample audio to 16kHz mono                             │
│  - Run pipeline('automatic-speech-recognition', ...)        │
│  - Return transcription text                                │
│                                                             │
│  State tracking:                                            │
│  - `clientSTT.loaded` — model ready?                        │
│  - `clientSTT.slowDevice` — inference > 15s detected?       │
│  - `clientSTT.reRecordCount` — consecutive re-recordings    │
│  - `clientSTT.mode` — 'client' | 'server' | 'auto'          │
└─────────────────────────────────────────────────────────────┘
```

### 13.9 Implementation Plan

#### Phase 1: Core Client-Side STT (2-3 days)
1. **Add CDN script tag** — Load `@huggingface/transformers@3` from jsdelivr CDN (lazy-loaded on first mic press)
2. **Create `clientSTT.js` module** — Encapsulate:
   - Model loading (`pipeline('automatic-speech-recognition', ...)`)
   - Audio resampling (16kHz mono via `OfflineAudioContext`)
   - Inference execution with progress callbacks
   - Performance measurement (inference time tracking)
   - Error handling + auto-fallback logic
3. **Modify `sendTranscription()`** — Rewrite to:
   - Try client-side first (if model loaded)
   - On failure/slow → POST `/api/transcribe` (server fallback)
   - Track re-record count for auto-fallback trigger
4. **Add state management** — `clientSTT` global object tracking loaded/slow/mode state

#### Phase 2: UX & Fallback Logic (1-2 days)
5. **Progress indicators** — Show "Loading model..." / "Transcribing..." / progress bar
6. **Fallback notifications** — "Switching to server mode for better accuracy" (toast message)
7. **Re-record detection** — Count consecutive recordings; auto-switch to server after 3 failures
8. **Model caching indicator** — Show "Model cached (78 MB)" badge after first download
9. **Error messages** — Clear messages for OOM, slow devices, unsupported browsers

#### Phase 3: Polish & Optimization (2-3 days)
10. **WebGPU acceleration** — Enable when available (Chrome 113+, Safari 17.4+)
11. **Adaptive model selection** — Auto-choose model size based on device capability:
    - Desktop/flagship → `small.en` (~285 MB) for better quality
    - Mid-range mobile → `base.en` (~78 MB) for speed
    - Low-end mobile → `tiny.en` (~50 MB) for reliability
12. **Offline indicator** — Show "Offline mode active" when Cache API has models
13. **Settings panel** — Allow user to force server mode or client mode

### 13.10 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **OOM on low-end mobile** | Medium | High | Use `tiny.en` (~50 MB) as smallest option; auto-fallback to server |
| **Slow inference (>15s)** | Medium | Medium | Auto-fallback to server; show progress bar; allow cancel |
| **Model download timeout** | Low | Medium | Retry with exponential backoff; show download progress; fallback to server |
| **WASM compatibility** | Low | Medium | Graceful auto-fallback to server STT; clear error message |
| **Cache API quota exceeded** | Low | Low | Use `navigator.storage.estimate()` to check quota; fallback to server |
| **CDN downtime** | Low | Medium | Self-host transformers.js bundle as fallback; server STT always available |
| **Poor client WER → user frustration** | Medium | Medium | Re-record count trigger → auto-switch to server (better quality) |

### 13.11 Model Selection Matrix (Adaptive)

| Device Tier | Client Model | Reason | Fallback Trigger |
|---|---|---|---|
| **Desktop (M1+/modern CPU)** | `small.en` q4f16 (~285 MB) | Fast inference, good quality | Inference > 15s |
| **Desktop (older CPU)** | `base.en` q4f16 (~78 MB) | Balanced quality/speed | Inference > 15s |
| **iPhone 14+ (Safari 17.4+)** | `base.en` q4f16 (~78 MB) | WebGPU available, fast enough | Inference > 15s |
| **iPhone 12-13 (Safari 16-17)** | `base.en` q4f16 (~78 MB) | Acceptable WASM performance | Inference > 15s |
| **iPhone SE / older iOS** | `tiny.en` q4f16 (~50 MB) | Slow CPU, smaller model needed | Inference > 10s (stricter) |
| **Android flagship** | `base.en` q4f16 (~78 MB) | Good CPU, fast enough | Inference > 15s |
| **Android mid-range** | `tiny.en` q4f16 (~50 MB) | Slower CPU, smaller model needed | Inference > 10s (stricter) |
| **Low-end Android / old browser** | Auto-fallback to server | WASM may be slow/missing | Immediate |
| **Multilingual (any device)** | `base` q4f16 (~80 MB) | Same size as base.en, 99 languages | Same thresholds |

### 13.12 Verdict

**✅ Feasible as the primary STT layer.** The `transcribe-asr` approach using Transformers.js + ONNX Runtime Web is well-tested, privacy-first, and works across modern browsers. Key advantages of client-first:
- **Zero VRAM usage** — server GPU fully dedicated to ComfyUI
- **Zero network latency** — no audio upload/download round-trip
- **Privacy-first** — audio never leaves the device
- **Offline-capable** — works after initial model download
- **Scales infinitely** — each user's device does its own work

**Key constraints:**
- **Initial download cost** (~50-285 MB depending on model) — one-time, cached after
- **CPU-bound inference** (3-25s for typical 5-15s prompts) — acceptable UX with progress indicator
- **Memory usage** (~200-350 MB during inference) — may OOM on low-end devices → auto-fallback
- **Quality gap** — client base.en (WER ~3-5%) vs server large-v3-turbo (WER ~2%) → re-record trigger bridges this

**Recommendation:** Implement as **Phase 1** (core client-side STT) with `base.en` q4f16 as the default model. Server-side `faster-whisper` remains as the **fallback engine**, activated automatically on slow devices, OOM, re-record threshold, or explicit user choice.

**Estimated effort:** 3-5 days for Phase 1 (core client-side STT + fallback logic), 5-8 days for full integration with adaptive model selection & UX polish.

**Server-side `faster-whisper` status:** Retained but deprioritized. The existing `stt_engine.py` and `/api/transcribe` endpoint remain unchanged — they serve as the fallback path only. VRAM eviction logic remains useful for when fallback is triggered during active ComfyUI generation.

---

## 14. Change Log & Current Architecture Status

### 14.1 Deployment (2026-06-21 to 2026-06-22)

| Item | Status | Notes |
|---|---|---|
| **Backend STT Engine** | ✅ Deployed | `stt_engine.py` — `faster-whisper` INT8 (`int8_float16`), lazy-load, 120s active eviction |
| **Flask `/api/transcribe`** | ✅ Deployed | Accepts FormData audio + base64 JSON; language param; temp file cleanup |
| **Frontend 🎤 Mic Button** | ✅ Deployed | MediaRecorder API, dynamic mimeType (WebM→MP4→default), 30s auto-stop, 0-byte guard |
| **HTTPS (mkcert)** | ✅ Deployed | Auto-detects cert files; `SSL=0` env var to disable; `https://100.89.22.74:7860` |
| **faster-whisper 1.2.1** | ✅ Installed | `qwen_webui` conda env; CTranslate2 weights cached at `~/.cache/faster-whisper/` |
| **Production Server** | ✅ Running | `https://100.89.22.74:7860` (auto-SSL); backup at `/home/xh97-ml/qwen_image_edit_web_backup/` |

### 14.2 Measured VRAM Behavior

| State | VRAM Usage | Notes |
|---|---|---|
| **Base idle (ComfyUI State 2)** | ~21.8 GB | Qwen workflow loaded |
| **STT model loaded (INT8)** | ~23.2 GB | +1.4 GB (model weights only, no inference) |
| **STT active inference** | ~23.2-23.5 GB | +1.4-1.7 GB (weights + activations) |
| **120s eviction complete** | ~22.1 GB | ~1.1 GB freed (partial — PyTorch caching allocator retains some) |
| **Peak concurrent (4 users)** | ~24.1 GB | 1.5 GB (shared model) + ~0.8 GB (4×200 MB activations) — **fits safely** |

**Measured idle model weight size: 1394 MB** (confirmed via `nvidia-smi` after STT load, before inference).

### 14.3 Bug Fixes Applied

#### 14.3.1 Re-edit Button — Output Image Not Loading (2026-06-22)

**Symptom:** Clicking "🔄 Re-edit" failed to display the current output image as the new input image.

**Root cause:** Server returns images as **data URLs** (`data:image/png;base64,...`) in `result_info['images']`. The re-edit code tried to `fetch()` the data URL, producing a malformed URL like `https://100.89.22.74:7860data:image/png;base64,...`.

**Fix (v6):**
1. Added `lastResultUrl` global variable — set whenever `resultImage.src` is assigned (3 locations in code)
2. Re-edit handler checks if `lastResultUrl` starts with `data:` — if so, uses it directly; otherwise fetches and converts
3. **Files modified:** `templates/index.html` (lines 1236, 1505, 1651, 1985, 1678-1705)

```javascript
// Before: fetch(resultImage.src) → malformed URL for data URLs
// After:
if (lastResultUrl.startsWith('data:')) {
    selectedImageData = lastResultUrl;  // Use directly
    showImagePreview(selectedImageData);
} else {
    const fullUrl = lastResultUrl.startsWith('http') ? lastResultUrl : (window.location.origin + lastResultUrl);
    fetch(fullUrl).then(r => r.blob()).then(blob => { /* convert to data URL */ });
}
```

#### 14.3.2 HEIC Conversion Error — "The string did not match the expected pattern" (2026-06-22)

**Symptom:** Uploading certain HEIC files produced: `"Conversion error: The string did not match the expected pattern."`

**Root cause (two-fold):**
1. **Server-side:** `pillow_heif.read_heif()` threw a cryptic pattern error for certain HEIC file structures (rare but known issue with some camera/app-generated HEIC files)
2. **HTTP 413:** The server returned HTTP 413 (Request Entity Too Large) because Flask's `MAX_CONTENT_LENGTH` was 100MB. The frontend tried to `res.json()` on the HTML error page, which threw the pattern error.

**Fix:**
1. **Client-side decode + downsample (primary):** All images (including HEIC) are now decoded via `<img>` + `<canvas>` in the browser. Modern browsers (Safari, Chrome 124+, Edge, Firefox 125+) decode HEIC natively. No server upload needed!
2. **Server-side fallback (secondary):** Only triggered when the browser can't decode the format. Added Pillow fallback after `pillow_heif` failure.
3. **MAX_CONTENT_LENGTH:** Increased from 100MB → 500MB for large HEIC/DNG files.
4. **413 error handling:** Frontend shows clear message: `"File too large (XXX MB). Max 500 MB."`

**Files modified:**
- `templates/index.html` — `handleFile()` rewritten to use `URL.createObjectURL()` → `<img>` → `<canvas>` → `toDataURL()` (client-side). `handleFileConvert()` retained as server-side fallback.
- `server_comfy.py` — `MAX_CONTENT_LENGTH` increased to 500MB; HEIC conversion block wrapped with Pillow fallback.

```javascript
// Before: Check file extension → if HEIC, POST to server for conversion
// After:
const img = new Image();
const objUrl = URL.createObjectURL(file);
img.onload = () => {
    // Browser decoded successfully → downsample + convert to JPEG
    const canvas = document.createElement('canvas');
    // ... resize to 2048 max edge ...
    selectedImageData = canvas.toDataURL('image/jpeg', 0.85);
    showImagePreview(selectedImageData);
};
img.onerror = () => {
    // Browser couldn't decode → try server conversion
    handleFileConvert(file);
};
img.src = objUrl;
```

#### 14.3.3 STT Replace Mode (2026-06-22)

**Symptom:** Voice transcription appended to existing prompt text, causing duplicate/compound prompts.

**Fix:** Changed `sendTranscription` to **replace** entire prompt instead of appending.

```javascript
// Before: if (promptEl.value.trim()) { promptEl.value += ' ' + data.text; }
// After:
promptEl.value = data.text;  // Replace entirely
```

#### 14.3.4 History & All Jobs — Slow Loading (2026-06-23)

**Symptom:** "My History" and "All Jobs" tabs took several seconds to load.

**Root cause:** The `result_info` column stores **full base64-encoded images** (~1.6 MB per record). The history endpoints returned ALL records with the full base64 data, resulting in ~60 MB of JSON being transferred for a single query.

**Fix:**
1. **`/api/history` (All Jobs):** Strips `result_info['images']` from list view (was already stripping `settings` and `negative_prompt`)
2. **`/api/my-history` (My History):** Strips `result_info['images']` from list view (was returning everything)
3. **Database indexes:** Added indexes on `ip` and `queued_at DESC` for faster queries as the DB grows

**Performance improvement:**
| Endpoint | Before | After |
|---|---|---|
| `/api/history` (All Jobs) | ~60 MB | **15 KB** |
| `/api/my-history` (My History) | ~60 MB | **<1 KB** |

**Files modified:**
- `server_comfy.py` — `history_endpoint()` and `my_history_endpoint()` now strip `result_info['images']`; `_ensure_db()` creates indexes on `ip` and `queued_at DESC`

### 14.4 Code Review Fixes (Applied During Deployment)

The following fixes were applied during the 9 review cycles before deployment:

| Fix | Location | Description |
|---|---|---|
| `threading.Lock()` → `threading.RLock()` | `stt_engine.py` | Nested deadlock prevention (mid-inference OOM handler re-enters lock) |
| `_last_load_time` → `_last_active_time` | `stt_engine.py` | Sliding window TTL (updated on every use, not just load) |
| `_active_requests` counter | `stt_engine.py` | Parallel inference — lock only protects state, not inference execution |
| Eviction race: `_active_requests == 0` check | `stt_engine.py` | Prevents mid-inference eviction/reload |
| Mid-inference OOM recovery | `stt_engine.py` | Nested try/except → `POST /free` → retry inference |
| Timer leak: `_eviction_timer = None` inside `if` | `stt_engine.py` | Only clears reference if eviction actually occurred |
| Flask scope: `tmp_path = None` pre-declaration | `server_comfy.py` | Prevents `UnboundLocalError` in `finally` block |
| Cross-platform: `/tmp/` → `tempfile.gettempdir()` | `server_comfy.py` | Works on Windows/Linux |
| Centralized `onstop` state resets | `templates/index.html` | Prevents UI flicker on recording stop |
| `typeof currentLang` guards | `templates/index.html` | Safe scope evaluation for i18n |
| `fileExt` parameter decoupling | `templates/index.html` | Prevents race condition from rapid clicks |

### 14.5 Current Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  qwen_image_edit_web (Production — https://100.89.22.74:7860)       │
│                                                                     │
│  Flask (port 7860, SSL enabled via mkcert)                          │
│  ├── templates/index.html                                           │
│  │   ├── 🎤 Mic button → MediaRecorder → /api/transcribe            │
│  │   ├── Image upload → client-side decode + downsample (canvas)    │
│  │   ├── 🔄 Re-edit → lastResultUrl → data URL → showImagePreview   │
│  │   └── History tabs → /api/history, /api/my-history (stripped)    │
│  ├── server_comfy.py                                                │
│  │   ├── /api/transcribe → stt_engine.py (faster-whisper INT8)      │
│  │   ├── /api/history → stripped (no base64 images)                 │
│  │   ├── /api/my-history → stripped (no base64 images)              │
│  │   ├── /api/convert-image → HEIC/DNG fallback (Pillow + heif)     │
│  │   └── /outputs/<filename> → serve generated images               │
│  ├── stt_engine.py                                                  │
│  │   ├── faster-whisper large-v3-turbo (INT8, ~1.4 GB idle)         │
│  │   ├── Lazy-load, 120s active eviction, RLock thread safety       │
│  │   └── Mid-inference OOM recovery (POST /free → retry)            │
│  └── memory_engine.py (unchanged)                                   │
│                                                                     │
│  Conda env: qwen_webui (Python 3.12, Flask 3.1.3, faster-whisper 1.2.1) │
│  SQLite DB: history.db (~57 MB, 38 records, indexed on ip/queued_at)│
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
   ComfyUI (8188) — Qwen Image Edit 2511 workflow (~21.5 GB VRAM)
```

### 14.6 Remaining Work

| Item | Status | Priority |
|---|---|---|
| **Client-side ASR (Section 13)** | 📋 Planned (not started) | Medium |
| **gunicorn production deployment** | 📋 Planned | Low (dev server works for now) |
| **HTTPS trust documentation** | 📋 Planned | Low (mkcert root CA needs to be trusted on each client device) |
| **VRAM fragmentation monitoring** | 🔍 Monitoring | Low (stable so far) |
| **History DB archival** | 📋 Planned | Low (38 records, 57 MB — manageable for now) |
