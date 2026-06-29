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
