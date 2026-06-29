# Deployment Comparison: 5090 (local) vs 3090 (remote)

## Summary

| Aspect | 5090 (local) | 3090 (remote) | Impact |
|--------|-------------|---------------|--------|
| **server_comfy.py** | ✅ Same (md5 match) | ✅ Same | None |
| **ComfyUI version** | 0.24.0 | 0.24.0 | None |
| **PyTorch** | 2.12.0.dev+cu130 | 2.12.0+cu130 | Negligible |
| **Python** | 3.13 | 3.14.4 | None |
| **cuDNN** | 91900 | 92000 | None |
| **VRAM state** | NORMAL_VRAM | NORMAL_VRAM | None |
| **llama-server** | ⚠️ RUNNING (~29.6GB) | ✅ NOT RUNNING | **MAJOR** |
| **Model files** | ⚠️ Symlinks + extras | ✅ Clean originals | Minor |
| **ComfyUI dir** | ~/comfy-blackwell | ~/ComfyUI | None |
| **Conda envs** | comfy-bw / qwen_web | comfyui / qwen_webui | None |

---

## Key Differences

### 1. llama-server (MAJOR)

| | 5090 | 3090 |
|---|------|------|
| **Status** | RUNNING (PID 856328) | NOT RUNNING |
| **GPU Memory** | 29.6 GB | 0 GB |
| **Available for ComfyUI** | ~2.4 GB | 24.1 GB |

**Impact:** On the 5090, ComfyUI must share the GPU with llama-server. With only ~2.4GB available, ComfyUI uses heavy CPU offloading, resulting in:
- Higher reported VRAM usage (~29GB total)
- Slower inference
- Potential OOM errors

### 2. Model Files

| | 5090 | 3090 |
|---|------|------|
| **diffusion_models** | Symlink `qwen_image_edit_2511_fp8_e4m3fn.safetensors` → `qwen_image_edit_fp8_e4m3fn.safetensors` (19.5GB) | Original file (19.5GB) |
| **loras** | Symlink `Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors` → `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (810MB) | Original file (810MB) |
| **Extra models** | Many (flux, ideogram, wan, chroma, etc.) | Only qwen models |

**Impact:** The symlinks resolve to the same files. The extra models on the 5090 could cause ComfyUI to load additional models at startup, consuming more VRAM.

### 3. ComfyUI Directory

| | 5090 | 3090 |
|---|------|------|
| **Path** | ~/comfy-blackwell | ~/ComfyUI |
| **Custom Nodes** | Many (ComfyUI-Manager, etc.) | Fewer |

**Impact:** More custom nodes on the 5090 could add overhead at startup.

---

## GPU Memory Breakdown

### 5090 (32.6GB total)
```
llama-server:     29.6 GB  (91%)
ComfyUI (idle):    0.5 GB  (1.5%)
Other:             0.0 GB
Free:              1.9 GB  (6%)
```

### 3090 (24.6GB total)
```
ComfyUI (idle):   21.7 GB  (88%)
Other:             0.0 GB
Free:              2.4 GB  (10%)
```

---

## Root Cause of VRAM Difference

The 5090 uses ~29GB because **llama-server is running**, not because of any ComfyUI configuration difference. When llama-server is stopped:

```
5090 without llama-server:
  ComfyUI (idle):  ~0.5 GB
  Free:           ~32.1 GB (98%)
```

The 3090 uses ~21.7GB because ComfyUI loads the full model stack into VRAM without any CPU offloading needed (24GB is sufficient).

---

## Recommendations

1. **Stop llama-server on 5090** when running image editing jobs:
   ```bash
   ~/qwen_image_edit_web/kill.sh llama
   ```

2. **Clean up extra models on 5090** if they're not needed:
   - Move non-qwen models to a separate directory
   - Reduces ComfyUI startup time and VRAM usage

3. **Fix symlinks on 5090** to point to original files (not symlinks to symlinks):
   ```bash
   cd ~/comfy-blackwell/models/diffusion_models
   rm qwen_image_edit_2511_fp8_e4m3fn.safetensors
   ln -s qwen_image_edit_fp8_e4m3fn.safetensors qwen_image_edit_2511_fp8_e4m3fn.safetensors
   ```
