# Deployment Analysis — Qwen-Image-Edit-2511 Web UI

**Date:** 2026-06-19
**Scope:** Choosing between RTX 5090 (local) vs RTX 3090 (remote) for testing and production deployment

---

## Hardware Comparison

| Spec | Local (5090) | Remote (3090) |
|------|-------------|---------------|
| **GPU** | RTX 5090, 32GB VRAM | RTX 3090, 24GB VRAM |
| **GPU Perf** | ~40-60% faster (Ada Lovelace 2nd gen) | Baseline |
| **CPU** | Ryzen 7 7800X3D (8C/16T, 5.05GHz, 96MB L3) | i5-12600K (10C/16T, 4.9GHz, 20MB L3) |
| **RAM** | 64GB | 32GB |
| **Disk Free** | 328GB (82% used) | 1.6TB (11% used) |
| **LAN** | 192.168.1.223 | 192.168.90.129 |
| **Tailscale** | 100.121.112.83 | 100.89.22.74 |
| **Inter-machine latency** | \multicolumn{2}{c|}{~22ms (Tailscale), ~8.7MB/s throughput} |
| **Current GPU usage** | llama-server (Qwen3.6-27B, 30GB) | ComfyUI + Qwen-Image-Edit model (21.8GB) |
| **Current services** | llama-server :8080 | ComfyUI :8188, Flask :7860 |

---

## Model VRAM Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Transformer (FP8) | 20GB | `qwen_image_edit_2511_fp8_e4m3fn.safetensors` |
| Text Encoder (FP8) | 8.8GB | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | 243MB | `qwen_image_vae.safetensors` |
| Lightning LoRA | 811MB | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0.safetensors` |
| **Static weights total** | **~29.1GB** | |
| **Runtime activations** | **~3-4GB** | During inference (KSampler, VAEDecode) |
| **Total needed** | **~32-33GB** | Peak during generation |

### VRAM Fit Analysis

| GPU | FP8 fit? | BF16 fit? | Notes |
|-----|----------|-----------|-------|
| **3090 (24GB)** | ✅ Yes (with CPU offload) | ❌ No (needs ~55GB) | Current setup: loads fine, uses 21.8GB |
| **5090 (32GB)** | ✅ Yes (tight, ~32-33GB) | ⚠️ Maybe (with aggressive offload) | Needs ~1-2GB overhead for OS/drivers |

**Key insight:** FP8 is the only viable quantization on both GPUs. The 5090's 8GB advantage is mostly consumed by activation memory, leaving ~2-4GB headroom for larger batch sizes or higher resolution.

---

## The Fundamental Constraint: One Workload Per GPU

Both GPUs are at ~90% capacity with their current workloads:

```
3090: [████████████████████░░░░] 21.8/24GB  (Qwen-Image-Edit)
5090: [████████████████████████░] 30/32GB    (Qwen3.6-27B LLM)
```

**Neither machine can run BOTH the LLM and the image-edit model simultaneously.**

This means any deployment strategy requires a **choice** or a **swap**.

---

## Deployment Options

### Option A: Keep Current (3090 = Production, 5090 = LLM)

**Setup:** 3090 serves the web UI, 5090 serves the LLM.

```
User → [Web UI :7860] → [ComfyUI :8188] → RTX 3090
User → [llama-server :8080] → RTX 5090
```

**Pros:**
- ✅ **Zero setup** — already running and proven
- ✅ **Model pre-loaded** — no cold start penalty (118s first run)
- ✅ **Dedicated GPU** — no contention with LLM
- ✅ **Plenty of disk** — 1.6TB free for history/outputs
- ✅ **Lightning mode works** — 14s inference (4 steps)
- ✅ **Proven stable** — running since Jun 16

**Cons:**
- ❌ **3090 is slower** — ~40-60% slower than 5090 for same model
- ❌ **24GB VRAM is tight** — limited to 1 image at a time, FP8 only
- ❌ **Normal mode slow** — 116s for 20 steps (vs ~70-80s estimated on 5090)
- ❌ **No BF16 option** — quality ceiling is FP8

**Best for:** Immediate production, low traffic (1-5 users), cost-sensitive

**Verdict:** 🟢 **Best for production RIGHT NOW** — proven, stable, zero setup

---

### Option B: Swap Roles (5090 = Image Edit, 3090 = LLM)

**Setup:** Move image-edit to 5090, move LLM to 3090.

```
User → [Web UI :7860] → [ComfyUI :8188] → RTX 5090  (image edit)
User → [llama-server :8080] → RTX 3090              (LLM)
```

**Pros:**
- ✅ **Faster inference** — 5090 is ~40-60% faster (est. 70s normal, 9s lightning)
- ✅ **More VRAM headroom** — 32GB allows larger batches, higher resolution
- ✅ **BF16 possible** — if model fits with offloading (quality boost)
- ✅ **Better CPU** — 7800X3D's 96MB L3 cache helps with preprocessing

**Cons:**
- ❌ **Needs full setup** — ComfyUI install, model download (~30GB), Flask setup
- ❌ **Disk is tight** — 328GB free, model takes 30GB → 298GB remaining
- ❌ **llama-server must move** — Qwen3.6-27B (22GB) fits on 3090 but barely
- ❌ **Downtime during migration** — ~30-60 minutes of setup + testing
- ❌ **3090 for LLM is tight** — 22GB model + 2.2GB free = risk of OOM with long contexts

**Migration effort:**
1. Kill llama-server on 5090 (`~30GB freed`)
2. Install ComfyUI on 5090 (use existing `comfy-bw` env)
3. Transfer model files via Tailscale (~30GB, ~3-5 min)
4. Set up Flask + web UI on 5090
5. Move llama-server to 3090 (kill ComfyUI, transfer model, restart)
6. Test both services

**Verdict:** 🟡 **Best for performance** — but requires ~1-2 hours migration

---

### Option C: Hybrid (3090 = Production, 5090 = Testing/Staging)

**Setup:** Keep 3090 as production. Deploy a parallel instance on 5090 for testing.

```
Production:  User → [3090 :7860]  (always available)
Testing:     Dev → [5090 :7861]   (code changes, new features)
```

**Pros:**
- ✅ **Zero production downtime** — 3090 stays live
- ✅ **Safe testing** — break anything on 5090 without affecting users
- ✅ **A/B testing** — compare FP8 vs BF16, different step counts, new features
- ✅ **Gradual migration** — test on 5090, promote to 3090 when ready
- ✅ **Code review fixes** — apply and test all 53 findings safely

**Cons:**
- ❌ **Duplication** — two instances to maintain
- ❌ **5090 needs model download** — 30GB of disk used
- ❌ **llama-server must be paused** during image-edit testing (can't run both on 5090)
- ❌ **Extra setup** — need separate ComfyUI + Flask on 5090

**Workflow:**
```
1. Deploy staging on 5090 (kill llama-server temporarily)
2. Apply code review fixes, test thoroughly
3. Copy tested code to 3090 production
4. Restart 3090, bring llama-server back on 5090
5. Repeat for each feature cycle
```

**Verdict:** 🟢 **Best for development workflow** — safest path forward

---

### Option D: 5090 for Image Edit, LLM on CPU/Cloud

**Setup:** 5090 for image edit, move LLM to a cloud API (OpenAI, Anthropic) or CPU inference.

```
User → [Web UI :7860] → [ComfyUI :8188] → RTX 5090  (image edit)
User → [Cloud API]                                       (LLM)
```

**Pros:**
- ✅ **Best image-edit performance** — 5090 dedicated
- ✅ **No LLM migration** — cloud handles it
- ✅ **Scalable** — LLM scales independently

**Cons:**
- ❌ **LLM cost** — cloud API per-token pricing
- ❌ **Latency** — cloud API adds 200-500ms per request
- ❌ **Privacy** — prompts sent to external provider
- ❌ **Still needs setup** on 5090

**Verdict:** 🔵 **Best for long-term scaling** — if LLM is optional or can be cloud-hosted

---

## Recommendation Matrix

| Scenario | Recommended Option | Why |
|----------|-------------------|-----|
| **Ship today** | **A** (3090 production) | Already running, proven |
| **Best performance** | **B** (swap to 5090) | 40-60% faster inference |
| **Safe development** | **C** (hybrid) | Test on 5090, serve from 3090 |
| **Long-term scaling** | **D** (5090 + cloud LLM) | Independent scaling |

---

## Recommended Path: C → B (Hybrid, then Swap)

### Phase 1: Deploy Staging on 5090 (1-2 hours)

```bash
# On 5090 (local):
# 1. Kill llama-server temporarily
kill $(pgrep llama-server)

# 2. Clone ComfyUI into comfy-bw env
cd /home/xh97
git clone https://github.com/comfyanonymous/ComfyUI.git

# 3. Transfer model files from 3090
rsync -avz --progress xh97-ml@100.89.22.74:/home/xh97-ml/ComfyUI/models/ \
    /home/xh97/ComfyUI/models/

# 4. Start ComfyUI on 5090
/home/xh97/miniconda3/envs/comfy-bw/bin/python main.py --port 8188

# 5. Deploy Flask on 5090 (port 7861 to avoid conflict)
cd /home/xh97/qwen-image-edit-web
# Update server_comfy.py COMFYUI_URL to localhost:8188
/home/xh97/miniconda3/envs/comfy-bw/bin/python server_comfy.py
```

### Phase 2: Apply Code Review Fixes on Staging (ongoing)

- Fix all 53 findings from CODE_REVIEW.md
- Test on staging (5090) without affecting production (3090)
- Verify mobile/iOS Safari compatibility
- Benchmark performance differences

### Phase 3: Swap to Production (when ready)

```bash
# On 3090 (remote):
# 1. Kill ComfyUI + Flask
kill $(pgrep -f "main.py --port 8188")
kill $(pgrep -f "server_comfy.py")

# 2. Start llama-server (move LLM to 3090)
# Transfer Qwen3.6-27B model if not present
llama-server -hf HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive:Q6_K_P \
    --port 8080 --n-gpu-layers 99

# On 5090 (local):
# 3. Kill staging Flask
kill $(pgrep -f "server_comfy.py")

# 4. Start production Flask on 5090
cd /home/xh97/qwen-image-edit-web
python server_comfy.py  # now on port 7860
```

### Phase 4: Update DNS / Tailscale Routes

- Update any bookmarks/links from `100.89.22.74:7860` → `100.121.112.83:7860`
- Or set up a Tailscale MagicDNS name for easy routing

---

## Performance Comparison (Estimated)

| Metric | 3090 (current) | 5090 (estimated) | Improvement |
|--------|---------------|-----------------|-------------|
| **Lightning (4 steps)** | 14s | ~9-10s | 30% faster |
| **Normal (20 steps)** | 116s | ~70-80s | 40% faster |
| **Max batch size** | 1 image | 1-2 images | 2x throughput |
| **Max resolution** | 1024×1024 | 1024-1536 | 20-50% larger |
| **Model quality** | FP8 only | FP8 (+ BF16 possible) | Higher ceiling |
| **First-run cold start** | 118s | ~80-90s | 25% faster |

---

## Network Considerations

| Path | Latency | Bandwidth | Notes |
|------|---------|-----------|-------|
| **Same machine (localhost)** | <1ms | ~10GB/s | ComfyUI ↔ Flask |
| **Tailscale (cross-machine)** | ~22ms | ~8.7MB/s | For model transfer, API calls |
| **LAN 192.168.1.x** | ~1-2ms | ~100MB/s | Local network users |
| **LAN 192.168.90.x** | ~1-2ms | ~100MB/s | Remote machine LAN |

**For the web UI:** The user's browser connects to Flask (port 7860), which connects to ComfyUI (port 8188) on the **same machine**. The ~22ms Tailscale latency only matters for:
- Model file transfers (one-time, ~30GB)
- Cross-machine API calls (if using Option D)
- Admin management (SSH between machines)

**For end users:** They connect via Tailscale or LAN. The 22ms latency is negligible for the web UI (inference takes 14-116s).

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **3090 OOM during swap** | Medium | High | Keep 3090 running until 5090 is verified |
| **5090 disk fills up** | Low | Medium | Monitor disk, cleanup outputs/history |
| **Model transfer corruption** | Low | Medium | Verify file hashes after rsync |
| **Downtime during migration** | Medium | Medium | Phase C → B approach minimizes downtime |
| **Tailscale connectivity drops** | Low | Low | Both machines have LAN fallback |

---

## Quick Decision Guide

```
Is the web UI already serving users?
├── YES → Stay with Option A (3090), add Option C (5090 staging) for dev
└── NO  → Go with Option B (swap to 5090) for best performance

Do users need BOTH LLM chat + image editing simultaneously?
├── YES → Option A (split: 3090=image, 5090=LLM) or Option D (cloud LLM)
└── NO  → Either machine works, pick based on performance needs

Is this for personal use or public-facing?
├── Personal → Option B (5090, best performance)
└── Public   → Option A → C → B (gradual, safe migration)
```
