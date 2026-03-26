# Phase 8: RADIO Integration - Research

**Researched:** 2026-03-25
**Domain:** NVIDIA C-RADIOv4 model integration as teacher for knowledge distillation
**Confidence:** HIGH

## Summary

Phase 8 integrates C-RADIOv4 models (SO400M and H variants) as teacher models in the autoresearch ReID distillation pipeline. RADIO is an agglomerative vision foundation model that distills multiple teachers (SigLIP2, DINOv3, SAM3) into a single backbone, exposing their capabilities through adaptor heads. Each C-RADIOv4 model provides 3 adaptor outputs (backbone, dino_v3, siglip2-g) plus raw backbone features, giving the agent a rich set of distillation targets.

The primary technical challenges are: (1) loading RADIO from the cloned local repository via `torch.hub.load`, (2) caching both summary and spatial features per adaptor, and (3) implementing spatial distillation loss that aligns the student's pre-GAP spatial features with RADIO's spatial features. A critical finding is that spatial feature caching at full resolution is infeasible with current disk space (329GB available, but a single variant's spatial cache for one adaptor would require ~417GB for 495k samples at 196 tokens x 1152d). The plan must address this constraint -- options include computing spatial features on-the-fly during training, using a smaller input resolution, or caching only for the distillation subset.

**Primary recommendation:** Load RADIO via `torch.hub.load` from local `RADIO/` directory. Cache all summary features to disk (2-3GB per adaptor). For spatial features, use on-the-fly inference during cache-building with batched memory-mapped writes, but strongly consider computing spatial distillation loss online (re-run RADIO per batch during training) given disk constraints, or cache spatial features only for the distillation dataset subset rather than the full 495k samples.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Cache ALL 3 adaptor outputs (backbone, dino_v3, siglip2-g) for each RADIO variant. Disk is cheap (~10GB per adaptor), re-running RADIO inference is expensive.
- **D-02:** Summary features cached as .npy files per sample, native dimension (1152d for SO400M, 1280d for H). Same pattern as other teachers.
- **D-03:** Spatial features cached as memory-mapped .npy files. Format: `{sample_id}_spatial.npy` shape `(N, D)` where N=num_spatial_tokens, D=feature_dim. Memory-mapped to avoid loading entire cache into RAM.
- **D-04:** `RADIO_VARIANT = "so400m"` constant in train.py (or "h"). `RADIO_ADAPTORS = ["backbone"]` list constant -- agent selects which adaptor outputs to distill from.
- **D-05:** Each adaptor gets its own projection head in train.py (because dims may differ across adaptors). Agent can tune projection architectures.
- **D-06:** Spatial distillation loss aligns student's pre-GAP spatial features (from Phase 5 custom LCNet) with RADIO's spatial features. Student spatial is bilinear-interpolated to match RADIO's spatial resolution.
- **D-07:** `SPATIAL_DISTILL_WEIGHT = 0.0` default (disabled). Agent enables by setting positive value. Can be used alongside summary distillation.
- **D-08:** Spatial adapter in train.py: Conv1x1 projecting student channels to RADIO spatial dim, followed by BatchNorm.
- **D-09:** RADIO teachers register as `"radio_so400m"` and `"radio_h"` in the Phase 6 teacher registry. Each variant x adaptor combination is a separate cache.
- **D-10:** Cache directory structure: `workspace/output/teacher_cache/radio_so400m/backbone/`, `workspace/output/teacher_cache/radio_so400m/dino_v3/`, etc.
- **D-11:** RADIO teacher class loads from cloned `RADIO/` directory (user already cloned NVlabs/RADIO).

### Claude's Discretion
- Spatial feature interpolation method (bilinear vs nearest)
- Whether to L2-normalize spatial features before loss computation
- Batch loading strategy for memory-mapped spatial features (pre-load batch or lazy per-sample)
- RADIO model initialization (torch.hub.load vs direct import from cloned repo)
- Cache building batch size for RADIO (balance VRAM vs speed)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RADIO-01 | RADIOTeacher class supporting all C-RADIO variants with adaptor selection | RADIO loading via torch.hub.load verified; hubconf.py `radio_model()` function accepts `version` and `adaptor_names` params; local source loading confirmed in test_example.py |
| RADIO-02 | 3 adaptor outputs: backbone, dino_v3, siglip2-g -- agent selects which to distill from | When adaptors loaded, forward() returns dict with 'backbone', 'dino_v3', 'siglip2-g' keys, each containing RadioOutput(summary, features) NamedTuple |
| RADIO-03 | Each adaptor's summary features cached with native dim, projection in train.py | Backbone summary dim = embed_dim * num_summary_idxs (3456d SO400M, 3840d H); adaptor summary dims determined by MLP output layer at runtime; ~2-3GB per adaptor cache for 495k samples |
| RADIO-04 | Spatial features cached separately with memory-mapped storage | Spatial features shape (N_tokens, D) per sample; 196 tokens for 224x224 input; **CRITICAL: 417-463GB per adaptor per variant -- exceeds available 329GB disk**; must scope to distillation subset or use alternative strategy |
| RADIO-05 | Spatial distillation loss in train.py | Student pre-GAP features from Phase 5 LCNet.encode_with_spatial(); bilinear interpolation to match RADIO spatial grid; Conv1x1+BN adapter; MSE or smooth-L1 loss on aligned features |
| RADIO-06 | RADIO_VARIANT and RADIO_ADAPTORS as tunable constants | Straightforward module-level constants in train.py; RADIO_VARIANT selects version string for torch.hub.load; RADIO_ADAPTORS selects which dict keys to cache/distill from |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.1 | Training framework, model loading | Already installed, pinned |
| torch.hub | (bundled) | Load RADIO models from local clone | Official RADIO loading mechanism per hubconf.py |
| einops | latest | Required by RADIO adaptor_module_factory.py | RADIO internal dependency; already in INFRA-09 scope |
| timm | latest | Used internally by RADIO for ViT creation | Already installed |
| numpy | >=2.2.6 | Summary cache (.npy), spatial cache (mmap) | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| transformers | latest | Required by SigLIP2 adaptor (text encoding) | Only if loading siglip2-g adaptor -- it imports AutoModel |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.hub.load (local) | Direct import of RADIO package | torch.hub.load is simpler and matches official examples; direct import requires sys.path manipulation |
| Memory-mapped .npy | HDF5 / zarr | .npy mmap is simpler, no extra deps; HDF5 would allow chunked access but adds dependency |
| Per-sample .npy files | Single large .npy mmap | Per-sample is simpler for cache invalidation and parallel writes; single file has better sequential read but harder to extend |

## Architecture Patterns

### Recommended Cache Directory Structure
```
workspace/output/teacher_cache/
├── radio_so400m/
│   ├── backbone/
│   │   ├── metadata.json          # {model_version, adaptor, embed_dim, spatial_dim, ...}
│   │   ├── {md5_hash}.npy         # summary: shape (D,) float32
│   │   └── {md5_hash}_spatial.npy # spatial: shape (N, D) float32 (memory-mapped)
│   ├── dino_v3/
│   │   ├── metadata.json
│   │   ├── {md5_hash}.npy
│   │   └── {md5_hash}_spatial.npy
│   └── siglip2-g/
│       ├── metadata.json
│       ├── {md5_hash}.npy
│       └── {md5_hash}_spatial.npy
├── radio_h/
│   ├── backbone/
│   ├── dino_v3/
│   └── siglip2-g/
```

### Pattern 1: RADIO Model Loading (torch.hub.load from local clone)
**What:** Load C-RADIOv4 models using torch.hub.load with local source
**When to use:** In prepare.py RADIOTeacher.__init__()
**Example:**
```python
# Source: RADIO/test_example.py and RADIO/hubconf.py
import torch

# Map user-friendly names to hubconf version strings
RADIO_VERSION_MAP = {
    "so400m": "c-radio_v4-so400m",
    "h": "c-radio_v4-h",
}

def load_radio_model(variant: str, adaptor_names: list[str], device: str = "cuda"):
    """Load a C-RADIOv4 model with specified adaptors."""
    repo = "RADIO"  # local clone path
    version = RADIO_VERSION_MAP[variant]

    model = torch.hub.load(
        repo,
        "radio_model",
        source="local",
        version=version,
        progress=True,
        skip_validation=True,
        adaptor_names=adaptor_names,
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
```

### Pattern 2: RADIO Forward Pass and Feature Extraction
**What:** Extract summary + spatial features from all requested adaptors
**When to use:** During cache building in prepare.py
**Example:**
```python
# Source: RADIO/radio/radio_model.py _extract_final() and test_example.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def extract_radio_features(model, images: torch.Tensor):
    """
    Extract features from RADIO model.

    Args:
        model: RADIOModel loaded with adaptors
        images: tensor [B, 3, H, W] in range [0, 1]

    Returns:
        dict of adaptor_name -> (summary, spatial_features)
        When adaptors loaded, model returns dict:
          {'backbone': RadioOutput(summary, features),
           'dino_v3': RadioOutput(summary, features),
           'siglip2-g': RadioOutput(summary, features)}
    """
    # Ensure resolution is compatible with patch_size
    nearest_res = model.get_nearest_supported_resolution(*images.shape[-2:])
    if images.shape[-2:] != nearest_res:
        images = F.interpolate(images, nearest_res, mode='bilinear', align_corners=False)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(images)  # dict when adaptors loaded

    results = {}
    for name, radio_output in output.items():
        summary = radio_output.summary.float()    # [B, summary_dim]
        features = radio_output.features.float()   # [B, N_tokens, feature_dim] (NLC format)
        results[name] = (summary, features)
    return results
```

### Pattern 3: Spatial Feature Memory-Mapped Caching
**What:** Save spatial features as memory-mapped .npy files for lazy loading
**When to use:** During cache building and training data loading
**Example:**
```python
import numpy as np
from pathlib import Path

def save_spatial_cache(features: np.ndarray, cache_path: Path):
    """Save spatial features as .npy for memory-mapped loading.
    features: shape (N_tokens, D) float32
    """
    np.save(str(cache_path), features.astype(np.float32))

def load_spatial_cache(cache_path: Path) -> np.ndarray:
    """Load spatial features via memory mapping (no RAM allocation)."""
    return np.load(str(cache_path), mmap_mode='r')  # read-only mmap

def load_spatial_batch(cache_paths: list[Path]) -> torch.Tensor:
    """Load a batch of spatial features. Pre-load batch for GPU transfer."""
    batch = []
    for p in cache_paths:
        arr = np.load(str(p), mmap_mode='r')
        batch.append(torch.from_numpy(np.array(arr)))  # copy from mmap to RAM
    return torch.stack(batch)  # [B, N_tokens, D]
```

### Pattern 4: Spatial Distillation Loss
**What:** Align student pre-GAP spatial features with RADIO spatial features
**When to use:** In train.py loss computation
**Example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAdapter(nn.Module):
    """Conv1x1 + BN to project student spatial features to RADIO dim."""
    def __init__(self, student_channels: int, radio_spatial_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, radio_spatial_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(radio_spatial_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H_s, W_s] student spatial features
        return self.bn(self.proj(x))

def spatial_distillation_loss(
    student_spatial: torch.Tensor,   # [B, C, H_s, W_s] from LCNet pre-GAP
    teacher_spatial: torch.Tensor,   # [B, N, D] RADIO spatial (NLC format)
    adapter: SpatialAdapter,
    teacher_grid_h: int,
    teacher_grid_w: int,
) -> torch.Tensor:
    """
    Compute spatial distillation loss between student and teacher spatial features.

    Student spatial is bilinear-interpolated to match teacher spatial resolution,
    then projected via Conv1x1+BN to teacher dim.
    """
    # Reshape teacher from NLC to NCHW for comparison
    B = teacher_spatial.shape[0]
    teacher_nchw = teacher_spatial.reshape(B, teacher_grid_h, teacher_grid_w, -1)
    teacher_nchw = teacher_nchw.permute(0, 3, 1, 2)  # [B, D, H_t, W_t]

    # Interpolate student spatial to match teacher resolution
    student_interp = F.interpolate(
        student_spatial,
        size=(teacher_grid_h, teacher_grid_w),
        mode='bilinear',
        align_corners=False,
    )  # [B, C, H_t, W_t]

    # Project student to teacher dim
    student_proj = adapter(student_interp)  # [B, D, H_t, W_t]

    # MSE loss (or smooth-L1 as in RADIO tech report hybrid loss)
    loss = F.mse_loss(student_proj, teacher_nchw)
    return loss
```

### Pattern 5: RADIOTeacher Class for Phase 6 Registry
**What:** Teacher class following Phase 6 multi-teacher pattern
**When to use:** In prepare.py, registered in teacher registry
**Example:**
```python
class RADIOTeacher:
    """RADIO teacher for multi-teacher distillation pipeline."""

    def __init__(self, variant: str, adaptor_names: list[str], device: str = "cuda"):
        self.variant = variant
        self.adaptor_names = adaptor_names
        self.device = device
        self.model = load_radio_model(variant, adaptor_names, device)

        # Determine dims via a dummy forward pass
        dummy = torch.randn(1, 3, 224, 224, device=device)
        dummy.div_(255.0).clamp_(0, 1)  # RADIO expects [0, 1]
        with torch.no_grad():
            output = self.model(dummy)

        self.feature_dims = {}
        for name, radio_output in output.items():
            self.feature_dims[name] = {
                'summary_dim': radio_output.summary.shape[-1],
                'spatial_dim': radio_output.features.shape[-1],
                'spatial_tokens': radio_output.features.shape[1],
            }

    def get_feature_dim(self, adaptor: str = "backbone") -> int:
        return self.feature_dims[adaptor]['summary_dim']

    @torch.no_grad()
    def encode_batch(self, images: list, adaptor: str = "backbone"):
        """Encode a batch of PIL images, returning summary embeddings."""
        # ... transform images to [0, 1] tensors, run model, return summary
        pass
```

### Anti-Patterns to Avoid
- **Loading RADIO inside train.py:** RADIO model loading and caching belongs in prepare.py (immutable). train.py only reads cached features.
- **Caching adaptor features in a single combined file:** Each adaptor must have its own cache directory because the agent selects which adaptors to use per experiment.
- **Using RADIO's InputConditioner on already-normalized images:** RADIO expects raw images in [0, 1] range and applies its own normalization (CLIP mean/std). Do not pre-normalize.
- **Storing spatial features in float16:** Precision matters for spatial distillation loss. Cache in float32 (the memory overhead is on disk, not in RAM thanks to mmap).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RADIO model loading | Custom weight loading | `torch.hub.load("RADIO", "radio_model", source="local", ...)` | hubconf.py handles spectral reparam, EMA weights, adaptor creation, feature normalizer -- 200+ lines of non-trivial setup |
| Image preprocessing for RADIO | Custom normalization | RADIO's built-in `InputConditioner` (auto-applied in forward()) | Uses CLIP mean/std with specific scaling; model handles this internally |
| Adaptor MLP architecture | Reimplementation of adaptor heads | Load via `adaptor_names` parameter in `radio_model()` | Adaptor weights and architecture are part of the checkpoint; creating them from scratch would be wrong |
| Resolution alignment | Manual padding/cropping | `model.get_nearest_supported_resolution(H, W)` + `F.interpolate` | RADIO has specific resolution step constraints (patch_size * window_size) |

## Common Pitfalls

### Pitfall 1: Spatial Cache Disk Exhaustion
**What goes wrong:** Spatial features for 495k samples at 196 tokens x 1152d float32 = 417GB per adaptor. With 329GB available disk, even one complete spatial cache cannot fit.
**Why it happens:** User estimate of ~10GB per adaptor was for summary features only, not spatial.
**How to avoid:** Three strategies: (a) Cache spatial features only for the distillation training subset (if smaller than full dataset), (b) compute spatial features on-the-fly during training by running RADIO per batch (feasible if batch is small; ~2-4GB VRAM for RADIO), (c) use float16 for spatial cache (halves to ~208GB, still tight). **Recommendation:** Cache summaries to disk for all samples, compute spatial features on-the-fly during training batches when SPATIAL_DISTILL_WEIGHT > 0.
**Warning signs:** Disk fills up during cache building; training crashes with IOError.

### Pitfall 2: RADIO Input Range Mismatch
**What goes wrong:** Passing images normalized with ImageNet mean/std to RADIO, which expects [0, 1] range images and applies its own CLIP normalization internally.
**Why it happens:** Existing teacher (Trendyol/DINOv2) uses ImageNet normalization. RADIO has a built-in InputConditioner that expects raw [0, 1] pixel values.
**How to avoid:** In the RADIOTeacher.encode_batch(), transform PIL images to tensors in [0, 1] range (just `pil_to_tensor(img).float().div_(255.0)`), do NOT apply ImageNet normalize. RADIO's InputConditioner handles normalization.
**Warning signs:** RADIO outputs are garbage (very high loss, random features).

### Pitfall 3: Backbone Summary Dim Confusion
**What goes wrong:** Assuming backbone summary is embed_dim (1152 for SO400M), but it is actually embed_dim * num_summary_idxs (3456 for SO400M with 3 teachers). The summary is flattened CLS tokens from all teacher slots.
**Why it happens:** `radio_model.py` line `RadioOutput(bb_summary.flatten(1), ...)` flattens the multi-CLS-token summary. With 3 teachers (SigLIP2, DINOv3, SAM3), there are 3 summary indices.
**How to avoid:** Always determine dims via a dummy forward pass at initialization time (as shown in Pattern 5). Never hardcode dims.
**Warning signs:** Linear projection layer size mismatch error.

### Pitfall 4: Resolution Step Constraint
**What goes wrong:** Passing 224x224 images to RADIO when min_resolution_step requires a different alignment, causing a ValueError.
**Why it happens:** C-RADIOv4 with ViTDet has `min_resolution_step = patch_size * window_size`. Without ViTDet (the default loading), `min_resolution_step = patch_size = 16`. 224 is divisible by 16, so this works. But if someone uses a different input size (e.g., 240), it would fail.
**How to avoid:** Always use `model.get_nearest_supported_resolution(H, W)` before inference. For our 224x224 images, this returns (224, 224) which is fine.
**Warning signs:** ValueError about resolution not being a multiple of min_resolution_step.

### Pitfall 5: SigLIP2 Adaptor Requires transformers
**What goes wrong:** Loading the siglip2-g adaptor triggers `from transformers import AutoModel, AutoProcessor` which downloads the SigLIP2 text model (~2.4GB). This is unnecessary for our vision-only distillation use case.
**Why it happens:** The SigLIP2Adaptor class always downloads the text model for text encoding support.
**How to avoid:** For vision-only distillation, the siglip2-g adaptor's **visual** features (summary + spatial) are what we want, and these come from the adaptor's MLP head, not from the SigLIP2 text model. The text model download is triggered in `__init__`. Options: (a) accept the one-time download, (b) use only backbone and dino_v3 adaptors initially, (c) monkey-patch to skip text model loading. **Recommendation:** Accept the download -- it is a one-time cost and the siglip2-g visual features are valuable.
**Warning signs:** Slow first load, unexpected HuggingFace downloads, potential OOM if text model is loaded on GPU.

### Pitfall 6: Spatial Feature Grid Shape Assumption
**What goes wrong:** Assuming spatial features are always 14x14 (196 tokens for 224x224 input), but if input resolution changes or FeatSharp upsampling is active, the grid changes.
**Why it happens:** Some adaptors may have `upsample_factor > 1` (from FeatSharp), doubling or tripling the spatial resolution.
**How to avoid:** Always read the actual spatial token count from the forward pass output, not from a formula. Store `spatial_tokens`, `grid_h`, `grid_w` in metadata.json.
**Warning signs:** Shape mismatch in spatial distillation loss.

## Code Examples

### Loading RADIO with All 3 Adaptors
```python
# Source: RADIO/test_example.py, RADIO/hubconf.py
import torch
from torch.nn import functional as F

model = torch.hub.load(
    "RADIO",           # local clone directory
    "radio_model",
    source="local",
    version="c-radio_v4-so400m",
    progress=True,
    skip_validation=True,
    adaptor_names=["dino_v3", "siglip2-g"],  # backbone is always included
)
model.cuda().eval()

# Input: [0, 1] range, batch dimension required
x = torch.randn(4, 3, 224, 224, device="cuda").clamp(0, 1)

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(x)

# output is a dict when adaptors are loaded
for name, radio_out in output.items():
    print(f"{name}: summary={radio_out.summary.shape}, spatial={radio_out.features.shape}")
# Expected output (dims need runtime verification):
# backbone: summary=torch.Size([4, 3456]), spatial=torch.Size([4, 196, 1152])
# dino_v3: summary=torch.Size([4, DIM_D3]), spatial=torch.Size([4, N_D3, DIM_D3])
# siglip2-g: summary=torch.Size([4, DIM_SIG]), spatial=torch.Size([4, N_SIG, DIM_SIG])
```

### Memory-Mapped Spatial Feature Loading During Training
```python
# Efficient batch loading for spatial distillation
import numpy as np
import torch

def load_spatial_batch_for_training(
    sample_ids: list[str],
    cache_dir: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Load spatial features for a training batch via memory mapping."""
    batch = []
    for sid in sample_ids:
        path = f"{cache_dir}/{sid}_spatial.npy"
        # mmap_mode='r' maps file into virtual memory without loading
        arr = np.load(path, mmap_mode='r')
        # Copy to contiguous array for torch conversion
        batch.append(torch.from_numpy(np.array(arr)))
    return torch.stack(batch).to(device)  # [B, N_tokens, D]
```

### RADIO-Specific Cache Building
```python
# Source: derived from RADIO API + existing teacher cache pattern
import torch
import numpy as np
from pathlib import Path
import hashlib
import json

@torch.no_grad()
def build_radio_cache(
    model,               # RADIOModel with adaptors
    image_paths: list[str],
    cache_base: str,      # e.g., "workspace/output/teacher_cache/radio_so400m"
    batch_size: int = 32, # Conservative for VRAM
    device: str = "cuda",
    cache_spatial: bool = False,  # WARNING: huge disk usage
):
    """Build summary (and optionally spatial) caches for all loaded adaptors."""
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    # Create adaptor cache dirs
    for adaptor_name in model.adaptors.keys():
        (Path(cache_base) / adaptor_name).mkdir(parents=True, exist_ok=True)
    (Path(cache_base) / "backbone").mkdir(parents=True, exist_ok=True)

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        # Load and preprocess images: RADIO expects [0, 1]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            t = pil_to_tensor(img).float().div_(255.0)
            tensors.append(t)

        images = torch.stack(tensors).to(device)
        nearest_res = model.get_nearest_supported_resolution(*images.shape[-2:])
        if images.shape[-2:] != tuple(nearest_res):
            images = torch.nn.functional.interpolate(
                images, nearest_res, mode='bilinear', align_corners=False
            )

        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)

        # Save per-adaptor per-sample
        for adaptor_name, radio_out in output.items():
            summary = radio_out.summary.float().cpu().numpy()
            spatial = radio_out.features.float().cpu().numpy()

            for j, path in enumerate(batch_paths):
                file_id = hashlib.md5(path.encode()).hexdigest()
                cache_dir = Path(cache_base) / adaptor_name

                # Summary cache (always)
                np.save(str(cache_dir / f"{file_id}.npy"), summary[j])

                # Spatial cache (optional due to disk constraints)
                if cache_spatial:
                    np.save(str(cache_dir / f"{file_id}_spatial.npy"), spatial[j])
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DFN CLIP as RADIO teacher | SigLIP2-g-384 | C-RADIOv4 (Jan 2026) | Better text alignment, more ubiquitous |
| DINOv2 as RADIO teacher | DINOv3-7B | C-RADIOv4 (Jan 2026) | Improved SSL and dense representations |
| SAM as RADIO teacher | SAM3 | C-RADIOv4 (Jan 2026) | Better segmentation capability |
| Cosine summary loss | L_angle (angular dispersion normalized) | C-RADIOv4 (Jan 2026) | Balanced Summary Loss prevents one teacher dominating; relevant for Phase 9 |
| Bilinear spatial upsampling | FeatSharp (learnable upsampling) | RADIOv2.5/FeatSharp (2025) | Sharper spatial features; some adaptors may have upsample_factor > 1 |
| Fixed 2-resolution training | Stochastic resolution sampling | C-RADIOv4 (Jan 2026) | Better multi-resolution support |

## Open Questions

1. **Exact adaptor output dimensions**
   - What we know: Backbone summary is embed_dim * num_summary_idxs. Adaptor summary dims are set by the trained MLP output layer.
   - What's unclear: Exact dims for dino_v3 and siglip2-g adaptors on C-RADIOv4 models (DINOv3-7B output is 1536d but adaptor may project differently).
   - Recommendation: Determine at runtime via dummy forward pass. Store in metadata.json during cache build.

2. **Spatial cache disk budget**
   - What we know: 495k samples, 196 tokens, 1152d float32 = 417GB for one adaptor of SO400M. Only 329GB disk available.
   - What's unclear: Whether user can free disk space, or whether distillation subset is smaller.
   - Recommendation: Default to spatial-features-on-the-fly (RADIO runs per training batch when spatial loss is enabled). Cache only summaries. If disk becomes available later, spatial caching can be added.

3. **SigLIP2 adaptor text model download behavior**
   - What we know: SigLIP2Adaptor.__init__ downloads google/siglip2-giant-opt-patch16-384 text model (~2.4GB).
   - What's unclear: Whether it loads to GPU by default, whether it can be skipped for vision-only use.
   - Recommendation: Test during implementation. If problematic, defer siglip2-g adaptor or pre-download on CPU.

4. **FeatSharp upsample_factor on adaptors**
   - What we know: Some adaptors may have upsample_factor > 1, multiplying spatial tokens.
   - What's unclear: Whether C-RADIOv4 SO400M/H adaptors use FeatSharp upsampling.
   - Recommendation: Check at runtime from loaded model. If upsampling is active, spatial token count will be larger than (H/patch_size)^2.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | Model loading/inference | Yes | 2.9.1 | -- |
| einops | RADIO adaptor_module_factory | Yes | just installed | Must be in pyproject.toml (INFRA-09) |
| timm | RADIO model creation | Yes | installed | -- |
| RADIO/ clone | Teacher loading | Yes | local clone at RADIO/ | -- |
| transformers | SigLIP2 adaptor text model | Yes | installed | Can skip siglip2-g if problematic |
| Disk space | Spatial feature caching | 329GB free | -- | On-the-fly spatial inference; summary-only caching fits easily |

**Missing dependencies with no fallback:**
- None -- all required libraries are available.

**Missing dependencies with fallback:**
- Disk space for full spatial caching (329GB < 417GB needed) -- fallback: on-the-fly spatial inference during training.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (if available) or manual smoke tests |
| Config file | none -- see Wave 0 |
| Quick run command | `python -c "import sys; sys.path.insert(0,'RADIO'); from radio.common import RESOURCE_MAP; print('OK')"` |
| Full suite command | `python train.py` (10-epoch run with RADIO teacher) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RADIO-01 | RADIOTeacher loads and produces embeddings | smoke | `python -c "from prepare import RADIOTeacher; t = RADIOTeacher('so400m', ['backbone']); print(t.get_feature_dim('backbone'))"` | Wave 0 |
| RADIO-02 | All 3 adaptors return features | smoke | Load model with all 3 adaptor_names, verify output dict has 3 keys + backbone | Wave 0 |
| RADIO-03 | Summary cache matches expected dims | smoke | Build cache for 10 samples, verify .npy shapes | Wave 0 |
| RADIO-04 | Spatial features have correct shape | smoke | Forward pass, verify features shape is [B, N, D] | Wave 0 |
| RADIO-05 | Spatial distillation loss computes without error | unit | Forward student + load teacher spatial, compute loss, verify scalar output | Wave 0 |
| RADIO-06 | Constants RADIO_VARIANT and RADIO_ADAPTORS control behavior | smoke | Set constants, run, verify correct teacher is loaded | Wave 0 |

### Sampling Rate
- **Per task commit:** Quick smoke test (load model, verify shapes)
- **Per wave merge:** Full 10-epoch training run with RADIO teacher
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Smoke test script for RADIO model loading and feature extraction
- [ ] einops must be installed (INFRA-09 dependency)

## Sources

### Primary (HIGH confidence)
- `RADIO/hubconf.py` -- Model loading API, adaptor creation, weight handling (direct code analysis)
- `RADIO/radio/radio_model.py` -- RADIOModel forward, feature extraction, summary_dim property (direct code analysis)
- `RADIO/radio/common.py` -- RESOURCE_MAP with C-RADIOv4 configs: patch_size=16, preferred_res=512x512 (direct code analysis)
- `RADIO/radio/adaptor_base.py` -- RadioOutput(summary, features) NamedTuple (direct code analysis)
- `RADIO/radio/adaptor_generic.py` -- GenericAdaptor with separate summary/feature MLPs (direct code analysis)
- `RADIO/radio/input_conditioner.py` -- Uses OPENAI_CLIP_MEAN/STD normalization (direct code analysis)
- `RADIO/test_example.py` -- Official usage example with adaptors (direct code analysis)
- `RADIO/RADIOv4.0_tech_report.pdf` -- C-RADIOv4 architecture: SO400M (412M), H (631M), teachers: SigLIP2-g-384, DINOv3-7B, SAM3 (direct reading)

### Secondary (MEDIUM confidence)
- `timm` model registry -- ViT-SO400M embed_dim=1152, ViT-H embed_dim=1280 (verified via timm.create_model)
- Disk usage calculations -- 495k samples verified via filesystem count; cache sizes computed from verified dims

### Tertiary (LOW confidence)
- Exact adaptor output dimensions for dino_v3 and siglip2-g -- must be verified at runtime from loaded checkpoint weights
- FeatSharp upsample_factor on C-RADIOv4 adaptors -- unknown without loading model

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- RADIO code directly analyzed, all dependencies verified
- Architecture: HIGH -- Patterns derived from actual RADIO source code and official examples
- Pitfalls: HIGH -- Disk constraint verified numerically (495k * 196 * 1152 * 4 bytes); input normalization verified from InputConditioner code
- Open questions: MEDIUM -- Require runtime model loading to fully resolve

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- RADIO repo is a published release)
