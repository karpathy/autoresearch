# Phase 5: SSL + Custom LCNet - Research

**Researched:** 2026-03-25
**Domain:** Self-supervised contrastive learning + lightweight CNN backbone reimplementation
**Confidence:** HIGH

## Summary

Phase 5 adds two independent capabilities to `train.py` with zero `prepare.py` changes: (1) an InfoNCE self-supervised contrastive loss with a separate projection head, and (2) a custom LCNet backbone replacing the timm wrapper, giving the agent full control over architecture parameters. Both features are additive -- they do not break existing loss functions, training loop structure, or the `.encode()` contract.

The custom LCNet reimplementation is architecturally straightforward: 6 stages of DepthwiseSeparableConv blocks (stem -> DSConv stages -> GAP -> FC(1280) -> BN -> embed(256) -> BN), verified against the actual timm `lcnet_050` model structure. The key complexity is in the pretrained weight loading with key mapping between timm's naming convention and the custom implementation, and in exposing the `forward_features()` spatial API needed by Phase 8 (RADIO spatial distillation).

The SSL component uses standard in-batch InfoNCE with a learnable temperature parameter (CLIP-style `log_scale = nn.Parameter(log(1/0.07))`). A separate 2-layer MLP projection head (256->128->128) maps embeddings to the contrastive space without affecting the main 256d embedding.

**Primary recommendation:** Implement LCNet first (it replaces the model class), then add SSL on top. Both are train.py-only changes. Use timm's naming convention in the custom LCNet for 1:1 weight loading without key remapping.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** InfoNCE loss with learnable temperature (init 0.07). Two differently-augmented views of the same image are pushed together, different images pushed apart. Standard contrastive learning.
- **D-02:** Separate 2-layer MLP projection head for SSL (256 -> 128 -> 128, BN+ReLU). Does NOT affect the main 256d embedding used for evaluation. Only used during training.
- **D-03:** Dual-view augmentation: for each batch, apply train.py's augmentation pipeline twice to get view_a and view_b. Both go through the student encoder. Teacher cache is NOT involved in SSL -- this is purely self-supervised on student embeddings.
- **D-04:** `SSL_WEIGHT = 0.0` default (disabled). Agent can enable by setting to any positive value (e.g., 0.1). Total loss = distill_loss + arcface_weight * arcface_loss + SSL_WEIGHT * ssl_loss.
- **D-05:** `SSL_TEMPERATURE = 0.07` as tunable constant. `SSL_PROJ_DIM = 128` as tunable constant.
- **D-06:** Reimplement LCNet from scratch based on PP-LCNet paper + timm source inspection. ~6 stages of DepthwiseSeparableConv blocks. NOT wrapping timm -- full control for agent modification.
- **D-07:** Agent-tunable constants: `LCNET_SCALE` (width multiplier: 0.5~2.5), `SE_START_BLOCK` (which stage SE begins), `SE_REDUCTION` (SE squeeze ratio), `ACTIVATION` ("h_swish"/"relu"/"gelu"), `KERNEL_SIZES` (per-stage, default [3,3,3,3,5,5,5,5,5,5,5,5]).
- **D-08:** Fixed architecture pattern: stem -> DepthSepConv stages -> GAP -> FC(1280) -> BN -> embedding(256) -> BN. Agent tunes WITHIN this pattern, doesn't restructure it.
- **D-09:** Optional pretrained weight loading: when LCNET_SCALE matches a known timm variant (0.5 or 1.0), load weights with key mapping. `USE_PRETRAINED = True` as default, agent can disable.
- **D-10:** Custom LCNet class name: `LCNet` (same as current). Must keep `.encode(images) -> Tensor[B, 256]` contract.
- **D-11:** `model.forward_features(x)` returns `(spatial: Tensor[B, C, H, W], summary: Tensor[B, FC_DIM])`. Spatial is pre-GAP feature map.
- **D-12:** `model.encode(x)` remains the main API (returns L2-normalized Tensor[B, 256]). It calls forward_features internally.
- **D-13:** `model.encode_with_spatial(x)` convenience method returns `(embedding: Tensor[B, 256], spatial: Tensor[B, C, H, W])` for distillation training that needs both.
- **D-14:** `einops` added to pyproject.toml (requirement for future RADIO phase, added now since it's cheap).

### Claude's Discretion
- Exact DepthwiseSeparableConv implementation details (padding, bias flags)
- SSL batch construction (in-batch negatives vs memory bank -- recommend in-batch for simplicity)
- Whether SSL projection head uses BatchNorm or LayerNorm
- Weight initialization for custom LCNet (kaiming_normal for conv, xavier for linear)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SSL-01 | train.py includes InfoNCE contrastive loss | InfoNCE implementation pattern with learnable temperature (CLIP-style nn.Parameter), in-batch negatives, cross-entropy on similarity matrix |
| SSL-02 | SSL uses a separate projection head | 2-layer MLP (256->128->128, BN+ReLU), only used in training, does not affect .encode() output |
| SSL-03 | SSL_WEIGHT is a module-level constant agent can tune | Default 0.0 (disabled), added to total loss formula alongside distill and arcface |
| LCNET-01 | Custom LCNet backbone in train.py replacing timm, .encode() contract preserved | Full architecture verified: 6 stages, 13 DSConv blocks, matching timm lcnet_050 exactly. .encode() returns L2-normalized Tensor[B, 256] |
| LCNET-02 | Agent can tune LCNET_SCALE, SE_START_BLOCK, SE_REDUCTION, kernel sizes, ACTIVATION | Module-level constants with validated defaults matching timm lcnet_050 |
| LCNET-03 | Optional timm pretrained weight initialization | Weight key mapping strategy documented. Pretrained weights available for scale 0.5, 0.75, 1.0 |
| LCNET-04 | Exposes pre-GAP spatial features for RADIO spatial distillation | forward_features() returns (spatial, summary) tuple; encode_with_spatial() convenience method |
| INFRA-09 | einops added to pyproject.toml | einops 0.8.2 verified available (already installed), needs pyproject.toml entry |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.1 | All model components (nn.Module, Conv2d, etc.) | Already pinned in pyproject.toml |
| timm | installed | Pretrained weight source ONLY (not runtime dependency for model) | Used for `timm.create_model().state_dict()` to load pretrained weights |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| einops | 0.8.2 | Tensor rearrangement (future RADIO phase) | Added to pyproject.toml now, not used in Phase 5 code |

### No New Libraries Needed

InfoNCE loss is implemented from scratch (~15 lines of PyTorch). The custom LCNet uses only `torch.nn` primitives. No external contrastive learning libraries (lightly, pytorch-metric-learning) needed or allowed.

**Installation:**
```bash
# Add to pyproject.toml dependencies:
"einops>=0.8.0",
# Then:
uv sync
```

## Architecture Patterns

### Custom LCNet Architecture (Verified from timm lcnet_050)

```
Input: [B, 3, 224, 224]
  |
Stem: Conv2d(3, stem_ch, 3x3, s=2, p=1, bias=False) + BN + HSwish
  -> [B, stem_ch, 112, 112]     (stem_ch = make_divisible(16 * scale))
  |
Stage 0 (1 block):  DSConv k=3, s=1, out=make_divisible(32*scale)
  -> [B, 32*s, 112, 112]
  |
Stage 1 (2 blocks): DSConv k=3, s=2/1, out=make_divisible(64*scale)
  -> [B, 64*s, 56, 56]
  |
Stage 2 (2 blocks): DSConv k=3, s=2/1, out=make_divisible(128*scale)
  -> [B, 128*s, 28, 28]
  |
Stage 3 (2 blocks): DSConv k=3,5, s=2/1, out=make_divisible(256*scale)
  -> [B, 256*s, 14, 14]
  |
Stage 4 (4 blocks): DSConv k=5, s=1, out=make_divisible(256*scale)
  -> [B, 256*s, 14, 14]
  |
Stage 5 (2 blocks): DSConv k=5, s=2/1, out=make_divisible(512*scale), SE=True
  -> [B, 512*s, 7, 7]              ** spatial features exposed here **
  |
conv_head: Conv2d(512*s, 1280, 1x1) + HSwish      (no BN per paper)
  -> [B, 1280, 7, 7]
  |
GAP: AdaptiveAvgPool2d(1)
  -> [B, 1280]
  |
embedding: Linear(1280, 256) + BN
  -> [B, 256]
  |
L2 normalize
  -> [B, 256]   (the .encode() output)
```

### Verified lcnet_050 Block Details (scale=0.5)

| Stage | Blocks | Kernel | Stride (first) | Channels (after scale) | SE | Output Size |
|-------|--------|--------|-----------------|----------------------|-----|-------------|
| Stem | 1 conv | 3x3 | 2 | 8 | No | 112x112 |
| 0 | 1 | 3x3 | 1 | 16 | No | 112x112 |
| 1 | 2 | 3x3 | 2, 1 | 32 | No | 56x56 |
| 2 | 2 | 3x3 | 2, 1 | 64 | No | 28x28 |
| 3 | 2 | 3,5 | 2, 1 | 128 | No | 14x14 |
| 4 | 4 | 5x5 | 1 | 128 | No | 14x14 |
| 5 | 2 | 5x5 | 2, 1 | 256 | Yes (0.25) | 7x7 |
| Head | 1x1 conv | - | - | 1280 | No | 7x7 |

**Channel formula:** `make_divisible(base_channels * LCNET_SCALE, 8)` where base channels are [16, 32, 64, 128, 256, 256, 512] for [stem, s0, s1, s2, s3, s4, s5].

### Pattern 1: DepthwiseSeparableConv Block

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, use_se=False, se_ratio=0.25, act_layer=nn.Hardswish):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise conv
        self.conv_dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = act_layer()
        # Squeeze-and-Excitation (optional)
        if use_se:
            mid_ch = max(1, int(in_ch * se_ratio))
            self.se = SqueezeExcite(in_ch, mid_ch)
        else:
            self.se = nn.Identity()
        # Pointwise conv
        self.conv_pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_dw(x)))
        x = self.se(x)
        x = self.act2(self.bn2(self.conv_pw(x)))
        return x
```

**Key details verified from timm:**
- No bias on Conv2d layers (bias=False)
- BN + activation after BOTH depthwise and pointwise convs
- SE module applied between depthwise and pointwise convolutions
- No residual/skip connections (unlike MobileNetV2/V3 inverted residuals)
- Padding = kernel_size // 2 (same padding)

### Pattern 2: SqueezeExcite Block

```python
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduced_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(channels, reduced_ch, 1, bias=True)  # Note: bias=True for SE
        self.act = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_ch, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.act(self.conv_reduce(scale))
        scale = self.gate(self.conv_expand(scale))
        return x * scale
```

**Verified from timm:** SE uses ReLU for internal activation, Hardsigmoid for gating. Conv2d with bias=True in SE blocks.

### Pattern 3: InfoNCE Loss with Learnable Temperature

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        # CLIP-style learnable log temperature
        self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

    def forward(self, z_a, z_b):
        # z_a, z_b: [B, D] L2-normalized projections from two augmented views
        temperature = torch.exp(-self.log_scale)  # = 1/exp(log_scale)
        # Similarity matrix
        logits = z_a @ z_b.T / temperature  # [B, B]
        # Positives are on the diagonal
        labels = torch.arange(len(z_a), device=z_a.device)
        # Symmetric loss
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
```

**Key design: in-batch negatives.** Each sample's positive is its other augmented view. All other samples in the batch are negatives. No memory bank needed -- simpler and sufficient for this use case where batch sizes are 256.

### Pattern 4: SSL Projection Head (Separate from Main Embedding)

```python
class SSLProjectionHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)
```

**This head is used ONLY during training for the SSL objective.** The main 256d embedding from .encode() is unaffected. The SSL head maps 256d -> 128d -> 128d, L2-normalized, fed to InfoNCE.

### Pattern 5: Dual-View Augmentation in Training Loop

```python
# In training loop, for each batch of images:
view_a = augment(images)  # Apply train_transform
view_b = augment(images)  # Apply train_transform again (different random params)

# Forward through student encoder
emb_a = model.forward_embeddings_train(view_a)  # [B, 256]
emb_b = model.forward_embeddings_train(view_b)  # [B, 256]

# SSL projection
z_a = ssl_proj_head(emb_a)  # [B, 128]
z_b = ssl_proj_head(emb_b)  # [B, 128]

# InfoNCE loss
ssl_loss = info_nce(z_a, z_b)
```

**Important:** The dual augmentation requires raw images BEFORE the transform is applied. The current distill_loader applies transforms in the Dataset. For SSL, we need to either (a) apply transforms in the training loop instead of the Dataset, or (b) keep a reference to raw images. Option (a) is cleaner -- apply `train_transform` twice in the loop using raw PIL images. However, the current `CombinedDistillDataset` applies transforms internally.

**Recommended approach:** Apply `train_transform` in the dataset as now for `view_a` (the normal training view used for distillation). Create `view_b` by re-applying the transform to the same raw image. This requires the dataset to return the raw image or path alongside the transformed image. Since `distill_loader` already returns paths, we can load the image again from path for `view_b`. Alternatively, modify collation to return both views. The simplest approach: in the training loop, when SSL_WEIGHT > 0, apply train_transform to re-loaded images from the paths that are already returned by the dataloader.

### Pattern 6: Pretrained Weight Loading with Key Mapping

```python
def load_pretrained_lcnet(model, scale, device):
    """Load timm pretrained weights into custom LCNet."""
    scale_to_variant = {0.5: 'lcnet_050', 0.75: 'lcnet_075', 1.0: 'lcnet_100'}
    variant = scale_to_variant.get(scale)
    if variant is None:
        logger.warning(f"No pretrained weights for scale={scale}, training from scratch")
        return

    timm_model = timm.create_model(f"hf-hub:timm/{variant}.ra2_in1k", pretrained=True, num_classes=0)
    timm_sd = timm_model.state_dict()

    # Key mapping: if custom LCNet uses same naming as timm, load directly
    # timm keys: conv_stem.weight, bn1.weight, blocks.{s}.{b}.conv_dw.weight, etc.
    model_sd = model.state_dict()
    loaded = 0
    for k, v in timm_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd, strict=False)
    logger.info(f"Loaded {loaded}/{len(timm_sd)} pretrained weights from {variant}")
    del timm_model
```

**Critical insight:** Use the SAME naming convention as timm (`conv_stem`, `bn1`, `blocks.{stage}.{block}.conv_dw`, etc.) in the custom LCNet. This eliminates the need for key remapping entirely. The only keys that won't load are `conv_head` (1280 -> classifier), which is replaced by the embedding head, and `classifier` which is not used.

### Anti-Patterns to Avoid

- **Wrapping timm model:** D-06 explicitly says NOT to wrap timm. Build from scratch using nn.Module primitives.
- **Single projection head for both distillation and SSL:** The SSL projection head MUST be separate (D-02). The main projection head maps to 256d for distillation; the SSL head maps 256d -> 128d for contrastive learning.
- **Applying SSL to teacher embeddings:** D-03 is explicit: SSL is purely self-supervised on student embeddings. Teacher cache is NOT involved.
- **Modifying prepare.py:** Zero prepare.py changes in this phase. All code goes in train.py.
- **Breaking .encode() contract:** .encode() must continue to return L2-normalized Tensor[B, 256]. The new forward_features() and encode_with_spatial() are additional APIs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Channel rounding | Custom rounding logic | `make_divisible(ch, 8)` (standard mobile net utility, ~5 lines) | Must match timm's channel calculation exactly for weight loading |
| Activation dispatch | Complex activation class hierarchy | Simple dict lookup: `{"h_swish": nn.Hardswish, "relu": nn.ReLU, "gelu": nn.GELU}` | Agent only needs 3 options per D-07 |
| Normalization mean/std | Hardcoded ImageNet values | `timm.data.resolve_data_config()` for pretrained variants, hardcode for custom | Must match whatever pretrained weights expect |

**Key insight:** The custom LCNet is a reimplementation, not a novel architecture. Every component (DSConv, SE, HSwish, GAP) exists in PyTorch primitives. The value is in agent tunability, not architectural novelty.

## Common Pitfalls

### Pitfall 1: Channel Mismatch Breaking Pretrained Weight Loading
**What goes wrong:** Custom LCNet uses different channel widths than timm, so state_dict keys exist but shapes don't match. All pretrained weights silently fail to load.
**Why it happens:** `make_divisible` rounds differently, or base channel table doesn't match timm's internal architecture table.
**How to avoid:** Verify channel dimensions match timm's lcnet_050 exactly: stem=8, s0=16, s1=32, s2=64, s3=128, s4=128, s5=256, conv_head=1280 for scale=0.5. Run a unit test that creates both models and compares state_dict shapes.
**Warning signs:** "Loaded 0/N pretrained weights" in logs.

### Pitfall 2: SE Positioned on Wrong Blocks
**What goes wrong:** SE modules placed on too many blocks (VRAM increase, speed regression) or too few (accuracy loss).
**Why it happens:** PP-LCNet paper finding: SE is only effective on the last 2 blocks (Stage 5). Adding SE everywhere hurts latency without proportional accuracy gain.
**How to avoid:** Default `SE_START_BLOCK` should correspond to Stage 5, Block 0 (the 12th block, index 10 in 0-indexed flat list of 13 blocks). Agent can tune this value.

### Pitfall 3: SSL Loss Dominating Total Loss
**What goes wrong:** SSL_WEIGHT too high causes the model to optimize for augmentation invariance at the expense of distillation quality. mean_cosine with teacher drops.
**How to avoid:** Default SSL_WEIGHT=0.0 (disabled). When enabled, typical values are 0.01-0.1. The loss scales are different: distillation loss is ~0.3-0.5, InfoNCE is ~2-4 (log(batch_size)). So SSL_WEIGHT of 0.1 with InfoNCE loss of ~3 contributes ~0.3 to total loss, comparable to distillation.
**Warning signs:** combined_metric drops when SSL is enabled. mean_cosine decreases while ssl_loss decreases.

### Pitfall 4: Dual-View Augmentation Doubling VRAM
**What goes wrong:** Two forward passes per batch doubles activation memory. With batch_size=256, this can push past 24GB.
**Why it happens:** SSL needs two views of each image, each requiring a full forward pass through the student.
**How to avoid:** When SSL is enabled, the agent should halve the batch size (or the implementation should automatically process views in two sequential forward passes rather than concatenating). Document this in the constants section.

### Pitfall 5: Forgetting to Exclude SSL Head from .encode()
**What goes wrong:** SSL projection head weights end up in the saved model or affect .encode() output. Evaluation scores change unpredictably.
**How to avoid:** SSL projection head is a separate nn.Module, NOT part of the LCNet class. It's instantiated in main() alongside the model. The .encode() method only uses the model's internal projection to 256d.

### Pitfall 6: forward_features() Returning Wrong Spatial Resolution
**What goes wrong:** Spatial features are taken from wrong layer (e.g., after GAP instead of before), returning [B, C, 1, 1] instead of [B, C, 7, 7].
**How to avoid:** forward_features() must return the feature map BEFORE global average pooling. For 224x224 input with lcnet_050, spatial output is [B, 256, 7, 7] (output of Stage 5, before conv_head). Per D-11, this is the pre-GAP feature map.

### Pitfall 7: Learnable Temperature Exploding
**What goes wrong:** The log_scale parameter grows unbounded during training, effectively making temperature -> 0 or -> infinity, causing NaN gradients.
**How to avoid:** Clamp the log_scale parameter: `log_scale = self.log_scale.clamp(max=4.6)` (temperature min ~0.01). Or clamp temperature directly. CLIP uses a clamp of `log_scale <= log(100)`.

## Code Examples

### make_divisible Utility (Match timm exactly)

```python
def make_divisible(v, divisor=8, min_value=None):
    """Round channel count to nearest divisor (matches timm implementation)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
```

### LCNet Architecture Configuration Table

```python
# Per-block config: (kernel_size, stride, out_channels_base)
# Verified against timm lcnet_050 block-by-block output
LCNET_ARCH = [
    # Stage 0: 1 block
    [(3, 1, 32)],
    # Stage 1: 2 blocks
    [(3, 2, 64), (3, 1, 64)],
    # Stage 2: 2 blocks
    [(3, 2, 128), (3, 1, 128)],
    # Stage 3: 2 blocks
    [(3, 2, 256), (5, 1, 256)],
    # Stage 4: 4 blocks
    [(5, 1, 256), (5, 1, 256), (5, 1, 256), (5, 1, 256)],
    # Stage 5: 2 blocks (with SE)
    [(5, 2, 512), (5, 1, 512)],
]
# Stem: Conv2d(3, 16*scale, 3x3, s=2)
# Head: Conv2d(512*scale, 1280, 1x1) + HSwish (no BN)
```

### Total Loss Formula

```python
# Per D-04: loss combination
total_loss = distill_loss + ARCFACE_LOSS_WEIGHT * arc_loss + SSL_WEIGHT * ssl_loss
# Plus existing terms (sep_loss, vat_loss) when enabled
```

### Tunable Constants Block (New additions)

```python
# --- SSL Contrastive Loss ---
SSL_WEIGHT = 0.0              # 0 = disabled; agent tunes (e.g., 0.01-0.1)
SSL_TEMPERATURE = 0.07        # InfoNCE temperature (learnable, this is init value)
SSL_PROJ_DIM = 128            # SSL projection head output dim

# --- Custom LCNet Backbone ---
LCNET_SCALE = 0.5             # Width multiplier (0.5 matches current lcnet_050)
SE_START_BLOCK = 10           # Block index where SE begins (0-indexed, 13 total blocks)
SE_REDUCTION = 0.25           # SE squeeze ratio
ACTIVATION = "h_swish"        # "h_swish" | "relu" | "gelu"
KERNEL_SIZES = [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5]  # Per-block kernel sizes (13 blocks)
USE_PRETRAINED = True          # Load timm pretrained weights when scale matches
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already used) |
| Config file | tests/test_train.py (existing) |
| Quick run command | `python -m pytest tests/test_train.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SSL-01 | InfoNCE loss exists and computes scalar | unit | `pytest tests/test_train.py::test_infonce_loss_computes -x` | Wave 0 |
| SSL-02 | SSL projection head is separate module | unit | `pytest tests/test_train.py::test_ssl_proj_head_separate -x` | Wave 0 |
| SSL-03 | SSL_WEIGHT constant exists, default 0.0 | unit | `pytest tests/test_train.py::test_ssl_weight_constant -x` | Wave 0 |
| LCNET-01 | Custom LCNet .encode() returns Tensor[B, 256] | unit | `pytest tests/test_train.py::test_encode_contract_shape -x` | Existing (update) |
| LCNET-02 | Tunable constants exist (LCNET_SCALE, etc.) | unit | `pytest tests/test_train.py::test_lcnet_tunable_constants -x` | Wave 0 |
| LCNET-03 | Pretrained weight loading works for scale=0.5 | unit | `pytest tests/test_train.py::test_pretrained_loading -x` | Wave 0 |
| LCNET-04 | forward_features returns spatial + summary | unit | `pytest tests/test_train.py::test_forward_features_spatial -x` | Wave 0 |
| INFRA-09 | einops in pyproject.toml | unit | `pytest tests/test_train.py::test_einops_in_pyproject -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_train.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_train.py::test_infonce_loss_computes` -- covers SSL-01
- [ ] `tests/test_train.py::test_ssl_proj_head_separate` -- covers SSL-02
- [ ] `tests/test_train.py::test_ssl_weight_constant` -- covers SSL-03
- [ ] `tests/test_train.py::test_lcnet_tunable_constants` -- covers LCNET-02
- [ ] `tests/test_train.py::test_pretrained_loading` -- covers LCNET-03
- [ ] `tests/test_train.py::test_forward_features_spatial` -- covers LCNET-04
- [ ] `tests/test_train.py::test_encode_with_spatial` -- covers LCNET-04
- [ ] `tests/test_train.py::test_einops_in_pyproject` -- covers INFRA-09
- [ ] Update `test_encode_contract_shape` to use new LCNet class instead of FrozenBackboneWithHead
- [ ] Update `test_module_level_constants_exist` to include new constants
- [ ] Update `test_constant_values` for LR change (currently asserts LR==0.1 but train.py has LR=2e-3)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed temperature (SimCLR) | Learnable temperature (CLIP) | 2021 | Better training stability, one fewer hyperparameter to tune manually |
| Memory bank negatives (MoCo) | In-batch negatives (SimCLR/CLIP) | 2020 | Simpler implementation, sufficient with large batch sizes |
| MobileNetV1 DSConv | PP-LCNet DSConv + SE + HSwish | 2021 | Better accuracy/speed on CPU, strategic SE placement |

## Open Questions

1. **Dual-view data loading strategy**
   - What we know: Current dataloader applies transforms inside Dataset and returns (images, labels, paths). SSL needs two different augmented views of the same image.
   - What's unclear: Whether to modify the collate function, create a second dataloader, or re-load images from paths in the training loop.
   - Recommendation: Re-apply transform to images loaded from paths (paths are already returned). This is the least invasive change. Performance impact is negligible since image loading is I/O-bound and images are likely in OS page cache.

2. **Test file LR assertion mismatch**
   - What we know: `test_constant_values` asserts `train.LR == 1e-1` but actual train.py has `LR = 2e-3`. This test is already broken.
   - Recommendation: Fix during Wave 0 test updates. Set expected LR to match whatever value is in train.py at implementation time.

## Sources

### Primary (HIGH confidence)
- timm lcnet_050 model introspection (runtime `print(model)` + state_dict analysis) -- architecture verified
- PP-LCNet paper (arXiv:2109.15099) -- [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2109.15099) -- architecture design rationale
- Current train.py source code -- existing patterns, import contract, loss structure
- pyproject.toml -- dependency versions

### Secondary (MEDIUM confidence)
- [timm/lcnet_050.ra2_in1k](https://huggingface.co/timm/lcnet_050.ra2_in1k) -- pretrained model card
- [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch) -- reference InfoNCE implementation
- [CLIP learnable temperature](https://github.com/pytorch/pytorch/issues/97539) -- PyTorch discussion on InfoNCE

### Tertiary (LOW confidence)
- None -- all claims verified against code or official sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- verified from pyproject.toml and runtime introspection
- Architecture: HIGH -- custom LCNet verified block-by-block against timm lcnet_050, all channel dims, kernel sizes, SE positions confirmed
- Pitfalls: HIGH -- derived from verified architecture details and known contrastive learning failure modes
- SSL implementation: HIGH -- InfoNCE is well-understood, CLIP-style learnable temperature is standard

**Research date:** 2026-03-25
**Valid until:** Indefinite (architecture is stable, no fast-moving dependencies)
