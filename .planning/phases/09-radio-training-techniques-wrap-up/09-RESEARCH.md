# Phase 9: RADIO Training Techniques + Wrap-up - Research

**Researched:** 2026-03-25
**Domain:** Multi-teacher knowledge distillation training techniques + agent instruction authoring
**Confidence:** HIGH

## Summary

Phase 9 implements seven RADIO-inspired training techniques as agent-tunable ENABLE_* toggles in train.py, and rewrites program.md for the full v2.0 search space. The techniques span three categories: (1) distribution balancing (PHI-S, Feature Normalizer), (2) loss formulation improvements (L_angle, Hybrid Loss), and (3) architecture upgrades (Adaptor MLP v2, FeatSharp, Shift Equivariant Loss).

The mathematical formulations for all seven techniques are fully specified in the RADIO papers (PHI-S paper arXiv:2410.01680, RADIOv2.5 arXiv:2412.07679, C-RADIOv4 tech report, FeatSharp ICML 2025). The key implementation challenge is adapting techniques designed for large-scale ViT-to-ViT distillation (600k iterations on DataComp-1B) to our small-scale CNN student setup (10 epochs on product ReID data). All techniques must be self-contained in train.py using only torch, numpy, and einops -- no scipy (not available), no new dependencies.

**Primary recommendation:** Implement techniques in priority order from D-01 (PHI-S first, then Feature Normalizer, Hybrid Loss, L_angle, Adaptor MLP v2, FeatSharp, Shift Equivariant Loss). Each technique is an nn.Module or function with an ENABLE_* flag. The program.md rewrite should document the full v2.0 search space including all 5 teachers, custom LCNet, SSL, and these techniques.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Implementation priority (mandatory to optional): PHI-S > Feature Normalizer > Hybrid Loss > L_angle > Adaptor MLP v2 > FeatSharp > Shift Equivariant Loss
- **D-02:** Each technique is a toggleable module with an `ENABLE_*` flag in train.py. Agent can enable/disable independently.
- **D-03:** Techniques that modify loss computation (PHI-S, L_angle, Hybrid Loss) are applied inside the existing distillation_loss function via conditional branches.
- **D-04:** Techniques that modify feature processing (Feature Normalizer, FeatSharp) are nn.Module classes that wrap student/teacher features before loss computation.
- **D-05:** Adaptor MLP v2 replaces simple Linear projection heads when enabled. Agent can toggle between simple and MLP v2 per teacher.
- **D-06:** PHI-S applies Hadamard rotation followed by standardization (zero mean, unit variance) to each teacher's features independently. Uses a fixed Hadamard matrix (no learnable params). Applied before loss computation.
- **D-07:** Feature Normalizer computes running mean and covariance of teacher features during first epoch (warmup). Then applies whitening transform for the rest of training. Per-teacher, computed from training set.
- **D-08:** program.md gets a complete rewrite for v2.0 search space (5 teachers, SSL, custom LCNet, RADIO adaptors, training techniques, expanded experiment playbook).
- **D-09:** Evaluation metric UNCHANGED: 0.5*recall@1 + 0.5*mean_cosine. Trust boundary preserved.
- **D-10:** Hard constraints updated: never edit prepare.py, never add dependencies (except einops already added), never exceed epoch budget, never stop.

### Claude's Discretion
- Exact Hadamard matrix size for PHI-S (match teacher embedding dim or use closest power of 2)
- Feature Normalizer warmup strategy (how many batches to accumulate statistics)
- FeatSharp implementation details (if VRAM allows)
- Shift Equivariant Loss exact formulation
- program.md experiment phase ordering (which techniques to try first)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAIN-01 | PHI-S distribution balancing (Hadamard isotropic standardization) | PHI-S paper Section 2.2.5 provides full math; Sylvester construction for Hadamard matrix (Eq. 18); PHI-S transform formula (Eq. 25-26); RADIOv2.5 Section 4.4 confirms adoption |
| TRAIN-02 | Feature Normalizer (per-teacher whitening + rotation) | PHI-S paper Sections 2.2.1-2.2.2 covers standardization and whitening; RADIOv2.5 Table A9 confirms PHI-S used as feature normalization; FeatSharp Section 3.2 describes using PHI-S for feature normalization |
| TRAIN-03 | Balanced Summary Loss L_angle (normalize by angular dispersion) | C-RADIOv4 Section 2.5 provides full formulation (Eq. 3-7); Table 3 shows angular dispersions differ significantly between teachers (SigLIP2: 0.694, DINOv3: 2.120) |
| TRAIN-04 | Hybrid Loss (0.9*cosine + 0.1*smooth-L1 for spatial features) | PHI-S paper Section 2.1 Eq. 3 defines L_hyb-sml1; AM-RADIO uses beta=0.9 for cosine + 0.1 smooth-L1; RADIOv2.5 Table A9 confirms MSE for features, Cosine for summary |
| TRAIN-05 | Per-teacher adaptor MLP v2 (LayerNorm+GELU+residual) | RADIOv2.5 Section 2.2 describes 2-layer MLP with LayerNorm and GeLU; C-RADIOv4 inherits this design |
| TRAIN-06 | FeatSharp spatial feature sharpening | FeatSharp paper (ICML 2025) provides full architecture; Section 3.3 describes tile-guided attentional refinement; marked OPTIONAL per D-01 |
| TRAIN-07 | Shift Equivariant Loss for spatial distillation | C-RADIOv4 Section 2.3.1 Eq. 1 defines L_spatial with shift mapping; random shifts in patch-size increments prevent learning fixed-pattern noise |
| INFRA-08 | program.md updated with expanded search space | Current program.md is v1.0 single-teacher; v2.0 needs 5 teachers, custom LCNet, SSL, RADIO techniques, multi-teacher phases |
| INFRA-10 | Evaluation metric unchanged -- trust boundary preserved | Metric is 0.5*recall@1 + 0.5*mean_cosine, computed in immutable prepare.py; no changes needed |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.9.1+cu128 | All technique implementations (modules, losses, Hadamard) | Already available; all papers use PyTorch |
| numpy | 2.2.6 | Hadamard matrix construction, covariance computation | Already available; needed for matrix operations |
| einops | 0.8.2 | Spatial feature rearrangement (FeatSharp, spatial distillation) | Already added in Phase 5 (INFRA-09) |

### Not Available (must hand-roll)
| Library | Would Use For | Hand-Roll Strategy |
|---------|-------------|-------------------|
| scipy | `scipy.linalg.hadamard()` for Hadamard matrices | Sylvester recursive construction in pure Python/torch (Eq. 18 from PHI-S paper) |
| scipy | `scipy.linalg.eigh()` for eigendecomposition in PHI-S | `torch.linalg.eigh()` -- available in PyTorch 2.x |

## Architecture Patterns

### Recommended Integration Structure

All techniques integrate into train.py as conditional wrappers around existing code:

```
train.py (after Phase 9)
  |
  +-- ENABLE_* flags (module-level constants)
  |     ENABLE_PHI_S = True
  |     ENABLE_FEATURE_NORMALIZER = False
  |     ENABLE_HYBRID_LOSS = True
  |     ENABLE_L_ANGLE = True
  |     ENABLE_ADAPTOR_MLP_V2 = False
  |     ENABLE_FEATSHARP = False
  |     ENABLE_SHIFT_EQUIVARIANT = False
  |
  +-- Technique modules (nn.Module classes)
  |     class PHISTransform        # stateless Hadamard rotation + scaling
  |     class FeatureNormalizer     # stateful running stats + whitening
  |     class AdaptorMLPv2         # LayerNorm+GELU+residual projection
  |     class FeatSharpModule      # spatial feature sharpening (OPTIONAL)
  |
  +-- Loss functions
  |     def hybrid_loss()          # 0.9*cosine + 0.1*smooth-L1
  |     def l_angle_loss()         # angular dispersion-normalized summary loss
  |     def shift_equivariant_loss()  # shifted spatial MSE
  |
  +-- Integration in run_train_epoch()
        # Before loss: apply PHI-S and/or Feature Normalizer to teacher features
        # Loss computation: use hybrid_loss or l_angle instead of cosine
        # Spatial: apply shift equivariant loss if spatial features available
```

### Pattern 1: ENABLE_* Toggle Pattern
**What:** Each technique is gated by a module-level boolean constant. Zero runtime overhead when disabled.
**When to use:** Every technique in this phase.
**Example:**
```python
# Module-level constants
ENABLE_PHI_S = True
ENABLE_HYBRID_LOSS = True

# In loss computation
if ENABLE_PHI_S:
    teacher_emb = phi_s_transform(teacher_emb)

if ENABLE_HYBRID_LOSS:
    distill_loss = hybrid_loss(student_emb, teacher_emb, beta=0.9)
else:
    distill_loss = (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=1)).mean()
```

### Pattern 2: Stateful Feature Normalizer with Warmup
**What:** Accumulate running statistics during warmup, then apply transform. Must handle the transition.
**When to use:** Feature Normalizer (TRAIN-02).
**Example:**
```python
class FeatureNormalizer(nn.Module):
    def __init__(self, feature_dim, warmup_batches=100):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.register_buffer('count', torch.tensor(0))
        self.ready = False

    def update_stats(self, features):
        """Call during warmup to accumulate statistics."""
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0)
        n = self.count.item()
        # Welford's online algorithm for running mean/variance
        self.count += 1
        delta = batch_mean - self.running_mean
        self.running_mean += delta / (n + 1)
        self.running_var = (n * self.running_var + batch_var * features.shape[0]) / (n * 1 + features.shape[0])
        if self.count >= self.warmup_batches:
            self.ready = True

    def forward(self, features):
        if not self.ready:
            self.update_stats(features.detach())
            return features  # passthrough during warmup
        return (features - self.running_mean) / (self.running_var.sqrt() + 1e-6)
```

### Pattern 3: Hadamard Matrix Construction (Sylvester)
**What:** Build normalized Hadamard matrices for PHI-S without scipy.
**When to use:** PHI-S implementation (TRAIN-01).
**Example:**
```python
def build_hadamard(n: int) -> torch.Tensor:
    """Build normalized Hadamard matrix of size n (must be power of 2).
    Sylvester construction per PHI-S paper Eq. 18."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)
    return H
```

### Anti-Patterns to Avoid
- **Learnable PHI-S parameters:** PHI-S uses a FIXED Hadamard matrix. Do NOT make it learnable. The entire point is that the isotropic standardization is data-independent at inference time.
- **Computing covariance every batch for PHI-S:** Covariance + eigendecomposition is computed ONCE from training data statistics (or first epoch), then the transform is fixed. Do NOT recompute per batch.
- **FeatSharp without spatial features:** FeatSharp operates on spatial (patch) feature maps. It requires the student CNN to expose pre-GAP spatial features (LCNET-04). If spatial features are not cached or available, FeatSharp cannot be applied. Mark OPTIONAL and skip if unavailable.
- **Applying PHI-S to already-L2-normalized embeddings:** PHI-S operates on raw teacher features BEFORE L2 normalization. If teacher embeddings are already cached as L2-normalized 256d vectors, PHI-S needs the pre-normalized features. This is a key distinction -- the current single-teacher setup caches normalized embeddings. Multi-teacher setup from Phase 6/8 may need to cache raw features for PHI-S.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hadamard matrix | scipy.linalg.hadamard | Sylvester recursive construction (10 lines) | scipy not available; Sylvester is trivial and exact |
| Eigendecomposition | Custom SVD/eigen solver | torch.linalg.eigh() | Available in PyTorch 2.x, numerically stable |
| Smooth L1 loss | Custom implementation | torch.nn.functional.smooth_l1_loss() | Built into PyTorch |
| Layer normalization | Manual mean/var computation | torch.nn.LayerNorm | Built into PyTorch, handles edge cases |

## Common Pitfalls

### Pitfall 1: PHI-S on Pre-normalized Embeddings
**What goes wrong:** PHI-S is applied to teacher embeddings that were already L2-normalized to unit sphere. The Hadamard rotation + scaling has no effect on distribution shape because all vectors are already on the unit sphere.
**Why it happens:** Current train.py caches teacher embeddings as L2-normalized 256d vectors. PHI-S is designed to operate on RAW teacher features (varying magnitudes, different distributions per teacher).
**How to avoid:** PHI-S must be applied BEFORE L2 normalization. For our setup, since we only have summary features (not spatial), and the current cosine loss is already direction-only, PHI-S primarily helps when multiple teachers have vastly different activation magnitudes. With a single teacher, the benefit may be limited -- but the technique should still be implemented correctly for multi-teacher use.
**Warning signs:** If enabling PHI-S produces zero metric change, verify it is operating on raw (non-normalized) features.

### Pitfall 2: Hadamard Matrix Size Mismatch
**What goes wrong:** Teacher embedding dimension (e.g., 768 for DINOv2, 1152 for C-RADIO) is not a power of 2, so Sylvester construction fails.
**Why it happens:** Hadamard matrices via Sylvester construction only exist for powers of 2.
**How to avoid:** Use closest power of 2 >= dim and pad/truncate, OR use the PHI-S paper's Appendix A.1.1 which shows how to construct Hadamard-like matrices for non-power-of-2 sizes (768, 1024, 1152, 1280, 1408). For 256d (our embedding dim): 256 = 2^8, so Sylvester works directly.
**Warning signs:** RuntimeError with shape mismatch in matmul.

### Pitfall 3: Feature Normalizer Numerical Instability
**What goes wrong:** Covariance matrix has near-zero eigenvalues, causing divide-by-zero or NaN in whitening transform.
**Why it happens:** Teacher features may be low-rank (many dimensions with near-zero variance), especially after projection to 256d.
**How to avoid:** Add epsilon (1e-6) to eigenvalues before inverting. Use standardization (PHI-S) instead of full whitening if instability occurs. PHI-S paper Section 6 notes that standardization outperforms whitening because teacher distributions may not be full rank.
**Warning signs:** NaN loss values, training divergence.

### Pitfall 4: L_angle Division by Zero for Single Teacher
**What goes wrong:** Angular dispersion Disp(theta_y) is zero or near-zero when teacher features are tightly clustered (small cone radius).
**Why it happens:** L_angle normalizes by angular dispersion to balance across teachers. With a single teacher or very aligned features, dispersion approaches zero.
**How to avoid:** Add epsilon to Disp(theta_y) denominator. For single-teacher mode, L_angle provides no balancing benefit but the formula still works with epsilon protection.
**Warning signs:** Loss explosion (very large values).

### Pitfall 5: Hybrid Loss Applied to Summary vs Spatial
**What goes wrong:** Using smooth-L1 on summary (CLS/global average) features where cosine is more appropriate, or using pure cosine on spatial features where MSE/smooth-L1 captures magnitude information.
**Why it happens:** Confusing the two feature types. AM-RADIO uses cosine for summary, hybrid (cosine+smooth-L1) for spatial features (patches).
**How to avoid:** Per RADIOv2.5 Table A9: Summary Loss = Cosine, Feature Distillation Loss = MSE. The hybrid loss (0.9*cosine + 0.1*smooth-L1) from AM-RADIO is for spatial/patch features specifically. For summary features, pure cosine or L_angle is standard.
**Warning signs:** Metric regression when enabling hybrid loss on summary features.

### Pitfall 6: Shift Equivariant Loss Without Spatial Feature Cache
**What goes wrong:** Shift equivariant loss (TRAIN-07) requires spatial (patch-level) teacher features with known spatial positions. Without cached spatial features from Phase 8 (RADIO-04), this loss cannot be computed.
**Why it happens:** Shift equivariant loss operates on dense feature maps, not summary embeddings.
**How to avoid:** Gate ENABLE_SHIFT_EQUIVARIANT on availability of spatial features. Mark as OPTIONAL per D-01. If spatial cache from Phase 8 does not exist at runtime, skip silently.
**Warning signs:** KeyError or missing file when trying to load spatial features.

## Code Examples

### PHI-S Transform (from PHI-S paper Eq. 18, 21-26)
```python
# Source: PHI-S paper (arXiv:2410.01680) Section 2.2.5
class PHISTransform(nn.Module):
    """PCA-Hadamard Isotropic Standardization (PHI-S).

    Rotates teacher features so all dimensions have equal variance,
    then scales uniformly. Prevents any single teacher from dominating
    gradient updates in multi-teacher distillation.

    Key property: transform is invertible and isotropic -- errors of equal
    magnitude in normalized space map to equal magnitude in original space.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # Will be set after computing statistics
        self.register_buffer('R', torch.eye(feature_dim))  # rotation: H @ U^T
        self.register_buffer('alpha', torch.tensor(1.0))    # scale: 1/phi
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.ready = False

    def fit(self, features: torch.Tensor):
        """Compute PHI-S parameters from a batch of teacher features.
        features: (N, D) tensor of raw teacher features.
        """
        D = self.feature_dim
        # Compute mean
        mu = features.mean(dim=0)

        # Compute covariance
        centered = features - mu
        cov = (centered.T @ centered) / (features.shape[0] - 1)

        # Eigendecomposition: cov = U @ diag(lambda) @ U^T
        eigenvalues, U = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp(min=1e-8)  # numerical stability

        # Build Hadamard matrix (power of 2 only via Sylvester)
        # For non-power-of-2 dims, pad to next power of 2
        n_pad = 1
        while n_pad < D:
            n_pad *= 2
        H = _build_hadamard(n_pad)
        H = H[:D, :D]  # truncate if padded (approximate but functional)

        # PHI-S: R = H @ U^T, alpha = 1/phi
        # phi = sqrt(1/C * sum(lambda_i))
        phi = torch.sqrt(eigenvalues.sum() / D)
        self.R.copy_(H @ U.T)
        self.alpha.copy_(1.0 / phi)
        self.mean.copy_(mu)
        self.ready = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            return features  # passthrough before fit
        # X' = alpha * R @ (X - mu)
        centered = features - self.mean
        return self.alpha * (centered @ self.R.T)


def _build_hadamard(n: int) -> torch.Tensor:
    """Sylvester construction of normalized Hadamard matrix."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)
    return H
```

### L_angle Loss (from C-RADIOv4 Section 2.5, Eq. 3-7)
```python
# Source: C-RADIOv4 tech report Section 2.5
def l_angle_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    teacher_mean_dir: torch.Tensor,
    teacher_angular_dispersion: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Balanced summary loss normalized by angular dispersion.

    Prevents teachers with large angular spread from dominating the loss.

    Args:
        student: (B, D) student predictions
        teacher: (B, D) teacher targets
        teacher_mean_dir: (D,) mean direction of teacher features (E[y]/||E[y]||)
        teacher_angular_dispersion: scalar Disp(theta_y)
        eps: numerical stability
    """
    # cos(x, y) = x^T y / (||x|| ||y||)
    cos_sim = F.cosine_similarity(student, teacher, dim=1)
    # theta(x, y) = arccos(cos(x, y))
    theta = torch.acos(cos_sim.clamp(-1 + eps, 1 - eps))
    # L_angle = theta^2 / Disp(theta_y)
    return (theta ** 2).mean() / (teacher_angular_dispersion + eps)
```

### Hybrid Loss (from PHI-S paper Eq. 3, AM-RADIO)
```python
# Source: PHI-S paper (arXiv:2410.01680) Section 2.1, Eq. 3
def hybrid_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    beta: float = 0.9,
) -> torch.Tensor:
    """Hybrid cosine + smooth-L1 loss for spatial features.

    AM-RADIO uses beta=0.9 (90% cosine, 10% smooth-L1).
    Cosine captures direction; smooth-L1 captures magnitude.
    """
    cos_loss = (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    sml1_loss = F.smooth_l1_loss(student, teacher)
    return beta * cos_loss + (1 - beta) * sml1_loss
```

### Adaptor MLP v2 (from RADIOv2.5 Section 2.2)
```python
# Source: RADIOv2.5 (arXiv:2412.07679) Section 2.2
class AdaptorMLPv2(nn.Module):
    """2-layer MLP adaptor with LayerNorm + GELU + residual.

    Replaces simple linear projection when ENABLE_ADAPTOR_MLP_V2=True.
    Per RADIOv2.5: "Our adaptor is a 2-layer MLP with a LayerNorm
    and a GeLU in between."
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(out_features, out_features)
        # Residual only if dims match
        self.use_residual = (in_features == out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.layer2(h)
        if self.use_residual:
            h = h + x
        return h
```

### Shift Equivariant Loss (from C-RADIOv4 Section 2.3.1, Eq. 1)
```python
# Source: C-RADIOv4 tech report Section 2.3.1, Eq. 1
def shift_equivariant_loss(
    student_spatial: torch.Tensor,
    teacher_spatial: torch.Tensor,
    max_shift: int = 2,
    patch_size: int = 16,
) -> torch.Tensor:
    """Spatial distillation loss with random shift for equivariance.

    Randomly shifts student and teacher spatial features independently
    to prevent the student from learning fixed positional noise patterns.
    Shifts are in increments of patch_size to avoid interpolation.

    Args:
        student_spatial: (B, C, H, W) student spatial features
        teacher_spatial: (B, C, H, W) teacher spatial features (PHI-S normalized)
        max_shift: maximum shift in patches
        patch_size: size of one patch in pixels
    """
    B, C, H, W = student_spatial.shape

    # Random shift for student (in patch increments)
    sh_s = torch.randint(-max_shift, max_shift + 1, (2,))
    sh_t = torch.randint(-max_shift, max_shift + 1, (2,))

    # Compute overlap region
    h_start = max(sh_s[0].item(), sh_t[0].item(), 0)
    h_end = min(H + sh_s[0].item(), H + sh_t[0].item(), H)
    w_start = max(sh_s[1].item(), sh_t[1].item(), 0)
    w_end = min(W + sh_s[1].item(), W + sh_t[1].item(), W)

    if h_end <= h_start or w_end <= w_start:
        return torch.tensor(0.0, device=student_spatial.device)

    # Extract overlapping regions
    s_region = student_spatial[:, :,
        h_start - sh_s[0].item():h_end - sh_s[0].item(),
        w_start - sh_s[1].item():w_end - sh_s[1].item()]
    t_region = teacher_spatial[:, :,
        h_start - sh_t[0].item():h_end - sh_t[0].item(),
        w_start - sh_t[1].item():w_end - sh_t[1].item()]

    return F.mse_loss(s_region, t_region)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Equal loss weights for all teachers | PHI-S distribution balancing | PHI-S paper (Oct 2024) | Balances gradient contribution; prevents SAM (191x larger variance than DFN CLIP) from dominating |
| Cosine-only summary loss | L_angle (angular dispersion normalization) | C-RADIOv4 (Jan 2026) | Normalizes by cone radius; prevents DINOv3 (dispersion 2.12) from dominating vs SigLIP2 (0.69) |
| Simple linear projection heads | 2-layer MLP with LayerNorm+GELU | RADIOv2.5 (Feb 2025) | Better feature transformation capacity; adopted in all subsequent RADIO versions |
| MSE for spatial features | Hybrid cosine+smooth-L1 for spatial | AM-RADIO (2024) | Captures both direction and magnitude; smooth-L1 robust to outliers |
| Bilinear upsampling for hi-res features | FeatSharp tile-guided attention | FeatSharp (ICML 2025) | +0.39% average MTL gain; cleaner high-res features |
| Fixed spatial loss positions | Shift equivariant loss with random offsets | C-RADIOv4 (Jan 2026) | Prevents learning fixed positional noise patterns from teachers |

**Deprecated/outdated:**
- **PCA Whitening (PCA-W) for distribution balancing:** PHI-S paper Table 1 shows PHI-S consistently outranks PCA-W. PCA-W places disproportionate weight on high-variance axes, amplifying outlier errors.
- **ZCA Whitening:** Also outranked by PHI-S. Error profile is proportional to covariance matrix, amplifying errors in high-variance directions.
- **MSE loss for all features:** AM-RADIO baseline. PHI-S paper Table 3 shows MSE has worst fidelity. Cosine or hybrid losses are strictly better for distillation.

## Adaptation Notes for Our Setup

The RADIO papers target large-scale ViT-to-ViT distillation (DataComp-1B, 600k iterations, ViT-B/L/H students). Our setup differs significantly:

| Aspect | RADIO Papers | Our Setup | Adaptation |
|--------|-------------|-----------|------------|
| Student | ViT-B/16 to ViT-H/16 | CNN (LCNet050, ~2M params) | Techniques are architecture-agnostic; work on feature tensors |
| Training scale | 600k iterations, DataComp-1B | 10 epochs, ~50k product images | Statistics warmup must be shorter (1 epoch vs 100k iters) |
| Teachers | 4-5 large VFMs (DFN CLIP, DINOv2, SAM, SigLIP) | 1-5 teachers (Trendyol ONNX, DINOv2, DINOv3-ft, C-RADIO variants) | PHI-S and L_angle most valuable with multi-teacher |
| Spatial features | Dense patch tokens from ViT | Pre-GAP CNN feature maps (if LCNET-04 implemented) | FeatSharp and Shift Equivariant need spatial features from Phase 8 |
| Feature dim | 768-1280 (native teacher dim) | 256 (student projection to embedding space) | PHI-S operates on teacher features before student projection; 256 is power of 2 (good for Hadamard) |
| Summary features | CLS token | Global average pooled features | L_angle and cosine loss work identically |

**Key insight for single-teacher mode:** PHI-S and L_angle provide minimal benefit with a single teacher since there is no cross-teacher balancing to do. However, implementing them correctly now means they immediately activate when multi-teacher mode (Phase 6) is enabled. This is the right architectural choice.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual validation (autoresearch pattern) |
| Config file | None -- no automated test suite for this domain |
| Quick run command | `python train.py > run.log 2>&1` |
| Full suite command | `python train.py > run.log 2>&1` (full 10-epoch run IS the test) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | PHI-S transform applied to teacher features | smoke | `python -c "from train import PHISTransform; ..."` | Wave 0 |
| TRAIN-02 | Feature Normalizer accumulates stats and transforms | smoke | `python -c "from train import FeatureNormalizer; ..."` | Wave 0 |
| TRAIN-03 | L_angle loss computes correctly | smoke | `python -c "from train import l_angle_loss; ..."` | Wave 0 |
| TRAIN-04 | Hybrid loss combines cosine + smooth-L1 | smoke | `python -c "from train import hybrid_loss; ..."` | Wave 0 |
| TRAIN-05 | Adaptor MLP v2 forward pass works | smoke | `python -c "from train import AdaptorMLPv2; ..."` | Wave 0 |
| TRAIN-06 | FeatSharp processes spatial features | smoke | (OPTIONAL, only if spatial features available) | Wave 0 |
| TRAIN-07 | Shift equivariant loss with random shifts | smoke | `python -c "from train import shift_equivariant_loss; ..."` | Wave 0 |
| INFRA-08 | program.md contains v2.0 search space | manual-only | Visual inspection of program.md content | N/A |
| INFRA-10 | Metric computation unchanged | manual-only | Run baseline, verify metric matches pre-phase value | N/A |

### Sampling Rate
- **Per task commit:** Quick smoke test of new module (import + forward pass with dummy data)
- **Per wave merge:** Full 10-epoch training run to verify no regression
- **Phase gate:** Full training run with default ENABLE_* settings produces comparable-or-better metric

### Wave 0 Gaps
- [ ] Smoke tests for each technique module (TRAIN-01 through TRAIN-07) -- verify import, forward pass, gradient flow
- [ ] Verify ENABLE_* flags default to values that do not regress baseline metric

## Open Questions

1. **PHI-S on already-projected features?**
   - What we know: Current setup projects teacher features to 256d and L2-normalizes. PHI-S paper operates on raw teacher features (varying dim per teacher).
   - What's unclear: Whether PHI-S on the 256d projected embeddings provides meaningful benefit, since cosine loss already handles magnitude invariance.
   - Recommendation: Implement PHI-S to operate on raw teacher features (before projection). For the current single-teacher mode where features are loaded from cache as normalized 256d vectors, PHI-S will have limited effect. When multi-teacher mode is enabled (Phase 6 teachers provide raw features), PHI-S activates properly. Add a comment documenting this.

2. **Feature Normalizer warmup duration in 10-epoch regime?**
   - What we know: RADIO papers use 100k iterations warmup. We have ~200 batches per epoch (50k images / 256 batch size).
   - What's unclear: Is 1 epoch (200 batches) sufficient for stable covariance estimation?
   - Recommendation: Default to full first epoch (~200 batches) for warmup. This is Claude's discretion per CONTEXT.md. Add `NORMALIZER_WARMUP_BATCHES = 200` as a tunable constant.

3. **FeatSharp VRAM impact?**
   - What we know: FeatSharp requires tiled inference + attention block, which adds significant memory overhead. Our RTX 4090 has 24GB with ~18-19GB typical usage.
   - What's unclear: Whether FeatSharp + existing training fits in 24GB.
   - Recommendation: Mark OPTIONAL per D-01. Default `ENABLE_FEATSHARP = False`. Agent can try enabling it -- if OOM, the crash recovery system handles it.

## Sources

### Primary (HIGH confidence)
- PHI-S paper (arXiv:2410.01680) -- Sections 2.2.1-2.2.5 for all normalization/standardization formulas, Table 1 for method rankings, Eq. 18 for Hadamard construction, Eq. 25-26 for PHI-S transform
- RADIOv2.5 paper (arXiv:2412.07679) -- Section 2.2 for baseline model architecture and adaptor MLP design, Section 4.4 for PHI-S adoption, Table A9 for hyperparameters
- C-RADIOv4 tech report -- Section 2.3.1 Eq. 1 for shift equivariant loss, Section 2.5 Eq. 3-7 for L_angle, Table 3 for angular dispersions
- FeatSharp paper (ICML 2025) -- Full architecture for spatial feature sharpening, Section 3.2 for feature normalization, Section 3.3 for tile-guided refinement

### Secondary (MEDIUM confidence)
- Current train.py (direct code analysis) -- existing loss structure, projection head design, training loop
- Current program.md (direct reading) -- v1.0 agent instructions structure to serve as template for v2.0

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all techniques use only torch/numpy/einops (verified available)
- Architecture: HIGH -- formulas directly from papers with equations numbered, adapted for our setup
- Pitfalls: HIGH -- derived from paper analysis (distribution issues, numerical stability) and code analysis (cached normalized embeddings)
- Code examples: HIGH -- translated directly from paper equations with paper references

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- these are published techniques, not evolving APIs)
