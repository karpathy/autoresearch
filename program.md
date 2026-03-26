# ReID Autoresearch v2.0 -- Agent Instructions

This is an autonomous AI-driven experimentation system for ReID (Re-Identification) model training. You modify `train.py`, run experiments, evaluate results, and keep or discard changes in a continuous loop. The human writes these instructions -- you write the training code.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current working branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` -- these instructions (you are reading it)
   - `prepare.py` -- IMMUTABLE: data loading, teacher inference, evaluation, caching. Do not modify.
   - `train.py` -- YOUR FILE: student model, losses, optimizer, scheduler, augmentations, teacher selection, technique toggles. Everything here is fair game within constraints.
4. **Verify teacher caches exist**: Check that teacher caches are present:
   - `workspace/output/trendyol_teacher_cache2/` for Trendyol ONNX (default teacher)
   - Additional caches are built automatically on first use (takes 10-30 min, excluded from experiment budget)
   - einops is available as a dependency (required by RADIO components)
5. **Initialize results.tsv** with just the header row:
   ```
   commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
   ```
6. **Run baseline**: `python train.py > run.log 2>&1` -- your first run is always the unmodified train.py to establish the baseline.
7. **Record baseline** in results.tsv, commit as baseline.
8. **Begin experiment loop**.

Once the baseline is recorded, kick off the experimentation loop below.

## Experimentation

Each experiment runs on a single GPU (RTX 4090, 24GB VRAM). The training script runs for a **fixed budget of 10 epochs** (NOT wall-clock time -- you optimize WHAT happens in 10 epochs). You launch it simply as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, loss functions, loss weights, augmentations, training loop, batch size, projection head design, teacher selection, technique toggles, RADIO adaptor selection. All within the constraints below.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, teacher inference, and caching logic. Modifying it breaks the trust boundary and makes all experiments non-comparable.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`. Available: torch, timm, onnxruntime, transformers, torchvision, numpy, PIL, einops.
- Modify the evaluation harness. The `compute_combined_metric` function in `prepare.py` is the ground truth metric.
- Exceed the edge deployment limits (see Hard Constraints below).

**The goal is simple: get the highest combined_metric.** The combined metric is `0.5 * recall@1 + 0.5 * mean_cosine`. Higher is better. recall@1 measures retrieval accuracy (can the model find the right product?). mean_cosine measures teacher alignment (does the student agree with the teacher?).

**VRAM** is a HARD constraint. The RTX 4090 has 24GB. If peak VRAM exceeds 22GB on a successful run, do NOT increase batch size or model size further. OOM = crash = discard.

**Simplicity criterion**: All else being equal, simpler is better. But always maintain edge-deployability -- a simpler model that's too big to deploy is useless.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Hard Constraints -- NEVER VIOLATE

These are absolute rules. Violating any one invalidates all experiments.

1. **NEVER edit prepare.py** -- it contains evaluation, data loading, and teacher inference. Modifying it breaks the trust boundary. All experiments become non-comparable. If you need something from prepare.py, import it.

2. **NEVER install new packages** -- only use what's in `pyproject.toml`. Available: torch, timm, onnxruntime, transformers, torchvision, numpy, PIL, einops. If you want a feature from a missing library, implement it yourself in train.py using only these packages.

3. **NEVER exceed 10 epochs** -- this is the fixed experiment budget. You optimize WHAT happens in 10 epochs, not how many. The epoch count is enforced by prepare.py.

4. **NEVER modify the evaluation metric** -- the combined metric is `0.5 * recall@1 + 0.5 * mean_cosine`, computed by `compute_combined_metric` in prepare.py. This is immutable. Do not redefine, rescale, or replace it.

5. **NEVER stop the loop** -- run until manually interrupted. The human may be asleep. Do NOT ask "should I continue?" or "is this a good stopping point?" See NEVER STOP section below.

6. **NEVER exceed edge deployment limits** -- the student model must remain edge-deployable:
   - Embedding dimension MUST remain 256.
   - Parameter count must stay lightweight (LCNet-class, not ResNet-50 class).
   - If you change the backbone, verify it is still comparable in size to lcnet_050 (~2M params).

7. **NEVER remove or modify the checkpoint saving block** -- the code between `# Save model checkpoint` and `# Compute final metrics` in train.py MUST remain intact.

8. **NEVER remove quality degradation augmentation** -- `RandomQualityDegradation` must remain active. You may tune its parameters but NEVER set prob to 0 or remove the transform.

## Evaluation Metric (UNCHANGED)

The combined metric is computed in `prepare.py` and is IMMUTABLE:

```
combined_metric = 0.5 * recall@1 + 0.5 * mean_cosine
```

- **recall@1**: Retrieval accuracy -- can the model find the correct product as its top-1 nearest neighbor?
- **mean_cosine**: Teacher alignment -- does the student's embedding agree with the teacher's?
- Higher is better for both components and for the combined metric.

You do NOT compute this metric -- `compute_combined_metric()` from prepare.py handles it. Your job is to maximize it through training improvements.

## Search Space Reference

The v2.0 search space is dramatically expanded from v1.0. All tunable constants are module-level variables in train.py. Always read train.py to confirm exact variable names before editing.

### A. Teachers (5 available)

| Constant | Default | Notes |
|----------|---------|-------|
| `TEACHER` | `"trendyol_onnx"` | Single-teacher mode (backward compatible) |
| `TEACHERS` | `None` | Multi-teacher dict: `{"trendyol_onnx": 0.5, "dinov2": 0.5}`. Overrides TEACHER. |

**Available teachers** (registered in prepare.py's TEACHER_REGISTRY):

| Teacher Name | Embedding Dim | Type | Notes |
|-------------|---------------|------|-------|
| `trendyol_onnx` | 256 | ONNX | Default baseline teacher, fast inference |
| `dinov2` | 256 | HuggingFace | Trendyol DINOv2 ecommerce, strong generalization |
| `dinov3_ft` | 1280 | LoRA fine-tuned | DINOv3 ViT-H+ fine-tuned on product data |
| `radio_so400m` | varies | RADIO | C-RADIOv4-SO400M, use with RADIO_ADAPTORS |
| `radio_h` | varies | RADIO | C-RADIOv4-H, use with RADIO_ADAPTORS |

**Guidance:**
- Start with single Trendyol (baseline) -- this is the strongest single teacher for product ReID.
- Try DINOv2 as single teacher to compare.
- Multi-teacher: start with 2 teachers (e.g., `{"trendyol_onnx": 0.6, "dinov2": 0.4}`).
- RADIO teachers require RADIO_VARIANT and RADIO_ADAPTORS configuration.
- Enable PHI-S and Feature Normalizer when using 2+ teachers (see RADIO Techniques section).

### B. Student Architecture (Custom LCNet)

| Constant | Default | Range | Notes |
|----------|---------|-------|-------|
| `LCNET_SCALE` | `0.5` | 0.25-1.0 | Width multiplier. 0.5 = lcnet_050. Higher = more params. |
| `SE_START_BLOCK` | `10` | 0-12 | Block index where Squeeze-Excitation begins (13 total blocks) |
| `SE_REDUCTION` | `0.25` | 0.1-0.5 | SE squeeze ratio |
| `ACTIVATION` | `"h_swish"` | `"h_swish"`, `"relu"`, `"gelu"` | Activation function |
| `KERNEL_SIZES` | `[3,3,3,3,3,3,5,5,5,5,5,5,5]` | 3, 5, 7 per block | Per-block depthwise conv kernel sizes (13 blocks) |
| `USE_PRETRAINED` | `True` | bool | Load timm pretrained weights (only for scale 0.5, 0.75, 1.0) |
| `MODEL_NAME` | `"hf-hub:timm/lcnet_050.ra2_in1k"` | timm model name | Fallback model name (used for val transforms) |

**Guidance:**
- Start from default LCNet050 (scale=0.5), tune incrementally.
- Try `SE_START_BLOCK=8` to add SE to more blocks (adds minimal params, may improve).
- Try larger kernels in early blocks: `[5,5,5,3,3,3,5,5,5,5,5,5,5]`.
- `LCNET_SCALE=0.75` is a good step up if VRAM allows.
- The custom LCNet enables architecture search that timm models cannot.

### C. SSL Contrastive Loss

| Constant | Default | Range | Notes |
|----------|---------|-------|-------|
| `SSL_WEIGHT` | `0.0` | 0.0-0.5 | 0 = disabled. Enables dual-view InfoNCE contrastive loss. |
| `SSL_TEMPERATURE` | `0.07` | 0.03-0.15 | InfoNCE temperature (learnable, this is init value) |
| `SSL_PROJ_DIM` | `128` | 64-256 | SSL projection head output dimension |

**Guidance:**
- Start with SSL_WEIGHT=0 (disabled). It is an auxiliary signal, not the primary objective.
- Try SSL_WEIGHT=0.1 first. If it helps, tune in range 0.05-0.3.
- WARNING: Enabling SSL approximately doubles forward passes per batch (VRAM ~1.5x). Reduce BATCH_SIZE if OOM.
- SSL creates a dual-view setup: view_a from normal augmentation, view_b re-loaded with different augmentation.

### D. RADIO Training Techniques (7 toggles)

These are advanced techniques from the RADIO paper series. Each is gated by an `ENABLE_*` flag. All default to `False` for zero regression on existing behavior.

| Constant | Default | Category | Notes |
|----------|---------|----------|-------|
| `ENABLE_PHI_S` | `False` | Distribution | Hadamard isotropic standardization. Prevents one teacher from dominating gradients. **Enable with 2+ teachers.** |
| `ENABLE_FEATURE_NORMALIZER` | `False` | Distribution | Per-teacher whitening with Welford warmup. **Enable with 2+ teachers.** |
| `NORMALIZER_WARMUP_BATCHES` | `200` | Distribution | Batches to accumulate stats before normalizing (~1 epoch). |
| `ENABLE_L_ANGLE` | `False` | Loss | Angular dispersion-normalized summary loss. Balances teachers with different angular spreads. **Enable with 2+ teachers.** |
| `ENABLE_HYBRID_LOSS` | `False` | Loss | 0.9*cosine + 0.1*smooth-L1 for spatial features. **Enable if SPATIAL_DISTILL_WEIGHT > 0.** |
| `HYBRID_LOSS_BETA` | `0.9` | Loss | Cosine weight in hybrid loss (1-beta = smooth-L1 weight). |
| `ENABLE_ADAPTOR_MLP_V2` | `False` | Architecture | LayerNorm+GELU+residual projection heads. Replaces simple Linear+BN. **Safe upgrade -- try after establishing good baseline.** |
| `ENABLE_FEATSHARP` | `False` | Spatial | Attention-based spatial feature sharpening. OPTIONAL, may OOM on RTX 4090. |
| `ENABLE_SHIFT_EQUIVARIANT` | `False` | Spatial | Random-shift spatial MSE. Prevents learning fixed positional noise. **Enable if SPATIAL_DISTILL_WEIGHT > 0.** |
| `SHIFT_EQUIVARIANT_MAX_SHIFT` | `2` | Spatial | Maximum shift in patches for shift equivariant loss. |

**Guidance -- Priority Order:**
1. **PHI-S + Feature Normalizer** -- most impactful for multi-teacher distillation. Enable both together when using 2+ teachers.
2. **L_angle** -- helps balance teachers with very different angular dispersions. Enable with 2+ teachers.
3. **Adaptor MLP v2** -- safe upgrade, try after establishing a good multi-teacher baseline.
4. **Hybrid Loss** -- for spatial distillation. Enable when SPATIAL_DISTILL_WEIGHT > 0.
5. **FeatSharp** -- OPTIONAL spatial sharpening. May OOM. Try only when spatial distillation is working well.
6. **Shift Equivariant** -- OPTIONAL spatial regularizer. Enable when spatial distillation is active.

**Key rules:**
- PHI-S, Feature Normalizer, and L_angle provide minimal benefit with a single teacher. They activate properly with multi-teacher mode.
- Hybrid Loss and Shift Equivariant Loss are for spatial distillation only (SPATIAL_DISTILL_WEIGHT > 0).
- FeatSharp adds VRAM overhead. If OOM, disable it.
- All flags default False -- no regression on existing behavior.

### E. RADIO Adaptors

| Constant | Default | Notes |
|----------|---------|-------|
| `RADIO_VARIANT` | `"so400m"` | `"so400m"` (SO400M) or `"h"` (ViT-H). SO400M is lighter. |
| `RADIO_ADAPTORS` | `["backbone"]` | Subset of `["backbone", "dino_v3", "siglip2-g"]` |
| `RADIO_CACHE_BASE` | `"workspace/output/teacher_cache"` | Base directory for RADIO cached embeddings |
| `SPATIAL_DISTILL_WEIGHT` | `0.0` | 0.0 = disabled. Weight for spatial distillation loss from RADIO. |

**Guidance:**
- Try backbone adaptor first (default) -- lowest overhead.
- Then try `["backbone", "dino_v3"]` for richer feature combination.
- Then `["backbone", "dino_v3", "siglip2-g"]` for maximum multi-adaptor coverage.
- SPATIAL_DISTILL_WEIGHT requires a RADIO teacher in TEACHERS dict. Start with 0.1-0.5.

### F. Standard Hyperparameters

| Constant | Default | Range | Notes |
|----------|---------|-------|-------|
| `LR` | `2e-3` | 1e-4 to 1e-1 | Learning rate -- high impact |
| `WEIGHT_DECAY` | `1e-5` | 1e-6 to 1e-3 | Regularization |
| `BATCH_SIZE` | `256` | 64-512 | Distillation batch size. Watch VRAM. |
| `ARCFACE_BATCH_SIZE` | `128` | 32-256 | ArcFace batch size. Watch VRAM. |
| `ARCFACE_LOSS_WEIGHT` | `0.03` | 0.01-0.2 | Weight of ArcFace loss -- high impact |
| `ARCFACE_S` | `32.0` | 16-64 | ArcFace scale parameter |
| `ARCFACE_M` | `0.50` | 0.3-0.7 | ArcFace margin -- affects class separation |
| `ARCFACE_PHASEOUT_EPOCH` | `0` | 0 (disabled), 5-8 | Epoch to begin phasing out ArcFace loss |
| `VAT_WEIGHT` | `0.0` | 0.0-0.1 | Virtual adversarial training weight (0 = disabled) |
| `VAT_EPSILON` | `8.0` | 1.0-16.0 | VAT perturbation magnitude |
| `SEP_WEIGHT` | `1.0` | 0.0-5.0 | Separation loss weight |
| `UNFREEZE_EPOCH` | `0` | 0-7 | When to unfreeze backbone (0 = from start) |
| `BACKBONE_LR_MULT` | `0.1` | 0.01-1.0 | Backbone LR = LR * this multiplier |
| `QUALITY_DEGRADATION_PROB` | `0.5` | 0.1-0.9 | Quality degradation probability (NEVER set to 0) |
| `DROP_HARD_RATIO` | `0.2` | 0.0-0.5 | ArcFace hard negative mining ratio |
| `ARCFACE_MAX_PER_CLASS` | `100` | 50-500 | Max samples per class for ArcFace dataset |

## Experiment Playbook -- Prioritized Phases

Work through these phases in order. Each phase builds on the previous one. Skip phases that don't apply.

### Phase A: Establish Strong Single-Teacher Baseline (start here)

**Goal:** Maximize combined_metric with single Trendyol teacher before adding complexity.

1. Run unmodified train.py to get baseline metric.
2. Tune `LR` (try 1e-3, 5e-3, 1e-2).
3. Tune `ARCFACE_LOSS_WEIGHT` (try 0.01, 0.02, 0.05, 0.1).
4. Tune `BACKBONE_LR_MULT` (try 0.05, 0.1, 0.2).
5. Try `ARCFACE_PHASEOUT_EPOCH=7` (phase out ArcFace in final epochs).
6. Tune `SEP_WEIGHT` (try 0.5, 1.0, 2.0).
7. Try enabling VAT: `VAT_WEIGHT=0.01`.

### Phase B: SSL Contrastive Loss

**Goal:** Add self-supervised contrastive signal to improve representation quality.

1. Enable SSL: `SSL_WEIGHT=0.1`. May need to reduce BATCH_SIZE for VRAM.
2. Tune `SSL_WEIGHT` in range 0.05-0.3.
3. Try different `SSL_TEMPERATURE` (0.05, 0.07, 0.1).
4. If SSL helps, try keeping it enabled for all future phases.

### Phase C: Custom LCNet Architecture Search

**Goal:** Find better student architecture within the LCNet family.

1. Try `SE_START_BLOCK=8` (more SE blocks).
2. Try larger kernels: `KERNEL_SIZES=[5,5,5,3,3,3,5,5,5,5,5,5,5]`.
3. Try `ACTIVATION="gelu"`.
4. Try `LCNET_SCALE=0.75` (wider, more params -- watch VRAM and param count).
5. Combine best architecture changes.

### Phase D: Multi-Teacher Distillation

**Goal:** Learn from multiple teachers simultaneously.

1. Try 2 teachers: `TEACHERS={"trendyol_onnx": 0.6, "dinov2": 0.4}`.
2. Enable `ENABLE_PHI_S=True` and `ENABLE_FEATURE_NORMALIZER=True` for gradient balancing.
3. Tune teacher weight ratios.
4. Try 3 teachers: add `"radio_so400m"` with RADIO_ADAPTORS=["backbone"].
5. Enable `ENABLE_L_ANGLE=True` for angular dispersion balancing.

### Phase E: RADIO Techniques

**Goal:** Apply advanced training techniques from the RADIO paper series.

1. Enable `ENABLE_ADAPTOR_MLP_V2=True` (replace linear projection with MLP).
2. If multi-teacher: ensure PHI-S + Feature Normalizer + L_angle are all enabled.
3. Tune `NORMALIZER_WARMUP_BATCHES` (try 100, 200, 400).

### Phase F: Spatial Distillation

**Goal:** Distill spatial (patch-level) features from RADIO teachers.

1. Set `SPATIAL_DISTILL_WEIGHT=0.1` with a RADIO teacher active.
2. Enable `ENABLE_HYBRID_LOSS=True` for better spatial loss.
3. Tune `HYBRID_LOSS_BETA` (try 0.8, 0.9, 0.95).
4. Try `ENABLE_SHIFT_EQUIVARIANT=True` for spatial regularization.

### Phase G: Advanced / Optional Techniques

**Goal:** Squeeze out remaining gains with advanced techniques.

1. Try `ENABLE_FEATSHARP=True` (spatial sharpening -- may OOM).
2. Try multiple RADIO adaptors: `RADIO_ADAPTORS=["backbone", "dino_v3"]`.
3. Try RADIO-H variant: `RADIO_VARIANT="h"` (larger, may be slower to cache).
4. Combine best techniques from all phases.

## The Experiment Loop

LOOP FOREVER:

1. **Read history**: `cat results.tsv`. Analyze what worked, what didn't, what patterns emerge.

2. **Choose next experiment**: Based on history and the playbook, pick an idea. One change per experiment for clear attribution.

3. **Edit train.py**: Make your changes. Keep diffs minimal and focused.

4. **git commit**: Commit with a descriptive message.

5. **Run**: `python train.py > run.log 2>&1`

6. **Read results**: `grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log`
   - If grep is empty: run crashed. `tail -n 50 run.log` for the stack trace.

7. **Log to results.tsv**: Record all 7 columns. Do NOT git-track results.tsv.

8. **Keep or discard**:
   - combined_metric improved? **KEEP** -- advance the branch.
   - Same or worse? **DISCARD** -- `git reset --hard HEAD~1`
   - Crash? Log as crash, `git reset --hard HEAD~1`
   - 3+ consecutive crashes on the same idea? **SKIP** that direction entirely.

9. **GOTO 1**

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" The human might be asleep. You are autonomous. If you run out of ideas, think harder -- re-read the playbook, re-read results.tsv for patterns, re-read train.py for overlooked opportunities, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## Output Format

After each run, the script prints a summary block:

```
---
status:           success
combined_metric:  0.654321
recall@1:         0.432100
mean_cosine:      0.876543
peak_vram_mb:     18432.1
epochs:           10
elapsed_seconds:  342.5
```

Extract key metrics:

```
grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
a1b2c3d	0.654321	0.432100	0.876543	18432.1	keep	baseline
b2c3d4e	0.672100	0.460200	0.884000	18500.3	keep	reduce ArcFace weight to 0.02
c3d4e5f	0.640000	0.410000	0.870000	18400.0	discard	enable SSL weight=0.1
d4e5f6g	0.000000	0.000000	0.000000	0.0	crash	double batch size (OOM)
```

## Domain Context: ReID Knowledge Distillation

- **ReID (Re-Identification)**: Given a query product image, find matching products in a gallery using embedding similarity.
- **Knowledge distillation**: Large teacher model(s) produce high-quality embeddings. A small student (LCNet050) learns to match them while remaining edge-deployable.
- **The combined metric** balances retrieval accuracy (recall@1) and teacher alignment (mean_cosine).
- **The student pipeline**: Image -> Custom LCNet backbone -> projection head(s) -> L2-normalized 256-dim embedding.
- **Multi-teacher mode**: Multiple teachers provide diverse supervision signals. Per-teacher projection heads handle different embedding dimensions. PHI-S and Feature Normalizer balance gradient contributions.
- **Training losses**: Distillation (cosine or L_angle), ArcFace (angular margin), VAT (adversarial), Separation (push blacklist away), SSL (contrastive), Spatial distillation (patch features from RADIO).

## When You Are Stuck

1. **Minor plateau (3-5 no-improvement)**: Switch parameter dimension. If tuning LR, try loss weights. If tuning losses, try architecture.
2. **Medium plateau (5-10 no-improvement)**: Jump to next playbook phase. Combine best configs from different experiments.
3. **Major plateau (10+ no-improvement)**: Try radical changes -- multi-teacher, RADIO techniques, spatial distillation, different backbone scale.
4. **Decompose metrics**: If recall@1 is good but mean_cosine lags, focus on distillation improvements. If mean_cosine is good but recall@1 lags, focus on ArcFace and augmentation.

## Crash Handling

1. `tail -n 50 run.log` for the error.
2. **OOM**: Reduce BATCH_SIZE, disable ENABLE_FEATSHARP, reduce LCNET_SCALE, disable SSL.
3. **Bug**: Fix and re-run (does not count as failed experiment).
4. **3+ consecutive crashes**: Skip that direction, log as crash, move on.
5. **VRAM budget rule**: If peak_vram_mb > 22000, do NOT add more compute. Maintain or reduce.
