# Requirements: ReID Autoresearch

**Defined:** 2026-03-25
**Core Value:** AI agent autonomously discovers better ReID model configurations without human intervention

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Refactoring

- [x] **REFAC-01**: Monolith `finetune_trendyol_arcface3.py` is split into `prepare.py` (immutable) and `train.py` (agent-editable)
- [x] **REFAC-02**: Evaluation logic (retrieval recall@1/k, mean_cosine) lives exclusively in `prepare.py` and cannot be modified by the agent
- [x] **REFAC-03**: Teacher embedding cache (ONNX + DINOv2) is extracted to `prepare.py` with disk + memory caching
- [x] **REFAC-04**: Dataset loading (product_code, retail, commodity, negatives) is extracted to `prepare.py` with fixed train/val splits
- [x] **REFAC-05**: `train.py` exposes all tunable parameters as module-level constants (no argparse, no config files)
- [x] **REFAC-06**: `train.py` model implements `.encode(images) -> Tensor[B, 256]` (L2-normalized) contract with `prepare.py`
- [x] **REFAC-07**: `prepare.py` computes the single combined metric: `0.5 * recall@1 + 0.5 * mean_cosine`

### Infrastructure

- [x] **INFRA-01**: Each experiment is a git commit; improvement = keep, regression = `git reset --hard`
- [x] **INFRA-02**: `results.tsv` logs every experiment: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status (kept/discarded/crash), description
- [x] **INFRA-03**: OOM and runtime crashes are caught, logged as "crash" in results.tsv, git reset performed, loop continues
- [x] **INFRA-04**: After 3 consecutive crashes on the same idea, agent skips that direction
- [x] **INFRA-05**: Peak VRAM (`torch.cuda.max_memory_allocated()`) is tracked and logged per experiment
- [x] **INFRA-06**: Run output includes decomposed sub-metrics (recall@1, recall@5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss) in greppable format
- [x] **INFRA-07**: Fixed budget of 10 epochs per experiment, teacher cache build time excluded from budget

### Agent

- [x] **AGNT-01**: `program.md` contains ReID-specific experiment strategy, constraints, and search space documentation
- [x] **AGNT-02**: `program.md` includes prioritized experiment hints: loss weights, backbone unfreezing, augmentation, LR schedule, projection head design
- [x] **AGNT-03**: `program.md` encodes hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop
- [x] **AGNT-04**: Agent runs autonomously in a never-stop loop -- modify train.py -> run -> evaluate -> keep/discard -> repeat
- [x] **AGNT-05**: Agent reads results.tsv history to reason about what to try next

### Validation

- [x] **VALD-01**: Baseline run of unmodified train.py produces a valid combined metric and logs to results.tsv
- [x] **VALD-02**: At least one full autonomous loop cycle completes: agent modifies train.py, runs, evaluates, keeps or discards
- [x] **VALD-03**: Crash recovery verified: intentionally trigger OOM, confirm system logs crash and continues

## v2 Requirements -- Expanded Search Space

### SSL Contrastive Loss
- [x] **SSL-01**: train.py includes InfoNCE contrastive loss
- [x] **SSL-02**: SSL uses a separate projection head
- [x] **SSL-03**: `SSL_WEIGHT` is a module-level constant agent can tune

### Custom LCNet Backbone
- [x] **LCNET-01**: Custom LCNet backbone in train.py replacing timm, `.encode()` contract preserved
- [x] **LCNET-02**: Agent can tune LCNET_SCALE, SE_START_BLOCK, SE_REDUCTION, kernel sizes, ACTIVATION
- [x] **LCNET-03**: Optional timm pretrained weight initialization
- [x] **LCNET-04**: Exposes pre-GAP spatial features for RADIO spatial distillation

### Multi-Teacher Infrastructure
- [x] **TEACH-01**: prepare.py supports 5+ teachers: Trendyol ONNX, DINOv2, DINOv3-ft, all C-RADIO variants
- [x] **TEACH-02**: Each teacher has independent cache directory with metadata
- [x] **TEACH-03**: Cache building sequential per teacher (VRAM safety)
- [ ] **TEACH-04**: `TEACHER` is a module-level constant -- agent can switch teachers
- [ ] **TEACH-05**: Multi-teacher mode with per-teacher loss weights as tunable constants

### RADIO Integration
- [ ] **RADIO-01**: RADIOTeacher class supporting all C-RADIO variants with adaptor selection
- [ ] **RADIO-02**: 3 adaptor outputs: backbone, dino_v3, siglip2-g -- agent selects which to distill from
- [ ] **RADIO-03**: Each adaptor's summary features cached with native dim, projection in train.py
- [ ] **RADIO-04**: Spatial features cached separately with memory-mapped storage
- [ ] **RADIO-05**: Spatial distillation loss in train.py
- [ ] **RADIO-06**: `RADIO_VARIANT` and `RADIO_ADAPTORS` as tunable constants

### DINOv3 Fine-tuned Teacher
- [ ] **DINO3-01**: Fine-tune largest DINO variant fitting RTX 4090 (ViT-g 1.1B + LoRA) on product dataset
- [ ] **DINO3-02**: Uses autoresearch pattern (prepare_dino.py + train_dino.py)
- [ ] **DINO3-03**: Fine-tuned model exported and integrated as teacher
- [ ] **DINO3-04**: Embeddings cached to disk like other teachers

### RADIO Training Techniques
- [ ] **TRAIN-01**: PHI-S distribution balancing (Hadamard isotropic standardization)
- [ ] **TRAIN-02**: Feature Normalizer (per-teacher whitening + rotation)
- [ ] **TRAIN-03**: Balanced Summary Loss L_angle (normalize by angular dispersion)
- [ ] **TRAIN-04**: Hybrid Loss (0.9*cosine + 0.1*smooth-L1 for spatial features)
- [ ] **TRAIN-05**: Per-teacher adaptor MLP v2 (LayerNorm+GELU+residual)
- [ ] **TRAIN-06**: FeatSharp spatial feature sharpening (deferred if VRAM-constrained)
- [ ] **TRAIN-07**: Shift Equivariant Loss for spatial distillation

### Updated Infrastructure
- [ ] **INFRA-08**: program.md updated with expanded search space
- [x] **INFRA-09**: `einops` added to pyproject.toml
- [ ] **INFRA-10**: Evaluation metric unchanged -- trust boundary preserved

## v3 Requirements (Deferred)

- **MULTI-01**: Multiple agents in parallel on separate GPU branches
- **MULTI-02**: Results from parallel agents merged
- **METR-01**: Per-category recall breakdown
- **METR-02**: Embedding visualization (t-SNE/UMAP)
- **EXPRT-01**: Best model exported to ONNX
- **EXPRT-02**: Inference latency benchmark
- **OPT-01**: Fine-tune larger DINO on multi-GPU
- **OPT-02**: Quantization-aware training (INT8)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-GPU / distributed training | Single RTX 4090 constraint |
| DINOv3-7B fine-tune | 7B requires 50GB+ VRAM even with LoRA; use ViT-g 1.1B |
| RADIO training code reproduction | Training code not released; use inference weights only |
| Spectral reparametrization | Low value for small CNN student |
| Per-teacher CLS token | Student is CNN not ViT |
| DAMP weight noise | Low priority regularization |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REFAC-01 | Phase 1 | Complete |
| REFAC-02 | Phase 1 | Complete |
| REFAC-03 | Phase 1 | Complete |
| REFAC-04 | Phase 1 | Complete |
| REFAC-05 | Phase 1 | Complete |
| REFAC-06 | Phase 1 | Complete |
| REFAC-07 | Phase 1 | Complete |
| INFRA-01 | Phase 2 | Complete |
| INFRA-02 | Phase 2 | Complete |
| INFRA-03 | Phase 2 | Complete |
| INFRA-04 | Phase 2 | Complete |
| INFRA-05 | Phase 2 | Complete |
| INFRA-06 | Phase 2 | Complete |
| INFRA-07 | Phase 2 | Complete |
| AGNT-01 | Phase 3 | Complete |
| AGNT-02 | Phase 3 | Complete |
| AGNT-03 | Phase 3 | Complete |
| AGNT-04 | Phase 3 | Complete |
| AGNT-05 | Phase 3 | Complete |
| VALD-01 | Phase 4 | Complete |
| VALD-02 | Phase 4 | Complete |
| VALD-03 | Phase 4 | Complete |
| SSL-01 | Phase 5 | Complete |
| SSL-02 | Phase 5 | Complete |
| SSL-03 | Phase 5 | Complete |
| LCNET-01 | Phase 5 | Complete |
| LCNET-02 | Phase 5 | Complete |
| LCNET-03 | Phase 5 | Complete |
| LCNET-04 | Phase 5 | Complete |
| INFRA-09 | Phase 5 | Complete |
| TEACH-01 | Phase 6 | Complete |
| TEACH-02 | Phase 6 | Complete |
| TEACH-03 | Phase 6 | Complete |
| TEACH-04 | Phase 6 | Pending |
| TEACH-05 | Phase 6 | Pending |
| DINO3-01 | Phase 7 | Pending |
| DINO3-02 | Phase 7 | Pending |
| DINO3-03 | Phase 7 | Pending |
| DINO3-04 | Phase 7 | Pending |
| RADIO-01 | Phase 8 | Pending |
| RADIO-02 | Phase 8 | Pending |
| RADIO-03 | Phase 8 | Pending |
| RADIO-04 | Phase 8 | Pending |
| RADIO-05 | Phase 8 | Pending |
| RADIO-06 | Phase 8 | Pending |
| TRAIN-01 | Phase 9 | Pending |
| TRAIN-02 | Phase 9 | Pending |
| TRAIN-03 | Phase 9 | Pending |
| TRAIN-04 | Phase 9 | Pending |
| TRAIN-05 | Phase 9 | Pending |
| TRAIN-06 | Phase 9 | Pending |
| TRAIN-07 | Phase 9 | Pending |
| INFRA-08 | Phase 9 | Pending |
| INFRA-10 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 22 total (all complete)
- v2 requirements: 32 total
- Mapped to phases: 32/32
- Unmapped: 0

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after v2.0 roadmap creation*
