# Roadmap: ReID Autoresearch

## Milestones

- [x] **v1.0 MVP** - Phases 1-4 (shipped)
- [ ] **v2.0 Expanded Search Space** - Phases 5-9 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-4) - SHIPPED</summary>

### Phase 1: Core Refactoring
**Goal**: The monolith is cleanly split so that prepare.py owns all immutable concerns (data, teacher, evaluation, caching) and train.py is a self-contained, agent-editable training script
**Depends on**: Nothing (first phase)
**Requirements**: REFAC-01, REFAC-02, REFAC-03, REFAC-04, REFAC-05, REFAC-06, REFAC-07
**Plans:** 3 plans
Plans:
- [x] 01-01-PLAN.md -- Create prepare.py with all immutable infrastructure
- [x] 01-02-PLAN.md -- Create train.py with all agent-editable components
- [x] 01-03-PLAN.md -- Smoke tests and boundary verification

### Phase 2: Experiment Infrastructure
**Goal**: A complete experiment loop harness exists that can run train.py, log results, manage git state, and recover from crashes
**Depends on**: Phase 1
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07
**Plans:** 2 plans
Plans:
- [x] 02-01-PLAN.md -- Metrics output, crash handling, VRAM tracking
- [x] 02-02-PLAN.md -- Infrastructure test suite

### Phase 3: Agent Instructions
**Goal**: program.md gives an LLM agent everything it needs to autonomously run effective ReID experiments
**Depends on**: Phase 2
**Requirements**: AGNT-01, AGNT-02, AGNT-03, AGNT-04, AGNT-05
**Plans:** 1 plan
Plans:
- [x] 03-01-PLAN.md -- Write complete program.md

### Phase 4: Validation
**Goal**: The complete system is proven to work end-to-end
**Depends on**: Phase 3
**Requirements**: VALD-01, VALD-02, VALD-03
**Plans:** 2 plans
Plans:
- [x] 04-01-PLAN.md -- Baseline run + one full autonomous cycle
- [x] 04-02-PLAN.md -- Crash recovery verification + launch overnight run

</details>

### v2.0 Expanded Search Space (In Progress)

**Milestone Goal:** Dramatically expand the agent's search space with multi-teacher distillation (5 teachers), SSL contrastive loss, custom LCNet architecture, RADIO training techniques, and DINOv3 fine-tuning -- all as agent-tunable parameters.

**Phase Numbering:**
- Integer phases (5, 6, 7, 8, 9): Planned milestone work
- Decimal phases (e.g. 5.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 5: SSL + Custom LCNet** - Add InfoNCE contrastive loss and custom LCNet backbone with agent-tunable architecture (zero prepare.py changes) (completed 2026-03-25)
- [x] **Phase 6: Multi-Teacher Infrastructure** - Expand prepare.py to support 5+ teachers with independent caches and multi-teacher combos (completed 2026-03-25)
- [x] **Phase 7: DINOv3 Fine-tune** - Fine-tune DINOv3 ViT-H+ (840M) on product dataset using autoresearch pattern (separate sub-project) (completed 2026-03-25)
- [x] **Phase 8: RADIO Integration** - Add RADIO teacher with adaptor selection and spatial distillation via memory-mapped cache (completed 2026-03-25)
- [ ] **Phase 9: RADIO Training Techniques + Wrap-up** - Implement RADIO-inspired training techniques and update program.md with expanded search space

## Phase Details

### Phase 5: SSL + Custom LCNet
**Goal**: The agent has two new independent capabilities in train.py -- a self-supervised contrastive loss and a fully custom LCNet backbone with tunable architecture -- without any changes to prepare.py
**Depends on**: Phase 4 (v1.0 complete)
**Requirements**: SSL-01, SSL-02, SSL-03, LCNET-01, LCNET-02, LCNET-03, LCNET-04, INFRA-09
**Success Criteria** (what must be TRUE):
  1. train.py includes an InfoNCE contrastive loss with a separate projection head, and `SSL_WEIGHT` is a module-level constant the agent can set to 0.0 (disabled) or any positive value
  2. train.py defines a custom LCNet backbone that preserves the `.encode(images) -> Tensor[B, 256]` contract, with `LCNET_SCALE`, `SE_START_BLOCK`, `SE_REDUCTION`, kernel sizes, and `ACTIVATION` as module-level constants
  3. Custom LCNet supports optional timm pretrained weight initialization (when scale matches a known timm variant)
  4. Custom LCNet exposes pre-GAP spatial feature maps (for future RADIO spatial distillation) via a `.encode_spatial()` or equivalent method
  5. `einops` is added to pyproject.toml and importable
**Plans:** 2/2 plans complete
Plans:
- [x] 05-01-PLAN.md -- Custom LCNet backbone with tunable architecture + einops dependency
- [x] 05-02-PLAN.md -- SSL InfoNCE contrastive loss with projection head and training loop integration

### Phase 6: Multi-Teacher Infrastructure
**Goal**: prepare.py supports 5+ teachers with independent caches, and the agent can select which teacher(s) to distill from via module-level constants in train.py
**Depends on**: Phase 5 (custom LCNet needed for spatial features)
**Requirements**: TEACH-01, TEACH-02, TEACH-03, TEACH-04, TEACH-05
**Success Criteria** (what must be TRUE):
  1. prepare.py can initialize and cache embeddings for all 5 teachers: Trendyol ONNX, DINOv2, DINOv3-ft, C-RADIOv4-SO400M, C-RADIOv4-H -- each with its own cache directory and metadata
  2. Cache building runs sequentially per teacher (never two teachers on GPU simultaneously) to stay within 24GB VRAM
  3. `TEACHER` is a module-level constant in train.py that the agent can set to any single teacher name, and the system loads the correct cached embeddings
  4. Multi-teacher mode works: `TEACHER_WEIGHTS` dict in train.py maps teacher names to loss weights, and training uses the weighted combination
**Plans:** 2/2 plans complete
Plans:
- [x] 06-01-PLAN.md -- Teacher registry, DINOv2 fix, cache infrastructure in prepare.py
- [x] 06-02-PLAN.md -- TEACHER/TEACHERS constants and multi-teacher training loop in train.py

### Phase 7: DINOv3 Fine-tune
**Goal**: A DINOv3 ViT-H+ (840M) model is fine-tuned on the product dataset using its own autoresearch loop, producing a teacher checkpoint that integrates into the main system
**Depends on**: Phase 6 (multi-teacher infra to integrate the result)
**Requirements**: DINO3-01, DINO3-02, DINO3-03, DINO3-04
**Success Criteria** (what must be TRUE):
  1. A separate autoresearch sub-project exists (prepare_dino.py + train_dino.py + program_dino.md) that fine-tunes DINOv3 ViT-H+ 840M with LoRA on the product dataset within RTX 4090 VRAM
  2. The fine-tuned DINOv3 model is exported to a format loadable by the main system's teacher infrastructure
  3. DINOv3-ft embeddings are cached to disk like other teachers and produce valid cosine similarities with student embeddings
**Plans:** 3/3 plans complete
Plans:
- [x] 07-01-PLAN.md -- Create dino_finetune/ sub-project (prepare_dino.py + train_dino.py)
- [x] 07-02-PLAN.md -- Create program_dino.md agent instructions
- [x] 07-03-PLAN.md -- Integrate DINOv3FTTeacher into main prepare.py + add peft dependency

### Phase 8: RADIO Integration
**Goal**: RADIO models are fully integrated as teachers with adaptor selection, spatial feature caching, and spatial distillation loss -- all agent-tunable
**Depends on**: Phase 5 (LCNet spatial features), Phase 6 (multi-teacher infra)
**Requirements**: RADIO-01, RADIO-02, RADIO-03, RADIO-04, RADIO-05, RADIO-06
**Success Criteria** (what must be TRUE):
  1. A RADIOTeacher class supports both C-RADIOv4-SO400M and C-RADIOv4-H, with 3 adaptor outputs (backbone, dino_v3, siglip2-g) selectable at init time
  2. Each adaptor's summary features are cached with native dimension, and projection to student dimension happens in train.py (not in the cache)
  3. Spatial features are cached separately using memory-mapped storage (.npy), enabling spatial distillation without re-running RADIO inference
  4. train.py includes a spatial distillation loss that aligns student spatial features (from custom LCNet) with teacher spatial features (from RADIO cache)
  5. `RADIO_VARIANT` and `RADIO_ADAPTORS` are module-level constants in train.py that the agent can tune
**Plans**: TBD

### Phase 9: RADIO Training Techniques + Wrap-up
**Goal**: RADIO-inspired training techniques are available as agent-tunable options, and program.md is updated with the full v2.0 search space documentation
**Depends on**: Phase 8 (RADIO integration)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, INFRA-08, INFRA-10
**Success Criteria** (what must be TRUE):
  1. PHI-S (Hadamard isotropic standardization) and Feature Normalizer (per-teacher whitening + rotation) are implemented in train.py as toggleable modules with enable flags
  2. L_angle (balanced summary loss normalized by angular dispersion) and Hybrid Loss (cosine + smooth-L1 for spatial) are available as loss function options the agent can select
  3. Per-teacher adaptor MLP v2 (LayerNorm+GELU+residual) replaces simple linear projection when enabled, and FeatSharp is implemented (deferred if VRAM-constrained)
  4. Shift Equivariant Loss for spatial distillation is implemented and toggleable
  5. program.md documents the full v2.0 search space (5 teachers, SSL, custom LCNet params, RADIO adaptors, all training techniques) with prioritized experiment strategy, and the evaluation metric remains unchanged (trust boundary preserved)
**Plans:** 2/2 plans complete
Plans:
- [ ] 09-01-PLAN.md -- Feature processing modules (PHI-S, Feature Normalizer, Adaptor MLP v2)
- [ ] 09-02-PLAN.md -- Loss functions (L_angle, Hybrid Loss, FeatSharp, Shift Equivariant)
- [ ] 09-03-PLAN.md -- Wire techniques into training loop + rewrite program.md for v2.0

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7 -> 8 -> 9

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Core Refactoring | v1.0 | 3/3 | Complete | - |
| 2. Experiment Infrastructure | v1.0 | 2/2 | Complete | - |
| 3. Agent Instructions | v1.0 | 1/1 | Complete | - |
| 4. Validation | v1.0 | 2/2 | Complete | - |
| 5. SSL + Custom LCNet | v2.0 | 0/2 | Complete    | 2026-03-25 |
| 6. Multi-Teacher Infrastructure | v2.0 | 2/2 | Complete    | 2026-03-25 |
| 7. DINOv3 Fine-tune | v2.0 | 1/3 | Complete    | 2026-03-25 |
| 8. RADIO Integration | v2.0 | 2/2 | Complete   | 2026-03-25 |
| 9. RADIO Training Techniques + Wrap-up | v2.0 | 0/3 | Planned | - |
