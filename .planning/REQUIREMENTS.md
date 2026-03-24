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

- [ ] **AGNT-01**: `program.md` contains ReID-specific experiment strategy, constraints, and search space documentation
- [ ] **AGNT-02**: `program.md` includes prioritized experiment hints: loss weights, backbone unfreezing, augmentation, LR schedule, projection head design
- [ ] **AGNT-03**: `program.md` encodes hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop
- [ ] **AGNT-04**: Agent runs autonomously in a never-stop loop -- modify train.py -> run -> evaluate -> keep/discard -> repeat
- [ ] **AGNT-05**: Agent reads results.tsv history to reason about what to try next

### Validation

- [ ] **VALD-01**: Baseline run of unmodified train.py produces a valid combined metric and logs to results.tsv
- [ ] **VALD-02**: At least one full autonomous loop cycle completes: agent modifies train.py, runs, evaluates, keeps or discards
- [ ] **VALD-03**: Crash recovery verified: intentionally trigger OOM, confirm system logs crash and continues

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Multi-Agent

- **MULTI-01**: Multiple agents experiment in parallel on separate GPU branches
- **MULTI-02**: Results from parallel agents are merged and compared

### Advanced Metrics

- **METR-01**: Per-category recall breakdown (e.g., by product type)
- **METR-02**: Embedding visualization (t-SNE/UMAP) generated after each kept experiment

### Model Export

- **EXPRT-01**: Best model automatically exported to ONNX format
- **EXPRT-02**: Best model benchmarked for inference latency

## Out of Scope

| Feature | Reason |
|---------|--------|
| Optuna / Ray Tune / NAS frameworks | LLM agent IS the search algorithm -- HPO frameworks restrict to parameter sweeps |
| MLflow / W&B experiment tracking | results.tsv + git history is sufficient; adds infra complexity |
| Multi-GPU / distributed training | Single RTX 4090 constraint; distributed adds NCCL/sync complexity |
| Configuration files (YAML/JSON) | Code IS the config -- agent edits Python constants directly |
| Warm-starting from previous weights | Breaks fair comparison between experiments |
| Dashboard / visualization UI | No one watches at 3am; results.tsv is human-readable |
| New pip dependencies | Breaks reproducibility and security boundary |
| Agent editing prepare.py | Evaluation must be immutable -- trust boundary |

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
| AGNT-01 | Phase 3 | Pending |
| AGNT-02 | Phase 3 | Pending |
| AGNT-03 | Phase 3 | Pending |
| AGNT-04 | Phase 3 | Pending |
| AGNT-05 | Phase 3 | Pending |
| VALD-01 | Phase 4 | Pending |
| VALD-02 | Phase 4 | Pending |
| VALD-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after roadmap creation*
