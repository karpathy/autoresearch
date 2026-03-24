# Requirements: ReID Autoresearch

**Defined:** 2026-03-25
**Core Value:** AI agent autonomously discovers better ReID model configurations without human intervention

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Refactoring

- [ ] **REFAC-01**: Monolith `finetune_trendyol_arcface3.py` is split into `prepare.py` (immutable) and `train.py` (agent-editable)
- [ ] **REFAC-02**: Evaluation logic (retrieval recall@1/k, mean_cosine) lives exclusively in `prepare.py` and cannot be modified by the agent
- [ ] **REFAC-03**: Teacher embedding cache (ONNX + DINOv2) is extracted to `prepare.py` with disk + memory caching
- [ ] **REFAC-04**: Dataset loading (product_code, retail, commodity, negatives) is extracted to `prepare.py` with fixed train/val splits
- [ ] **REFAC-05**: `train.py` exposes all tunable parameters as module-level constants (no argparse, no config files)
- [ ] **REFAC-06**: `train.py` model implements `.encode(images) -> Tensor[B, 256]` (L2-normalized) contract with `prepare.py`
- [ ] **REFAC-07**: `prepare.py` computes the single combined metric: `0.5 * recall@1 + 0.5 * mean_cosine`

### Infrastructure

- [ ] **INFRA-01**: Each experiment is a git commit; improvement = keep, regression = `git reset --hard`
- [ ] **INFRA-02**: `results.tsv` logs every experiment: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status (kept/discarded/crash), description
- [ ] **INFRA-03**: OOM and runtime crashes are caught, logged as "crash" in results.tsv, git reset performed, loop continues
- [ ] **INFRA-04**: After 3 consecutive crashes on the same idea, agent skips that direction
- [ ] **INFRA-05**: Peak VRAM (`torch.cuda.max_memory_allocated()`) is tracked and logged per experiment
- [ ] **INFRA-06**: Run output includes decomposed sub-metrics (recall@1, recall@5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss) in greppable format
- [ ] **INFRA-07**: Fixed budget of 10 epochs per experiment, teacher cache build time excluded from budget

### Agent

- [ ] **AGNT-01**: `program.md` contains ReID-specific experiment strategy, constraints, and search space documentation
- [ ] **AGNT-02**: `program.md` includes prioritized experiment hints: loss weights, backbone unfreezing, augmentation, LR schedule, projection head design
- [ ] **AGNT-03**: `program.md` encodes hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop
- [ ] **AGNT-04**: Agent runs autonomously in a never-stop loop — modify train.py → run → evaluate → keep/discard → repeat
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
| Optuna / Ray Tune / NAS frameworks | LLM agent IS the search algorithm — HPO frameworks restrict to parameter sweeps |
| MLflow / W&B experiment tracking | results.tsv + git history is sufficient; adds infra complexity |
| Multi-GPU / distributed training | Single RTX 4090 constraint; distributed adds NCCL/sync complexity |
| Configuration files (YAML/JSON) | Code IS the config — agent edits Python constants directly |
| Warm-starting from previous weights | Breaks fair comparison between experiments |
| Dashboard / visualization UI | No one watches at 3am; results.tsv is human-readable |
| New pip dependencies | Breaks reproducibility and security boundary |
| Agent editing prepare.py | Evaluation must be immutable — trust boundary |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REFAC-01 | — | Pending |
| REFAC-02 | — | Pending |
| REFAC-03 | — | Pending |
| REFAC-04 | — | Pending |
| REFAC-05 | — | Pending |
| REFAC-06 | — | Pending |
| REFAC-07 | — | Pending |
| INFRA-01 | — | Pending |
| INFRA-02 | — | Pending |
| INFRA-03 | — | Pending |
| INFRA-04 | — | Pending |
| INFRA-05 | — | Pending |
| INFRA-06 | — | Pending |
| INFRA-07 | — | Pending |
| AGNT-01 | — | Pending |
| AGNT-02 | — | Pending |
| AGNT-03 | — | Pending |
| AGNT-04 | — | Pending |
| AGNT-05 | — | Pending |
| VALD-01 | — | Pending |
| VALD-02 | — | Pending |
| VALD-03 | — | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22 ⚠️

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after initial definition*
