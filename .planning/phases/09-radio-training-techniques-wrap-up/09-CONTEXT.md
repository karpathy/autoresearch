# Phase 9: RADIO Training Techniques + Wrap-up - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement RADIO-inspired training techniques as agent-tunable options in train.py, and update program.md with the full v2.0 expanded search space documentation.

</domain>

<decisions>
## Implementation Decisions

### Training Techniques — Priority & Scope
- **D-01:** Implementation priority (mandatory → optional):
  1. **PHI-S** (TRAIN-01) — MANDATORY. Hadamard isotropic standardization prevents any single teacher from dominating gradients. Critical for multi-teacher setup.
  2. **Feature Normalizer** (TRAIN-02) — MANDATORY. Per-teacher whitening + rotation aligns feature distributions before loss computation.
  3. **Hybrid Loss** (TRAIN-04) — MANDATORY. 0.9*cosine + 0.1*smooth-L1 for spatial features. Proven in RADIO papers.
  4. **L_angle** (TRAIN-03) — IMPORTANT. Balanced summary loss normalized by angular dispersion. Prevents loss imbalance.
  5. **Adaptor MLP v2** (TRAIN-05) — IMPORTANT. LayerNorm+GELU+residual projection head. Better than simple linear.
  6. **FeatSharp** (TRAIN-06) — OPTIONAL. Spatial feature sharpening. Defer if VRAM-constrained.
  7. **Shift Equivariant Loss** (TRAIN-07) — OPTIONAL. Spatial distillation regularizer.

### Implementation Pattern
- **D-02:** Each technique is a toggleable module with an `ENABLE_*` flag in train.py (e.g., `ENABLE_PHI_S = True`). Agent can enable/disable independently.
- **D-03:** Techniques that modify loss computation (PHI-S, L_angle, Hybrid Loss) are applied inside the existing distillation_loss function via conditional branches.
- **D-04:** Techniques that modify feature processing (Feature Normalizer, FeatSharp) are nn.Module classes that wrap student/teacher features before loss computation.
- **D-05:** Adaptor MLP v2 replaces simple Linear projection heads when enabled. Agent can toggle between simple and MLP v2 per teacher.

### PHI-S Specifics
- **D-06:** PHI-S applies Hadamard rotation followed by standardization (zero mean, unit variance) to each teacher's features independently. Uses a fixed Hadamard matrix (no learnable params). Applied before loss computation.

### Feature Normalizer Specifics
- **D-07:** Computes running mean and covariance of teacher features during first epoch (warmup). Then applies whitening transform for the rest of training. Per-teacher, computed from training set.

### program.md Update
- **D-08:** program.md gets a complete rewrite for v2.0 search space:
  - 5 teachers with selection guidance
  - SSL contrastive loss usage patterns
  - Custom LCNet architecture search hints
  - RADIO adaptor selection strategy
  - Training technique enable/disable recommendations
  - Expanded experiment playbook with multi-teacher phases
- **D-09:** Evaluation metric UNCHANGED: 0.5*recall@1 + 0.5*mean_cosine. Trust boundary preserved.
- **D-10:** Hard constraints updated: never edit prepare.py, never add dependencies (except einops already added), never exceed epoch budget, never stop.

### Claude's Discretion
- Exact Hadamard matrix size for PHI-S (match teacher embedding dim or use closest power of 2)
- Feature Normalizer warmup strategy (how many batches to accumulate statistics)
- FeatSharp implementation details (if VRAM allows)
- Shift Equivariant Loss exact formulation
- program.md experiment phase ordering (which techniques to try first)

</decisions>

<canonical_refs>
## Canonical References

### Source Code
- `train.py` — Phase 8 version with RADIO distillation + custom LCNet + SSL
- `prepare.py` — Phase 6 multi-teacher infrastructure
- `program.md` — v1.0 version to be rewritten

### Research
- `.planning/research/ARCHITECTURE.md` — Training technique integration analysis
- `.planning/research/FEATURES.md` — Technique descriptions and dependencies

### External (RADIO papers — user provided, in RADIO/ directory)
- `RADIO/RADIOv2.5_tech_report.pdf` — PHI-S, Feature Normalizer, L_angle original descriptions
- `RADIO/RADIOv4.0_tech_report.pdf` — C-RADIOv4 improvements, Adaptor MLP v2
- `RADIO/amradio.pdf` — AM-RADIO original paper
- `RADIO/phi2.5.pdf` — PHI-S 2.5 standardization method
- `RADIO/FeatSharp.pdf` — FeatSharp spatial feature sharpening

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing distillation_loss function — extend with technique wrappers
- Existing projection heads — replace with Adaptor MLP v2 when enabled
- Existing program.md — template structure for v2.0 rewrite

### Established Patterns
- Module-level enable flags (ENABLE_*)
- Loss combination: weighted sum of loss terms
- program.md structure: Setup, Constraints, Search Space, Loop, Never Stop

### Integration Points
- train.py distillation_loss — wrap with PHI-S, Feature Normalizer, L_angle
- train.py projection heads — swap with Adaptor MLP v2
- train.py spatial distillation — add FeatSharp + Shift Equivariant Loss
- program.md — complete rewrite with v2.0 search space

</code_context>

<specifics>
## Specific Ideas

User wants all RADIO training techniques from the technical reports available as agent-tunable options. The papers are already in the RADIO/ directory for reference.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 09-radio-training-techniques-wrap-up*
*Context gathered: 2026-03-25*
