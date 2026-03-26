# Phase 5: SSL + Custom LCNet - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Add two independent capabilities to train.py with zero prepare.py changes:
1. InfoNCE self-supervised contrastive loss (augmentation invariance signal)
2. Custom LCNet backbone replacing timm, with agent-tunable architecture params

Both are purely train.py changes. The `.encode(images) -> Tensor[B, 256]` contract is preserved.

</domain>

<decisions>
## Implementation Decisions

### SSL Contrastive Loss
- **D-01:** InfoNCE loss with learnable temperature (init 0.07). Two differently-augmented views of the same image are pushed together, different images pushed apart. Standard contrastive learning.
- **D-02:** Separate 2-layer MLP projection head for SSL (256 → 128 → 128, BN+ReLU). Does NOT affect the main 256d embedding used for evaluation. Only used during training.
- **D-03:** Dual-view augmentation: for each batch, apply train.py's augmentation pipeline twice to get view_a and view_b. Both go through the student encoder. Teacher cache is NOT involved in SSL — this is purely self-supervised on student embeddings.
- **D-04:** `SSL_WEIGHT = 0.0` default (disabled). Agent can enable by setting to any positive value (e.g., 0.1). Total loss = distill_loss + arcface_weight * arcface_loss + SSL_WEIGHT * ssl_loss.
- **D-05:** `SSL_TEMPERATURE = 0.07` as tunable constant. `SSL_PROJ_DIM = 128` as tunable constant.

### Custom LCNet Backbone
- **D-06:** Reimplement LCNet from scratch based on PP-LCNet paper + timm source inspection. ~6 stages of DepthwiseSeparableConv blocks. NOT wrapping timm — full control for agent modification.
- **D-07:** Agent-tunable constants: `LCNET_SCALE` (width multiplier: 0.5~2.5), `SE_START_BLOCK` (which stage SE begins), `SE_REDUCTION` (SE squeeze ratio), `ACTIVATION` ("h_swish"/"relu"/"gelu"), `KERNEL_SIZES` (per-stage, default [3,3,3,3,5,5,5,5,5,5,5,5]).
- **D-08:** Fixed architecture pattern: stem → DepthSepConv stages → GAP → FC(1280) → BN → embedding(256) → BN. Agent tunes WITHIN this pattern, doesn't restructure it.
- **D-09:** Optional pretrained weight loading: when LCNET_SCALE matches a known timm variant (0.5 or 1.0), load weights with key mapping. `USE_PRETRAINED = True` as default, agent can disable.
- **D-10:** Custom LCNet class name: `LCNet` (same as current). Must keep `.encode(images) -> Tensor[B, 256]` contract.

### Spatial Feature API
- **D-11:** `model.forward_features(x)` returns `(spatial: Tensor[B, C, H, W], summary: Tensor[B, FC_DIM])`. Spatial is pre-GAP feature map.
- **D-12:** `model.encode(x)` remains the main API (returns L2-normalized Tensor[B, 256]). It calls forward_features internally.
- **D-13:** `model.encode_with_spatial(x)` convenience method returns `(embedding: Tensor[B, 256], spatial: Tensor[B, C, H, W])` for distillation training that needs both.

### Infrastructure
- **D-14:** `einops` added to pyproject.toml (requirement for future RADIO phase, added now since it's cheap).

### Claude's Discretion
- Exact DepthwiseSeparableConv implementation details (padding, bias flags)
- SSL batch construction (in-batch negatives vs memory bank — recommend in-batch for simplicity)
- Whether SSL projection head uses BatchNorm or LayerNorm
- Weight initialization for custom LCNet (kaiming_normal for conv, xavier for linear)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code
- `train.py` — Current train.py with existing model (FrozenBackboneWithHead wrapping timm lcnet_050)
- `prepare.py` — Immutable infrastructure (DO NOT modify in this phase)
- `finetune_trendyol_arcface3.py` — Original monolith with LCNet architecture reference

### Research
- `.planning/research/ARCHITECTURE.md` — Custom LCNet integration analysis, component boundaries
- `.planning/research/FEATURES.md` — SSL and LCNet feature specs with dependencies
- `.planning/research/STACK.md` — einops requirement, LCNet block details from timm inspection
- `.planning/research/PITFALLS.md` — SSL+distillation augmentation conflict, multi-loss weighting risks

### External
- PP-LCNet paper (arXiv:2109.15099) — Original architecture specification

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FrozenBackboneWithHead` in train.py — current model wrapper, will be replaced by custom LCNet
- `build_transforms()` in train.py — augmentation pipeline, reusable for SSL dual-view
- Existing loss functions (distillation_loss, arcface) — SSL loss adds alongside these

### Established Patterns
- Module-level constants for hyperparameters (no argparse)
- `.encode()` contract returning L2-normalized Tensor[B, 256]
- Loss combination: `total_loss = distill_loss + ARCFACE_WEIGHT * arcface_loss`

### Integration Points
- train.py model class — replace FrozenBackboneWithHead with custom LCNet
- train.py training loop — add SSL loss computation (dual-view forward pass)
- train.py constants section — add SSL_WEIGHT, SSL_TEMPERATURE, LCNET_SCALE, etc.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-ssl-custom-lcnet*
*Context gathered: 2026-03-25*
