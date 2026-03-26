# Phase 7: DINOv3 Fine-tune - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Create a separate autoresearch sub-project that fine-tunes a DINOv3 ViT-g (1.1B) model on the product dataset using LoRA, producing a teacher checkpoint that integrates into the main multi-teacher infrastructure.

</domain>

<decisions>
## Implementation Decisions

### Training Approach
- **D-01:** Supervised contrastive fine-tune with LoRA — NOT self-supervised DINO-style. Reason: DINO-style requires dual student/teacher models (2x VRAM), LoRA fine-tune fits ViT-g 1.1B on 24GB with batch_size=4-8.
- **D-02:** Loss: InfoNCE contrastive loss on product embeddings. Same product images are positives, different products are negatives. This teaches the model domain-specific visual discrimination.
- **D-03:** LoRA rank=16, alpha=32, applied to attention Q/V matrices. These are the standard settings for vision transformer fine-tuning.
- **D-04:** Data: same product_code_dataset from `/data/mnt/mnt_ml_shared/`. Train/val split inherited from main project.
- **D-05:** Fine-tune for a fixed number of epochs (budget TBD based on VRAM profiling). Agent can tune LR, LoRA rank, augmentation.

### Sub-Project Structure
- **D-06:** Same repo, subdirectory: `dino_finetune/` containing `prepare_dino.py`, `train_dino.py`, `program_dino.md`.
- **D-07:** Follows exact autoresearch pattern: prepare_dino.py is immutable (data, base model loading, evaluation), train_dino.py is agent-editable (LoRA config, optimizer, augmentation), program_dino.md is agent instructions.
- **D-08:** Metric: cosine similarity between same-product embeddings (higher = better). Recall@1 on held-out validation set.
- **D-09:** Output: saved LoRA adapter weights + base model reference. Main system's prepare.py loads base DINOv3 + LoRA adapter to produce the "dinov3_ft" teacher.

### Integration with Main System
- **D-10:** After fine-tuning completes, the adapter is saved to `dino_finetune/output/best_adapter/`. Main prepare.py has a `DINOv3FTTeacher` class that loads base model + adapter.
- **D-11:** DINOv3-ft embeddings are cached to disk like other teachers via the multi-teacher infrastructure from Phase 6.

### Claude's Discretion
- Exact LoRA configuration (which layers, whether to include MLP)
- Training schedule (warmup ratio, LR range)
- Augmentation pipeline for DINOv3 fine-tuning
- Whether to use mixed precision (bf16 vs fp16)
- Evaluation frequency during fine-tuning

</decisions>

<canonical_refs>
## Canonical References

### Source Code
- `prepare.py` — Main system teacher infrastructure (Phase 6 multi-teacher)
- `train.py` — Main system training (reference for autoresearch pattern)
- `program.md` — Main system agent instructions (template for program_dino.md)

### Research
- `.planning/research/FEATURES.md` — DINOv3 fine-tune feature spec
- `.planning/research/PITFALLS.md` — VRAM constraints for ViT-g 1.1B with LoRA

### External
- Trendyol DINOv2 (huggingface.co/Trendyol/trendyol-dino-v2-ecommerce-256d) — reference for e-commerce DINO fine-tuning
- PEFT/LoRA documentation — adapter training patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DINOv2Teacher` class in prepare.py — base DINOv2 loading pattern, extend for DINOv3
- Product dataset loading in prepare.py — reuse for fine-tuning data
- Autoresearch pattern (prepare.py/train.py/program.md) — proven template

### Established Patterns
- Module-level constants, no argparse
- Git commit/reset experiment loop
- results.tsv logging

### Integration Points
- `dino_finetune/output/best_adapter/` → main prepare.py `DINOv3FTTeacher`
- Phase 6 multi-teacher registry → "dinov3_ft" teacher entry

</code_context>

<specifics>
## Specific Ideas

User specifically wants:
- Teacher quality maximized — largest model that fits (ViT-g 1.1B)
- Autoresearch pattern for the fine-tuning itself (AI agent tunes the LoRA config autonomously)
- Training data is the same product dataset already mounted

</specifics>

<deferred>
## Deferred Ideas

- **Multi-GPU DINOv3 fine-tune**: If ViT-g is too constrained on 24GB, could use multi-GPU setup — deferred to v3.0 (OPT-01)
- **Larger DINOv3 variants**: 7B model requires multi-GPU — deferred to v3.0

</deferred>

---

*Phase: 07-dinov3-fine-tune*
*Context gathered: 2026-03-25*
