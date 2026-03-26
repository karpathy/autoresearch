# ReID Autoresearch

## What This Is

An autonomous AI-driven experimentation system for ReID (Re-Identification) model training, inspired by Karpathy's autoresearch project. An AI agent autonomously modifies the training code — model architecture, optimizer, loss functions, hyperparameters, augmentations — runs experiments, evaluates results, and keeps or discards changes in a continuous loop. The human writes the agent instructions (`program.md`), not the training code.

## Core Value

The AI agent can autonomously discover better ReID model configurations (higher recall@1 + cosine alignment) without human intervention, running experiments overnight and producing a log of what worked and what didn't.

## Current Milestone: v2.0 Expanded Search Space

**Goal:** Dramatically expand the agent's search space with multi-teacher distillation (5 teachers), SSL contrastive loss, custom LCNet architecture, RADIO training techniques, and DINOv3 fine-tuning — all as agent-tunable parameters.

**Target features:**
- SSL contrastive loss (InfoNCE) as additional training signal
- Custom LCNet backbone with agent-tunable architecture (width, SE, kernels, activation)
- 5 teachers: Trendyol ONNX, DINOv2, DINOv3-ft (ViT-g 1.1B), C-RADIOv4-SO400M, C-RADIOv4-H
- RADIO adaptor outputs (backbone, dino_v3, siglip2-g) as selectable distillation targets
- RADIO spatial distillation with memory-mapped cache
- RADIO training techniques: PHI-S, Feature Normalizer, L_angle, Hybrid Loss, Adaptor MLP v2, FeatSharp
- Teacher selection and multi-teacher combos as agent-tunable parameters
- DINOv3 fine-tuned on product dataset using autoresearch pattern

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Refactor `finetune_trendyol_arcface3.py` into `prepare.py` (fixed: data loading, teacher, evaluation, caching) + `train.py` (agent-editable: student model, optimizer, losses, augmentations)
- [ ] Fixed experiment budget of 10 epochs per run, with teacher cache time excluded from budget
- [ ] Single combined metric: 0.5 * recall@1 + 0.5 * mean_cosine (lower/higher = better clearly defined)
- [ ] Experiment loop: modify train.py → run → evaluate → keep/discard → repeat
- [ ] Results logging in `results.tsv` (commit hash, metric, VRAM, status, description)
- [ ] Git branch management: each experiment is a commit, discard = git reset
- [ ] `program.md` agent instructions tailored for ReID domain
- [ ] Agent runs autonomously and never stops until manually interrupted

### Out of Scope

- Multi-GPU / distributed training — single RTX 4090 only
- DINOv3-7B fine-tune — requires 50GB+ VRAM; use ViT-g 1.1B instead
- RADIO training code reproduction — use inference weights only
- Modifying evaluation logic — ground truth metric must be immutable
- Changing dataset paths or data preparation — fixed infrastructure

## Context

- **Base project**: Karpathy's autoresearch — AI agent experiments with GPT training code autonomously
- **ReID training script**: `finetune_trendyol_arcface3.py` — knowledge distillation (Trendyol teacher → LCNet050 student) + ArcFace classification + VAT regularization + separation loss
- **Student backbone**: `hf-hub:timm/lcnet_050.ra2_in1k` (frozen initially, then last stages unfreezen)
- **Teacher**: Trendyol ONNX model (`distill_qwen_lcnet050_retail_2.onnx`) or DINOv2 (`Trendyol/trendyol-dino-v2-ecommerce-256d`)
- **Datasets**: product_code_dataset (train/val), retail_product_checkout_crop, commodity, negatives — all mounted at `/data/mnt/mnt_ml_shared/`
- **Evaluation**: Retrieval recall@1 and recall@k on validation set + teacher-student cosine alignment
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) — batch sizes and model sizes must respect this limit
- **Key difference from original autoresearch**: Uses fixed epoch count (10) instead of fixed wall-clock time, because ReID distillation convergence needs consistent epoch-level comparison

## Constraints

- **Hardware**: Single RTX 4090 24GB — agent must be VRAM-aware, OOM = crash/discard
- **Teacher cache**: First run builds teacher embedding cache (disk + memory); subsequent runs reuse it. Cache build time is excluded from experiment budget.
- **Data immutability**: Dataset paths and preparation are fixed infrastructure, not experiment variables
- **Package lockdown**: Only dependencies in existing `pyproject.toml` — no new installs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Split into prepare.py + train.py | Follows autoresearch pattern: immutable eval/data + editable training code | — Pending |
| 10 epochs per experiment | Enough for convergence signal on ReID distillation, allows ~4-6 experiments/hour | — Pending |
| Combined metric (recall@1 + cosine) | recall@1 alone is noisy in short runs; cosine provides stable distillation signal | — Pending |
| Exclude teacher cache from budget | Cache build is one-time I/O, not a training variable — shouldn't penalize experiments | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-25 after initialization*
