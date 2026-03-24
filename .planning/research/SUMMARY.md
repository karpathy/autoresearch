# Research Summary: ReID Autoresearch

**Domain:** Autonomous AI-driven experimentation for ReID model training
**Researched:** 2026-03-25
**Overall confidence:** HIGH

## Executive Summary

This project adapts Karpathy's autoresearch pattern (March 2026, 34k+ GitHub stars) to the domain of Re-Identification model training via knowledge distillation. The pattern is proven: Karpathy demonstrated 700 autonomous experiments over two days with 20 additive improvements that transferred to larger models. The core idea is radical simplicity -- three files (prepare.py, train.py, program.md), one metric, one loop.

The existing codebase (`finetune_trendyol_arcface3.py`, ~1400 lines) already implements all the ReID-specific components needed: knowledge distillation from a Trendyol teacher, ArcFace classification, VAT regularization, separation loss, retrieval evaluation, and teacher embedding caching. The work is not building new ML capabilities but refactoring the monolith into the autoresearch split and writing domain-specific agent instructions.

The critical constraint is **no new pip dependencies**. PyTorch 2.9.1, timm, torchvision, transformers, and onnxruntime-gpu are already available and sufficient. The agent's "search algorithm" is the LLM itself -- no Optuna, no Ray Tune, no NAS frameworks. This is the correct approach for this problem because the search space includes architectural changes (which backbone stages to unfreeze, projection head design, loss function composition) that are not expressible in traditional HPO parameter spaces.

The key adaptation from Karpathy's original is **epoch-based budget** (10 epochs) instead of wall-clock time (5 minutes). This is essential because ReID distillation convergence needs epoch-level consistency -- wall-clock varies with batch size and augmentation changes, making experiments non-comparable. The combined metric (0.5 * recall@1 + 0.5 * mean_cosine) provides a stable hill-climbing signal even in short runs.

## Key Findings

**Stack:** No new dependencies needed. PyTorch 2.9.1 + timm + torchvision + onnxruntime-gpu + transformers. The LLM agent replaces HPO/NAS frameworks entirely.

**Architecture:** Standard autoresearch three-file split. prepare.py (immutable: data, teacher, eval, cache) imports into train.py (agent-editable: model, losses, optimizer, loop). program.md carries ReID-specific experiment strategy.

**Critical pitfall:** Evaluation leaking into train.py. If the agent can touch evaluation code, the entire experiment history becomes unreliable. This is the #1 thing to get right in the refactoring.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 1: Core Refactoring** - Split the monolith, establish the trust boundary
   - Addresses: prepare.py/train.py split, immutable evaluation, teacher cache extraction, combined metric
   - Avoids: Wrong split boundary (Pitfall 5), circular imports (Pitfall 6)
   - This is the critical path. Everything else depends on it.

2. **Phase 2: Experiment Loop Infrastructure** - Git management, results logging, crash recovery
   - Addresses: results.tsv format, git branch management, OOM handling, VRAM tracking
   - Avoids: OOM cascade (Pitfall 3), non-deterministic baseline (Pitfall 7)
   - Should be built and tested with a manual experiment before giving to the agent.

3. **Phase 3: Agent Instructions (program.md)** - ReID-specific search strategy
   - Addresses: Domain-specific experiment hints, search space documentation, constraint encoding
   - Avoids: Metric gaming (Pitfall 4), aimless experimentation
   - The "research org code" that makes the agent effective at ReID specifically.

4. **Phase 4: First Autonomous Run** - Baseline + overnight experiments
   - Addresses: End-to-end validation, baseline establishment
   - This is the validation phase. Everything before this is setup.

**Phase ordering rationale:**
- Phase 1 must come first because it creates the files that Phases 2-4 operate on
- Phase 2 before Phase 3 because the infrastructure must work before the agent uses it
- Phase 3 before Phase 4 because domain-specific instructions make experiments effective
- Phase 4 is pure execution -- the system should be complete before starting

**Research flags for phases:**
- Phase 1: Likely needs careful design review. The split boundary is the most important decision and hardest to change later. Research confidence is HIGH on what goes where, but implementation details (exact API between prepare.py and train.py) need phase-specific design.
- Phase 2: Standard patterns, unlikely to need research. Git management and results logging are well-documented in the autoresearch repo.
- Phase 3: Would benefit from deeper ReID domain research. What specific experiment ideas should program.md suggest? What loss functions are most promising? This research covers the landscape but a ReID-specialist review of program.md would be valuable.
- Phase 4: No research needed. Pure execution and observation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All dependencies verified from repo files. No new deps needed -- confirmed by PROJECT.md constraint and autoresearch philosophy |
| Features | HIGH | Table stakes derived directly from Karpathy's autoresearch (verified) + PROJECT.md. Differentiators based on code analysis |
| Architecture | HIGH | Three-file pattern is well-documented. Split boundary derived from code analysis of current monolith |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (eval leakage, cache invalidation) are derived from first principles + autoresearch community reports. Some ReID-specific pitfalls (augmentation gaming) are based on domain knowledge, not direct evidence |

## Gaps to Address

- **Exact timm version installed**: Currently unpinned. Should verify and pin before starting experiments for reproducible baselines
- **Teacher model selection**: Current code supports both ONNX TrendyolEmbedder and DINOv2Teacher. Which to use as the default teacher for autoresearch? Probably ONNX (faster inference, disk-cacheable) but this needs a decision
- **Validation set quality**: Does the current val set include degraded images? If not, the agent might game augmentations (Pitfall 4)
- **program.md experiment ideas**: This research maps the landscape but does not prescribe a specific experiment backlog for the agent. Phase 3 should include a prioritized list of experiments based on ReID literature (circle loss, subcenter ArcFace, proxy-anchor, MixUp/CutMix, label smoothing, etc.)
- **10-epoch budget validation**: Is 10 epochs sufficient for convergence signal in this specific distillation setup? May need a quick manual test before committing to this budget

## Sources

- [Karpathy autoresearch GitHub](https://github.com/karpathy/autoresearch)
- [Fortune: Karpathy's autonomous AI research agent](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
- [VentureBeat: Autoresearch overnight experiments](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [Kingy AI: Autoresearch Minimal Agent Loop](https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [FastReID: PyTorch Toolbox for Re-identification](https://github.com/JDAI-CV/fast-reid)
- Codebase analysis: `finetune_trendyol_arcface3.py`, `pyproject.toml`, `requirements.txt`, `program.md`
