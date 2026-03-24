# Feature Landscape

**Domain:** Autonomous ML Experimentation for ReID Knowledge Distillation
**Researched:** 2026-03-25

## Table Stakes

Features the system must have or it simply does not function as an autoresearch-style loop.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **prepare.py / train.py split** | Immutable eval + mutable training is the core autoresearch contract. Without it the agent can game metrics or break evaluation. | High | Current `finetune_trendyol_arcface3.py` is a ~1400-line monolith -- must be split. prepare.py owns teacher, data loading, eval, caching. train.py owns student model, losses, optimizer, augmentations. |
| **Single combined metric** | Agent needs one unambiguous scalar to hill-climb. Multi-objective confuses the keep/discard gate. | Low | `0.5 * recall@1 + 0.5 * mean_cosine` per PROJECT.md. Must be computed in prepare.py (immutable). |
| **Fixed epoch budget (10 epochs)** | Consistent comparison across experiments. Without fixed budget, agent can cheat by training longer. | Low | ReID distillation needs epoch-level consistency, not wall-clock (unlike original autoresearch's 5-min budget). |
| **Keep/discard gate with git** | Each experiment = git commit. Improvement = keep. Regression = `git reset`. Branch history shows only improvements. | Low | Standard autoresearch pattern. Agent commits before running, resets on failure. |
| **results.tsv logging** | Persistent record of all experiments (kept, discarded, crashed). Human reviews in morning. | Low | Tab-separated: commit, metric, VRAM, status, description. Must NOT be git-tracked. |
| **Teacher embedding cache** | Teacher inference is expensive (ONNX or DINOv2). Without caching, most experiment time is wasted on repeated teacher forward passes. | Med | Already implemented: in-memory dict + disk .npy files. Cache build time excluded from budget. Must live in prepare.py. |
| **OOM crash recovery** | On 24GB RTX 4090, agent WILL hit OOM. System must catch crashes, log them, revert, and continue. | Low | Crash = log "crash" status, `git reset`, move on. After 3+ consecutive crashes on same idea, skip it. |
| **Never-stop loop** | Agent runs autonomously and never pauses to ask. Human is asleep. | Low | Enforced by program.md instructions: "NEVER STOP". Structural, not optional. |
| **Immutable evaluation function** | Ground truth metric cannot be touched by agent. This is the trust boundary that prevents reward hacking. | Med | Retrieval recall@1/k + teacher-student cosine alignment. Must be in prepare.py only. |
| **VRAM tracking** | Every experiment logs peak VRAM. Agent must be VRAM-aware to avoid constant OOM on 24GB limit. | Low | `torch.cuda.max_memory_allocated()` after each run. Logged in results.tsv. |
| **Retrieval evaluation** | recall@1 and recall@k on validation set -- the actual product metric. | Med | Already implemented. Must be extracted to prepare.py (immutable). |
| **Agent instructions (program.md)** | Domain-specific guidance for ReID experimentation strategy, constraints, and what to try. | Med | The "research org code" -- encodes human research judgment. |

## Differentiators

Features that make this ReID-specific autoresearch more effective than a naive port of Karpathy's original.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Epoch-based budget (not wall-clock)** | ReID distillation convergence needs consistent epoch-level comparison. Wall-clock varies with augmentation and batch size changes, making comparison unfair. | Low | Key philosophical difference from original autoresearch. |
| **Dual-metric (retrieval + alignment)** | recall@1 alone is noisy in short (10-epoch) runs. Cosine alignment provides a stable distillation signal even when retrieval hasn't converged. The combination gives the agent a reliable hill-climbing signal. | Low | Already designed in PROJECT.md. |
| **ReID-aware search space hints in program.md** | Agent needs domain guidance: what losses to try (triplet, contrastive, proxy-anchor, circle loss, subcenter ArcFace), what augmentations matter for ReID (color jitter, random erasing, quality degradation), backbone unfreezing strategies. Generic agent wastes experiments on irrelevant changes. | Med | program.md is the "research org code" -- encode ReID-specific research judgment. Include a prioritized experiment idea backlog for when the agent gets stuck. |
| **Staged backbone unfreezing as experiment axis** | Current code freezes backbone then unfreezes last 4 stages. Making unfreezing schedule an explicit experiment variable gives the agent a high-leverage axis unique to distillation. Number of stages, epoch to unfreeze, per-stage LR multiplier. | Low | Already partially implemented (`unfreeze_last_stage`). Expose as train.py variables. |
| **Loss composition search** | Agent can adjust relative weights of distillation, ArcFace, VAT, and separation losses. This is the richest search dimension -- small weight changes can have outsized impact on convergence. | Low | Already implemented. Expose all weights as train.py constants. |
| **Quality degradation as tunable augmentation** | `RandomQualityDegradation` simulates real-world low-res and JPEG artifacts. ReID models must be robust to this. Degradation probability, downsample ratio, and quality range are domain-specific tuning knobs. | Low | Already implemented. Move params to train.py constants. |
| **Separation loss for blacklist/whitelist** | Pushing blacklist embeddings away from whitelist centroid is a domain-specific loss. Agent can tune sep_weight, EMA decay, margin. Not in original autoresearch. | Low | Already implemented. Expose in train.py. |
| **VAT regularization tuning** | Feature-level VAT (Miyato et al., 2018) for embedding robustness. Agent can tune epsilon, xi, power iterations, weight. Unique to this ReID distillation setup. | Low | Already implemented. Expose in train.py. |
| **Teacher cache exclusion from budget** | First-run cache build is one-time I/O overhead, not a training variable. Should not penalize experiments. | Low | Time the cache phase separately from training epochs. |
| **Metric decomposition in run output** | Log not just combined metric but recall@1, recall@5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss separately. Agent can see WHICH sub-metric improved/regressed to reason about next experiment. | Low | Already computed in EpochStats. Ensure they appear in greppable run output. |
| **Cosine warmup + LR schedule as search variable** | Distillation benefits from warmup (avoid destroying teacher alignment early). Making schedule type and warmup fraction agent-tunable is high-value, low-cost. | Low | Standard PyTorch schedulers. Expose in train.py. |

## Anti-Features

Things to deliberately NOT build. The autoresearch philosophy is radical simplicity. These add complexity without proportional value.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Experiment tracking platform (MLflow, W&B, Neptune)** | Adds infrastructure dependency, config complexity, and network I/O to the tight loop. Autoresearch deliberately avoids this. A TSV file + git history IS the tracking system. | results.tsv + `git log` on the experiment branch. Human reads TSV in the morning. |
| **Hyperparameter search framework (Optuna, Ray Tune)** | These impose a search algorithm (Bayesian, TPE, etc.) that fights with the agent's own reasoning. The whole point is the LLM IS the search algorithm. Optuna restricts to parameter sweeps; the agent can do architectural changes. | Let the agent decide what to try. It reads results.tsv, sees what worked, and reasons about next steps. |
| **Multi-GPU / distributed training** | Single RTX 4090 is the constraint. Distributed adds complexity (NCCL, gradient sync, batch size scaling) without benefit on one card. | Single GPU. Agent must fit everything in 24GB. |
| **Configuration files (YAML/TOML/JSON)** | Indirection between config and code makes agent diffs harder to review and increases config-code mismatch risk. Code IS the config. Karpathy's pattern: constants at the top of train.py. | Agent edits Python constants directly in train.py. |
| **Model checkpointing between experiments** | Each experiment trains 10 epochs from scratch. Checkpointing adds "which checkpoint to resume from?" complexity. Fresh start ensures fair comparison. | Train from scratch every experiment. Only save final model if it beats the overall best. |
| **Warm-starting from previous experiment weights** | Conflates "this architecture is better" with "this architecture benefits from pre-training on previous weights." Breaks fair comparison and makes the hill-climbing signal unreliable. | Always start from pretrained backbone (timm) + fresh projection head. Never carry over experiment-specific weights. |
| **Automated dataset curation** | Data is fixed infrastructure. Letting the agent modify data loading, dataset paths, or create synthetic data breaks the immutability contract and makes experiments non-comparable. | Data loading in prepare.py (immutable). Agent can only change augmentation transforms in train.py. |
| **Dashboard / visualization UI** | No one watches a dashboard at 3am. Web server, frontend dependencies, and maintenance burden for zero value in autonomous mode. | results.tsv is human-readable. `cat results.tsv \| column -t` after waking up. |
| **Multi-agent coordination** | Karpathy says this is "the next step" -- not the current step. Single-agent, single-GPU is the proven pattern. Multi-agent adds coordination complexity (merge conflicts, result aggregation). | One agent, one GPU, one branch. Scale to multi-agent later if needed. |
| **Ensemble methods / model selection** | Goal is one good student model, not an ensemble. Ensembles defeat the purpose of distilling to a lightweight LCNet050. | Single best model wins. |
| **Dynamic epoch budget** | Letting agent choose how many epochs to train breaks comparability. 10 epochs is the fixed contract. | Fixed 10 epochs, always. Agent optimizes WHAT happens in 10 epochs, not how many. |
| **New pip dependency installation** | Breaks reproducibility and security boundary. Could introduce conflicts or vulnerabilities. | Only what is in pyproject.toml. torch, timm, onnxruntime, transformers are already available. |
| **Complex rollback (rewind N experiments)** | Agent should almost never rewind. Hill-climbing means HEAD is always the best known state. Rewinding wastes experiments re-exploring dead ends. | Linear forward progress. If stuck, try radically different approaches rather than rewinding. |
| **Agent editing prepare.py** | Evaluation must be immutable for fair comparison. If agent touches eval code, the entire trust boundary collapses. | Enforce structurally via program.md instructions. The agent edits only train.py. |
| **Automatic report generation** | Over-engineering. results.tsv + git log IS the lab notebook. | Plain text descriptions in results.tsv. |

## Feature Dependencies

```
prepare.py/train.py split (CRITICAL PATH -- everything depends on this)
  |
  +-> Immutable evaluation function (must be in prepare.py)
  |     +-> Single combined metric (computed by eval)
  |           +-> Keep/discard gate (needs one number to compare)
  |           +-> results.tsv logging (logs the metric)
  |
  +-> Teacher embedding cache (must be in prepare.py)
  |     +-> Teacher cache exclusion from budget
  |
  +-> Data loading extraction (must be in prepare.py)
  |
  +-> All train.py-editable features:
        +-> Loss composition search (weights as constants)
        +-> Backbone unfreezing schedule
        +-> Quality degradation params
        +-> Separation loss tuning
        +-> VAT regularization tuning
        +-> LR schedule / warmup

program.md with ReID hints
  +-> Experiment idea backlog (embedded in program.md)
  +-> Search space guidance (what to try, what to avoid)

OOM crash recovery
  +-> VRAM tracking (agent needs to see what caused OOM)
  +-> Keep/discard gate (crash = discard + revert)

Metric decomposition in run output
  +-> results.tsv (combined metric logged)
  +-> Agent reasoning (sub-metrics inform next experiment)
```

## MVP Recommendation

Build in this order:

**Phase 1 -- Core Loop (must ship together, system does not function without all of these):**
1. prepare.py / train.py split -- this unlocks everything else
2. Single combined metric in prepare.py
3. Immutable evaluation function in prepare.py
4. Fixed 10-epoch budget
5. Teacher embedding cache in prepare.py
6. Keep/discard gate with git
7. results.tsv logging
8. OOM crash recovery
9. VRAM tracking
10. Never-stop loop

**Phase 2 -- ReID Domain Optimization (makes the agent effective at ReID specifically):**
1. program.md with ReID-aware experiment hints and idea backlog
2. Metric decomposition in greppable run output
3. Expose all existing knobs as train.py constants (backbone unfreezing, loss weights, augmentation params, LR schedule, separation loss, VAT params)

**Defer indefinitely:** All anti-features. The system's value comes from what it does NOT have. Simplicity is the feature.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table stakes | HIGH | Directly derived from Karpathy's autoresearch (verified via GitHub, DeepWiki, multiple articles) + PROJECT.md requirements |
| Differentiators | MEDIUM | Based on analysis of current training code + ReID domain knowledge from academic literature. Some features (e.g., proxy-anchor loss as experiment idea) are research-informed but untested in this specific setup |
| Anti-features | HIGH | Strong consensus across multiple sources that autoresearch's power comes from simplicity. Every anti-feature listed contradicts the core philosophy |
| Feature dependencies | HIGH | Derived directly from code analysis and architectural constraints |

## Sources

- [Karpathy autoresearch GitHub](https://github.com/karpathy/autoresearch)
- [DeepWiki: karpathy/autoresearch](https://deepwiki.com/karpathy/autoresearch)
- [Kingy AI: Autoresearch Minimal Agent Loop](https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/)
- [Fortune: Why everyone is talking about Karpathy's autonomous AI research agent](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
- [VentureBeat: Autoresearch lets you run hundreds of AI experiments a night](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [Ken Huang: Exploring Karpathy's Autoresearch](https://kenhuangus.substack.com/p/exploring-andrej-karpathys-autoresearch)
- [DataCamp: Guide to AutoResearch](https://www.datacamp.com/tutorial/guide-to-autoresearch)
- [SOTAAZ: Build Your Own autoresearch](https://www.sotaaz.com/post/autoresearch-part3-en)
- [AgentHPO: LLM Agent for Hyper-Parameter Optimization (arXiv 2402.01881)](https://arxiv.org/abs/2402.01881)
- Current training code: `finetune_trendyol_arcface3.py` (direct analysis)
- Project context: `.planning/PROJECT.md`, `program.md`
