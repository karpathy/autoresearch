# Domain Pitfalls

**Domain:** Autonomous ML experimentation for ReID model training
**Researched:** 2026-03-25

## Critical Pitfalls

Mistakes that cause rewrites, broken experiment loops, or unreliable results.

### Pitfall 1: Evaluation Leaking Into train.py
**What goes wrong:** Agent modifies evaluation logic (intentionally or accidentally) to inflate metrics. The combined metric goes up but the model is not actually better.
**Why it happens:** If evaluation code lives in train.py, the agent edits it as part of its normal workflow. Even unintentional changes (import order, normalization, data subset selection) can shift metrics.
**Consequences:** All experiments become non-comparable. Entire experiment history is unreliable. Must restart from scratch.
**Prevention:** Evaluation MUST live in prepare.py (immutable). train.py imports evaluation as a black box. program.md must explicitly state: "Never modify anything imported from prepare.py."
**Detection:** If combined_metric jumps dramatically (>0.1) in a single experiment, inspect whether evaluation code changed via `git diff`.

### Pitfall 2: Teacher Cache Invalidation
**What goes wrong:** Teacher cache becomes stale or corrupted. Experiments use wrong teacher embeddings. Cosine alignment metric becomes meaningless.
**Why it happens:** Multiple scenarios: (a) teacher model changed but cache not cleared, (b) image paths changed, (c) disk corruption, (d) cache key collision (MD5 of path).
**Consequences:** Distillation trains against wrong targets. Metrics are unreliable. Hours of experiments wasted.
**Prevention:** Cache key should include teacher model hash + image path. On startup, validate a sample of cached embeddings against fresh inference. Cache lives in prepare.py (immutable), keyed by model identity + image path.
**Detection:** If mean_cosine suddenly drops or spikes without a corresponding train.py change, suspect cache issues. Add a cache validation step to prepare.py startup.

### Pitfall 3: OOM Cascade (Agent Repeatedly Tries OOM-Prone Ideas)
**What goes wrong:** Agent tries increasing batch size or model size, hits OOM, crashes. Reverts. Then tries a slightly different OOM-prone change. Crashes again. Spends hours in a crash-revert loop with zero progress.
**Why it happens:** Agent lacks memory of why previous attempts failed. results.tsv shows "crash" but not specifically "OOM at batch_size=128." Agent reasons "maybe the problem was something else" and tries again.
**Consequences:** Wasted experiment budget. Agent produces no improvements overnight.
**Prevention:** (1) Log peak VRAM in results.tsv for every experiment including crashes. (2) In program.md, explicitly state VRAM limits: "RTX 4090 = 24GB. If peak VRAM > 22GB on a successful run, do NOT increase batch size or model size further." (3) After OOM crash, log the specific cause in the description column.
**Detection:** Multiple consecutive "crash" entries in results.tsv.

### Pitfall 4: Metric Gaming via Augmentation Suppression
**What goes wrong:** Agent discovers that removing all training augmentations improves recall@1 on the validation set. Keeps the change. Model overfits to clean validation images and fails on real-world degraded inputs.
**Why it happens:** Validation set is clean images. Removing augmentations lets the model memorize clean image features. Short 10-epoch runs do not show overfitting clearly.
**Consequences:** Model appears improved but degrades in production on low-quality surveillance/retail images.
**Prevention:** (1) Include quality-degraded images in the validation set. (2) In program.md, warn: "Do NOT remove quality degradation augmentation. Real-world ReID images are low-resolution and JPEG-compressed." (3) Consider adding a degraded-validation metric to the combined score (future).
**Detection:** recall@1 improves but mean_cosine stays flat or drops. Or: recall@1 on degraded test images drops.

### Pitfall 5: Incorrect prepare.py / train.py Split Boundary
**What goes wrong:** Too much goes into prepare.py (agent has nothing to change) or too little (evaluation is editable). The most common mistake: putting the training loop in prepare.py and only leaving hyperparameters in train.py.
**Why it happens:** Unclear about what "agent-editable" means. Tendency to put "infrastructure" in prepare.py, but the training loop IS the experiment.
**Consequences:** If too much is fixed: agent can only tune numbers, not architecture or training strategy. System degenerates into a worse version of Optuna. If too little is fixed: agent can break evaluation integrity.
**Prevention:** Clear rule: prepare.py owns data, teacher, evaluation, caching. train.py owns model definition, losses, optimizer, scheduler, augmentations, and the full training loop. The training loop is agent-editable because the agent should be able to change gradient accumulation, epoch structure, warmup logic, etc.
**Detection:** Review the split before starting experiments. Ask: "Can the agent change the model architecture? The loss function? The optimizer? The training loop structure?" All must be yes.

## Moderate Pitfalls

### Pitfall 6: Circular Import Between prepare.py and train.py
**What goes wrong:** prepare.py needs to call model.encode() for evaluation. train.py imports from prepare.py. Circular dependency.
**Prevention:** prepare.py should accept the model as a parameter to evaluate_retrieval(), not import it. Pattern: `evaluate_retrieval(model, val_loader, device)` where model is passed in from train.py.

### Pitfall 7: Non-Deterministic Baseline
**What goes wrong:** First run (baseline) gets different results each time due to random augmentations, data shuffling, or CUDA non-determinism. Agent discards good experiments because baseline was lucky.
**Prevention:** Set random seeds in prepare.py (immutable). Use `torch.backends.cudnn.deterministic = True` for the baseline run. Accept that subsequent runs may have slight variance, but baseline must be reproducible.

### Pitfall 8: Git History Pollution
**What goes wrong:** Agent commits every experiment (even discarded ones) to the branch. After 100 experiments, git log has 100 commits but only 15 were kept. Hard to review what actually improved.
**Prevention:** Use the standard autoresearch pattern: commit before running, reset if discarded. Only kept experiments remain in git history. results.tsv (untracked) records everything including discards.

### Pitfall 9: ArcFace Class Count Mismatch After Split
**What goes wrong:** ArcFace head size depends on number of classes in the dataset. If dataset loading is in prepare.py but ArcFace head is in train.py, the class count must be passed correctly.
**Prevention:** prepare.py should expose `NUM_ARCFACE_CLASSES` as a constant or return it from dataset construction. train.py uses this to initialize ArcMarginProduct. Never hardcode class counts.

### Pitfall 10: Agent Breaks Import Contract
**What goes wrong:** Agent modifies train.py in a way that changes what it imports from prepare.py (e.g., removes an import it still needs, or adds an import of a private function).
**Prevention:** prepare.py should have a clean public API. Document the import contract in comments at the top of both files. In program.md, list the exact imports from prepare.py that are available.

### Pitfall 11: Forgetting to Clear CUDA Cache Between Runs
**What goes wrong:** Agent runs experiment A, then experiment B. VRAM from A is still allocated. B hits OOM not because B is too big, but because A's tensors were not freed.
**Prevention:** In the experiment loop (run by the agent), ensure each run is a fresh Python process (`python train.py > run.log 2>&1`). Do NOT run experiments in the same process. Each experiment = new process = clean VRAM.

## Minor Pitfalls

### Pitfall 12: results.tsv Tab vs Comma Confusion
**What goes wrong:** Agent accidentally uses commas instead of tabs. TSV parsing breaks. Descriptions with commas corrupt the format.
**Prevention:** Use tabs explicitly. In program.md, specify: "Tab-separated, NOT comma-separated." Show exact format with example.

### Pitfall 13: run.log Growing Unbounded
**What goes wrong:** After 100+ experiments, disk fills up with large log files if not managed.
**Prevention:** Overwrite run.log each experiment (standard: `> run.log 2>&1` overwrites). Only results.tsv persists across experiments.

### Pitfall 14: Agent Trying to Modify Backbone Architecture Directly
**What goes wrong:** Agent tries to add layers to LCNet050 backbone by modifying timm internals. Fails because timm models have frozen architecture.
**Prevention:** In program.md, clarify: "You can change which backbone to use (from timm registry), which stages to unfreeze, and the projection head. You cannot add layers inside the backbone."

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| prepare.py/train.py split | Wrong boundary (Pitfall 5), circular imports (Pitfall 6) | Follow the clear rule: prepare=data+teacher+eval, train=model+losses+optimizer+loop |
| First baseline run | Non-deterministic baseline (Pitfall 7) | Set seeds, verify reproducibility with 2 identical runs |
| Agent loop start | OOM cascade (Pitfall 3) | Log VRAM in results.tsv, document VRAM limits in program.md |
| Program.md authoring | Agent gaming augmentations (Pitfall 4) | Warn about augmentation removal, include degraded images in val |
| Long overnight runs | Evaluation leakage (Pitfall 1), cache corruption (Pitfall 2) | Immutable prepare.py, cache validation on startup |
| ArcFace setup | Class count mismatch (Pitfall 9) | Expose NUM_ARCFACE_CLASSES from prepare.py |

## Sources

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) -- pattern analysis, common issues
- [Karpathy tweet on 700 experiments](https://x.com/karpathy/status/2031135152349524125) -- scale issues
- [Fortune: Karpathy's autonomous AI research agent](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) -- adoption challenges
- Analysis of `finetune_trendyol_arcface3.py` -- code-specific pitfalls
- ReID domain knowledge: ArcFace margin sensitivity, distillation convergence patterns
