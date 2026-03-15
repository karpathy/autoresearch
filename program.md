# Medical Imaging AutoResearch — Agent Instructions

## Context
You are an AI research agent running automated DL experiments on medical
image classification using MedMNIST+ datasets. Your goal: maximize val_auc
within a fixed 5-minute training budget per experiment.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `medical`.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model, optimizer, training loop.
4. **Verify data exists**: Check that `~/.medmnist/` contains the dataset. If not, run `python prepare.py` first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

## Rules
- You may ONLY edit `train.py`
- You may NOT edit `prepare.py` or any other file
- You may NOT install new packages
- Each experiment must complete within the 5-minute time budget
- After each run, record: hypothesis, change, val_auc, val_acc, accepted/rejected

## What to edit in train.py

Sections marked with `# === ... (agent can modify) ===` are fair game:

- **MODEL ARCHITECTURE**: ResNet variants, EfficientNet, ViT, hybrid models
- **DATA AUGMENTATION**: transforms pipeline (RandomCrop, ColorJitter, domain-specific medical augmentations)
- **OPTIMIZER & SCHEDULER**: AdamW, SGD+momentum, cosine/step/warmup schedules
- **HYPERPARAMETERS**: batch_size, learning_rate, weight_decay, image resolution
- **TRAINING LOOP**: mixup, cutmix, label smoothing, gradient accumulation
- **ALTERNATIVE APPROACH**: Foundation Model (DINO/CLIP) embeddings + linear probe

## Evaluation (priority order)
1. `val_auc` (primary metric, higher is better)
2. `val_acc` (secondary)

## Key insights from literature (Bamberg xAILab, 2024-2025)
- Higher image resolution does NOT always improve performance
- Foundation Model + linear probe is a strong and compute-efficient baseline
- Domain-specific augmentation outperforms generic augmentation for robustness
- CNNs are competitive with ViTs on MedMNIST scale datasets
- Pretrained weights significantly help, especially with limited data

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	val_auc	val_acc	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.812345) — use 0.000000 for crashes
3. val_acc achieved (e.g. 0.734567) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## Experiment protocol

LOOP FOREVER:

1. Read current `train.py`
2. Formulate ONE hypothesis
3. Make ONE focused change
4. `git commit`
5. Run: `python train.py > run.log 2>&1`
6. Read results: `grep "val_auc\|val_acc" run.log`
7. If val_auc improved → keep the commit. If not → `git reset --hard HEAD~1`
8. Record results in `results.tsv` (do NOT commit this file)
9. Repeat

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Run until manually interrupted.

**Crashes**: If a run crashes (OOM, bug), fix obvious issues and re-run once. If the idea is fundamentally broken, log `crash` and move on.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.
