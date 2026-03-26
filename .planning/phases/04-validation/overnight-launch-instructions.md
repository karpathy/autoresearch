# Overnight Autonomous Run Launch Instructions

## Prerequisites (all verified)
- [x] Clean git state (no uncommitted changes)
- [x] No OOM trigger in train.py
- [x] train.py in last-kept state (ARCFACE_LOSS_WEIGHT=0.03, combined_metric=0.706)
- [x] results.tsv has 3 data rows (baseline + experiment + crash)
- [x] Teacher cache exists at workspace/output/trendyol_teacher_cache2/
- [x] All 3 validations passed (VALD-01, VALD-02, VALD-03)

## Launch Command

From the project directory `/home/whiskey/workspace/project/central/v2/training/autoresearch`, run in a persistent terminal (tmux/screen recommended):

### Option 1: Claude Code CLI (recommended)
```bash
cd /home/whiskey/workspace/project/central/v2/training/autoresearch
claude --print "Read program.md for your full instructions. You are starting an autonomous experiment run. Read results.tsv for history, then begin the experiment loop. NEVER stop -- run indefinitely until manually interrupted."
```

### Option 2: Interactive Claude Code session
```bash
cd /home/whiskey/workspace/project/central/v2/training/autoresearch
claude
```
Then paste:
```
Read program.md for your full instructions. You are starting an autonomous experiment run. Read results.tsv for history, then begin the experiment loop. NEVER stop -- run indefinitely until manually interrupted.
```

### Option 3: tmux session (recommended for overnight)
```bash
tmux new-session -s autoresearch
cd /home/whiskey/workspace/project/central/v2/training/autoresearch
claude --print "Read program.md for your full instructions. You are starting an autonomous experiment run. Read results.tsv for history, then begin the experiment loop. NEVER stop -- run indefinitely until manually interrupted."
# Press Ctrl+B then D to detach
```

## What the Agent Does
1. Reads program.md (experiment loop instructions)
2. Reads results.tsv (experiment history)
3. Reads train.py (current best configuration)
4. Chooses next experiment based on history
5. Edits train.py, commits, runs training
6. Evaluates results, keeps/discards
7. Logs to results.tsv
8. Repeats indefinitely

## Monitoring
- Check results: `cat results.tsv`
- Check current run: `tail -f run.log`
- Check git history: `git log --oneline -20`
- Check VRAM: `nvidia-smi`

## Stopping
- Kill the Claude process (Ctrl+C in terminal)
- The agent will stop after current experiment completes
