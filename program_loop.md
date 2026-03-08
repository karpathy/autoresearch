# Closed-loop autoresearch

This is the unified research loop: train a model, stress-test it for failure modes, use those failures to inform the next training change, repeat. The model and its tests co-evolve.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `loop-mar8`). The branch `loop/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b loop/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — model architecture, optimizer, training loop. You modify this.
   - `test_protocol.py` — adversarial test functions. You may also modify this.
   - `run_dojo.py` — adversarial test runner. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Establish baseline**: Run the full cycle once:
   ```
   uv run train.py > train.log 2>&1
   uv run run_dojo.py > dojo.log 2>&1
   ```
   Record both val_bpb and robustness_gap in `loop_results.tsv`.
6. **Confirm and go**.

## The loop

Each experiment has TWO phases:

### Phase 1: Train (5 min)

Modify `train.py` with your experimental idea, then:

```
uv run train.py > train.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" train.log
```

This trains the model and saves a checkpoint to `~/.cache/autoresearch/checkpoints/`.

### Phase 2: Stress-test (5 min)

Immediately run the adversarial tests against the freshly trained model:

```
uv run run_dojo.py > dojo.log 2>&1
grep "^robustness_gap:\|^worst_test:\|^baseline_bpb:\|^worst_case_bpb:" dojo.log
```

Also read the full test summary to understand WHERE the model is weak:

```
grep "tests_summary:" -A 10 dojo.log
```

### Keep/revert decision

An experiment is a **keep** if:
- val_bpb improved (lower) or stayed within 0.01 of the previous best, AND
- robustness_gap did not increase by more than 0.1 (the model didn't get significantly more fragile)

An experiment is **especially good** if:
- val_bpb improved AND robustness_gap decreased (model got better AND more robust)
- OR robustness_gap decreased significantly (>0.05) even if val_bpb barely changed

An experiment is a **discard** if:
- val_bpb regressed significantly (>0.01 worse)
- OR robustness_gap increased significantly (>0.1 worse) even if val_bpb improved

The key insight: **a model that scores well on val_bpb but fails adversarial tests is not actually a good model.** Use the DOJO results to reject changes that optimize the average at the expense of the worst case.

## What you CAN modify

- `train.py` — model architecture, optimizer, hyperparameters, training loop, batch size, model size. Everything is fair game.
- `test_protocol.py` — adversarial test functions. You can add new tests, improve subgroup definitions, or change perturbation strategies. But only modify this if you have a specific hypothesis about a failure mode to probe — don't change tests just for the sake of it.

## What you CANNOT modify

- `prepare.py` — read-only. Fixed evaluation, data loading, tokenizer.
- `run_dojo.py` — read-only. Adversarial test runner.
- Dependencies — only use what's in `pyproject.toml`.

## Using adversarial results to guide training

After each stress-test, read the per-test breakdown. Use the findings to inform your next training change:

- **High subgroup_disparity on numbers/code?** The model underperforms on structured content. Try: different tokenization handling, adjusted learning rates, or architecture changes that help with numerical patterns.
- **High noise_robustness gap?** The model is fragile to perturbations. Try: data augmentation during training, regularization, or dropout.
- **High adversarial_search gap?** The model has exploitable worst-case inputs. Try: gradient clipping, logit temperature, or architecture changes.
- **High memorization_leakage?** The model is overfitting to training data. Try: more regularization, smaller model, or early stopping.

## Logging results

Log to `loop_results.tsv` (tab-separated). The TSV has these columns:

```
commit	val_bpb	robustness_gap	worst_case_bpb	worst_test	memory_gb	status	description
```

Example:

```
commit	val_bpb	robustness_gap	worst_case_bpb	worst_test	memory_gb	status	description
abc1234	2.014	1.105	3.119	window_boundary	20.7	keep	baseline
def5678	1.982	1.203	3.185	adversarial_search	21.1	keep	lower val_bpb, slightly higher gap
ghi9012	1.950	1.890	3.904	adversarial_search	22.3	discard	val_bpb great but robustness collapsed
```

## The experiment loop

LOOP FOREVER:

1. Look at the current results: last kept val_bpb, robustness_gap, and per-test breakdown
2. Decide what to change in `train.py` (and optionally `test_protocol.py`) based on the adversarial findings
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run Phase 1: `uv run train.py > train.log 2>&1`
5. Check training: `grep "^val_bpb:\|^peak_vram_mb:" train.log`
6. If training crashed: read `tail -n 50 train.log`, fix or skip
7. Run Phase 2: `uv run run_dojo.py > dojo.log 2>&1`
8. Check adversarial: `grep "^robustness_gap:\|^worst_test:" dojo.log`
9. Read full breakdown: `grep "tests_summary:" -A 10 dojo.log`
10. Apply keep/revert decision (see rules above)
11. If keep: `git add loop_results.tsv && git commit --amend --no-edit`
12. If discard: record in tsv, then `git reset --hard <previous kept commit>`

**Total time per experiment: ~12 minutes** (5 min train + ~2 min compile/eval + 5 min adversarial). Expect ~5 experiments/hour, ~40 overnight.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human. The loop runs until manually interrupted. If you run out of training ideas, read the adversarial test breakdown for inspiration — the model's weaknesses are your roadmap.
