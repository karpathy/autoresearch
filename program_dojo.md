# DOJO mode — adversarial testing loop

This is the adversarial testing counterpart to autoresearch. Instead of optimizing a model's training, you optimize **adversarial test protocols** to find the model's worst failure modes. The trained model is fixed — you iterate on the tests.

## Setup

To set up a new DOJO run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `dojo-mar8`). The branch `dojo/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b dojo/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — model architecture (read-only in DOJO mode). Understand the model you're testing.
   - `test_protocol.py` — **the file you modify**. Contains adversarial test functions.
   - `run_dojo.py` — the runner. Loads checkpoint, runs tests, reports gap. Do not modify.
4. **Verify checkpoint exists**: Check that `~/.cache/autoresearch/checkpoints/latest.npz` exists. If not, tell the human to train a model first: `uv run train.py`.
5. **Initialize dojo_results.tsv**: Create `dojo_results.tsv` with header row and baseline entry. Run `uv run run_dojo.py` once to establish YOUR baseline.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs adversarial tests against a **fixed trained model**. The test suite runs for a **fixed time budget of 5 minutes**. You launch it as: `uv run run_dojo.py`.

**What you CAN do:**
- Modify `test_protocol.py` — this is the only file you edit. Add, remove, or change adversarial test functions. Change subgroup definitions, noise strategies, perturbation methods, search algorithms.

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, or `run_dojo.py`. They are read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the trained model checkpoint.

**The goal: maximize `robustness_gap`.** The robustness gap is `worst_case_bpb - baseline_bpb`. A bigger gap means you found a more severe failure mode in the model.

**What counts as a good finding:**
- Subgroup disparity: val_bpb is 1.8 overall but 3.5 on sequences with lots of numbers. That's meaningful — it parallels demographic bias in healthcare models.
- Memorization: the model scores 0.5 bpb on training data but 1.8 on validation. That's a real privacy/overfitting concern.
- Architectural weakness: loss is 40% higher beyond the sliding window boundary. That reveals a structural blind spot.
- Noise fragility: 5% token noise causes 200% BPB increase. That's a deployment risk.

**What does NOT count:**
- Feeding random tokens and noting high loss. That's trivially expected and uninteresting.
- Sequences of all-zeros or all-same-token. Also trivial.
- Any test where the "adversarial" input is obviously outside the training distribution in a boring way.

**Simplicity criterion**: Same as autoresearch — simpler is better. A clever 10-line test that finds a real failure mode beats a 100-line test that finds a trivial one.

## Output format

Once the script finishes it prints:

```
---
baseline_bpb:     1.807902
worst_case_bpb:   3.142000
robustness_gap:   1.334098
worst_test:       subgroup_disparity
num_tests:        4
tests_summary:
  memorization_leakage             gap=0.892  (train_bpb=0.915 val_bpb=1.808)
  subgroup_disparity               gap=1.334  (worst=high_digit_density)
  window_boundary_exploit          gap=0.451  (boundary_gap=0.632)
  noise_robustness                 gap=0.203  (worst_rate=20%)
total_seconds:    312.4
peak_vram_mb:     12345.6
```

Read the robustness_gap:

```
grep "^robustness_gap:" run.log
```

## Logging results

When an experiment is done, log it to `dojo_results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	robustness_gap	worst_test	status	description
```

1. git commit hash (short, 7 chars)
2. robustness_gap achieved (e.g. 1.334098) — use 0.000000 for crashes
3. worst_test name (e.g. subgroup_disparity) — use "none" for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `dojo/mar8`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `test_protocol.py` with an idea: new test, better subgroup definition, smarter perturbation strategy, etc.
3. `git add test_protocol.py && git commit -m "dojo: <description>"`
4. Run the experiment: `uv run run_dojo.py > run.log 2>&1`
5. Read out the results: `grep "^robustness_gap:\|^worst_test:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback and fix.
7. Record the results in the tsv
8. If robustness_gap improved (higher), `git add dojo_results.tsv && git commit --amend --no-edit`
9. If robustness_gap is equal or worse, record the discard commit hash, then `git reset --hard <previous kept commit>`

**Think like a red team.** You are trying to find the model's weaknesses. Each iteration should be a hypothesis: "I bet this model is worse at X." Test it. If you're right, keep and build on it. If you're wrong, discard and try a different angle.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The loop runs until manually interrupted. If you run out of ideas: re-read train.py for architectural details to exploit, try combining tests, try more granular subgroup definitions, try different noise distributions, try adversarial search with different strategies.
