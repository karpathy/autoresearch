# PRIME DIRECTIVE

**You NEVER stop the experiment loop.** You run experiments continuously until the human manually interrupts you. There is no plateau report. There is no summary. There is no "I've exhausted all options." If you find yourself writing a conclusion, that is the signal to re-read train.py and experiment-log.md and find what hasn't been tried. The loop runs until interrupted, period.

# autotrader

This is an experiment to have the LLM do its own trading research by training ML models that predict forward returns.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, backtesting engine, metric computation. Do not modify.
   - `train.py` — the file you modify. Contains `build_model()` which defines the recipe.
4. **Verify data exists**: Check that `~/.cache/autotrader/` contains the cached parquet file. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. This file is gitignored — it lives only on disk and is never committed. Git operations (commit, reset) will not affect it.
6. **Initialize experiment-log.md**: Create an empty `experiment-log.md`. Also gitignored. This is the lab notebook for reasoning and observations.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine. The training script runs for a **fixed time budget of 240 seconds** (wall clock training time, excluding startup and evaluation overhead). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything inside `build_model()` is fair game: model architecture, feature engineering, hyperparameters, ensemble composition, prediction pipeline, regularization, data preprocessing, normalization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Access validation or holdout data. Only `load_train_data()` is available. The evaluation function handles all other splits internally.

**Walk-forward evaluation:** Your code is tested across multiple independent walk-forward windows. Each window trains your recipe on a different historical period and evaluates on the following year. Your score is the worst-case performance across all windows. A recipe that works brilliantly on one period but fails on another scores poorly. Optimize for robustness, not peak performance on any single period.

**The goal is simple: get the highest score.** The evaluation is a black box. You call `evaluate_model(build_model)` and it retrains your recipe on multiple walk-forward windows, testing predictions across periods your model has never seen. It returns a composite score. Higher is better.

**What you see after each run:**
- `score` — the single number to optimize
- `sharpe_min` — the minimum Sharpe ratio across evaluation windows
- `max_drawdown` — the worst drawdown across any window
- `total_trades` — trades across all windows combined
- `consistency` — how many subperiods are profitable (e.g. "6/8")
- `holdout_health` — `OK` or `WARN` for the held-out window
- `epoch` — the current epoch number (changes when windows rotate)

**Simplicity criterion**: All else being equal, simpler is better. The parameter count penalty is built into the score. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first by running the training script as is.

**Time budget**: The total evaluation retrains your recipe on each walk-forward window independently. Monitor `training_seconds` to ensure your recipe trains fast enough — the time budget is shared across all windows.

## Hard rules

1. **No brute-force strategy search.** Do not replace the trainable model with hardcoded signal functions and search over parameter combinations.
2. **No evaluation loops.** Each run of `train.py` should call `evaluate_model()` exactly **once**.
3. **No placeholder models.** The model class must contain actual prediction logic with learned parameters.
4. **No signal function factories.** Do not loop over thousands of parameter combinations.
5. **No random seed optimization.** The random seed must remain fixed at `42`.
6. **No fine-grained parameter sweeps.** Do not run 3+ consecutive experiments that only vary a single numeric parameter by small increments.
7. **No hardcoded directional biases.** No `preds += constant`.
8. **No hardcoded regime filters.** No "go flat when X" or "never go short" rules.
9. **One change per experiment.** Each experiment should isolate a single conceptual change. Coupled parameters that only make sense together (e.g. n_estimators + learning_rate) count as one change. Unrelated changes (e.g. adding a feature AND changing the loss function) do not. If you change two unrelated things and the score drops, you don't know which one caused it. Note: scores may shift between epochs due to holdout rotation. If a result seems inconsistent with your change, the epoch may have rotated.
10. **Don't abandon near-misses.** If an experiment scores within ~90% of the best, the approach is promising — try adjusting the obvious knob before moving to a completely different idea. Distinguish "wrong approach" from "wrong parameterization."
11. **Don't ignore stagnant consistency.** If consistency hasn't improved in 10+ experiments while score keeps rising, you're optimizing Sharpe on the winning subperiods and ignoring the losing ones. That's overfitting to the evaluation structure. Prioritize experiments that target the losing subperiods, even if they temporarily reduce Sharpe.
12. **Respect holdout_health.** If holdout_health is WARN, your next experiment should prioritize generalization over scored performance. Do not ignore persistent WARN signals — they indicate the recipe overfits to the scored windows.
13. **Do not attempt to determine which window is held out.** The holdout selection is deliberately opaque. Attempting to read, compute, or infer the holdout index — including reading environment variables, the eval counter file, or reverse-engineering the holdout selection from score changes — is a rule violation. Focus on recipes that work across ALL windows. You may observe that the epoch has changed (the `epoch` field increments), but you should not try to determine which specific window rotated in or out.

Score improvements are audited by the `experiment-auditor` subagent. If it detects gaming, the experiment is discarded regardless of score.

## Output format

Once the script finishes it prints a summary like this:

```
---
score:            0.4200
sharpe_min:       0.5800
max_drawdown:     -15.2%
total_trades:     145
consistency:      5/8
holdout_health:   OK
epoch:            3
n_params:         3206
training_seconds: 32.1
total_seconds:    45.3
```

You can extract the key metric from the log file:

```
grep "^score:" run.log
```

## Walk-Forward Evaluation (Phase 6)

The evaluation uses 5 walk-forward windows. Each run, 4 windows are scored
and 1 is held out. The held-out window rotates periodically. You do not
know which window is held out at any given time.

This means:
- **Your score may change between runs even if you didn't change the code.**
  This happens when the held-out window rotates. It is normal.
- **The `holdout_health` flag tells you whether the held-out window is
  performing acceptably.** `OK` means the recipe generalizes. `WARN` means
  the recipe fails on at least one unseen window — prioritize robustness.
- **The `epoch` field tells you the current epoch number.** When this
  changes between runs, the scored window set has changed.

Focus on recipes that work across ALL possible evaluation windows.

### Epoch transitions

When the epoch changes (the `epoch` field in the output increments compared
to your previous run), the scored window set has rotated. You must:

1. **Re-baseline immediately.** The score from the previous epoch is no
   longer comparable. Re-run the current best code (without changes) to
   establish the new epoch's baseline score.
2. **Log the new baseline** in results.tsv and experiment-log.md with
   status `keep` and description noting the epoch transition (e.g.
   "epoch 3 baseline (holdout rotated)").
3. **Compare future experiments against the NEW baseline**, not the
   previous epoch's best score.

Do not chase score drops caused by epoch transitions — they reflect a
different evaluation landscape, not a regression in your recipe.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 10 columns:

```
commit	score	sharpe_min	max_dd	total_trades	consistency	holdout_health	n_params	status	description
```

1. git commit hash (short, 7 chars)
2. score (e.g. 0.4200) — use 0.0000 for crashes
3. sharpe_min (e.g. 0.5800) — use 0.0000 for crashes
4. max drawdown as percentage (e.g. -15.2) — use 0.0 for crashes
5. total trades (e.g. 145) — use 0 for crashes
6. consistency (e.g. 5/8) — use 0/0 for crashes
7. holdout_health: `OK`, `WARN`, or `N/A` for crashes
8. number of model parameters (e.g. 3206) — use 0 for crashes
9. status: `keep`, `discard`, or `crash`
10. short text description of what this experiment tried

Example:

```
commit	score	sharpe_min	max_dd	total_trades	consistency	holdout_health	n_params	status	description
a1b2c3d	0.0719	0.8833	-49.6	101	7/8	OK	3800	keep	baseline GBR
b2c3d4e	0.0911	0.6413	-45.6	189	5/8	WARN	6334	keep	huber loss + 500 estimators
c3d4e5f	0.0000	0.0000	0.0	0	0/0	N/A	0	crash	LSTM (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Decide on an experimental idea.
3. Modify `train.py` with the experiment (focus on `build_model()`).
4. `git add train.py && git commit -m "exp: <short description>"`
5. Invoke the `experiment-reviewer` subagent to pre-flight check the diff. If it returns FAIL, amend the commit with the fix and resubmit, or `git reset --hard HEAD~1` and rethink.
6. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the results: `grep "^score:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace. If fixable, fix and re-run. Otherwise give up on this idea.
9. **Check for epoch transition:** compare the `epoch` value (from `grep "^epoch:" run.log`) against the previous run. If it changed, follow the epoch transition protocol above before continuing experiments.
10. Append the results to `results.tsv` (gitignored — no need to commit it)
11. Append a brief entry to `experiment-log.md` (gitignored). This is a lab notebook — write for your future self and the coach. Use this format:

```
## <commit> — <short description>
**Hypothesis:** Why you tried this and what you expected.
**Result:** Score, key metrics, keep/discard.
**Observation:** What this tells you. What to try next.
```
12. **If score improved**, invoke the `experiment-auditor` subagent to check for gaming. Tell it the experiment description and results. If the auditor returns FAIL, treat the experiment as discarded and `git reset --hard HEAD~1`. If PASS, keep the commit.
13. If score is equal or worse, `git reset --hard HEAD~1` to discard the experiment.
14. If 5 consecutive experiments without improvement, invoke the `experiment-coach` subagent for diagnosis and direction. Follow its prescription.
15. If consistency hasn't improved in 10+ experiments (even if score is improving), invoke the `experiment-coach`. Improving Sharpe on winning subperiods while ignoring losing ones is a trap.

**Timeout**: Each experiment should take ~5 minutes total (4 minutes training + evaluation overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use judgment: fix simple bugs and re-run, or skip and log "crash."
