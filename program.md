# autotrader

This is an experiment to have the LLM do its own trading research by training ML models that predict forward returns.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, backtesting engine, metric computation. Do not modify.
   - `train.py` — the file you modify. Model architecture, feature engineering, training loop.
4. **Verify data exists**: Check that `~/.cache/autotrader/` contains the cached parquet file. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. This file is gitignored — it lives only on disk and is never committed. Git operations (commit, reset) will not affect it.
6. **Initialize experiment-log.md**: Create an empty `experiment-log.md`. Also gitignored. This is the lab notebook for reasoning and observations.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine. The training script runs for a **fixed time budget of 4 minutes** (wall clock training time, excluding startup and evaluation overhead). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, feature engineering, loss function, optimizer, hyperparameters, training loop, regularization, data preprocessing, normalization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Access validation or holdout data. Only `load_train_data()` is available. The evaluation function handles all other splits internally.

**The goal is simple: get the highest score.** The evaluation is a black box. You call `evaluate_model(predict_on_data, n_params)` and it tests your predictions across multiple time periods — including periods your model has never seen. It returns a composite score. Higher is better.

**What you see after each run:**
- `score` — the single number to optimize
- `sharpe_min` — the minimum Sharpe ratio across evaluation periods
- `max_drawdown` — the worst drawdown across any period
- `total_trades` — trades across all periods combined
- `consistency` — how many subperiods are profitable (e.g. "5/7")

**Simplicity criterion**: All else being equal, simpler is better. The parameter count penalty is built into the score. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first by running the training script as is.

## Hard rules

1. **No brute-force strategy search.** Do not replace the trainable model with hardcoded signal functions and search over parameter combinations.
2. **No evaluation loops.** Each run of `train.py` should call `evaluate_model()` exactly **once**.
3. **No placeholder models.** The model class must contain actual prediction logic with learned parameters.
4. **No signal function factories.** Do not loop over thousands of parameter combinations.
5. **No random seed optimization.** The random seed must remain fixed at `42`.
6. **No fine-grained parameter sweeps.** Do not run 3+ consecutive experiments that only vary a single numeric parameter by small increments.
7. **No hardcoded directional biases.** No `preds += constant`.
8. **No hardcoded regime filters.** No "go flat when X" or "never go short" rules.
9. **One change per experiment.** Each experiment should isolate a single conceptual change. Coupled parameters that only make sense together (e.g. n_estimators + learning_rate) count as one change. Unrelated changes (e.g. adding a feature AND changing the loss function) do not. If you change two unrelated things and the score drops, you don't know which one caused it.
10. **Don't abandon near-misses.** If an experiment scores within ~90% of the best, the approach is promising — try adjusting the obvious knob before moving to a completely different idea. Distinguish "wrong approach" from "wrong parameterization."
11. **Don't ignore stagnant consistency.** If consistency hasn't improved in 10+ experiments while score keeps rising, you're optimizing Sharpe on the winning subperiods and ignoring the losing ones. That's overfitting to the evaluation structure. Prioritize experiments that target the losing subperiods, even if they temporarily reduce Sharpe.

Score improvements are audited by the `experiment-auditor` subagent. If it detects gaming, the experiment is discarded regardless of score.

## Output format

Once the script finishes it prints a summary like this:

```
---
score:            0.4200
sharpe_min:       0.5800
max_drawdown:     -15.2%
total_trades:     145
consistency:      5/7
n_params:         3206
training_seconds: 32.1
total_seconds:    45.3
```

You can extract the key metric from the log file:

```
grep "^score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 9 columns:

```
commit	score	sharpe_min	max_dd	total_trades	consistency	n_params	status	description
```

1. git commit hash (short, 7 chars)
2. score (e.g. 0.4200) — use 0.0000 for crashes
3. sharpe_min (e.g. 0.5800) — use 0.0000 for crashes
4. max drawdown as percentage (e.g. -15.2) — use 0.0 for crashes
5. total trades (e.g. 145) — use 0 for crashes
6. consistency (e.g. 5/7) — use 0/0 for crashes
7. number of model parameters (e.g. 3206) — use 0 for crashes
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	score	sharpe_min	max_dd	total_trades	consistency	n_params	status	description
a1b2c3d	0.0719	0.8833	-49.6	101	7/7	3800	keep	baseline GBR
b2c3d4e	0.0911	0.6413	-45.6	189	5/7	6334	keep	huber loss + 500 estimators
c3d4e5f	0.0000	0.0000	0.0	0	0/0	0	crash	LSTM (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Decide on an experimental idea.
3. Modify `train.py` with the experiment.
4. `git add train.py && git commit -m "exp: <short description>"`
5. Invoke the `experiment-reviewer` subagent to pre-flight check the diff. If it returns FAIL, amend the commit with the fix and resubmit, or `git reset --hard HEAD~1` and rethink.
6. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the results: `grep "^score:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace. If fixable, fix and re-run. Otherwise give up on this idea.
9. Append the results to `results.tsv` (gitignored — no need to commit it)
10. Append a brief entry to `experiment-log.md` (gitignored). This is a lab notebook — write for your future self and the coach. Use this format:

```
## <commit> — <short description>
**Hypothesis:** Why you tried this and what you expected.
**Result:** Score, key metrics, keep/discard.
**Observation:** What this tells you. What to try next.
```
11. **If score improved**, invoke the `experiment-auditor` subagent to check for gaming. Tell it the experiment description and results. If the auditor returns FAIL, treat the experiment as discarded and `git reset --hard HEAD~1`. If PASS, keep the commit.
12. If score is equal or worse, `git reset --hard HEAD~1` to discard the experiment.
13. If 5 consecutive experiments without improvement, invoke the `experiment-coach` subagent for diagnosis and direction. Follow its prescription.
14. If consistency hasn't improved in 10+ experiments (even if score is improving), invoke the `experiment-coach`. Improving Sharpe on winning subperiods while ignoring losing ones is a trap.

**Timeout**: Each experiment should take ~5 minutes total (4 minutes training + evaluation overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use judgment: fix simple bugs and re-run, or skip and log "crash."

**Autonomy**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Run experiments continuously. If you run out of ideas, think harder — re-read the in-scope files for new angles, revisit near-miss experiments and try them in isolation, try more radical architectural changes.

**Plateau detection**: If the coach has been invoked twice without producing an improvement (i.e. ~15+ consecutive non-improving experiments), stop the loop. Write a summary to `plateau-report.md` containing:
- Current best score and configuration
- What the last 15+ experiments tried and why they failed
- Which near-misses came closest and what they suggest
- Your assessment of what structural change is needed to break through (e.g. new data, evaluation change, architectural shift beyond what's available)

Then stop and wait for the human. Plateaus usually mean the agent has exhausted what's possible within the current system constraints — the next breakthrough requires a human decision (e.g. changing the evaluation system, the backtester, or the scoring formula).