# autoresearch

This is an experiment to have the LLM do its own quantitative trading research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearchQuant/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearchQuant/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `data.md` - documentation on how to pull the data that you have available for analysis.
   - `test.py` — the generalizable test script that will evaluate your algorithm. It runs on the train window (2017-2022), which is also the only data `algo.py` has access to. A held-out window exists for final validation, but you will never see or evaluate against it — do not try.
   - `algo.py` — the (currently minimal) strategy file you will iterate on. Its `strategy(data)` function is the contract `test.py` calls; study its docstring and the placeholder implementation before writing your first experiment.
4. **Test the placeholder strategy**: Before touching `algo.py`, verify the full pipeline runs end-to-end by evaluating the shipped placeholder. Run `uv run test.py > run.log 2>&1` and then `tail -n 30 run.log`. You should see a `TEST RESULTS` block with a Sharpe value (roughly in the ballpark of buy-and-hold SPY on 2017-2022). 
5. Create `experiment_log.md`, where you will log your work.
5. **Establish research direction**: Ask your human if they have initial ideas on research hypotheses. Go back and forth until you have a good idea of what is intended.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is a self-contained algorithmic trading strategy that you want to optimize for a high Sharpe Ratio. 

**What you CAN do:**
- Modify `algo.py` -  this is the only existing file you may edit. Feel free to pull any data from the provided data sources to hydrate your algorithm. 
- Write a throwaway `scratch.py` script to explore your data as you see necessary.

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Edit `test.py` or `load.py`. These are fixed so that back-testing is robust. 

**The goal is simple: get the highest Sharpe Ratio.** Everything is fair game: The only constraint is that the code runs without crashing. The Sharpe Ratio and other metrics will be returned by `test.py`. Other diagnostics are also logged by `test.py` for you to reference.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 Sharpe increase that adds 20 lines of hacky code? Probably not worth it. A 0.01 Sharpe increase from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.


## Output format

`uv run test.py > run.log 2>&1` writes a report to `run.log` and appends one tab-separated row to `results.tsv` (one line per run). The tail of `run.log` looks like:

```
============================================================
====================== TEST RESULTS ========================
============================================================
Split        : train  (2017-01-01 .. 2022-12-31)
Timeframe    : day
Periods      : 1509  (2017-01-04 ...  -->  2022-12-30 ...)
Symbols      : 1
Cost model   : 2.0 bps one-way per unit |dW|

RETURNS
  Sharpe              : 0.7321    <-- PRIMARY METRIC
  Total return        : 98.45%
  Ann. return         : 12.34%
  Ann. volatility     : 16.87%
  t-stat of mean      : 2.58
  Hit rate            : 53.12%

RISK
  Max drawdown        : -33.92%
  Avg gross exposure  : 1.00x
  Max gross exposure  : 1.00x
  Avg net exposure    : 1.00x

ACTIVITY
  Ann. turnover (1w)  : 0.00x
  Cost drag           : 0.001% / yr

Wall time    : 2.1s
============================================================
Appended row to results.tsv
```
(illustrative)

If the `TEST RESULTS` block is missing from `run.log`, the run crashed before producing a report. See the experiment loop below for how to recover.

## Logging results

When an experiment is done, write a concise summary to `experiment_log.md`. Each experiment you run should have its own entry and explain the motivation, testing approach, and results.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearchQuant/apr21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Write `algo.py` with an experimental idea by directly hacking the code. You may print additional diagnostics as you like in this file, which will be captured in the log.
3. git commit
4. Run the experiment: `uv run test.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read the results: `tail -n 30 run.log`. That captures the full `TEST RESULTS` block plus the `Appended row to results.tsv` confirmation. The line you're comparing against the previous run is `  Sharpe              : X.XXXX    <-- PRIMARY METRIC`.
6. If the `TEST RESULTS` block is missing from `tail -n 30 run.log`, the run crashed before producing metrics. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Write a concise summary of the motivation behind your test and results to `experiment_log.md`.
8. If the Sharpe improved, you "advance" the branch, keeping the git commit
9. If Sharpe is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes max (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
