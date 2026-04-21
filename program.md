# autoresearch

This is an experiment to have the LLM do its own quantitative trading research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearchQuant/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearchQuant/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `/experiment` — the folder where you will iterate on trading algorithms. The files are all currently empty. 
   - `/docs/alpaca/INDEX.md` - the entrypoint to your documentation on how to use Alpaca, your provided data source
4. **Verify data pulling works**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.md`. This will be where you write your experiments. 
6. **Establish research direction**: Ask your human if they have initial ideas on research hypotheses or classes. Go back and forth until you have a good idea of what is intended.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is a self-contained algorithmic trading strategy that you want to optimize for a high Sharpe Ratio. 

**What you CAN do:**
- Modify `experiment/load.py` - this is the file you edit to pull data from the provided data source Alpaca. Any data you pull should be  written to `experiment/data/`
- Modify `experiment/algo.py` — this is the only trading algorithm you edit. Given a fixed set of data from `experiment/data`, it should make algorithmic trading decisions. 
- Modify `experiment/test.py` - this is the evaluation harness. It should accept your `experiment/algo.py` and `experiment/data/` and test the Sharpe Ratio of the strategy.

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the highest Sharpe Ratio.** Everything is fair game: The only constraint is that the code runs without crashing.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 Sharpe increase that adds 20 lines of hacky code? Probably not worth it. A 0.01 Sharpe increase from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.


## Output format
The `experiment/test.py` can output whatever statistics would be helpful for you to evaluate the performance. Try to use statistical tests when appropriate in addition to Sharpe Ratio to determine the significance of the strategy.

## Logging results

When an experiment is done, write a concise summary to `experiment_log.md`. Each experiemnt you run should have its own entry and explain the motivation, testing approach, and results.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearchQuant/apr21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. In `experiment_log.md`, log your hypothesis for the experiment and what needs to be done to implement it.
2a. If necessary implement/edit `experiment/load.py` 
2b. If necessary implement/edit `experiment/test.py`
3. Write `algo.py` with an experimental idea by directly hacking the code.
4. git commit
5. Run the experiment: `uv run test.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results
7. If the results are empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in `experiment_log.md` (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If the Sharpe improved, you "advance" the branch, keeping the git commit
10. If Sharpe is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
