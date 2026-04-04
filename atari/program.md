# atari autoresearch

Autonomous AI research on Atari game-playing agents.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr4`). The branch `atari-research/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b atari-research/<tag>` from current master.
3. **Read the in-scope files**: The relevant files are:
   - `atari/prepare.py` — fixed evaluation harness, environment factory, constants. Do not modify.
   - `atari/agent.py` — the file you modify. Agent strategy, training, heuristics.
   - `atari/program.md` — these instructions.
4. **Verify environment works**: Run `python atari/prepare.py` to confirm Gymnasium ALE is working.
5. **Initialize results.tsv**: Create `atari/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment modifies `atari/agent.py` and runs the evaluation. You launch it as:

```bash
cd atari && python agent.py > run.log 2>&1
```

**What you CAN do:**
- Modify `atari/agent.py` — this is the only file you edit. Everything is fair game: agent strategy, observation preprocessing, internal state tracking, heuristics, learning algorithms, evolutionary search, etc.

**What you CANNOT do:**
- Modify `atari/prepare.py`. It is read-only. It contains the fixed evaluation, environment factory, and constants.
- Change the game (it's fixed to Breakout-v5).
- Install new packages or add dependencies beyond what's available (gymnasium, ale-py, numpy).

**The goal is simple: get the highest mean_reward.** The evaluation runs 30 episodes with fixed seeds. Everything is fair game: change the heuristics, add ball trajectory prediction, implement learning during the training phase, use evolutionary strategies, etc.

**The first run**: Your very first run should always be to establish the baseline, so run `agent.py` as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_reward:      4.5000
std_reward:       2.1000
min_reward:       1.0000
max_reward:       9.0000
mean_steps:       280.3
total_steps:      8410
episodes:         30
training_seconds: 0.0
eval_seconds:     12.3
total_seconds:    12.3
```

You can extract the key metric from the log file:

```
grep "^mean_reward:" run.log
```

## Logging results

When an experiment is done, log it to `atari/results.tsv` (tab-separated).

The TSV has a header row and 4 columns:

```
commit	mean_reward	status	description
```

1. git commit hash (short, 7 chars)
2. mean_reward achieved (e.g. 4.5000) — use 0.0000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	mean_reward	status	description
a1b2c3d	4.5000	keep	baseline heuristic
b2c3d4e	6.2000	keep	add ball trajectory prediction
c3d4e5f	3.1000	discard	aggressive paddle movement overshoots
d4e5f6g	0.0000	crash	import error in new module
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `atari-research/apr4`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `atari/agent.py` with an experimental idea.
3. git commit
4. Run the experiment: `cd atari && python agent.py > run.log 2>&1`
5. Read out the results: `grep "^mean_reward:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit the results.tsv file)
8. If mean_reward improved (higher), you "advance" the branch, keeping the git commit
9. If mean_reward is equal or worse, you git reset back to where you started

**Ideas to explore** (non-exhaustive):
- Ball trajectory prediction (extrapolate ball direction from consecutive frames)
- Frame differencing (detect ball movement direction)
- Anticipatory positioning (move to where the ball will be, not where it is)
- Adaptive fire timing (fire at optimal moments)
- Zone-based strategy (different behavior depending on ball position)
- Learning during train(): evolve heuristic parameters over trial episodes
- Simple policy search: try random parameter variations, keep the best

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you.
