# autoresearch - Collaborative Community Edition

This is an experiment to have LLMs do their own research, **collaboratively as a community**.

## The Vision

Rather than a single agent working alone, we are building a **research community** where multiple agents:
- Explore **different research directions** in parallel
- **Share findings** with each other
- **Build on** each other's work
- Collectively advance faster than any single agent could

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `research_hub.py` — community coordination (DO read).
   - `program_community.md` — these instructions.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize the research hub**: Run `python -c "from research_hub import ResearchHub, ResearchThread; h = ResearchHub(); h.register_thread(ResearchThread('autoresearch/<tag>', 'Baseline', 'Initial baseline run', ['baseline']))"`
7. **Check community**: Query the hub for existing research threads to understand what's been tried.
8. **Confirm and go**: Confirm setup looks good.

## Research Hub

Before starting research, you MUST check the community hub:

```python
from research_hub import ResearchHub

hub = ResearchHub()

# Get all active research threads
active = hub.get_active_threads()
for t in active:
    print(f"- {t.branch}: {t.hypothesis} (val_bpb: {t.val_bpb})")

# Find related work
related = hub.find_related_work(['optimizer', 'attention'])
```

This helps you avoid duplicating work others have already tried!

## Collaboration Guidelines

### Before Starting

1. **Query the hub**: Check existing threads to find related work
2. **Choose a direction**: Either:
   - Extend an existing active thread (build on their findings)
   - Start a NEW direction with different tags
3. **Read their paper**: If extending, read the prior agent's findings

### During Research

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Read other agents' work from the hub to get ideas

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.

### After Research (IMPORTANT!)

When your research session is complete (or you're stuck), you MUST contribute back:

1. **Generate a paper**: Use the research_hub to document findings:
   ```python
   from research_swarm import generate_paper
   from research_hub import ResearchHub
   
   hub = ResearchHub()
   thread = hub.get_thread("your-branch")
   # ... collect your results ...
   paper = generate_paper("your-agent-id", thread, your_results)
   ```

2. **Submit to community**: 
   ```python
   from github_integration import contribute_paper
   contribute_paper(f"Research: {thread.branch}", paper)
   ```

3. **Share your branch** (optional but recommended):
   ```python
   from github_integration import share_branch
   share_branch("your-branch", "Tried X, found Y")
   ```

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## Logging Results

When an experiment is done, log it to `results.tsv` AND update the research hub:

```python
hub = ResearchHub()
hub.contribute_findings(
    branch="your-branch",
    val_bpb=0.997900,
    commit="abc1234",
    key_findings=["Found that increasing LR by 2x improves convergence"],
    future_directions=["Try even higher LR", "Experiment with different optimizers"],
    improvement=-0.005,  # negative = improvement (lower val_bpb is better)
    peak_memory_gb=44.0,
    num_params_m=50.3,
    depth=8,
    status="active"  # or "exhausted" if no more ideas
)
```

The TSV format:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. **Check community first**: Look at hub.get_active_threads() and hub.find_related_work()
2. **Tune train.py** with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read out results: `grep "^val_bpb:" run.log`
6. Record in hub: hub.contribute_findings(...)
7. If val_bpb improved, keep the commit. If not, git reset back.
8. Every ~5 experiments, review key findings and update future_directions

**Crashes**: If a run crashes, try to fix once. If it keeps crashing, mark as "crash" in hub and try a different direction.

**NEVER STOP**: Continue working indefinitely. Document your findings as you go so the next agent can build on your work!

## Running Multiple Agents (Swarm Mode)

To run multiple agents in parallel:

```python
from research_swarm import ResearchSwarm

swarm = ResearchSwarm(num_agents=4)

# Each agent gets different research focus
tags = [
    ["optimizer"],      # Agent 0: focus on optimizers
    ["attention"],      # Agent 1: focus on attention
    ["architecture"],  # Agent 2: focus on architecture
    ["hyperparams"]    # Agent 3: focus on hyperparameters
]

results = swarm.run_sync(tags)
print(results)
```

## Tips for Collaboration

1. **Read before writing**: Always check what others have tried
2. **Build on success**: If someone found something good, extend it
3. **Document failures**: What didn't work is just as valuable
4. **Tag your work**: Use clear tags so others can find it
5. **Share early**: Open a PR even if not finished, share findings as discussions
6. **Think community**: You're not competing, you're collaborating

The goal is to emulate a **research community of many PhD students** — not just one!
