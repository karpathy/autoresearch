# autoresearch — prompt evolution for YC-bench

This is an experiment to have the LLM evolve its own system prompt to maximize performance on YC-bench, a business simulation benchmark.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prompt.txt` — the file you modify. This is the system prompt sent to the YC-bench agent.
   - `run_bench.py` — the evaluation harness. Do not modify.
   - `yc-bench-main/src/yc_bench/agent/prompt.py` — reference: the default prompt and turn context builders. Read this to understand what the agent sees.
4. **Set the API key**: `export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the YC-bench business simulation (~10 minutes). You launch it as: `uv run run_bench.py`

**What you CAN do:**
- Modify `prompt.txt` — this is the only file you edit. It contains the system prompt that guides the YC-bench agent's decisions about task selection, employee allocation, client management, etc.

**What you CANNOT do:**
- Modify `run_bench.py`, or anything in `yc-bench-main/`. The benchmark and evaluation harness are read-only.
- Install new packages or add dependencies.

**The goal is simple: maximize `final_funds`.** The agent starts with $200K and must run a consultancy — accepting tasks, assigning employees, managing payroll, building prestige and client trust. Bankruptcy (funds < 0) is the worst outcome. Surviving the full horizon with maximum funds is the best.

**Current baseline**: $-19,259.56 (bankruptcy). Anything positive is already an improvement.

## Key mechanics the prompt should address

Understanding these mechanics will help you write better prompts:

- **Salary bumps**: Each completed task raises salary by ~1% for ALL assigned employees. Assigning all 8 employees grows payroll 2.7x faster than assigning 3.
- **Throughput split**: Employees on N active tasks contribute at rate/N. Two tasks = 50% speed each.
- **Brooks's Law**: Only the first 4 employees on a task contribute. Beyond that is pure overhead (salary bump with no speed gain).
- **Deadlines**: Miss = prestige penalty + no reward + 35% clawback. Leave buffer for scope creep.
- **Trust**: Focus on fewer clients → builds trust → reduces work requirements → higher margins.
- **RAT clients** (~35%): Unreliable — scope creep, payment disputes. Detect via `client history`.
- **Prestige**: Grows per domain. Must climb broadly to unlock better-paying tasks.
- **Scratchpad**: Only 20 turns of conversation history. The scratchpad persists across truncation — agent should write reusable rules there, not observations.
- **Payroll timing**: Deducted monthly. Must have funds > 0 after each deduction.

## Output format

Once `run_bench.py` finishes it prints a summary:

```
---
final_funds:   $12,345.67
funds_cents:   1234567
tasks_done:    8
tasks_failed:  2
turns:         45
api_cost:      $6.52
elapsed:       561
outcome:       horizon_end
```

Extract the key metric: `grep "^final_funds:" run.log` or `grep "^funds_cents:" run.log`

## Logging results

Log each experiment to `results.tsv` (tab-separated).

Header and columns:

```
commit	final_funds	tasks_done	tasks_failed	outcome	description
```

1. git commit hash (short, 7 chars)
2. final_funds in dollars (e.g. -19259.56 or 12345.67)
3. tasks completed successfully
4. tasks failed
5. outcome: `horizon_end`, `bankruptcy`, `crash`, or `timeout`
6. short text description of what this prompt change tried

Example:

```
commit	final_funds	tasks_done	tasks_failed	outcome	description
a1b2c3d	-19259.56	1	15	bankruptcy	baseline (default prompt)
b2c3d4e	5230.00	6	3	horizon_end	added employee allocation rules (max 3-4 per task)
c3d4e5f	-8000.00	3	8	bankruptcy	aggressive prestige climbing (too many tasks)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state: the current branch/commit
2. Edit `prompt.txt` with a new idea for improving the agent's strategy
3. git commit
4. Run the experiment: `uv run run_bench.py > run.log 2>&1`
5. Read out the results: `grep "^final_funds:\|^tasks_done:\|^outcome:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.
7. Record the results in results.tsv (do not commit results.tsv)
8. If final_funds improved (higher), keep the git commit
9. If final_funds is equal or worse, git reset back to where you started

## Mutation strategies to try

Here are categories of prompt changes to explore, roughly ordered by expected impact:

1. **Employee allocation rules**: "Assign 3-4 employees per task, never all 8" — this is probably the single biggest lever (controls payroll growth)
2. **Explicit decision framework**: Step-by-step reasoning template for each turn (check funds, check runway, evaluate tasks, decide)
3. **RAT client detection**: "After any task failure, check `client history`. Avoid clients with >30% failure rate"
4. **Scratchpad templates**: Tell the agent exactly what to write to scratchpad on turn 1 (client ratings, employee skills, strategy rules)
5. **Prestige strategy**: "Climb prestige broadly across all domains, not deep in one"
6. **Deadline buffers**: "Only accept tasks where estimated completion is <70% of deadline"
7. **Revenue thresholds**: "Only accept tasks paying > monthly payroll / 4"
8. **Trust specialization**: "Focus on 2-3 reliable clients to build trust and reduce work"
9. **Concurrency management**: "Run 2-3 tasks concurrently, no more (throughput split penalty)"
10. **Bankruptcy prevention**: "If runway < 3 months, stop accepting new tasks and focus on completing current ones"

Try individual changes first to isolate what works. Then combine winners.

**The first run**: Your very first run should establish the baseline by running with the current prompt.txt as-is.

**Timeout**: Each experiment should take ~10 minutes. If a run exceeds 15 minutes, it will be killed automatically.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, re-read the mechanics above, look at your results.tsv for patterns, try combining near-misses, or try more radical prompt restructuring.
