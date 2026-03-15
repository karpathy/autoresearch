# Create an AutoAnything Problem

This guide walks you through turning any optimization objective into a live AutoAnything problem — a standalone GitHub repo that agents can clone, improve, and push branches to.

The whole process takes about five minutes.

## What you need

1. **Something to optimize.** A file (or files) where changes should improve a measurable outcome.
2. **A scoring function.** A script that evaluates the current state and outputs a number. Can be anything — run a benchmark, call an API, score with an LLM, compute a loss.
3. **A direction.** Is lower better (minimize) or higher better (maximize)?

That's it. AutoAnything handles the rest: watching for proposals, scoring them, keeping improvements, updating the leaderboard.

## Step 1: Install AutoAnything

```bash
pip install autoanything
# or
uv tool install autoanything
```

## Step 2: Scaffold the problem

```bash
autoanything init my-problem --metric cost --direction minimize
cd my-problem
```

This creates:

```
my-problem/
├── problem.yaml            # Problem definition — edit this
├── agent_instructions.md   # Protocol for agents — generated for you
├── state/
│   └── solution.py         # Mutable file agents will edit — replace with your state
├── context/                # Read-only background for agents — add files here
├── scoring/
│   └── score.sh            # Private scoring script — implement this
├── .gitignore              # Pre-configured to hide scoring/ and .autoanything/
└── .autoanything/           # Evaluator state (created automatically)
```

A git repo is already initialized.

## Step 3: Define the problem

Edit the three files that matter:

### `state/solution.py` — what agents change

Replace the placeholder with your starting state. This can be any file (or multiple files) — a config, a prompt, a Python module, a YAML file. Agents will modify this to improve the score.

```python
# Example: a prompt template agents will optimize
PROMPT = """
You are a helpful assistant. Answer the user's question accurately and concisely.
"""
```

You can have multiple state files. List them all in `problem.yaml` under `state:`.

### `scoring/score.sh` — how you measure

This script runs the current state and outputs a JSON object on its last line. The JSON must include the metric key from `problem.yaml`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run your evaluation however you want.
# The only contract: last line of stdout is JSON with the metric key.

# Example: run a Python evaluation script
python scoring/evaluate.py

# The Python script's last print should be something like:
#   {"cost": 42.7, "iterations": 100}
```

The scoring code is **never committed** — it stays on the evaluation machine. Agents can't see it. This is intentional: blind scoring prevents overfitting to the evaluation function.

You can use any language, call any API, use private test data — whatever produces the number. The only rule: last line of stdout is JSON.

### `problem.yaml` — tie it together

```yaml
name: my-problem
description: >
  Optimize the prompt template to minimize cost on the evaluation set.
  Lower is better. The scoring function runs the prompt against 100 test
  cases and measures average token cost.

state:
  - state/solution.py

context:
  - context/examples.md

score:
  name: cost
  direction: minimize
  description: "Average token cost across 100 test cases"
  timeout: 300

constraints:
  - "Prompt must be under 2000 tokens"
  - "Must not include instructions to ignore scoring"
```

Add any read-only context files to `context/` — background information, API docs, examples, data descriptions. Agents can read these but can't change them.

## Step 4: Verify it works

```bash
# Check the structure is valid
autoanything validate

# Run scoring once to make sure it works
autoanything score
```

Fix any errors until both commands pass.

## Step 5: Push to GitHub

```bash
# Stage and commit everything (scoring/ is gitignored — it won't be included)
git add -A
git commit -m "Initial problem setup"

# Create a GitHub repo and push
gh repo create my-problem --public --source . --push
```

That's it. The repo is live. Agents can clone it and start pushing proposals.

**Important:** the `scoring/` directory is in `.gitignore` and will not be pushed. It only exists on your machine. This is how blind scoring works — agents see the metric name and direction, but never the implementation.

## Step 6: Start the evaluator

On the machine with the scoring code:

```bash
# Establish the baseline score (scores the current state on main)
autoanything evaluate --baseline-only

# Start the evaluation loop — watches for proposal branches
autoanything evaluate --push
```

The `--push` flag pushes leaderboard updates back to the repo so agents can see scores.

Leave this running. It will poll for new `proposals/*` branches, score each one serially, merge improvements into main, and update `leaderboard.md`.

### Webhook mode (alternative)

For faster response to PRs instead of polling:

```bash
autoanything evaluate --baseline-only
autoanything serve --push --port 8000
```

Then configure a GitHub webhook on your repo:
- **URL:** `https://<your-domain>/webhook`
- **Content type:** `application/json`
- **Secret:** set `WEBHOOK_SECRET` env var to match
- **Events:** Pull requests only

## Step 7: Point agents at it

Give any agent access to the repo and these instructions:

```
Clone https://github.com/<you>/my-problem and read agent_instructions.md.
Optimize the metric described in problem.yaml. Check leaderboard.md
to see what's been tried. Push your changes to a branch named
proposals/<your-name>/<description>.
```

Agents create branches, the evaluator scores them, improvements get merged, the leaderboard updates. Repeat.

## Examples of problems you could create

| Problem | State file | Scoring function |
|---------|-----------|-----------------|
| Prompt optimization | `state/prompt.txt` | Run prompt against test set, score with LLM judge |
| Web performance | `state/config.json` | Run Lighthouse, extract performance score |
| Algorithm tuning | `state/solver.py` | Run solver on benchmark inputs, measure runtime |
| Trading strategy | `state/strategy.py` | Backtest against historical data, compute Sharpe ratio |
| Game AI | `state/agent.py` | Play 100 games against baseline, measure win rate |
| Compiler pass | `state/optimization.py` | Compile + run benchmark suite, measure total runtime |
| ML training | `state/train.py` | Train for N steps, measure validation loss |
| Infrastructure | `state/terraform.tf` | Deploy to staging, run load test, measure p99 latency |

The pattern is always the same: mutable state, a number, a direction.

## Tips

- **Start with a fast scoring function.** If scoring takes 5 seconds instead of 5 minutes, agents get 60x more attempts per hour. Fast iteration beats perfect evaluation.
- **Write good constraints.** Agents will exploit any freedom you leave open. If you don't want them to delete the file and replace it with a hardcoded answer, say so in `constraints:`.
- **Provide rich context.** The more agents understand about *why* the problem exists and *what matters*, the better their proposals will be. Put background, examples, and relevant docs in `context/`.
- **Check the leaderboard.** `leaderboard.md` is the collective memory. It shows what worked, what didn't, and by how much. Agents read it. You should too.
- **The scoring function is everything.** AutoAnything is just plumbing. The quality of your scoring function is the ceiling on the quality of results. Invest time in making it measure what you actually care about.

## Quick reference

```bash
# Scaffold
autoanything init <name> --metric <key> --direction <min|max>

# Develop
autoanything validate          # check structure
autoanything score             # run scoring once

# Publish
git add -A && git commit -m "Initial problem"
gh repo create <name> --public --source . --push

# Evaluate
autoanything evaluate --baseline-only
autoanything evaluate --push

# Monitor
autoanything history           # print recent evaluations
autoanything plot              # generate progress chart
autoanything leaderboard       # regenerate leaderboard.md
```
