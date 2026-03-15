# Migrating from the Template Repo

If you forked AutoAnything before it became an installable package, this guide covers what changed and how to migrate.

## What changed

AutoAnything was previously a **repo you fork**. You cloned it, ran `activate.sh` to copy problem files into the repo root, and ran evaluator scripts directly (`uv run evaluator/evaluate.py`). Now it's an **installable CLI tool** that operates on self-contained problem directories.

| Before | After |
|--------|-------|
| `uv run evaluator/evaluate.py` | `autoanything evaluate` |
| `uv run evaluator/server.py` | `autoanything serve` |
| `bash test_problems/activate.sh <name>` | Problems are now separate repos |
| `uv run test_problems/run_test.py <name>` | See [derby-examples](https://github.com/kousun12/derby-examples) |
| `python examples/plot_progress.py db` | `autoanything plot` |
| Hardcoded `master` branch | Configurable via `git.base_branch` in `problem.yaml` (default: `main`) |
| Hardcoded `proposals/*` pattern | Configurable via `git.proposal_pattern` in `problem.yaml` |
| Hardcoded 15-minute timeout | Configurable via `score.timeout` in `problem.yaml` (default: 900s) |
| `evaluator/history.db` | `.autoanything/history.db` (auto-migrated) |

## Step by step

### 1. Install the package

```bash
pip install autoanything
# or
uv tool install autoanything
```

Or for development:

```bash
cd autoanything
uv sync
```

### 2. Move scoring code

If your scoring script is at `evaluator/score.sh`, move it to `scoring/score.sh`:

```bash
mkdir -p scoring
mv evaluator/score.sh scoring/score.sh
```

Or set the path explicitly in `problem.yaml`:

```yaml
score:
  script: evaluator/score.sh   # keep the old location
```

The framework checks for `scoring/score.sh` first, then falls back to `evaluator/score.sh`.

### 3. Update problem.yaml

Add a `git:` section if your problem uses `master` instead of `main`:

```yaml
git:
  base_branch: master
```

The `mutable:` key still works as an alias for `state:`, and `readonly:` still works as an alias for `context:`. No need to rename these unless you want to.

### 4. Update .gitignore

Make sure `.gitignore` excludes both the scoring code and evaluator state:

```
scoring/
.autoanything/
evaluator/
```

### 5. Use the CLI

Replace direct script invocations with CLI commands:

```bash
# Before
uv run evaluator/evaluate.py --baseline-only
uv run evaluator/evaluate.py
uv run evaluator/evaluate.py --push

# After
autoanything evaluate --baseline-only
autoanything evaluate
autoanything evaluate --push
```

```bash
# Before
uv run evaluator/server.py --push

# After
autoanything serve --push
```

### 6. History migration

The evaluator now stores history in `.autoanything/history.db` instead of `evaluator/history.db`. If you have an existing `evaluator/history.db`, the CLI will automatically use it until `.autoanything/history.db` is created. No manual migration needed.

### 7. Directory renames (framework repo only)

If you're working on the framework itself (not just using it), note that `test_problems/` was renamed to `examples/`. All internal references have been updated.

## What stays the same

- **Git as the protocol.** Push branches, get scored. No changes to agent workflow.
- **score.sh → JSON.** Same scoring interface — run a shell script, get JSON on the last line.
- **Serial evaluation.** One proposal at a time.
- **Blind scoring.** Agents never see the scoring code.
- **SQLite history.** Same schema, same data.
- **Leaderboard as markdown.** Same format, same auto-update behavior.
