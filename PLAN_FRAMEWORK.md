# Plan: AutoAnything as an Installable Framework

## Design Principles

- **Use modern tooling (2026 standards).** `uv` over pip/conda/poetry. `uv_build` over setuptools/hatchling. Lean on conventions and zero-config defaults where possible. When choosing between tools or approaches, pick the one a new project would adopt today, not the legacy-compatible option.

## The Problem

AutoAnything is currently a **repo you fork**. To use it, you clone the whole thing, run `activate.sh` to copy problem files into the repo root, and run evaluator scripts from hardcoded paths. This works for developing the framework itself, but it doesn't scale to the vision: many people, many problems, a population of agents submitting solutions.

The pain points:

1. **Single problem at a time.** `activate.sh` overwrites the repo root — you can't work on two problems simultaneously.
2. **Framework and problem are entangled.** The evaluator code, test problems, and active problem all live in one repo. Users have to understand the framework internals to use it.
3. **No upgrade path.** If the evaluator improves, every user has to manually pull changes into their fork.
4. **Fragile configuration.** `problem.yaml` is parsed with string matching (no YAML library). The metric name defaults to `val_bpb` if parsing fails silently. Branch name `master` and path `proposals/*` are hardcoded constants.
5. **Monolithic dependencies.** GPU-heavy deps (torch, tiktoken) sit alongside instant-scoring problems in one `pyproject.toml`.
6. **No onboarding story.** There's no `init` command, no scaffolding. A new user reads the README, studies the test problems, and assembles their problem by hand.

## The Proposal

Transform AutoAnything from a template repository into an **installable CLI tool**. Problems become self-contained directories (or repos). The framework provides the evaluator runtime, and problems provide the scoring function and mutable state.

```
pip install autoanything       # or: uv tool install autoanything

autoanything init my-problem   # scaffold a new problem
cd my-problem
# ... set up scoring, state, context ...
autoanything score             # run score.sh once (sanity check)
autoanything evaluate          # start polling evaluator
autoanything serve             # start webhook server
```

This is the same evolution that happened with Jekyll (you install it, then create a site), pytest (you install it, then it discovers tests), and Cargo (you install it, then create a project). The common pattern: separate the runtime from the content it operates on.

## What a Problem Looks Like

A problem is a directory with a fixed structure. It can be its own git repo (the typical case — agents clone it and push branches) or a subdirectory of a larger project.

```
my-problem/
├── problem.yaml            # problem definition + framework config
├── agent_instructions.md   # protocol for agents (can be generated)
├── state/                  # mutable files agents edit
│   └── solution.py
├── context/                # read-only background for agents
│   └── background.py
├── scoring/                # GITIGNORED — private scoring code
│   └── score.sh            # outputs JSON on last line
├── leaderboard.md          # auto-updated by the evaluator
└── .autoanything/          # GITIGNORED — local evaluator state
    └── history.db          # SQLite evaluation history
```

This is almost identical to the current structure. The differences:

- `scoring/` replaces `evaluator/score.sh` as the conventional location for private scoring code. The name is clearer — "scoring" is what it does, "evaluator" is the tool that runs it.
- `.autoanything/` holds evaluator state (history.db, logs). Keeps it separate from the scoring code.
- No framework code in the repo. The evaluator, server, and leaderboard logic come from the installed `autoanything` package.
- `problem.yaml` carries all configuration (base branch, timeout, branch pattern) instead of relying on hardcoded constants.

## What the Framework Provides

The `autoanything` package provides:

### CLI Commands

| Command | Description |
|---------|-------------|
| `autoanything init <name>` | Scaffold a new problem directory with templates |
| `autoanything score` | Run `scoring/score.sh` once and print the result (sanity check) |
| `autoanything evaluate` | Start the polling evaluator (watches for proposal branches) |
| `autoanything evaluate --baseline-only` | Establish baseline score and exit |
| `autoanything serve` | Start the webhook server (receives PR events) |
| `autoanything leaderboard` | Regenerate `leaderboard.md` from history |
| `autoanything history` | Print evaluation history from the DB |
| `autoanything validate` | Check that the problem directory is well-formed |

All commands operate on the current directory by default (overridable with `--dir`).

### Package Structure

```
src/autoanything/
├── __init__.py
├── cli.py                  # CLI entry point (click)
├── evaluator.py            # polling evaluation loop
├── server.py               # webhook server (FastAPI)
├── scoring.py              # run score.sh, parse JSON output
├── problem.py              # parse + validate problem.yaml (proper YAML)
├── leaderboard.py          # render leaderboard.md from history
├── history.py              # SQLite history management
├── git.py                  # git operations (fetch, merge, branch listing)
└── templates/              # scaffolding templates for `init`
    ├── problem.yaml
    ├── agent_instructions.md
    ├── score.sh
    ├── solution.py
    └── gitignore            # template for problem repo .gitignore
```

### Dependencies (framework only)

```
pyyaml          # proper YAML parsing
fastapi         # webhook server
uvicorn         # ASGI server
click           # CLI framework
rich            # terminal formatting (tables, colors, progress)
```

No numpy, no torch, no tiktoken. The current `pyproject.toml` bundles several problem-specific deps that belong to individual problems, not the framework:

| Current dep | Used by | Framework needs it? |
|-------------|---------|-------------------|
| `numpy` | rastrigin, tsp, packing scoring | No |
| `tiktoken` | gpt problem | No |
| `kernels` | gpt problem | No |
| `matplotlib` | `plot_progress.py` | No (move to `[project.optional-dependencies]` dev/examples group) |
| `pandas` | `plot_progress.py` | No (same) |
| `pyarrow` | pandas dep | No (same) |
| `requests` | `server.py` GitHub API calls | Yes — keep |
| `fastapi` | `server.py` | Yes — keep |
| `uvicorn` | `server.py` | Yes — keep |

Phase 2 step 5 removes the problem-specific deps and adds an `[project.optional-dependencies]` section for examples/dev.

## Configuration: `problem.yaml`

`problem.yaml` becomes the single source of truth for both problem definition and evaluator behavior. Properly parsed with PyYAML, validated on load.

```yaml
name: my-problem
description: >
  One-paragraph description of what agents are optimizing.

# What agents can and cannot modify
state:
  - state/solution.py

context:
  - context/background.py

# Scoring configuration
score:
  name: cost                    # metric key in score.sh JSON output
  direction: minimize           # "minimize" or "maximize"
  description: "Total cost"     # human-readable, shown on leaderboard
  timeout: 900                  # seconds before scoring is killed (default: 900)
  script: scoring/score.sh      # path to scoring script (default: scoring/score.sh)
  bounded: true                 # whether the score has a known optimum (informational)

# Git configuration
git:
  base_branch: main             # branch that tracks the best state (default: main)
  proposal_pattern: "proposals/*"  # branch pattern for proposals (default: proposals/*)

# Constraints (documentation for agents, not enforced by framework)
constraints:
  - "All values must be finite"
  - "Solution must be a list of exactly 10 floats"
```

All fields under `git:` and `score:` have sensible defaults. A minimal `problem.yaml` is just:

```yaml
name: my-problem
description: Minimize the cost function.
state:
  - state/solution.py
score:
  name: cost
  direction: minimize
```

## What Stays the Same

The architectural choices that make AutoAnything elegant should not change:

- **Git as the protocol.** Anything that can `git push` can be an agent. This is the project's best design decision. No custom APIs, no agent SDKs, no registration. Push a branch, get scored.
- **Serial evaluation.** One proposal scored at a time. No race conditions, no stale comparisons. Simple and correct.
- **Blind scoring.** Agents never see the scoring code. This prevents overfitting to the test harness — agents optimize the actual objective, not artifacts of how it's measured.
- **score.sh → JSON.** The scoring interface is "run a shell script, get JSON on the last line." This is trivially implementable for any problem in any language. No framework-specific scoring API to learn.
- **Leaderboard as markdown.** Human-readable, version-controlled, auto-updated. Agents can parse it, humans can read it on GitHub.
- **SQLite history.** Lightweight, zero-config, portable. One file captures the full evaluation history.

## Blind Scoring: The Trust Model

A core design goal of AutoAnything is that **agents never see the scoring code**. If optimizers can see the evaluation function, they will overfit to it. Blind scoring forces agents to optimize the actual objective, not artifacts of how it's measured.

In the current repo, this guarantee is held together by a `.gitignore` entry. The scoring code lives in `evaluator/` in the same repo agents clone — it's only hidden because git doesn't track it. This works, but it's fragile. A misconfigured `.gitignore`, an accidental commit, or an agent with filesystem access to the evaluation machine breaks the wall.

The new framework design makes this separation **structural rather than conventional**. There are three distinct roles, and each one sees a different slice of the system:

### What agents see (the problem repo)

Agents clone the problem repo. It contains:

```
problem.yaml              # what to optimize, constraints, metric name
agent_instructions.md     # protocol
state/                    # mutable files — this is what they change
context/                  # read-only background
leaderboard.md            # scores and history of what's been tried
```

That's it. No scoring code. Not gitignored scoring code — *no* scoring code, ever, in any commit. The scoring implementation was never part of this repo's history. Agents know what metric they're optimizing (from `problem.yaml`) and what scores others have achieved (from `leaderboard.md`), but they have zero information about how the score is computed. They submit a branch, and a number comes back.

### What the evaluator operator has (local machine)

The person running the evaluator (the "problem author") clones the same repo, then adds the scoring code locally:

```
my-problem/                     # cloned from the problem repo
├── problem.yaml
├── state/
├── context/
├── leaderboard.md
├── scoring/                    # NEVER COMMITTED — exists only on this machine
│   └── score.sh                # the private scoring function
│   └── test_data/              # private test data, ground truth, etc.
└── .autoanything/              # NEVER COMMITTED — evaluator state
    └── history.db
```

The `scoring/` directory is in `.gitignore`. It was never committed. It exists only on the evaluation machine. The operator runs `autoanything evaluate`, which watches for agent branches, checks them out, runs `scoring/score.sh`, and posts results to the leaderboard.

The scoring code can be as complex as needed — it might call external APIs, use private test datasets, run hardware benchmarks, or invoke LLM judges. None of that is visible to agents. They see a score.

### What the framework provides (the autoanything package)

The `autoanything` CLI is the bridge. It knows how to:

- Parse `problem.yaml` to understand the metric and direction.
- Run `scoring/score.sh` and extract the JSON result.
- Manage the git workflow (fetch branches, merge improvements, update leaderboard).
- Record history in SQLite.

The framework is problem-agnostic and scoring-agnostic. It doesn't care what `score.sh` does internally — just that it outputs JSON with the named metric.

### How the wall is maintained

The separation isn't enforced by access controls — it's a consequence of the structure:

1. **The scoring code is never committed.** It's created on the evaluation machine, added to `.gitignore` by default (`autoanything init` sets this up), and never enters the repo's git history. There's no commit to find, no branch to check out, no reflog entry to recover.

2. **The problem repo is self-contained without it.** Agents can clone, read everything, and submit proposals without the scoring code being present. The repo is complete from their perspective.

3. **The evaluator is a local process.** It runs on the operator's machine, not as a service agents can probe. Agents interact with it only through git (push a branch, see a score on the leaderboard or a PR comment).

4. **`autoanything validate` checks the wall.** It warns if any files in `scoring/` are tracked by git, or if `.gitignore` doesn't exclude the scoring directory. This catches mistakes before they leak.

### Why this matters

The key difference from a competition model: AutoAnything is collaborative, not competitive. Each agent's proposal builds on the current best — it's closer to crowdsourced hill climbing than independent submissions. Blind scoring ensures agents optimize the real objective without gaming the evaluation function, while the shared state (each improvement merged into the base branch) means the population of agents collectively ratchets toward better solutions.

## What This Unlocks

### Multi-problem

A user can run evaluators for several problems simultaneously. Each problem is its own directory with its own history, leaderboard, and scoring. No conflicts.

```bash
# Terminal 1
cd problems/prompt-optimization && autoanything evaluate

# Terminal 2
cd problems/model-architecture && autoanything evaluate
```

### Shareable problems

A problem is a git repo. Onboarding is:

```bash
git clone https://github.com/someone/their-problem
cd their-problem
autoanything evaluate --baseline-only
autoanything evaluate
```

The problem author ships the repo with state/, context/, problem.yaml, and a .gitignore that excludes scoring/. They send the scoring code out-of-band (or keep it on the evaluation machine).

### Framework updates

```bash
pip install --upgrade autoanything
```

Evaluator improvements, new CLI commands, bug fixes — all delivered without touching problem repos.

### Problem ecosystem

People can publish problems as repos. The `autoanything` CLI is the common runtime. A community can form around shared problems — each one a self-hosted optimization target that any agent can push to.

## What Happens to the Current Test Problems

The four test problems (rastrigin, tsp, packing, gpt) move from `test_problems/` to `examples/` in the framework repo. They serve as:

1. **Examples** for people creating their own problems.
2. **Integration tests** for the framework.
3. **Demo problems** for trying out AutoAnything.

Each becomes a self-contained problem directory:

```
examples/
├── rastrigin/
│   ├── problem.yaml
│   ├── agent_instructions.md
│   ├── state/solution.py
│   ├── context/problem.py
│   └── scoring/score.sh
├── tsp/
│   ├── ...
├── packing/
│   ├── ...
└── gpt/
    ├── ...
```

The `activate.sh` script is removed. The `run_test.py` simulation script becomes `autoanything test` or stays as a standalone script in the examples.

## Backward Compatibility

The current test problems and evaluate.py/server.py continue to work during the transition. The migration is additive — the new CLI wraps the same logic, just decoupled from hardcoded paths.

The `scoring/score.sh` convention is new, but the framework should also look for `evaluator/score.sh` as a fallback during a transition period. This can be dropped in a future version.

---

## Test Suite (98 tests, all passing)

The test suite covers the full `src/autoanything/` package — 98 tests across 10 files. All tests pass.

| File | Tests | Covers |
|------|-------|--------|
| `test_problem.py` | 16 | YAML parsing, defaults, validation errors, `mutable:` → `state:` backward compat |
| `test_cli.py` | 20 | `init`, `validate`, `score`, `history`, `leaderboard` commands via CliRunner |
| `test_scoring.py` | 13 | `parse_score_output`, `run_score` (subprocess), timeout, `is_better` |
| `test_history.py` | 11 | `init_db`, incumbent CRUD, `record_evaluation`, `is_evaluated` |
| `test_server.py` | 11 | `/health`, `/webhook` routing, signature verification, PR file validation |
| `test_git.py` | 8 | `git()` helper, `get_proposal_branches` with configurable pattern, commit ops |
| `test_leaderboard.py` | 6 | `export_leaderboard` ordering for minimize/maximize, crash display |
| `test_evaluator.py` | 6 | `establish_baseline`, accept/reject/crash logic (mocked git+scoring) |
| `test_integration.py` | 5 | End-to-end init→validate→score, existing problem YAML compat, CLI entry point |

Key conventions the tests lock in:

- `load_problem(path)` returns a `ProblemConfig` with nested `.score` and `.git` objects
- `run_score(script, score_name, timeout, cwd)` — parameterized, no global state
- `git(..., cwd=)` and `get_proposal_branches(cwd=, pattern=)` — cwd and pattern are explicit args
- `create_app(problem_dir=, webhook_secret=)` — server is a factory, not a module-level singleton
- `export_leaderboard(conn, output_path, direction=)` — takes a connection and output path
- `validate_pr_files(modified=, mutable_files=)` — pure function, no `gh` subprocess call
- Backward compat: `mutable:` in YAML treated as alias for `state:`

---

## Implementation Plan

### Phase 1: Proper Configuration + Package Foundation [DONE]

**Goal:** Replace fragile string-matching YAML parsing with a real parser. Make hardcoded values configurable. Establish the installable package structure. This is a prerequisite for everything else and delivers immediate reliability gains.

**Steps:**

1. Add `pyyaml` to dependencies in `pyproject.toml`. Add `uv_build` build backend (uv's native build system — zero-config for standard `src/` layout) and `src/` layout for the installable package.

2. Create `src/autoanything/problem.py` — a module that loads and validates `problem.yaml`:
   - Parse with `yaml.safe_load()`.
   - Return dataclasses (`ProblemConfig`, `ScoreConfig`, `GitConfig`) with all fields, applying defaults:
     - `score.direction` → `"minimize"`
     - `score.timeout` → `900`
     - `score.script` → `"scoring/score.sh"` (fall back to `"evaluator/score.sh"` if it exists)
     - `git.base_branch` → `"main"`
     - `git.proposal_pattern` → `"proposals/*"`
   - Validate required fields: `name`, `state` (non-empty list), `score.name`, `score.direction`.
   - Raise clear `ValidationError` on missing or invalid fields.
   - Backward compatibility: `mutable:` treated as alias for `state:`, `readonly:` treated as alias for `context:`.

3. Update `evaluate.py` to use `autoanything.problem.load_problem` instead of `load_direction()` and `load_score_name()`:
   - Replace `MAIN_BRANCH = "master"` with value from problem config.
   - Replace hardcoded `"proposals/*"` pattern with value from problem config.
   - Replace hardcoded 15-minute timeout with value from problem config.
   - Replace hardcoded `SCORE_SCRIPT` path with value from problem config.
   - Cached config via `_get_config()` for consistent values across functions.

4. Update `server.py` to use the same config loader:
   - Replace its own `problem.yaml` parsing for `mutable:` validation (removed `load_mutable_files()`).
   - Use config for base branch, score direction, and score name throughout.

5. Update test problems' `problem.yaml` files to add `git: base_branch: master` section (since the default is now `main` but existing problems use `master`).

6. Establish `.autoanything/` as the evaluator state directory:
   - Update `evaluate.py` to write `history.db` to `.autoanything/history.db` (creates the directory if it doesn't exist).
   - Fall back to `evaluator/history.db` if it already exists (migration path).
   - Add `.autoanything/` and `scoring/` to `.gitignore`.

7. *(Deviation from plan)* Create the full `src/autoanything/` package with all modules, not just `problem.py`. This was pulled forward from Phase 2 because: (a) the test suite was already scaffolded against the full package API, and (b) creating the package structure is a prerequisite for `problem.py` to be importable. Modules created:
   - `history.py` — SQLite operations, parameterized by `db_path` (no global state)
   - `scoring.py` — `run_score()` and `parse_score_output()` as pure functions with explicit parameters
   - `git.py` — git subprocess wrapper with explicit `cwd` parameter
   - `leaderboard.py` — `export_leaderboard()` taking connection, output path, and direction
   - `evaluator.py` — `establish_baseline()` and `evaluate_proposal()` composing the above modules
   - `server.py` — `create_app()` factory returning a FastAPI instance (not a module-level singleton)
   - `cli.py` — click-based CLI with `init`, `validate`, `score`, `history`, `leaderboard`, `evaluate`, `serve`

8. Clean up `pyproject.toml`:
   - Remove problem-specific deps (`numpy`, `tiktoken`, `kernels`, `matplotlib`, `pandas`, `pyarrow`) from main dependencies.
   - Move them to `[project.optional-dependencies]` `examples` group.
   - Add `dev` group with `pytest`, `click`, `rich`, `httpx`.
   - Remove `[tool.uv.sources]` torch index (kept for backward compat, but torch is no longer a dependency).

**Validation:** 98 tests pass across 10 test files. All four existing problem YAMLs load correctly with the new parser.

#### Implementation Summary

**What was built:**
- Full `src/autoanything/` package (7 modules + `__init__.py`) — installable via `pip install -e .` or `uv sync`
- Proper YAML parsing with PyYAML, replacing fragile string-matching in `load_direction()` and `load_score_name()`
- All hardcoded values (branch name, timeout, proposal pattern, score script path) now come from `problem.yaml` config
- `.autoanything/` state directory with fallback to `evaluator/history.db` for migration
- CLI scaffolding with `init`, `validate`, `score`, `history`, `leaderboard` commands working
- 98/98 tests passing

**Key deviations from the original plan:**
- Created the full package structure (`src/autoanything/`) in Phase 1 instead of Phase 2. The original plan had Phase 1 creating just `autoanything/problem.py` and Phase 2 restructuring to `src/autoanything/`. Since the test suite was already scaffolded against the full package API (`from autoanything.history import ...`, etc.), and the package structure is needed for `problem.py` to be importable, it made sense to do this in one step.
- Implemented all package modules (history, scoring, git, leaderboard, evaluator, server, cli) rather than just problem.py. These are mostly extracted from the existing `evaluate.py` and `server.py` with the key improvement of parameterization (explicit `cwd`, `db_path`, `direction` parameters instead of module-level globals).
- Cleaned up `pyproject.toml` dependencies in Phase 1 rather than Phase 2 step 5. This was necessary because the heavy deps (`numpy`, `tiktoken`, `kernels`) would fail to install on machines without the right build tools.
- The `evaluate` and `serve` CLI commands are stubs that point users to the existing scripts. Full integration deferred to Phase 2.

**What this means for Phase 2:**
- Phase 2's "extract the package" work is largely done. The remaining Phase 2 work is:
  - Wire `autoanything evaluate` and `autoanything serve` CLI commands to the package modules (replacing the evaluator/ scripts)
  - Add `[project.scripts]` entry point to pyproject.toml
  - Rename `test_problems/` to `examples/`
  - Remove `evaluator/evaluate.py` and `evaluator/server.py` (they become the package modules)
  - Make evaluator state location configurable via `--db` flag or env var

### Phase 2: Wire CLI + Cleanup [DONE]

**Goal:** Complete the transition from evaluator scripts to the `autoanything` CLI. Rename directories, remove old scripts.

**Note:** Phase 1 already created the full `src/autoanything/` package with all modules extracted. The `[project.scripts]` entry point, `click`/`rich` in main deps — all done. What remains is wiring the `evaluate` and `serve` CLI commands, renaming directories, and removing the old evaluator scripts.

**Steps:**

1. Fill out `src/autoanything/git.py` with helpers needed by the polling loop:
   - `get_proposal_branches(cwd, pattern)` — changed from local to remote branch listing (`git branch -r`).
   - `get_branch_commit(branch, cwd)` — resolve a remote branch to a commit SHA.
   - `get_head_commit(cwd)` — resolve HEAD to a commit SHA (already existed).
   - `get_commit_message(commit_sha, cwd)` — first line of a commit message (already existed).
   - `merge_proposal(branch, base_branch, cwd)` — merge a proposal into base.
   - The existing `git(*args, cwd)` wrapper stays as the foundation.

2. Wire `autoanything evaluate` CLI command to `src/autoanything/evaluator.py`:
   - Added `run_evaluator()` function implementing the full polling loop: fetch, list proposal branches via `git.py` helpers, filter with `is_evaluated` from `history.py`, call `evaluate_proposal` for each.
   - `establish_baseline` and `evaluate_proposal` refactored to import from `autoanything.git` instead of defining a local `git()` wrapper.
   - CLI supports `--baseline-only`, `--poll-interval`, `--push`, `--db` flags.

3. Wire `autoanything serve` CLI command to `src/autoanything/server.py`:
   - Expanded `create_app()` factory to include the full evaluation pipeline:
     - `gh()` subprocess helper, `pr_comment`, `pr_merge`, `pr_close`, `pr_diff_files` — all with explicit `cwd` parameter
     - `format_results_comment` — markdown PR comment formatting with metrics table
     - `evaluation_worker` thread that drains the queue and calls scoring/merge logic
     - `_evaluate_one_pr` — checkout, validate files, score, comment, merge/close
     - `startup_scan` — `gh pr list` to enqueue unevaluated open PRs
     - Lifespan context manager to start the worker thread on app startup
   - All functions are closures inside `create_app()`, capturing `problem_dir`, `config`, etc. — no module-level globals.
   - CLI supports `--port`, `--host`, `--push`, `--db` flags.

4. Renamed `test_problems/` to `examples/` via `git mv`.

5. `evaluator/evaluate.py` and `evaluator/server.py` are not tracked by git (the `evaluator/` directory is gitignored). They continue to exist locally for backward compatibility but are superseded by the package modules. No git operation needed.

6. Made evaluator state location configurable:
   - Added `_resolve_db_path(problem_dir, db)` helper in `cli.py`.
   - Default: `.autoanything/history.db` in the problem directory.
   - Falls back to `evaluator/history.db` if it exists (migration path).
   - Configurable via `--db` flag on `evaluate`, `serve`, `history`, and `leaderboard` commands.

7. Updated `activate.sh` for the new directory layout:
   - Updated path references from `test_problems/` to `examples/`.
   - Updated help text to reference `autoanything evaluate` instead of `uv run evaluator/evaluate.py`.
   - Clears `.autoanything/` state on activation.

8. *(Additional)* Updated `CLAUDE.md` to reflect new structure and CLI commands.

9. *(Additional)* Updated all references from `test_problems/` to `examples/` in `run_test.py`, `plot_progress.py`, `README.md`, and `test_integration.py`.

**Validation:** 101 tests pass across 10 test files (up from 98 — added tests for `get_branch_commit`, `merge_proposal`, and `TestGetBranchCommit`).

#### Implementation Summary

**What was built:**
- Full polling evaluation loop in `src/autoanything/evaluator.py` via `run_evaluator()` — the package is now self-contained and doesn't need the standalone `evaluator/evaluate.py` script.
- Full webhook server in `src/autoanything/server.py` — `create_app()` now includes the evaluation worker, PR validation, GitHub interaction, startup scan, and leaderboard updates. The package doesn't need the standalone `evaluator/server.py` script.
- Complete `git.py` with `get_branch_commit()` and `merge_proposal()` helpers. Changed `get_proposal_branches()` to list remote branches (matching the real evaluator behavior).
- `--db` flag on `evaluate`, `serve`, `history`, and `leaderboard` commands for configurable database location.
- `test_problems/` renamed to `examples/` — all internal references updated.
- `CLAUDE.md` updated to reflect new structure and CLI commands.
- 101/101 tests passing.

**Key deviations from the original plan:**
- `evaluator/evaluate.py` and `evaluator/server.py` are not removed from disk because they were never tracked by git (`evaluator/` is gitignored). They still exist locally and work as before, but are superseded by the package modules. The plan said "remove them" but there's nothing to remove from git.
- The server's internal functions (`_evaluate_one_pr`, `evaluation_worker`, etc.) are closures inside `create_app()` rather than module-level functions. This eliminates all module-level global state — everything is captured from the factory's parameters.
- `format_results_comment` takes an explicit `score_name` parameter instead of using a cached config object, making it a pure function.
- Added `_resolve_db_path()` helper that encapsulates the db location logic (new default → old fallback → explicit --db) used by multiple CLI commands.

**What this means for downstream phases:**
- Phase 3 (scaffolding polish) is unchanged — `init` and `validate` still work, templates can still be externalized.
- Phase 4 (migration/cleanup) scope is reduced: `test_problems/` rename is done, `evaluator/` scripts don't need git removal (never tracked). Remaining: decide on `activate.sh` future, migrate `run_test.py`/`plot_progress.py`.
- Phase 5 (docs) is partially done: `CLAUDE.md` already updated. Remaining: `README.md` rewrite, `MIGRATING.md`.

### Phase 3: Scaffolding and Init (polish) [DONE]

**Goal:** Polish the `init` and `validate` commands (already functional from Phase 1). Add template files as separate resources.

**Note:** `autoanything init` and `autoanything validate` are already implemented and passing tests. What remains is polish.

**Steps:**

1. Moved template content from inline strings in `cli.py` to `src/autoanything/templates/` package:
   - `problem.yaml` — with `{{name}}`, `{{metric}}`, `{{direction}}` placeholders and explanatory comments for each field.
   - `agent_instructions.md` — expanded protocol (9 steps instead of 5), references problem.yaml, context/, and leaderboard.md.
   - `score.sh` — skeleton with JSON output convention and examples of additional metrics.
   - `solution.py` — minimal example state file with guidance comments.
   - `gitignore` — pre-configured to exclude `scoring/` and `.autoanything/`.
   - `__init__.py` — makes templates a proper Python package for `importlib.resources`.

2. Enhanced `autoanything init <name>`:
   - Templates loaded via `importlib.resources` and rendered with `{{key}}` placeholder substitution.
   - Initializes a git repo (`git init -b main`) in the new directory.
   - Prints next-steps instructions after scaffolding (edit files, validate, score).

3. `autoanything validate` was already complete from Phase 1 — no changes needed:
   - Checks that `problem.yaml` exists and parses correctly.
   - Checks that all files listed in `state:` exist.
   - Checks that `score.sh` (or configured script) exists and is executable.
   - Checks that `.gitignore` excludes scoring directory.
   - Warns if `scoring/` files are tracked by git.
   - Prints clear pass/fail with actionable error messages.

**Validation:** `autoanything init test-problem && cd test-problem && autoanything validate` passes. 104/104 tests passing (up from 101 — added 3 tests for git init, next-steps output, and template rendering).

#### Implementation Summary

**What was built:**
- `src/autoanything/templates/` package with 5 template files + `__init__.py` — templates are now separate, editable resources loaded at runtime via `importlib.resources`.
- `_load_template()` and `_render_template()` helpers in `cli.py` — simple `{{key}}` substitution, no external templating library needed.
- `git init -b main` runs automatically during `autoanything init`, so scaffolded problems are immediately git repositories.
- Next-steps instructions printed after init, guiding users through the edit→validate→score workflow.
- 3 new tests: `test_initializes_git_repo`, `test_prints_next_steps`, `test_templates_render_metric`.

**Key deviations from the original plan:**
- Used `{{key}}` placeholder syntax instead of Python f-strings or a templating library (Jinja2 etc.). This keeps templates as plain, readable files that can be edited without understanding Python string formatting. The `{{` / `}}` delimiters were chosen to avoid conflicts with YAML syntax and shell `${}`.
- `validate` command was already complete and didn't need any changes. The plan listed it as step 3 ("Implement `autoanything validate`") but it was fully implemented in Phase 1 with all the described checks.
- Templates are richer than the original inline strings — `agent_instructions.md` has 9 protocol steps (was 5), `score.sh` includes examples of additional metrics, `problem.yaml` has inline comments explaining each field.

**What this means for downstream phases:**
- Phase 4 (migration/cleanup) is unchanged — `activate.sh`, `run_test.py`, and `plot_progress.py` decisions remain.
- Phase 5 (docs) can reference `src/autoanything/templates/` as the canonical source for problem structure documentation. The `agent_instructions.md` template is now comprehensive enough to serve as the template referenced in Phase 5 step 3.

### Phase 4: Migration and Cleanup [DONE]

**Goal:** Clean up remaining legacy artifacts and migrate standalone scripts. Separate the framework (installable package) from operational problem tooling.

**Note:** Phase 2 already renamed `test_problems/` to `examples/` and updated `CLAUDE.md`. The `evaluator/evaluate.py` and `evaluator/server.py` scripts are not tracked by git (gitignored), so no git removal is needed. They continue to work locally but are superseded by the package.

**Steps:**

1. Moved `generate_chart()` into the package as `src/autoanything/plotting.py`:
   - Added `autoanything plot` CLI command with auto-detection of direction and score label from `problem.yaml`.
   - CLI supports `--db`, `-o`, `--title`, `--direction`, `--score-label` flags.
   - Changed `generate_chart()` to raise `ImportError`/`ValueError` instead of calling `sys.exit(1)` — callers handle user-facing errors.

2. Removed operational scripts from `examples/`:
   - Removed `activate.sh` — no longer needed now that problems are separate repos (see [derby-examples](https://github.com/kousun12/derby-examples)). The old design copied problem files into the framework repo root; the new design has each problem as its own repo.
   - Removed `run_test.py` — simulation harness with problem-specific scoring functions. Belongs with the problem repos, not the framework. Moved to derby-examples.
   - Removed `plot_progress.py` — superseded by `autoanything plot` CLI command. The `generate_chart()` function now lives in the package at `src/autoanything/plotting.py`.

3. Kept example problem directories (`rastrigin/`, `tsp/`, `packing/`, `gpt/`) as read-only reference — they show what a problem directory looks like without being operational. Integration tests still load their `problem.yaml` files.

4. Updated all documentation:
   - `CLAUDE.md` — removed activate.sh references, updated structure to show examples as reference-only, pointed to derby-examples.
   - `README.md` — removed activate.sh workflow and run_test.py section, pointed to derby-examples for runnable problems.
   - `MIGRATING.md` — updated command mapping table.
   - `examples/README.md` — removed activate.sh quick start and run_test.py section, added derby-examples link.

**Validation:** 106 tests pass across 10 test files.

#### Implementation Summary

**What was built:**
- `src/autoanything/plotting.py` — `generate_chart()` function extracted into the package with proper exception handling.
- `autoanything plot` CLI command — generates progress charts from evaluation history.
- 2 new tests: `TestPlot.test_no_history_fails`, `TestPlot.test_plot_help`.
- Clean separation: framework repo has the package + reference examples; operational tooling (activate, run_test, plot_progress) lives in derby-examples.
- 106/106 tests passing.

**Key decisions:**
- Example problems stay in the framework repo as reference (not deleted), but operational scripts that run evaluators or simulate submissions don't belong here — they belong with the problem repos.
- `activate.sh` was removed entirely rather than kept "for convenience." With `--dir` support on all CLI commands and problems in separate repos, there's no need to copy files into the framework repo root.
- `generate_chart()` raises exceptions instead of `sys.exit(1)`, making it usable as both a CLI command and a library function.

### Phase 5: Documentation Update [DONE]

**Goal:** Rewrite remaining documentation to reflect the new installable-framework structure.

**Note:** `CLAUDE.md` was already updated in Phases 2 and 4 to reflect the new structure and CLI commands (including the `plot` command).

**Steps:**

1. Rewrote `README.md`:
   - Added installation section (`pip install autoanything` / `uv tool install autoanything` / editable dev install).
   - New quick start flow: `autoanything init` → edit files → `autoanything validate` → `autoanything score` → `autoanything evaluate`.
   - Added problem structure overview showing the self-contained directory layout.
   - Added full CLI reference table with all 9 commands.
   - Added evaluator modes section (polling vs webhook) with CLI commands.
   - Added "Creating your own problem" section referencing `autoanything init`.
   - Removed all references to `evaluator/evaluate.py`, `evaluator/server.py`, `test_problems/`, and `activate.sh`.
   - Example problems section points to [derby-examples](https://github.com/kousun12/derby-examples) for runnable problems.

2. Added `MIGRATING.md`:
   - Before/after table mapping old commands to new CLI commands.
   - Step-by-step migration: install package, move scoring code, update `problem.yaml`, update `.gitignore`, use CLI, history migration.
   - Documented automatic fallback behaviors (e.g., `evaluator/score.sh` → `scoring/score.sh`, `evaluator/history.db` → `.autoanything/history.db`).
   - "What stays the same" section reassuring users the protocol hasn't changed.

3. `agent_instructions.md` template was already up to date — it references CLI commands and the correct protocol steps. No changes needed.

4. Reviewed all CLI `--help` output — all descriptions are accurate and consistent with the README.

5. *(Additional)* Updated `examples/README.md`:
   - Removed `activate.sh` quick start and `run_test.py` sections.
   - Added link to derby-examples for operational use.
   - Kept problem descriptions, "Creating Your Own Problem" section, and `autoanything plot` docs.

**Validation:** 106/106 tests pass. No mentions of `evaluator/evaluate.py`, `evaluator/server.py`, `activate.sh`, or `test_problems/` remain outside of historical planning docs (`PLAN_FRAMEWORK.md`, `WEB_LISTENER_PLAN.md`) and `MIGRATING.md` (where they appear intentionally in the migration context).

#### Implementation Summary

**What was built:**
- Complete `README.md` rewrite — install → quick start → how it works → problem structure → CLI reference → evaluator modes → examples → creating problems → design philosophy.
- `MIGRATING.md` — step-by-step migration guide with before/after command mapping.
- Updated `examples/README.md` — reference-only docs pointing to derby-examples for operational use.

### Phase 6: Extended Features (future, not blocking)

Ideas that become natural once the framework exists, but aren't needed for launch:

- **Scoring containers**: `autoanything evaluate --docker` to run score.sh in an isolated container. Without this, anyone accepting untrusted agent submissions is running arbitrary code unsandboxed on the evaluation machine.
- **Notifications**: Slack/Discord webhooks when a new best score is found. Tiny to implement (single HTTP POST), but it's the difference between a tool you babysit and one you leave running overnight.
