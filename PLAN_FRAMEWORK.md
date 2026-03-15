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
- **Blind scoring.** Agents never see the scoring code. Same reason Kaggle keeps the test set private — it prevents overfitting to the test harness.
- **score.sh → JSON.** The scoring interface is "run a shell script, get JSON on the last line." This is trivially implementable for any problem in any language. No framework-specific scoring API to learn.
- **Leaderboard as markdown.** Human-readable, version-controlled, auto-updated. Agents can parse it, humans can read it on GitHub.
- **SQLite history.** Lightweight, zero-config, portable. One file captures the full evaluation history.

## Blind Scoring: The Trust Model

A core design goal of AutoAnything is that **agents never see the scoring code**. This is the same principle behind Kaggle's private test set: if optimizers can see the evaluation function, they will overfit to it. Blind scoring forces agents to optimize the actual objective, not artifacts of how it's measured.

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

The person running the evaluator (the "problem author" or "competition host") clones the same repo, then adds the scoring code locally:

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

### The Kaggle analogy

This is exactly how Kaggle works:

| Kaggle | AutoAnything |
|--------|-------------|
| Public dataset | `state/`, `context/` |
| Private test set | `scoring/score.sh`, `scoring/test_data/` |
| Submission (upload CSV) | Push a branch or open a PR |
| Kaggle's scoring server | `autoanything evaluate` on the operator's machine |
| Public leaderboard | `leaderboard.md` |
| Competition rules | `problem.yaml`, `agent_instructions.md` |

The difference: Kaggle is a centralized platform. AutoAnything is self-hosted, open, and git-native. Anyone can host a competition. Anyone (or any agent) can compete. The scoring is as private as you make it — all you need is a machine that can run `score.sh` and push leaderboard updates.

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

People can publish problems as repos. The `autoanything` CLI is the common runtime. A community can form around shared problems, like Kaggle competitions but fully open and self-hosted.

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

### Phase 2: Wire CLI + Cleanup (reduced scope — extraction done in Phase 1)

**Goal:** Complete the transition from evaluator scripts to the `autoanything` CLI. Rename directories, add entry points, remove old scripts.

**Note:** Phase 1 already created the full `src/autoanything/` package with all modules extracted. What remains is wiring the `evaluate` and `serve` CLI commands, adding the entry point, renaming directories, and removing the old evaluator scripts.

**Steps:**

1. Wire `autoanything evaluate` CLI command to `src/autoanything/evaluator.py`:
   - Implement the polling loop (currently only `establish_baseline` and `evaluate_proposal` exist — need the main loop with fetch/poll).
   - Support `--baseline-only`, `--poll-interval`, `--push`, `--db` flags.

2. Wire `autoanything serve` CLI command to `src/autoanything/server.py`:
   - The `create_app()` factory exists. Need to add the evaluation worker thread and startup scan to the factory version.
   - Support `--port`, `--host`, `--push` flags.

3. Add `[project.scripts]` entry point to `pyproject.toml`:
   - `autoanything = "autoanything.cli:main"`
   - Move `click` from dev to main dependencies.

4. Rename `test_problems/` to `examples/`.

5. Remove `evaluator/evaluate.py` and `evaluator/server.py` (they now live in the package). Keep `evaluator/` in gitignore for problem repos' scoring code.

6. Make evaluator state location configurable:
   - Default: `.autoanything/history.db` in the problem directory.
   - Configurable via `--db` flag or env var.

7. Update `activate.sh` to work with the new directory layout, or remove it if the CLI replaces its function.

**Validation:** `pip install -e .` in the repo, then `cd examples/rastrigin && autoanything evaluate --baseline-only` works.

### Phase 3: Scaffolding and Init (polish)

**Goal:** Polish the `init` and `validate` commands (already functional from Phase 1). Add template files as separate resources.

**Note:** `autoanything init` and `autoanything validate` are already implemented and passing tests. What remains is polish.

**Steps:**

1. Move template content from inline strings in `cli.py` to `src/autoanything/templates/` files:
   - `problem.yaml` — with placeholder values and comments explaining each field.
   - `agent_instructions.md` — generic protocol, references problem.yaml for specifics.
   - `score.sh` — skeleton that shows the JSON output convention.
   - `solution.py` — minimal example state file.
   - `gitignore` — pre-configured to exclude `scoring/`, `.autoanything/`.

2. Enhance `autoanything init <name>`:
   - Initialize a git repo (`git init`) in the new directory.
   - Print next-steps instructions after scaffolding.

3. Implement `autoanything validate`:
   - Checks that `problem.yaml` exists and parses correctly.
   - Checks that all files listed in `state:` exist.
   - Checks that `score.sh` (or configured script) exists and is executable.
   - Checks that `.gitignore` excludes scoring directory.
   - Prints clear pass/fail with actionable error messages.

**Validation:** `autoanything init test-problem && cd test-problem && autoanything validate` passes.

### Phase 4: Migration and Cleanup

**Goal:** Remove the old structure, update documentation, ensure backward compatibility during transition.

**Steps:**

1. Remove `activate.sh`.

2. Remove `evaluator/evaluate.py` and `evaluator/server.py` from the repo root (they now live in the package). Keep `evaluator/` in the gitignore since problem repos will use `scoring/` instead.

3. Convert `test_problems/run_test.py` into either:
   - `autoanything test` CLI command, or
   - A standalone script in `examples/` that imports from the `autoanything` package.

4. Move `test_problems/plot_progress.py` to `src/autoanything/` or make it a CLI command (`autoanything plot`).

**Validation:** Old evaluator scripts removed, examples directory works standalone, `run_test.py` and `plot_progress.py` migrated.

### Phase 5: Documentation Update

**Goal:** Rewrite all documentation to reflect the new installable-framework structure. After Phase 4 removes the old code, the docs should match the current reality.

**Steps:**

1. Rewrite `README.md`:
   - Installation: `pip install autoanything` / `uv tool install autoanything`
   - Quick start: `autoanything init`, edit files, `autoanything score`, `autoanything evaluate`
   - Problem structure overview (what goes in a problem directory)
   - Link to `examples/` for reference problems
   - Remove all references to `activate.sh`, `evaluator/evaluate.py`, `evaluator/server.py`

2. Rewrite `CLAUDE.md`:
   - Update repository structure to reflect `src/autoanything/`, `examples/`, removal of `evaluator/`
   - Update commands section: replace `uv run evaluator/evaluate.py` with `autoanything evaluate`, etc.
   - Update problem structure to show `scoring/` instead of `evaluator/score.sh`
   - Update agent protocol if any steps changed
   - Remove references to `activate.sh`

3. Add `MIGRATING.md` for anyone who forked the old structure:
   - Move your problem files into their own directory
   - Move `evaluator/score.sh` to `scoring/score.sh`
   - Install `autoanything` and use CLI commands instead of running scripts directly
   - The `evaluator/` directory at repo root is no longer needed

4. Update `agent_instructions.md` template in `src/autoanything/templates/` to reference the CLI commands.

5. Review and update any inline help text in CLI commands (`--help` output) for accuracy.

**Validation:** All docs reference the new structure. No mentions of `activate.sh`, `evaluator/evaluate.py`, or `evaluator/server.py` remain outside of `MIGRATING.md`.

### Phase 6: Extended Features (future, not blocking)

Ideas that become natural once the framework exists, but aren't needed for launch:

- **`autoanything leaderboard --serve`**: Live-updating web leaderboard (the server already has FastAPI, add a simple HTML endpoint).
- **Problem registries**: `autoanything clone <url>` that clones a problem repo and validates its structure.
- **Agent helpers**: A small `autoanything-agent` package that agents can optionally install for utilities (parse leaderboard, read problem config, generate branch names).
- **Multi-evaluator**: `autoanything evaluate --problems ./p1 ./p2 ./p3` to run evaluators for multiple problems in one process.
- **Notifications**: Slack/Discord webhooks when a new best score is found.
- **Scoring containers**: `autoanything evaluate --docker` to run score.sh in an isolated container (security for untrusted agent submissions).
