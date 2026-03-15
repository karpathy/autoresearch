# Plan: Simplify Problem Definition

## Motivation

The current setup to create a new AutoAnything problem requires understanding and correctly wiring up multiple files: `problem.yaml` (with ~10 fields), `scoring/score.sh` (bash wrapper with a JSON-on-last-line convention), explicit `state:` file lists, `.gitignore`, and directory structure. This is too much ceremony for what is fundamentally a two-input system: **what can change** (state files) and **how to measure quality** (a scoring function).

The goal is to make the problem definition as close to zero-configuration as possible by leaning on directory conventions instead of explicit declarations.

## Design Decisions

### 1. `state/` is implicit

Everything in `state/` is mutable. No need to list files in `problem.yaml`. The framework discovers state files by listing the directory. Agents know to edit files in `state/` by convention.

This changes validation: instead of checking changed files against an explicit list, check that all changes have a `state/` path prefix. Simpler and more robust. This is intentionally more permissive than the old exact-set check — agents can create new files in `state/`, not just edit existing ones.

### 2. `scoring/score.py` replaces `scoring/score.sh`

The scoring convention becomes a Python function:

```python
# scoring/score.py
def score():
    """Return a dict with at least the primary metric."""
    from state.solution import x
    from context.problem import rastrigin
    return {"score": rastrigin(x)}
```

The framework runs this in a subprocess (fresh process each run, no import caching issues) and reads the returned dict. The user never thinks about JSON serialization, bash wrappers, or stdout conventions.

The subprocess invocation looks like:
```python
subprocess.run(
    [sys.executable, "-c",
     "import json; from scoring.score import score; print(json.dumps(score()))"],
    cwd=problem_dir, capture_output=True, text=True, timeout=timeout,
)
```

This works because Python 3 namespace packages resolve `scoring/score.py` from cwd without needing `__init__.py`. Same applies to `from state.solution import x` and `from context.problem import ...`.

Non-Python scoring (benchmarks, training runs, API calls) still works — users just call `subprocess.run()` or whatever they need inside their `score()` function.

### 3. `problem.yaml` gets smaller

Before (current):
```yaml
name: rastrigin
description: Minimize the Rastrigin function.
mutable:
  - state/solution.py
readonly:
  - context/problem.py
score:
  name: score
  direction: minimize
  description: "Rastrigin function value"
  script: scoring/score.sh
  timeout: 900
  bounded: true
git:
  base_branch: master
constraints:
  - "Must not modify files outside of state/"
```

After:
```yaml
name: rastrigin
description: Minimize the Rastrigin function.

score:
  direction: minimize
  description: "Rastrigin function value"
  bounded: true
```

Removed:
- `state:` / `mutable:` — implicit from `state/` directory
- `readonly:` / `context:` — implicit from `context/` directory
- `score.script` — implicit convention (`scoring/score.py`)
- `score.name` — defaults to `"score"`, only needed if your metric key differs

Kept (optional, have defaults):
- `score.timeout` — defaults to 900
- `git.base_branch` — defaults to `main`
- `git.proposal_pattern` — defaults to `proposals/*`
- `constraints` — documentation for agents

### 4. `score.name` defaults to `"score"`

Most problems use `"score"` as their metric key. If the `score()` function returns `{"score": 42.5}`, you don't need to specify `score.name` at all. Only set it when your primary metric has a different key (e.g., `val_bpb` for the GPT problem).

## File-by-File Changes

### `src/autoanything/problem.py`

- Remove `state` as a required field in validation. Make it optional — if present in YAML, use it; if absent, populate eagerly by calling `get_state_files(path)` inside `load_problem()`. This way `config.state` is always populated and all downstream code keeps working.
- Add a `get_state_files(problem_dir)` helper that lists `state/` contents (excluding `__pycache__`, `.pyc`, etc.).
- Remove the `score.script` field from `ScoreConfig`. The framework always uses `scoring/score.py`.
- Default `score.name` to `"score"` instead of requiring it.
- Remove `context:` / `readonly:` fields from validation. They were documentation-only anyway.
- Keep `ProblemConfig.state` as a field (populated at load time) and the `mutable` property as an alias.

### `src/autoanything/scoring.py`

- Add a new `run_score_py()` function that invokes `scoring/score.py` via subprocess:
  - Runs `python -c "import json; from scoring.score import score; print(json.dumps(score()))"` in the problem directory
  - Parses the JSON output
  - Returns `(score_value, metrics_dict, duration, error)` — same return shape as current `run_score()`
- Delete the old `run_score()` (no backward compat needed).
- New `run_score(problem_dir, score_name, timeout)` takes `problem_dir` instead of a script path. It runs `scoring/score.py` via `run_score_py()`. Discovery is internal — callers just pass the problem directory.
- `parse_score_output()` stays unchanged — still used to parse JSON from subprocess stdout.

### `src/autoanything/runner.py`

- State file validation: instead of `config.state` set comparison, check that all changed file paths start with `state/`. Remove the `state_files = set(config.state)` / `invalid = all_changes - state_files` logic, replace with a path-prefix check.
- Simplify hide/restore scoring: hardcode `scoring/` as the directory to hide. Remove the `script_path` parameter from `_scoring_dir()`, `_hide_scoring()`, `_restore_scoring()`, and `_recover_scoring()` — they always operate on `scoring/`.
- Update `run_score()` calls to use new signature: `run_score(problem_dir, score_name, timeout)` instead of `run_score(script, score_name, timeout, cwd)`.

### `src/autoanything/evaluator.py`

- Update `run_score()` calls to use new signature: `run_score(problem_dir, score_name, timeout)`. Remove `script = os.path.join(problem_dir, config.score.script)` lines in both `establish_baseline()` and `evaluate_proposal()`.
- The rest of the structure is unchanged — it still calls `run_score()` and handles the result the same way.

### `src/autoanything/server.py`

- **`validate_pr_files()`**: change from exact-file-set matching to path-prefix checking. Instead of `f not in mutable_files`, check `not f.startswith("state/")`. Remove the `mutable_files` parameter; the function just enforces the `state/` convention.
- **`create_app()`**: remove `mutable_files = config.state` and `score_script = os.path.join(problem_dir, config.score.script)`. Use the new `run_score(problem_dir, score_name, timeout)` signature.
- **Fallback block** (lines 170-175): remove the try/except fallback that hardcodes `scoring/score.sh` — the new `run_score()` handles everything internally.
- Update the call to `validate_pr_files()` in `_evaluate_one_pr()` — no longer passes `mutable_files`.

### `src/autoanything/cli.py`

**`init` command:**
- Remove `--metric` flag (default is `"score"`; users can edit `problem.yaml` if different).
- Keep `--direction` flag since it's one of the two fundamental inputs — default to `minimize`.
- Scaffold `scoring/score.py` instead of `scoring/score.sh`. No `chmod` needed (it's not executed directly).
- Updated `problem.yaml` template — smaller, no `state:` list.
- Updated `agent_instructions.md` template — refers to `state/` directory convention.
- Updated `.gitignore` template — same content, still hides `scoring/` and `.autoanything/`.

**`validate` command:**
- Check for `scoring/score.py` existence.
- Remove check for individual state files from `config.state`. Instead verify `state/` directory exists and is non-empty.
- Remove the executable permission check (`os.access(script_path, os.X_OK)`) — not relevant for `.py` files.
- Adjust `.gitignore` check to look for `scoring/` (unchanged).

**`score` command:**
- Use new `run_score(problem_dir, score_name, timeout)` signature. Remove `script_path` construction from `config.score.script`.

### `src/autoanything/templates/`

**`score.sh` → `score.py` (new file, delete `score.sh`):**
```python
"""Scoring function. The framework calls score() and reads the returned dict."""


def score():
    # TODO: implement your scoring logic
    # Import from state/ and context/ as needed:
    #   from state.solution import x
    #   from context.problem import evaluate
    #
    # Return a dict with at least the primary metric key.
    # Default key is "score" — change in problem.yaml if different.
    return {"score": 0.0}
```

**`problem.yaml` (updated):**
```yaml
name: {{name}}
description: >
  Describe what agents are optimizing.

score:
  direction: {{direction}}
  description: "Describe what this metric measures"
  # name: score             # metric key, default "score"
  # timeout: 900            # seconds before scoring is killed
  # bounded: false          # true if the metric has a known optimum

# git:
#   base_branch: main
# constraints:
#   - "Describe rules agents must follow"
```

**`solution.py` (unchanged)**

**`agent_instructions.md` (updated):**
- Replace "Modify only the files listed under `state:` in `problem.yaml`" with "You may create, modify, or delete files in `state/`"
- Keep the rest of the protocol

**`gitignore` (unchanged)**

### Examples: `examples/rastrigin/`

**`problem.yaml`** — remove `mutable:`, `readonly:`, `score.name` (it's the default `"score"`):
```yaml
name: rastrigin
description: >
  Minimize the Rastrigin function in 10 dimensions.

score:
  direction: minimize
  description: "Rastrigin function value — global minimum is 0.0 at the origin"
  bounded: true

git:
  base_branch: master

constraints:
  - "Solution vector must have exactly 10 elements"
```

**`evaluator/score.sh` → `scoring/score.py`:**
```python
def score():
    from context.problem import rastrigin
    from state.solution import x
    return {"score": round(rastrigin(x), 6)}
```

Delete `evaluator/` directory.

### Examples: `examples/tsp/`

**`problem.yaml`** — same treatment (remove `mutable:`, `readonly:`).

**`evaluator/score.sh` → `scoring/score.py`:**
```python
def score():
    from context.cities import tour_distance
    from state.tour import tour
    return {"score": tour_distance(tour)}
```

Delete `evaluator/` directory.

### Examples: `examples/packing/`

**`problem.yaml`** — same treatment.

**`evaluator/score.sh` → `scoring/score.py`:**
```python
def score():
    from context.problem import evaluate_packing
    from state.packing import placements
    return {"score": evaluate_packing(placements)}
```

Delete `evaluator/` directory.

### Examples: `examples/gpt/`

**`problem.yaml`** — remove `mutable:`, `readonly:`. Keep `score.name: val_bpb` since it differs from default:
```yaml
score:
  name: val_bpb
  direction: minimize
  ...
```

**`evaluator/score.sh` → `scoring/score.py`:**
```python
import re
import subprocess

def score():
    result = subprocess.run(
        ["uv", "run", "state/train.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr[-2000:]}")

    output = result.stdout
    def extract(key):
        m = re.search(rf"^{key}:\s+([\d.]+)", output, re.MULTILINE)
        return float(m.group(1)) if m else None

    return {
        "val_bpb": extract("val_bpb"),
        "peak_vram_mb": extract("peak_vram_mb"),
        "training_seconds": extract("training_seconds"),
        "total_seconds": extract("total_seconds"),
        "mfu_percent": extract("mfu_percent"),
        "total_tokens_M": extract("total_tokens_M"),
        "num_steps": extract("num_steps"),
        "num_params_M": extract("num_params_M"),
        "depth": extract("depth"),
    }
```

Delete `evaluator/` directory.

### Tests

**`tests/conftest.py`:**
- Update `minimal_problem_yaml` — remove `state:` and `score.name` fields.
- Update `full_problem_yaml` — remove `state:`, `context:`, `score.script`.
- Update `problem_dir` fixture — create `scoring/score.py` instead of `scoring/score.sh`.

**`tests/test_problem.py`:**
- Remove tests for missing/empty `state:` validation (it's no longer required in YAML).
- Remove `TestMutableField` backward compat tests.
- Remove `TestScoreFallback` tests for `evaluator/score.sh`.
- Remove `test_default_script` — no longer a field.
- Add tests for `get_state_files()` discovery.
- Keep validation tests for `name`, `score.direction`.

**`tests/test_scoring.py`:**
- Add `TestRunScorePy` class testing the new `run_score_py()` function with a `scoring/score.py` file.
- Keep `TestParseScoreOutput` unchanged.
- Keep `TestIsBetter` unchanged.
- Remove old `TestRunScore` (bash) tests — no backward compat needed.

**`tests/test_cli.py`:**
- `TestInit`: update to check for `scoring/score.py` instead of `scoring/score.sh`. Remove `test_custom_metric_and_direction` or update it. Update template rendering tests.
- `TestValidate`: update expectations for new structure. Check for `scoring/score.py`. Remove executable permission test.
- `TestScore`: update fixtures to use `scoring/score.py`.

**`tests/test_integration.py`:**
- `TestInitToScore`: update to write a `scoring/score.py` instead of `scoring/score.sh`.
- `TestExistingProblemStructure`: the examples will have been updated.

**`tests/test_evaluator.py`:**
- Mock config objects no longer need `score.script`. Update `_make_config()`.

**`tests/test_server.py`:**
- Update `validate_pr_files` tests to use path-prefix checking (no `mutable_files` parameter).
- Update mock config objects — no `score.script` field.
- Update any fixtures that construct score script paths.

### Documentation

**`CLAUDE.md`:**
- Update repository structure to show `scoring/score.py` instead of `scoring/score.sh`.
- Update "How Problems Work" section — show simplified `problem.yaml`, `scoring/score.py` convention.
- Update "Agent Protocol" — refer to `state/` directory not specific files. "You may create, modify, or delete files in `state/`."
- Remove `evaluator/` references.

**`CREATE_PROBLEM.md`:**
- Step 2 (`init`): remove `--metric` flag, simplify.
- Step 3 (define problem): show `scoring/score.py` function instead of `score.sh` bash script. Show simplified `problem.yaml`.
- Step 4 (verify): unchanged.
- Update quick reference at bottom.

## Execution Order

1. `src/autoanything/problem.py` — make `state:` optional, add `get_state_files()`, default `score.name`, remove `score.script`
2. `src/autoanything/scoring.py` — add `run_score_py()`, new `run_score()` with `problem_dir` signature
3. `src/autoanything/runner.py` — path-prefix validation, simplified hide/restore, new `run_score()` signature
4. `src/autoanything/evaluator.py` — new `run_score()` signature
5. `src/autoanything/server.py` — path-prefix validation, new `run_score()` signature, remove `mutable_files`
6. `src/autoanything/templates/` — new `score.py`, updated `problem.yaml`, updated `agent_instructions.md`
7. `src/autoanything/cli.py` — update `init`, `validate`, and `score`
8. `examples/` — convert all four examples, delete `evaluator/` directories
9. `tests/` — update all test files including `test_server.py`
10. `CLAUDE.md`, `CREATE_PROBLEM.md` — update docs

## Result

After this change, creating a new problem is:

```bash
pip install autoanything
autoanything init my-problem
cd my-problem
# edit state/solution.py — your starting state
# edit scoring/score.py — your score() function
autoanything run -a "claude -p 'improve'" -n 20
```

The problem directory:
```
my-problem/
├── problem.yaml          # name, description, direction — that's it
├── agent_instructions.md # auto-generated
├── state/                # mutable files (agents can create, modify, delete)
├── context/              # read-only files (optional)
├── scoring/
│   └── score.py          # implement score() → dict
└── .gitignore            # pre-configured
```
