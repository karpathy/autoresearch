# Plan: Branch-Based Scoring

## Context

Scoring code (`scoring/score.py`) is the black-box metric that drives optimization. Agents must never see it. Today this is enforced two ways:

1. **Gitignore**: `scoring/` is in `.gitignore`, so it's never committed to working branches.
2. **Move-once hiding** (local loop only): During `autoanything run`, `runner.py` moves `scoring/` to `.autoanything/_scoring/` once before entering the loop, and restores it when the loop ends. Scoring is loaded from the hidden location via sys.path injection.

Scoring is completely unversioned — it's an untracked file on disk with no history. It can't be diffed, distributed, or experimented with.

### What's already been done

The per-iteration hide/restore dance has been eliminated. `run_score()` now accepts a `scoring_dir` parameter and uses sys.path injection to load scoring from any location on disk (see `scoring.py:41-76`). The runner moves scoring once at the start of the loop and restores it in a `finally` block (see `runner.py:96-104, 254-259`). This reduced filesystem mutations from 2N (two per iteration) to 2 (one move, one restore), and the interrupt window is minimal.

This plan covers the next step: moving scoring to a dedicated git branch for versioning, distribution, and experimentation.

## Motivation

1. **Version-controlled scoring.** Currently if the evaluator's disk dies, scoring is gone. With a branch, scoring has full git history, diffs, and is distributed with `git clone`.

2. **Scoring experimentation.** Different scoring branches enable A/B testing of scoring functions. `--scoring-branch scoring/v2` on the CLI switches the metric entirely — no code changes, no file swaps. Two evaluators can run simultaneously with different scoring functions against the same problem.

3. **Multi-evaluator consistency.** Multiple machines running evaluation currently need `scoring/score.py` distributed out of band. With branch-based scoring, `git fetch` is all you need.

4. **Cleaner agent isolation.** Agents can't accidentally see scoring because it doesn't exist on any working branch. No filesystem tricks needed.

## Design

### Core concept

Scoring code lives on a dedicated orphan branch (default: `scoring`). Working branches (main, proposals/*) never have `scoring/` committed. The `.gitignore` still excludes `scoring/` as a safety net.

### Runtime mechanism

Scoring is extracted from the branch once (at evaluator startup or loop start) into `.autoanything/_scoring/` and loaded from there via the existing `scoring_dir` parameter on `run_score()`. No per-iteration git operations or filesystem mutations.

```python
# One-time extraction (at startup)
git("checkout", scoring_branch, "--", "scoring/", cwd=problem_dir)
shutil.move(os.path.join(problem_dir, "scoring"),
            os.path.join(problem_dir, ".autoanything", "_scoring"))
git("reset", "HEAD", "--", "scoring/", cwd=problem_dir, check=False)

# Per-iteration scoring (already works — no changes needed)
run_score(problem_dir, score_name=score_name, timeout=timeout,
          scoring_dir=os.path.join(problem_dir, ".autoanything", "_scoring"))
```

The `run_score` sys.path injection (already implemented) handles the import resolution. No `scoring_context` context manager is needed — the extraction happens once, the hidden directory persists for the session.

### Config changes

`problem.yaml` gains an optional field:

```yaml
score:
  name: score
  direction: minimize
  branch: scoring        # NEW — default "scoring"
```

CLI commands gain `--scoring-branch` to override:

```bash
autoanything run -a "./agent.sh" --scoring-branch scoring/v2
autoanything evaluate --scoring-branch scoring/experimental
autoanything serve --scoring-branch scoring/strict
autoanything score --scoring-branch scoring    # default
```

### `autoanything init` changes

Init creates the scoring branch as an orphan branch with the template `score.py`:

```bash
git checkout --orphan scoring
git rm -rf .              # clean slate
# write scoring/score.py
git add scoring/
git commit -m "Initial scoring function"
git checkout main
```

The result: `main` has no `scoring/` directory; `scoring` branch has only `scoring/score.py`. The `.gitignore` on `main` still lists `scoring/` as a safety net.

### Backwards compatibility

Support both modes with a simple priority:

1. If `--scoring-branch` is passed, use it (extract from branch).
2. If `score.branch` is set in `problem.yaml`, use it (extract from branch).
3. If neither is set and a local `scoring/` directory exists on disk, use it directly (legacy mode — no branch extraction needed, move-once still applies in the local loop).
4. If neither is set and no local `scoring/` exists, try the `scoring` branch by convention.

This means existing problems with `scoring/` on disk keep working with zero changes. The migration path is: move your `scoring/` onto a branch when you're ready.

---

## Implementation Plan

### Phase 1: Config + branch extraction utility

**Files:**
- `src/autoanything/problem.py` — add optional `branch` field to `ScoreConfig` (default: `None`)
- `src/autoanything/scoring.py` — add `extract_scoring_from_branch(problem_dir, branch)` that checks out scoring from the branch into `.autoanything/_scoring/` and returns the path. Add `resolve_scoring(problem_dir, scoring_branch_override, config)` that implements the priority logic above.

**`ScoreConfig` change:**
- Add `branch: str | None = None` field
- Parse from `score.branch` in `problem.yaml`

**`extract_scoring_from_branch` behavior:**
- Checkout `scoring/` from the branch into the working tree
- Move it to `.autoanything/_scoring/`
- `git reset HEAD -- scoring/` to unstage
- Return the path to `.autoanything/_scoring/`

**`resolve_scoring` behavior:**
- Check override → config → on-disk → convention branch
- Return the resolved `scoring_dir` path (or `None` for default)

**Tests:**
- `tests/test_scoring.py` — tests for `extract_scoring_from_branch` (requires a git repo fixture with a scoring branch)
- `tests/test_problem.py` — test that `score.branch` parses correctly from YAML

### Phase 2: Wire into runner.py, evaluator.py, server.py

**Files:**
- `src/autoanything/runner.py` — use `resolve_scoring()` at the top of `run_local()` to determine where scoring lives. If scoring comes from a branch, extract once; the existing move-once logic handles the rest.
- `src/autoanything/evaluator.py` — use `resolve_scoring()` in `establish_baseline()` and `evaluate_proposal()`, pass `scoring_dir` to `run_score()`
- `src/autoanything/server.py` — use `resolve_scoring()` in `create_app()`, pass `scoring_dir` through to `_evaluate_one_pr()` and its `run_score()` calls

**Key insight:** The `run_score(scoring_dir=...)` parameter already works (implemented). This phase is just about resolving _where_ scoring lives (branch vs disk) and passing that path through.

**Tests:**
- `tests/test_evaluator.py` — mock targets stay the same (`autoanything.evaluator.run_score`). These tests mock `run_score` so they don't exercise the branch extraction — that's tested in Phase 1.
- `tests/test_server.py` — same; server tests mock at a higher level

### Phase 3: CLI and init changes

**Files:**
- `src/autoanything/cli.py` — add `--scoring-branch` option to `run`, `evaluate`, `serve`, `score`, `validate`; update `init` to create scoring branch

**CLI option (shared):**
```python
@click.option("--scoring-branch", default=None,
              help="Git branch containing scoring/ (default: auto-detect)")
```

Added to: `run`, `evaluate`, `serve`, `score`, `validate`.

**`init` command changes:**
- After creating the problem directory and git repo on `main`:
  - Create orphan branch `scoring`
  - Write `scoring/score.py` from template
  - Commit
  - Switch back to `main`
- The `scoring/` directory on `main` is omitted (preferred — cleaner)
- Update printed "Next steps" to mention the scoring branch:
  ```
  # Edit scoring: git checkout scoring && edit scoring/score.py && git commit && git checkout main
  ```

**`score` command changes:**
- Use `resolve_scoring()` to find scoring, pass `scoring_dir` to `run_score`

**`validate` command changes:**
- Instead of checking for `scoring/score.py` on disk, also check the scoring branch
- If neither exists, error
- If scoring is on a branch, report that (not a warning)
- Adjust the "scoring tracked by git" check — on the scoring branch it's expected; on main it's still a warning

**Tests:**
- `tests/test_cli.py`:
  - `TestInit`: update `test_creates_scoring_dir` and `test_creates_score_py` — scoring now lives on a branch, not on disk on main. Add test that `git branch` output includes `scoring`.
  - `TestValidate`: update `test_missing_score_py_fails` to account for branch-based scoring. Add test for branch-based validation.
  - `TestScore`: update to work with scoring on a branch (may need a git repo fixture)
- `tests/test_integration.py`:
  - `TestInitToScore`: update — after init, scoring is on a branch. The test should checkout scoring, write the real `score()`, commit, checkout main, then run `autoanything score`.
- `tests/conftest.py`:
  - `problem_dir` fixture: add a `scoring` branch with `scoring/score.py` in addition to (or instead of) the on-disk scoring directory. For backwards-compat tests, keep the on-disk variant as a separate fixture.

### Phase 4: Template and doc updates

**Files:**
- `src/autoanything/templates/gitignore` — keep `scoring/` (now serves as a safety net rather than the primary mechanism)
- `src/autoanything/templates/problem.yaml` — add commented-out `branch: scoring` under `score:`
- `CLAUDE.md` — update "How Problems Work" section, evaluator design notes, and any references to scoring being gitignored
- `SCORE_DOCS.md` — update the "basics" section to explain branch-based scoring, add section on scoring experimentation via branches
- `agent_instructions.md` — no changes needed (agents already can't see scoring)
- `README` or repo docs — if they exist, update references

**Template changes:**

`templates/problem.yaml`:
```yaml
score:
  direction: {{direction}}
  description: "Describe what this metric measures"
  # name: score             # metric key, default "score"
  # timeout: 900            # seconds before scoring is killed
  # branch: scoring         # git branch with scoring/ (default "scoring")
  # bounded: false          # true if the metric has a known optimum
```

`templates/gitignore` (unchanged — scoring/ exclusion remains as safety net):
```
# Scoring code lives on the "scoring" branch — this is a safety net
scoring/

# Evaluator state (history database, logs)
.autoanything/
```

**CLAUDE.md updates:**
- Problem structure diagram: note that `scoring/` lives on a dedicated branch, not on main
- Evaluator design: replace "Blind scoring: agents never see scoring/ (gitignored; physically hidden during run)" with "Blind scoring: scoring code lives on a dedicated branch and is loaded via sys.path injection during evaluation"
- Add note about `--scoring-branch` for experimentation

**SCORE_DOCS.md updates:**
- "The basics" section: explain that `scoring/score.py` lives on the `scoring` branch
- Add new section: "Scoring branches" explaining how to use different branches for experimentation
- Update setup workflow: `git checkout scoring` → edit → commit → `git checkout main`

---

## Migration path for existing problems

Existing problems with `scoring/` on disk continue to work unchanged (legacy mode). To migrate:

```bash
cd my-problem

# Create the scoring branch from current scoring files
git checkout --orphan scoring
git rm -rf .
cp /path/to/your/scoring scoring/  # or move from backup
git add scoring/
git commit -m "Move scoring to dedicated branch"
git checkout main

# Verify
autoanything score   # should auto-detect the scoring branch
```

No changes to `problem.yaml` are required — the default branch name `scoring` is resolved by convention.

## Open questions

1. **Orphan branch vs regular branch?** Orphan keeps the scoring branch's history completely separate from main's history. Regular branch means scoring shares the initial commit with main. Orphan is cleaner but slightly more ceremony. Recommendation: orphan.

2. **Should `autoanything edit-scoring` be a convenience command?** Switching to the scoring branch, editing, committing, and switching back is more ceremony than editing a local file. A helper command could smooth this. Could be a fast follow-up, not required for the initial implementation.

3. **Remote scoring branches.** For problems hosted on GitHub, the scoring branch would be pushed. This means anyone with repo access can see the scoring code (on that branch). This is fine for most use cases but worth noting — if scoring must be truly secret, the branch can be kept local or in a separate private repo. Not a blocker.
