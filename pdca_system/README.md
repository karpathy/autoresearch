# PDCA System

A **portable orchestrator** for an autonomous Plan-Do–Check-Action loop. It drives an external AI agent (Claude, Codex, OpenCode, etc.) through file-based queues: the agent plans and implements in the **Plan-Do** stage, then adapts, runs, and evaluates in the **Check-Action** stage. State, logs, and worktrees live under `pdca_system/` so the host project stays clean.

**Designed to be copied into any repo.** Add `pdca_system` to your project, point the agent at the protocol, give it a prompt (e.g. “Improve X”, “Fix Y”), and run the daemon. The system adapts to your project’s layout and conventions via the protocol and stage docs you keep (or edit) inside `pdca_system/`.

---

## Use in a new project

All commands in this document are run from the **project root**. Use your project’s Python runner for executables (e.g. `uv run` or `python -m`) so the same environment is used everywhere.

1. **Copy** the `pdca_system` folder into your project root.
2. **Git:** If the project has no `.git` directory (not a git repo yet), run `git init`, then add all project files to git (e.g. `git add .`) so the project’s original files, `pdca_system/`, and `.gitignore` are tracked. If the project is already a repo, add `pdca_system/` and any new or changed project files to git (`git add pdca_system` or `git add .` as appropriate). Add the following to your project’s `.gitignore` so runtime artifacts are not committed:
   ```
   pdca_system/history/
   pdca_system/prompt_audit/
   pdca_system/baseline_branches.json
   pdca_system/baseline_metrics.json
   ```
3. **Install** PDCA dependencies from the project root (use your project's runner):
   ```bash
   uv pip install -r pdca_system/requirements.txt
   ```
   Or add them to your project’s `pyproject.toml` / `requirements.txt`. The system needs FastAPI, uvicorn, pydantic, jinja2, and python-multipart only. You do **not** need Node/npm to run the dashboard; the CSS is pre-built in `web/static/app.css`. To rebuild CSS after editing Tailwind source: from project root, run `npm install` then `npm run build:css` (from the directory that contains `package.json`).
4. **Optionally adapt** the protocol and stage docs to your project:
   - `protocol.md` — overall workflow, constraints, and what “baseline” and “promotion” mean in your context.
   - `PDCA-Plan-Do.md` — what the agent should do in the Plan-Do stage (e.g. which files to edit, which command to run).
   - `PDCA-Check-Action.md` — what the agent should do in the Check-Action stage (e.g. run tests, collect metrics, decide promote/discard).
   - `config.py` — promotion threshold, default baseline branch, paths if you change them.
5. **Run** the dashboard (optional) and daemon from the **project root**:
   - Dashboard: `uv run uvicorn pdca_system.web.app:app --reload` (or `python -m uvicorn pdca_system.web.app:app --reload`) → http://127.0.0.1:8000/pdca-system
   - Daemon: `uv run pdca_system/daemon.py` (or `python -m pdca_system.daemon`; set `PDCA_AGENT=codex` / `PDCA_AGENT=opencode` / `PDCA_AGENT=kimi` for other backends)
6. **Bootstrap:** Have your agent read `pdca_system/protocol.md`, create a seed from your prompt, queue it for Plan-Do, then start the daemon. The daemon will hand off tasks to the agent via the queue; do not run Plan-Do or Check-Action stages manually in the same session.

Seeds flow: **queue/pd/** → Plan-Do → **queue/ca/** → Check-Action → **state/**. View runs and status in the dashboard.

---

## Concepts

- **Seed** — One experiment or task, created from a human prompt. Has a branch, optional worktree, and a sequence of runs (Plan-Do then Check-Action).
- **Plan-Do (PD)** — Stage where the agent turns the prompt into a concrete plan and implements it (e.g. edit files, commit on the seed branch).
- **Check-Action (CA)** — Stage where the agent adapts/fixes, runs the canonical command (e.g. tests or training), and reports metrics; the system decides promote / keep / discard.
- **Daemon** — Long-lived process that polls `pdca_system/history/queue/pd/` and `pdca_system/history/queue/ca/`, dispatches tasks to your agent, and moves tasks to done/error.
- **Baseline** — Reference (e.g. a branch and metrics) that Check-Action compares against to decide promotion. Defined and updated by the workflow; see `protocol.md`.

---

## Layout (inside pdca_system)

```
pdca_system/
  requirements.txt      — Python deps for dashboard + daemon (from project root: uv pip install -r pdca_system/requirements.txt)
  protocol.md           — Workflow rules and constraints (customize for your project)
  PDCA-Plan-Do.md       — Plan-Do stage instructions
  PDCA-Check-Action.md  — Check-Action stage instructions
  daemon.py             — Entrypoint for the daemon
  task.py               — Queue and state helpers
  config.py             — Promotion threshold, target metric (TARGET_METRIC_KEY, TARGET_METRIC_LOWER_IS_BETTER, TARGET_METRIC_LABEL), default branch, paths
  web/                  — Dashboard (FastAPI + HTMX)
  domain/               — Models (Seed, Run, Stage, Status)
  services/             — Workflow and git logic
  repositories/         — State persistence
  history/              — Created at runtime: queue/, state/, logs/, worktrees/
  package.json          — Optional: only if you want to rebuild dashboard CSS (Tailwind)
  tailwind.config.js    — Optional: Tailwind config for npm run build:css
  postcss.config.js     — Optional: PostCSS config for npm run build:css
```

All runtime data (queue, state, logs, worktrees) lives under `pdca_system/history/` so the host repo can stay clean and the system remains self-contained.

**Shared data across worktrees:** Have the adapt agent modify the project’s data-prep process and path configuration to use `~/.cache/<project_name>/` (e.g. `~/.cache/my_project/`), and commit those changes. Then all worktrees share the same data naturally.

**npm is not required** to run the dashboard. The built `web/static/app.css` is included; the dashboard uses it as-is. Install Node/npm and run `npm install` + `npm run build:css` only if you edit `web/static/tailwind.input.css` and need to regenerate the CSS.

---

## Making it fit your project

- **Protocol and stage docs** — Edit `protocol.md`, `PDCA-Plan-Do.md`, and `PDCA-Check-Action.md` to describe your repo’s layout, **canonical run command** (the script or module to run, e.g. `train.py` or `pytest`), success metric, and promotion rules. The daemon injects the **Python executable** (the one running the daemon) into CA prompts; the agent uses that Python with the canonical command defined in your protocol/docs.
- **Config** — In `config.py` you can change the promotion threshold, the **target metric** used by Check-Action (`TARGET_METRIC_KEY`, `TARGET_METRIC_LOWER_IS_BETTER`, `TARGET_METRIC_LABEL` — e.g. `val_bpb` vs `val_accuracy`), the default baseline branch name, and (if needed) path overrides. The run must output that metric in stdout (or stderr) so the workflow can parse and record it. The dashboard and daemon prompt examples read these values from config, so editing only `config.py` (and docs) is enough to adapt the metric; no other code changes are required.
- **Agent backend** — Set `PDCA_AGENT=codex`, `PDCA_AGENT=opencode`, or `PDCA_AGENT=kimi` (default is Claude). The daemon invokes the agent per task; ensure the agent can read the worktree and run commands as required by your protocol.

No changes to the host project are required beyond adding the folder and dependencies; the system is a pure orchestrator and drives work *outside* `pdca_system` (e.g. your `train.py`, tests, or scripts).

### Prompt to auto-edit the docs for a new project

Give this to your AI agent (or use it as a checklist) so it rewrites the PDCA docs in the context of the current repo:

```
You are in a project that has pdca_system copied in. Absorb pdca_system into the project and adapt the PDCA docs to this project only.

1. Git: If the project has no `.git` directory (not a git repo yet), run `git init`, then add all project files to git (e.g. `git add .`) so the project's original files, `pdca_system/`, and `.gitignore` are tracked. If the project is already a repo, add `pdca_system/` and any new or changed files (`git add pdca_system` or `git add .` as appropriate). Add these lines to the project's `.gitignore` (create the file if it does not exist) so runtime artifacts are not committed: pdca_system/history/, pdca_system/prompt_audit/, pdca_system/baseline_branches.json, pdca_system/baseline_metrics.json
2. Read the repo: identify the canonical "run" command (run from project root, e.g. `uv run train.py` or `python -m pytest`), which files the agent is allowed to edit, which are read-only, and what the success metric is (e.g. test pass, val loss, benchmark score).
3. If the project uses local data (datasets, checkpoints, etc.): modify the data-prep process and any path configuration to use `~/.cache/<project_name>/`, and commit those file changes, so worktrees share data naturally.
4. Edit pdca_system/protocol.md: replace any project-specific examples (paths, commands, metric names) with this project’s. Keep the structure (Seed → PD → CA, queue, baseline, promotion). State the canonical run command (from project root), the metric the Check-Action stage must report.
5. Edit pdca_system/protocol.md: replace any project-specific examples (paths, commands, metric names) with this project's. Keep the structure (Seed → PD → CA, queue, baseline, promotion). State the canonical run command (from project root) and the metric the Check-Action stage must report.
6. Edit pdca_system/PDCA-Plan-Do.md: say which files or dirs the agent may change in Plan-Do, what “done” looks like (e.g. commit on seed branch, ready for Check-Action).
7. Edit pdca_system/PDCA-Check-Action.md: say how to run the canonical command from project root, how to parse the metric from output or logs, and how to decide promote vs keep vs discard (threshold, comparison to baseline).
8. If the project uses a different default branch, promotion threshold, or success metric, update pdca_system/config.py: DEFAULT_BASELINE_BRANCH, PROMOTION_THRESHOLD, and the target metric (TARGET_METRIC_KEY, and optionally TARGET_METRIC_LABEL, TARGET_METRIC_LOWER_IS_BETTER) so the dashboard and promotion logic use the correct key and direction.

Do not change pdca_system code (daemon, web, services). Only edit the markdown docs, config.py, and the project's .gitignore as above. The adapt agent may also modify the host project's data-prep process and commit those changes (step 3).
```
