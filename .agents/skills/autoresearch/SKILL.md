---
name: autoresearch
description: Use when asked to improve or investigate a code project through repeatable, metric-based experiments. Requires a measurable goal and a confirmed lab contract before the first measurement.
license: MIT
---

# AutoResearch

AutoResearch is a local experiment lab. Use its JSON CLI; Python 3.10+ and Git are required. Keep lab data outside the measured project under `AUTORESEARCH_HOME` (default `~/.autoresearch`).

For an existing SQLite lab, start with the compact canonical guide: `lab.py --lab <UUID> --agent guide --json`. It summarizes baseline, best result, goal criteria, evidence gaps, and next actions. Resolve a SQLite run's details only when needed with `guide --json --run <id>`. Its `lab.json`, `manifest.json`, and attempt logs are virtual references; use guide and API-backed run/evidence views. Legacy labs are archive-only: guide/status are read-only, and their saved runs cannot become measurements or a new baseline. Use the recommended ZIP export to preserve a legacy lab before starting fresh.

## Start or continue a lab

Read the brief or scout the project read-only: README, tests, evaluator, metric notes, prior results, and protected data. Ask only for missing facts. Get confirmation before creating a lab or measuring. Confirm a passport with project, goal, explicit success criteria, metric and direction, evaluation, constraints, and limits. Never create `program.md` in the measured project.

Run `preflight --spec <path>` before preparation; it validates the contract without evaluator or network calls. After the user approves the passport, prepare with `prepare --confirmed`, then read `contract_fingerprint` from the SQLite `guide --json` output. Before any evaluator operation, require the host-managed `AUTORESEARCH_HARNESS_PROFILE` described in [harness-profile.md](references/harness-profile.md), and pass `--confirmed` for that operation. The profile attests host-enforced approval, isolation, and network controls; its JSON is not proof that a host actually enforces them. AutoResearch does not limit spending. Keep paid calls disabled unless the user has approved them and set a spending limit on the API key. See [brief.md](references/brief.md) for passport/path rules and [safety.md](references/safety.md) for protected files, source boundaries, and promotion.

For ongoing work, reuse a lab only if brief, evaluator, metrics, and protected inputs still match. `prepare --resume` reuses a matching lab; evaluator `resume <run_id>` continues only supported runs with matching fingerprints. Preserve old labs with `--config <legacy-lab.json> storage export --output <archive.zip>`; do not import their runs, rewrite their history, or reuse their fingerprints for a new baseline. Put a new lab in a dedicated `AUTORESEARCH_HOME` outside the measured project. See [home.md](references/home.md) and [cli.md](references/cli.md) for storage and archive behavior.

The loopback dashboard service starts in the background with the first regular CLI command and is shared by `AUTORESEARCH_HOME`. Startup does not open or focus a browser; open one only after an explicit `--open-browser` request, such as `dashboard ensure --open-browser`. Use global `--headless` or `AUTORESEARCH_NO_DASHBOARD=1` to skip service startup. On Windows the service runs without opening a console window. SQLite labs use authenticated API routes. Legacy lab mutations and physical archive moves return `E_LEGACY_READ_ONLY`; archive export preserves contained files and lists skipped lock files.

## Measure and compare

Run and verify a baseline before trials; each evaluator run requires current operation confirmation plus a valid host profile. Pin evaluator, inputs, model conditions, metrics, repeats, usage, and hashes. Historical results are context. Fresh baselines and replay re-evaluate and may call a paid model; offline rescore uses stored responses without new calls. Never switch to `fresh` to bypass a mismatch. If the contract changes, open an epoch and explain why; refresh the profile fingerprint before the next evaluator run, and do not combine incomparable epochs. AutoResearch does not limit spending. Request and cost usage is recorded and reported; the user sets spending limits on the API keys.

Framework acceptance means a trial passed comparison with its parent and guards. Goal success means explicit thresholds and confirmation repeats passed. Missing or incomplete `goal_criteria` leave status exploratory or unknown. Record a source/hash for artifact-based criteria. Assess separate external gates independently; `best_id`, acceptance, a higher composite score, or replay does not prove goal success.

## Test one hypothesis

Use `guide --json --run <id>` to inspect a result and its evidence. Compare aggregate and parent-relative deltas to find bottlenecks, failed guards, or untested conditions. Before editing, state one explanation, evidence, one change, expected direction, and falsifier. Respect the confirmed budget.

Edit only `candidate/`. Run one change per trial and compare it to the declared parent using repeat rules and guards. Execute trials using zero-file inline hypothesis flags (`--hypothesis-json '{"title":"...","observation":"...","expected_effect":"..."}'` or `--title "..." --observation "..." --expected-effect "..."`) on `--confirmed step --run` or `--confirmed trial` / `--confirmed trial start` to avoid wasting tool calls on temporary `hypothesis.json` files.

Use the automatic diagnostics surfaced directly in the cockpit envelope instead of filesystem reads:
- `impact.error_tail` and `run_detail.error_tail`: last 10 lines (up to 600 chars) of evaluator stderr/stdout on crash, eliminating the need to read `run.log`.
- `case_samples` inside `case_regression` signal: up to 3 compact regressed case diagnostics (`id`, `status`, `error`, `metrics`), eliminating the need to read `cases.jsonl`.
- `dead_ends` in cockpit summary: up to 8 unique rejected/crashed hypotheses from the current epoch (backed by 100-run duplicate/similarity memory), protecting agents from repeating failed ideas after context compaction.

Keep improvements only when accepted; report goal criteria and confirmation repeats separately. For SQLite labs, use guide/run/job projections, not sidecar files. A missing CLI response does not prove an operation failed. When local iterations plateau or `repeated_rejection` occurs (3+ times), use the `openresearch` skill strictly for read-only literature retrieval (`orx discover keyword/embedding/openalex/biorxiv/pubmed` and `orx paper <id> [--full]`) and test the resulting idea inside `candidate/` via `lab.py` without running `orx` experiment/branch commands. See [autonomous-loop-protocol.md](references/autonomous-loop-protocol.md) for the complete protocol.

## Recover and report

Never blindly retry an evaluator. Inspect the run ID, result, checkpoint, and attempt data. SQLite stores these in database/API views; legacy labs use guide-referenced manifests and logs. Resume only when supported and fingerprints match. Use `cancel <run-id>` to stop an active run. `storage reconcile --lab <UUID>` previews only provable terminal-status repairs; add `--apply` to write them. It never invents missing results, attempts, metrics, or usage. Otherwise preserve history and choose a fresh run, offline rescore, or replay; record evidence class and model-call count. See [cli.md](references/cli.md), [measurement.md](references/measurement.md), and [evaluator-contract.md](references/evaluator-contract.md).

Report the baseline, best trial, parent-relative change, guard results, explicit goal status, confirmation count, evidence class, and actual or unknown usage. Keep historical, offline, replay, resumed, and fresh records distinct. Promote only when the user asked for it. Do not claim time or context savings without matched measurements of actual harness usage.
