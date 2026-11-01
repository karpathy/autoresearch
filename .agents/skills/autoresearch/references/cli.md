# CLI contracts

## Interface

Commands print one JSON envelope on stdout. Read `ok`, `data`, `warnings`, `next`, and `error`; errors may also be written to stderr. Exit codes: `0` success, `2` usage or contract error, `3` measurement failure, `4` human help needed, and `5` busy (`E_LOCKED`). Use the error hint before retrying.

```text
python SKILL_DIR/scripts/autoresearch.py doctor
python SKILL_DIR/scripts/autoresearch.py prepare --project <path> --brief-text - --spec <spec.json>
python SKILL_DIR/scripts/lab.py --config <lab.json> guide --json
python SKILL_DIR/scripts/lab.py --config <lab.json> status
```

For harnesses, prefix a lab command with global `--agent`:

```text
python SKILL_DIR/scripts/lab.py --agent --config <lab.json> guide --json
python SKILL_DIR/scripts/lab.py --agent --config <lab.json> status
```

The envelope keeps a small command outcome in `data` and adds one canonical bounded `cockpit`; it removes duplicate verbose command fields. Use that cockpit for the next decision instead of separately fetching a full report and guide.

The project-local skill works with a normal shell and filesystem. Do not assume a particular agent tool API. If a harness does not discover `.agents/skills/autoresearch/SKILL.md`, point it to that file directly.

## Approval and evaluator runs

Prepare a lab only after the user approves its passport. For a new SQLite lab, pass `--confirmed` to `scripts/prepare.py`; this does not run the evaluator. Read the `contract_fingerprint` from `guide --json`, configure `AUTORESEARCH_HARNESS_PROFILE` to a host-managed profile outside the project and lab state, then pass global `--confirmed` for each evaluator operation. The profile schema, offline/network modes, and harness responsibilities are documented in [harness-profile.md](harness-profile.md). A profile is a host attestation, not proof of sandbox or network enforcement. Read-only commands and `preflight` do not require it.

`prepare --auto` is deprecated. It returns `E_AUTO_MIGRATION_REQUIRED` before preparing a lab or starting an evaluator. Use `preflight`, then explicit `prepare.py --confirmed`, review the guide fingerprint, configure the host profile, and start a separate confirmed baseline.

## Legacy file-based labs and SQLite export

A lab stored as plain files (a real `lab.json` with no SQLite lab behind it) is no longer supported. Any command that loads one fails with `E_LEGACY_LAB` and leaves the files unchanged. Git tag `v0.8.0` can open and export such labs. Export a SQLite lab as JSON; the export needs no harness profile or evaluator confirmation:

```text
python SKILL_DIR/scripts/lab.py storage export --lab <sqlite-ref> --output <lab.json>
```

## Free checks and bounded guidance

```text
python SKILL_DIR/scripts/lab.py preflight --spec <spec.json> [--program <brief.md>] [--result <result.json>]
python SKILL_DIR/scripts/lab.py --config <lab.json> guide --json [--run <run_id>]
```

`preflight` validates the supplied contract, optional brief/budget provenance, and optional saved result without running the evaluator or making network requests. A spec that declares `budget.max_requests` or `budget.max_cost_usd` (or quotes them in `budget_sources`) is rejected by `preflight` and `prepare` with `E_CONTRACT_SPEND_LIMIT`: AutoResearch does not limit spending, so set the limit on the API key. A valid contract or result is not proof that the evaluator reached the intended provider: dry-run output remains `integration_status: "unverified"` and `new_requests: 0`. `doctor` checks the local environment and skill integrity, not provider integration. Test a provider adapter with a local substituted transport and assert endpoint, model, parsed result, attempts, and usage before relying on it. See [the evaluator contract](evaluator-contract.md).

`guide --json` is the canonical bounded summary for the next decision; `--run <run_id>` selects one run's evidence. For SQLite-backed labs, the global lab selector precedes the command: `python SKILL_DIR/scripts/lab.py --lab <UUID> guide --json`. Use its run IDs and the authenticated dashboard API's run/evidence views; SQLite manifests, results, checkpoints, and attempts are database projections, not virtual files to open. For legacy labs, read referenced files only when the card identifies a gap.

## Trials and resume

`step` without execution recommends an action. Execution requires an explicit hypothesis:

```text
python SKILL_DIR/scripts/lab.py --config <lab.json> step
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed step --run --hypothesis <hypothesis.json>
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed step --run --hypothesis-json '{"title":"...","observation":"...","expected_effect":"..."}'
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed step --run --title "..." --observation "..." --expected-effect "..."
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed resume <run_id>
```

For AI agents, inline hypothesis arguments (`--hypothesis-json` or `--title`/`--observation`/`--expected-effect`) are recommended over `--hypothesis <file>` to avoid disk write tool calls.

Native resume accepts only an interrupted, crashed, or timed-out run and validates the same frozen contract, candidate snapshot, pins, epoch, and parent before continuing that run ID. Valid saved results can be finalized without evaluator capability flags. If missing repeats require evaluator execution, verified `case_resume` and `offline_verified` capabilities with the pinned fixture are required; pending evaluations may still make provider calls. Before those evaluator calls, corrupt/torn journals or unresolved HTTP attempts block the resume path. Reconcile the logs before retrying; never make a second run look like continuation.

Background trials remain available:

```text
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed trial start --hypothesis <hypothesis.json>
python SKILL_DIR/scripts/lab.py --config <lab.json> --confirmed trial start --hypothesis-json '{"title":"...","observation":"...","expected_effect":"..."}'
python SKILL_DIR/scripts/lab.py --config <lab.json> trial wait --timeout 30
python SKILL_DIR/scripts/lab.py --config <lab.json> trial status
```

Evaluation and signal diagnostics are automatically attached to the cockpit envelope:
- `impact.error_tail` and `run_detail.error_tail`: last 10 lines (up to 600 chars) of evaluator stderr/stdout on crash, eliminating manual `run.log` inspection.
- `case_samples`: in `case_regression` signals, surfaces up to 3 compact failing cases (`id`, `status`, `error`, `metrics`), eliminating `cases.jsonl` reads.
- `dead_ends`: in the cockpit summary, lists up to 8 rejected/crashed hypotheses from the current epoch (backed by a 100-run history memory), guarding against cycling through failed ideas across context compaction.

Query the durable job through `trial status`/`trial wait` and inspect the stored guide/run projection instead of searching for a `manifest.json` path.

## Errors

Unknown `--lab` selectors return `E_LAB_NOT_FOUND`, malformed selected configs return `E_CONFIG_INVALID`, and ambiguous selectors stay explicit. SQLite `storage export` and `storage reconcile` remain available.

## Other commands

Use `prepare` to create a new SQLite-backed lab. `recover` remains available for SQLite labs. The local dashboard service starts in the background on the first regular lab command, before lab resolution or preparation, and is shared by commands using the same `AUTORESEARCH_HOME`. Startup does not open or focus a browser; a supported explicit `--open-browser` request, such as `dashboard ensure --open-browser`, is required. On Windows the service runs without a console window. Global `--headless` or `AUTORESEARCH_NO_DASHBOARD=1` skips service startup. `dashboard` supports `start`, `ensure`, `status`, and `stop`. Its stop token stays in protected home metadata and is never passed on the command line. See [home layout](home.md) for service files.

`cases diff --left <repeat-dir> --right <repeat-dir> [--out <path>]` compares response-body hashes for successful case IDs from two existing repeat directories. It reads their `cases.jsonl` and `sent-requests.jsonl`; both paths must be real directories, so this command does not query SQLite-backed labs' virtual repeat paths. SQLite checkpoint and attempt records stay in the database, are consumed by native `resume <run_id>`, and are included by explicit `storage export --lab REF --output <file>`.
