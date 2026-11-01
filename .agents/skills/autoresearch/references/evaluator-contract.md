# Evaluator, checkpoints, and recovery

The evaluator owns project data, case selection, provider transport, scoring, and `<run_dir>/result.json`. The lab owns snapshots, manifests, metric validation, budgets, and history. The evaluator measures the supplied candidate; it must not import a score from editable candidate code.

## Result and adapter checks

Write finite numeric metrics and numeric usage in `result.json`:

```json
{"metrics":{"quality":0.84},"usage":{"requests":1,"cost_usd":0.02}}
```

Keep unknown cost unknown; a zero known-cost subtotal does not prove a run was free. Put additional evidence in structured `details` and redact secrets from logs.

Before trusting a provider adapter, test it locally with the transport replaced by a deterministic stub. Assert the exact endpoint and model, parsed response and result metrics, request-attempt events, and reported usage. This tests the adapter contract without spending provider calls. `doctor` checks the local environment and skill integrity; it does not test evaluator/provider integration. `preflight` validates the contract and, optionally, a saved result without running the evaluator; a valid result still reports `integration_status: "unverified"` and `new_requests: 0`.

Optional `expected_attestations` is an object of string keys and values in lab config, `prepare` spec, or `epoch open --spec`; endpoint and model are useful keys, for example `{"endpoint":"https://sandbox.example/v1/chat/completions","model":"sandbox/model"}`. The result must return those same values under `attestations`. This checks reported result fields, not the actual transport destination or provider identity, so it does not replace the stubbed-transport test.

## Case and request checkpoints

The runtime exposes `case_scope(case_id)`, `completed_ids(path)`, `pending_ids(all_ids, done)`, and `append_case(path, record)` from `autoresearch_runtime.case_checkpoint`. It provides `AUTORESEARCH_CASE_LOG` and `AUTORESEARCH_ATTEMPTS_LOG` paths for the current repeat. Use stable case IDs from the frozen input set. At evaluator startup, derive `done` from the case log and process `pending_ids` in input order. Wrap each case in `case_scope(id)` so runtime HTTP hooks can associate requests with it. Append an `{"id":"…","status":"ok","usage":{…}}` case record only after its response is accepted and scored.

For SQLite-backed labs these environment paths are compatibility references, not files. Use the runtime helpers: they route checkpoints and attempt events into the database, and native `resume <run_id>` reads those stored projections.

For each provider attempt, record a durable `started` event with a unique `attempt_id` and `case_id` before sending; append `finished` with the same IDs and observed status afterward. Use `AUTORESEARCH_ATTEMPTS_LOG`. A 429 is a failed attempt, not a completed case; a later retry gets a new attempt ID. An unmatched `started` event means outcome and spend are uncertain, not free or failed. Do not issue a replacement request until the attempt is reconciled.

When native `resume <run_id>` must run the evaluator for missing repeats, it validates case and attempt records before making calls. For SQLite labs it reads typed database projections; an unfinished `started` attempt blocks resume before a replacement request. Review evidence through an explicit `storage export`.

## Capability proof

Capabilities default to false. A pinned local fixture report must include these required fields:

```json
{"schema":1,"offline":true,"case_resume":true,"evaluator_sha256":"<SHA-256 of the configured evaluator script>"}
```

Reference it by path and SHA-256 outside the source project and candidate:

```json
"evaluator_capabilities": {
  "case_resume": true,
  "offline_verified": true,
  "resume_fixture": {"path":"/external/proofs/resume.json","sha256":"<SHA-256 of resume.json>"}
}
```

The runtime checks the fixture hash, schema flags, and evaluator hash, then pins the fixture as an input; it does not execute the test described by the report. `offline_verified` records the result of a local stub test for this adapter and evaluator version. It is not a network sandbox or a promise that pending work on a real evaluator cannot contact a provider; inspect the actual attempt and usage logs.

## Resume boundaries

`prepare --resume` reuses a matching lab contract; it does not continue an interrupted evaluator run. Native `resume <run_id>` preserves the original ID and finalizes valid saved `result.json` repeats without requiring evaluator capability flags. It requires matching config/input/evaluator/candidate fingerprints, pins, epoch and parent. If a repeat still needs evaluator execution, `case_resume` and `offline_verified` plus the pinned fixture are required; only missing repeats are evaluated and that work may make provider calls.

Native resume keeps the same run and its decision gates; it does not create a second baseline/trial or silently change the parent, epoch, inputs, evaluator, candidate, or criteria. Do not describe ordinary `prepare --resume` or a new trial as continuation. An offline rescore is a separate evaluator-owned operation over preserved responses; only report zero new requests when logs establish that fact.
