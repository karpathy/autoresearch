# AutoResearch 0.8.0 migration notes

This release removes the spend-limit feature. AutoResearch does not limit spending. Set a spending limit on each API key that an evaluator uses. This page lists only the user-visible changes. It does not convert existing labs.

## Spend limits

- `budget.max_requests` and `budget.max_cost_usd`, and their `budget_sources` quotes, are rejected in a new spec. `preflight` and `prepare` fail with `E_CONTRACT_SPEND_LIMIT` (exit code 2). Remove the keys and set the limit on the API key instead.
- `max_seconds` and `max_trials` are unchanged.
- Usage and cost are still recorded and reported: request and cost counts, `usage.json`, run manifests, and the dashboard cost figures.
- The cockpit no longer reports `requests_remaining` or `cost_usd_remaining`. The dashboard no longer draws limit meters, and its detail payload no longer has `limit` fields.

## Existing labs

- Labs prepared before 0.8.0 still open. Their retired spend keys are ignored, and their stored `lab.json` is not rewritten.
- Their lab id and `contract_fingerprint` do not change, so harness approvals bound to them stay valid.
- Preparing the same spec again resumes the existing lab.

## Harness profiles

- `spend` is no longer required or documented. A profile without it is valid.
- A profile that still has a `spend` block is accepted, and the block is ignored.
- `network.mode: "allowlisted"` no longer needs a spend cap.
- The approval fingerprint is unchanged, so no re-approval is needed. The normalized profile no longer includes `spend`, so every profile's `profile_sha256` changes.

## Operations queued before the upgrade

A durable operation queued before the upgrade stores the old `profile_sha256`. Its policy recheck fails with `E_HARNESS_POLICY_STALE`, so it is not dispatched. Enqueue it again after the upgrade. Operations that already finished keep their stored snapshots.
