# AutoSaaS architecture

This repository centers on AutoSaaS's orchestration loop, which lives in `autosaas/main.py`. The loop wires together the helper modules that already exist so the system can take a request, scope a focused slice, and produce a redacted report even in dry-run mode.

## Top-level flow

1. `load_repo_context` infers what kind of project the target repo is (framework, package manager, scripts, sensitive files) without requiring any secrets.
2. `choose_next_slice` turns an operator request into a `TaskSlice`, establishing the allowed file patterns the implementation is permitted to touch.
3. `ImplementationExecutor` reinforces those patterns and tracks touched paths; the main loop uses it to prove the slice has a defined scope before validating outcomes.
4. `run_required_gates` (via `validation_pipeline`) attempts to run configured lint/typecheck/test/smoke gates in a deterministic order and reports results.
5. `decide_keep_or_revert` turns the gate verdicts into a user-facing disposition and `reporter.format_slice_run` yields the human-readable summary.
6. `privacy_guard.redact_text` scrubs any sensitive literals or configured tokens (e.g., `sk_live`) from that summary before it reaches the operator.

## Dry-run posture

Dry-run mode deliberately avoids touching real repositories or running commands. It still walks the same flow, but the validation stage is skipped and the slice status defaults to `keep`. This guarantees deterministic behavior for integration tests and early validation without deploying real change sets.
