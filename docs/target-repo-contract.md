# Target repository contract

AutoSaaS assumes the target repository exposes a tiny contract that lets the controller infer enough context to make decisions and run validations. The contract is intentionally narrow to keep the early loop deterministic and safe.

## Required artifacts (for real runs)

1. `project.autosaas.yaml` at the repo root.
   - It must define the `commands` mapping with `lint`, `typecheck`, `test`, `dev`, and `smoke` keys.
   - These commands are the ones AutoSaaS will attempt to execute via `validation_pipeline.run_required_gates` when `dry_run=False`.
      - (Currently the validation loop only considers `lint`, `typecheck`, `test`, `app_boot`, and `smoke`; the `dev` command is retained for future workflows but is not part of gate validation today.)
2. `package.json` (if present) is parsed by `repo_context_loader` to infer `scripts`, `framework`, and `package_manager`. Missing or malformed JSON is tolerated, but providing it improves context accuracy.

## Optional helpers

- The repo can list additional sensitive paths (`.env`, `.env.local`, etc.); AutoSaaS already redacts them via `privacy_guard.redact_text` when building reports.
- Future data such as branch metadata, git state, or `app_boot_url` can be added to the manifest gradually; the current code ignores unknown keys.

## Dry run tolerance

- When `dry_run=True`, none of the commands need to exist and the configuration file may be absent. The controller still walks the same pipeline in a minimal form (slice selection, executor scope, redacted reporting) so integration tests and documentation can rely on the same interfaces as a real run.
