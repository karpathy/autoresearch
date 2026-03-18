# Validation protocol

Validation is deterministic and localized to the target repository. The steps implemented in `autosaas/validation_pipeline.py` are as follows:

1. `run_required_gates` takes a `SliceRun`, a list of gate names (like `lint`, `typecheck`, `test`, `smoke`), and a command map derived from `project.autosaas.yaml`. The controller only passes the supported subset (`lint`, `typecheck`, `test`, `app_boot`, `smoke`); other entries such as `dev` are ignored until future flows need them. Providing `app_boot_url` in the manifest lets the controller treat `app_boot` as a health check rather than a runnable command.
2. Gates are executed following `_GATE_ORDER = ["lint", "typecheck", "test", "app_boot", "smoke"]`. The pipeline runs each gate in that order if it appears in the requested list, and stops as soon as one gate fails.
3. Each gate yields a `GateResult` (name, passed flag, summary, duration) that becomes part of the `SliceRun`.
4. When a gate fails, the run status is immediately set to `revert` and no further gates are launched.
5. After the deterministic loop, `decide_keep_or_revert` uses the gate pass/fail map to choose between `keep` and `revert`. An empty gate result map now resolves to `revert` so every active automation decision must name at least one gate.

For dry runs, none of these gates execute. The status stays `keep`, and the validation logic remains in place so the same code path can enforce command order once a real repo and commands are wired up.
