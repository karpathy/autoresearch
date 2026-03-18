# Validation protocol

Validation is deterministic and localized to the target repository. The steps implemented in `autosaas/validation_pipeline.py` are as follows:

1. `run_required_gates` takes a `SliceRun`, a list of gate names (like `lint`, `typecheck`, `test`, `smoke`), and a command map derived from `project.autosaas.yaml`.
2. Gates are executed following `_GATE_ORDER = ["lint", "typecheck", "test", "app_boot", "smoke"]`. The pipeline runs each gate in that order if it appears in the requested list, and stops as soon as one gate fails.
3. Each gate yields a `GateResult` (name, passed flag, summary, duration) that becomes part of the `SliceRun`.
4. When a gate fails, the run status is immediately set to `revert` and no further gates are launched.
5. After the deterministic loop, `decide_keep_or_revert` uses the gate pass/fail map to choose between `keep` and `revert`.

For dry runs, none of these gates execute. The status stays `keep`, and the validation logic remains in place so the same code path can enforce command order once a real repo and commands are wired up.
