# AutoSaaS private program notes

This private companion elaborates on the guarded main loop introduced in `autosaas/main.py`. It is intended for team members who need to understand how the automation interacts with target repositories when `dry_run=True` is insufficient.

- **Entry point:** `run_once(target_repo, request, dry_run=True)` runs a quick simulated execution that exercises `repo_context_loader`, `task_slicer`, `ImplementationExecutor`, and `reporter` without requiring a real target repo manifest.
- **Validation gate coverage:** When `dry_run=False`, the manifest can optionally provide `app_boot_url` so the pipeline treats `app_boot` as a health check and still only reports `keep` or `revert`. Any unexpected exception returns a `crash` status with a redacted summary.
- **Dry-run behavior:** Status defaults to `keep`, the scoped `TaskSlice` is instantiated but `ImplementationExecutor.run([])` leaves the touched-files list empty, and `privacy_guard.redact_text` ensures secrets do not leak into reports.
- **Next steps:** When real validations are added, the function will load `project.autosaas.yaml`, translate it to the validation gate map, and run `validation_pipeline.run_required_gates`. `decide_keep_or_revert` will then drive the final status.
