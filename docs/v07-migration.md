# AutoResearch 0.7.0 migration notes

This release changes how the skill manifest is handled, makes in-place promotion stricter, removes the bundled baseline pin, and adds error codes. This page lists only the user-visible breaking changes. It does not convert existing labs.

## Source checkouts and installed copies

- Source checkouts no longer contain `skill.manifest.json`. A source tree without a manifest is valid and is treated as an unmanaged source checkout.
- `install.py` writes the manifest into the target it installs. Run the installer again to update an installed copy.
- `LICENSE` is part of the skill package, so it is included in the file hashes and copied to installed targets.

## In-place promotion

- `promote --in-place` applies the patch and stages the changed files. It never commits.
- The `direct_sync` fallback is removed. It copied whole files over the source tree and committed the current branch, which could overwrite conflicting edits and commit unrelated files.
- A failed in-place apply restores only the patched paths.

New error codes, all with exit code 4:

- `E_CONFLICT`: the patch conflicts with the source tree. The patched paths are restored and the conflicting paths are listed. Use `--to-branch` to resolve the patch in a separate worktree.
- `E_RESTORE`: restoring the patched paths after a failed apply did not complete. The error lists the paths that need manual attention.
- `E_PREFIX_MISMATCH`: the lab's stored project prefix does not match where the source sits in its repository. Prepare the lab again from the matching location.
- `E_SOURCE_READ`: a patched path could not be read before the in-place apply started. No source file has changed. Fix the read problem and run `promote` again.

## Baseline pins

- The bundled pin is removed.
- Custom pins go in `AUTORESEARCH_HOME/pins` as `*.json` files.
- A damaged, unreadable, or unknown-schema pin now blocks preparation instead of being skipped, because skipping it would silently drop its guard.

## Existing runs

- Runs stamped by 0.6.0 cannot be resumed across the upgrade, because the skill fingerprint changes. Treat them as historical evidence, and establish a new baseline under 0.7.0 before running trials.
