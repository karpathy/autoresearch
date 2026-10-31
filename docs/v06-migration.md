# AutoResearch 0.6.0 migration notes

This source update changes legacy lab handling to an archive-and-new-start workflow. It does not convert or repair an existing lab. Existing SQLite records, migration bindings, baseline trust decisions, and archived fingerprints remain unchanged.

## Preserve a legacy lab

Legacy labs remain available through read-only `guide` and `status`. Those commands can summarize the saved state when `source_root` is missing, pins are stale, or old state lacks epoch metadata. They identify the lab as `ARCHIVE_ONLY`, do not offer baseline or trial execution, and recommend a byte-preserving ZIP export.

```text
python .agents/skills/autoresearch/scripts/lab.py --config <legacy-lab.json> guide --json
python .agents/skills/autoresearch/scripts/lab.py --config <legacy-lab.json> storage export --output <archive.zip>
```

The ZIP contains original regular files beneath `lab/` and a root `archive-index.json` with format `autoresearch-legacy-archive-v1`. The index records archive-relative paths, byte sizes, SHA-256 values, and excluded lock paths. The export preserves candidate `.git` files as bytes and never follows external `.gitdir` pointers. Known `.lock`, `.lock.guard`, and `.lock.*.tmp` files are excluded and listed. Symlinks, Windows reparse points, external state directories, output paths within the lab, source changes during export, and existing output paths cause a typed refusal. The ZIP is an archive, not a runtime restore format.

SQLite export remains JSON:

```text
python .agents/skills/autoresearch/scripts/lab.py storage export --lab <sqlite-ref> --output <lab.json>
```

SQLite `storage reconcile` remains separate from run recovery. It repairs only statuses contradicted by terminal manifests; it does not call `recover` or invent run data.

## Start a new lab

Choose a new project and goal, then prepare the lab under a dedicated `AUTORESEARCH_HOME` outside the measured project. Preparation still requires explicit user confirmation. A fresh baseline still requires the ordinary trusted host profile and operation confirmation. Archived runs, histories, and fingerprints cannot be imported into the new lab or treated as its baseline; create a new measurement record under the new contract.

## Retired operations and errors

`storage migrate`, `storage cleanup`, `history-import`, and `compat --migrate` remain recognized for a transition period and return `E_OPERATION_REMOVED` with archive/new-start guidance. `compat` without `--migrate` remains a read-only format audit. Attempts to write or physically move a legacy lab return `E_LEGACY_READ_ONLY`. Unknown `--lab` selectors return `E_LAB_NOT_FOUND`; malformed lab configs return `E_CONFIG_INVALID`; ambiguous selectors remain explicit.

The update removes the legacy import planner, importer, post-import cleanup, automatic project-history scanning/import, format mutation API, and physical legacy archive move routes. SQLite schema migrations, storage export/reconcile, native lifecycle operations, existing history rendering, and baseline trust checks remain.

## Verification scope

The focused regressions cover legacy archive byte preservation and containment, output no-clobber behavior, source drift refusal, guide-generated export argv, read-only legacy status, retired mutation refusals, SQLite export/reconcile, and preserved migration binding/trust behavior. No source lab or archived fingerprint is rewritten by this version.
