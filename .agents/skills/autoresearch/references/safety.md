# Project safety and promotion

AutoResearch keeps experiment state under `AUTORESEARCH_HOME` and treats the measured project as read-only unless you explicitly opt in.

## Snapshot boundaries

- **Service paths** (`.git`, virtualenvs, caches, build outputs, lab folders) never enter the candidate snapshot.
- **Secret-like files** are excluded by basename or extension: `.env*`, `.npmrc`, `.netrc`, `id_rsa*`, `*.pem`, `*.key`, `*.pfx`, `*.kdbx`, and anything under `.ssh` or `.aws`.
- Optional **`exclude_globs`** in `lab.json` add extra glob patterns; they must not be used to drop ordinary project source you intend to measure.
- Dirty git projects still snapshot **modified tracked files and untracked files** that pass the rules above. Ignored-only paths stay out. At init, **`included_untracked`** in `state.json` lists which untracked paths were copied.

## Read-only source git

`source_git` only runs `rev-parse`, `ls-files`, `status`, and `apply --check` on the user project, with `GIT_OPTIONAL_LOCKS=0` and `core.fsmonitor=false`, so listing and dry-run apply do not take the repo lock or rewrite files.

## Vault

Protected inputs live in the lab vault. `restore_vault` refuses to write back into `source_root` unless you pass **`--allow-source-write`**. Evaluators receive vault paths through **`AUTORESEARCH_INPUT_MAP`**.

If a protected input hash changes after init, runs fail with **`input_tampered`** (manifests may include `reason_code: input_tampered`).

## Promote

- **`promote`** writes `promotion/best.patch` (baseline commit → best commit), records **`project_prefix`** when the lab measures a subdirectory of a git repo, and runs **`git apply --check`** on the source repository without modifying it.
- **`promote --to-branch NAME`** creates a worktree at **`$AUTORESEARCH_HOME/promotions/NAME`**, applies with **`--3way`** and **`core.autocrlf=false`** on each git command (the shared `.git/config` is not changed), and returns merge **conflicts** in JSON instead of failing opaquely. It refuses when **`source_head`** drifted unless you pass **`--allow-drift`**.
- **`promote --in-place`** applies the patch to the source worktree and stages the changed files, but **never commits**. Before anything changes it refuses with **`E_DIRTY`** if a patched path has local changes (tracked, staged, untracked, or renamed), and with **`frozen_tampered`** or **`input_tampered`** if protected files were touched. If the patch conflicts, it restores only the patched paths and fails with **`E_CONFLICT`** (exit code 4). If that restore itself fails, it fails with **`E_RESTORE`** and lists the paths. When the apply conflicts, use **`--to-branch`** to resolve it in a separate worktree. The source must be a git repository. It does not switch branches.

Candidate git always uses **`core.autocrlf=false`** so byte-identical snapshots and binary patches stay stable on Windows.
