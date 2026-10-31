# AutoResearch 0.5.0 migration notes

This source tree contains the 0.5.0 production-hardening changes. It is not a published release. Existing 0.4.x run records, manifests, epochs, and contract fingerprints remain historical evidence and are not rewritten.

## Before upgrading

1. Stop all AutoResearch CLI commands and workers. Versions before 0.5.0 do not honor the persistent `.lock.guard` advisory lock.
2. Back up the AutoResearch home and any installed skill directories.
3. Install from a verified 0.5.0 package. If an existing skill target has a schema 1 manifest, modified managed file, symlink, or any unowned content, the installer stops before changing it. Preserve that copy and install into a clean target; there is no force overwrite.
4. Run doctor and inspect the new skill integrity result before starting work.

## Lab compatibility

SQLite is the only writable lab store. Legacy file-based labs remain readable and can be explicitly imported:

```text
python SKILL_DIR/scripts/lab.py storage migrate --lab <legacy-ref>
python SKILL_DIR/scripts/lab.py storage migrate --lab <legacy-ref> --apply
```

Review the preview before applying. Import does not remove source files and is idempotent after verification. Legacy lab-state mutations now return `E_STORAGE_MIGRATION_REQUIRED`; read-only views, explicit exports, previews, and unrelated global settings remain available. Cleanup is a separate explicit action after a verified import.

New SQLite lab preparation requires an explicit CLI `--confirmed`; API creation requires request `confirmed: true`. `prepare --auto` now returns `E_AUTO_MIGRATION_REQUIRED` before creating a lab, reading a harness profile, or running an evaluator. Use preflight, prepare a confirmed lab, and start a separate confirmed baseline.

## Harness approval and evaluator policy

Each durable measurement operation requires global CLI `--confirmed` (or API `confirmed: true`) and a valid host profile at `AUTORESEARCH_HARNESS_PROFILE`. Read the contract fingerprint from the prepared SQLite lab's `guide --json` output and bind it in the host profile. The profile is an attestation; the host harness must enforce its claimed sandbox, approval, network policy, and any provider cap. AutoResearch does not sandbox arbitrary subprocesses or enforce `max_requests`/`max_cost_usd` against a generic evaluator.

Wire bodies are omitted by default. Set `AUTORESEARCH_WIRE_LOG_BODY=1` only when body-dependent identity/replay checks require capture and sensitive evaluator data may be stored. Request-identity digests still require the captured request-body sequence; metadata, byte counts, and response-body hashes cannot reconstruct them. Bodyless runs are incomparable to a stored request-identity digest and fail that check explicitly; this does not prevent offline or file-input runs that do not use request-identity checks. Historical 0.4.x hashes stay unchanged; if the original sequence cannot be reproduced, establish new request-identity provenance with a body-captured baseline.

See [the harness profile contract](../.agents/skills/autoresearch/references/harness-profile.md), [installation guidance](../.agents/skills/autoresearch/references/installation.md), and [the CLI guide](../.agents/skills/autoresearch/references/cli.md) for the exact contract.
