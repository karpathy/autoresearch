# Security Policy

## Supported versions

Security fixes are made on the latest tagged release and on `main`. Older
releases may not receive backports. Upgrade to the latest release before
reporting an issue that may already be fixed.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting](https://github.com/mattmccarthy11/researchfs/security/advisories/new)
to share the affected version, impact, reproduction steps, and any proposed
mitigation. Remove API keys, database contents, paper corpora, and other
sensitive material from reports and attachments.

Maintainers will aim to acknowledge a report within five business days, assess
severity and scope, and coordinate a fix and disclosure timeline with the
reporter. Please allow time for a patched release before public disclosure.

## Security boundaries

- The stdlib HTTP API is an unauthenticated local integration surface. Keep it
  bound to `127.0.0.1`; do not expose it directly to an untrusted network.
- ResearchFS stores ingested artifacts, prompts, experiment data, and provenance
  in local SQLite databases. Operators own database permissions, backups,
  retention, and encryption at rest.
- Optional LLM extraction sends selected artifact text to the configured model
  provider. Review provider terms and data-handling requirements before enabling
  it on sensitive material.
- The NanoChat fixture under `benchmarks/nanochat/` is outside the lightweight
  core runtime and has its own GPU, Modal, and training dependencies.

General support requests and non-sensitive defects belong in the public issue
tracker.
