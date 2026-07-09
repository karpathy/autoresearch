# Contributing to ResearchFS

ResearchFS core is intentionally a small, synchronous Python and SQLite
product. Keep product changes dependency-light and keep benchmark/training work
inside `benchmarks/nanochat/`.

## Development setup

```bash
uv sync --group dev
uv run pytest -q
uv build
uv run python scripts/verify_distribution.py dist
```

Add focused regression tests under `tests/`. Use deterministic local fixtures,
explicit input validation, and the existing ResearchFS APIs instead of adding
raw SQL at product boundaries.

## Database changes

Append a new migration to `MIGRATIONS` in `researchfs/db.py`; never renumber,
rename, or rewrite a released migration. Update the canonical `SCHEMA`, cover
fresh and legacy upgrades, and document operator-visible changes in the
changelog.

## Pull requests

Keep changes scoped and explain the product behavior they alter. Include the
tests and package checks you ran. Never commit credentials, private corpora,
generated databases, run logs, or benchmark outputs.

Report vulnerabilities through the private process in `SECURITY.md`, not in a
public pull request or issue.
