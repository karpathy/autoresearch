# Changelog

All notable changes are listed here in reverse chronological order. The repo's evolution is otherwise visible in `git log`; this file calls out user-facing or contract-affecting changes that an agent or maintainer should be aware of.

## 2026-05-08

### Documentation

- Added a full `docs/` documentation set: architecture (with mermaid diagrams), getting-started guide, agent-workflow guide, reference pages for `prepare.py` / `train.py` / `results.tsv`, internals pages for the GPT model and `MuonAdamW`, operations pages for forking and analysis, and an agent-skills index.
- Added `llms.txt` (concise LLM index) and `llms-full.txt` (single-file ingestion bundle) so LLM agents can navigate the docs without crawling the tree.
- Added this `CHANGELOG.md`.
- README updated to link the new docs set, `llms.txt`, and `llms-full.txt`.

No code or contract changes — `prepare.py`, `train.py`, and `program.md` are unchanged.

## Earlier history

The repo was small and pre-CHANGELOG. Notable shifts from `git log`:

- **Notable-forks section** added to README, with MacOS, MLX, Windows, and AMD ROCm forks linked.
- **Forking guidance** added to README for tuning on smaller hardware (TinyStories dataset suggestion, `vocab_size`, `MAX_SEQ_LEN`, `DEPTH`, `WINDOW_PATTERN="L"`).
- **Beginner's guide link** added to README for newcomers to neural networks.
- **`results.tsv` made untracked** (`.gitignore`) and `program.md` updated to reflect this — the file now survives `git reset`.
- **Agent loop hardening** in `program.md`: explicit "NEVER STOP" rule, explicit `> run.log 2>&1` redirection (no tee), explicit `tail -n 50 run.log` on crash, explicit timeout.
- **`download_data` honors `--download-workers`** (was hardcoded to 8).
- **NaN/exploding-loss fast-fail** added to `train.py`: `print("FAIL"); exit(1)` if loss is NaN or > 100.
- **Guard against infinite loop when no training shards exist**: `make_dataloader` asserts `len(parquet_paths) > 0`.

## Conventions

- Append entries to the top, dated `YYYY-MM-DD`.
- Group changes under headings like **Agent**, **Harness**, **Model**, **Documentation**, **Operations**.
- For breaking changes, lead with **Breaking** and call out who is affected and what they need to do.
- Cross-link the relevant doc page or PR.
