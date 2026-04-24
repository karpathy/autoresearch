# autoresearch ROADMAP

This document lists what is deliberately **missing or incomplete** in the
v1.0.0 release. It exists so users can make informed choices before
depending on this package.

## Not implemented in v1.0.0

### Tribunal evaluation (NOT IMPLEMENTED)
The multi-judge Tribunal evaluation pipeline that scores autonomous-research
outputs is **not present** in this release. Only a stub lives at
`autoresearch/tribunal.py`; calling `autoresearch.tribunal.evaluate()` raises
`TribunalNotImplementedError`.

Planned for a future release. Design notes tracked separately.

### P2PCLAW Silicon integration (PARTIAL / INERT)
`silicon/grid_generator.py` is a **placeholder**. The Chess-Grid topology
builder referenced in `program.md` does not produce a usable knowledge grid
— generating the grid and reading from it are both stubbed. `silicon/` is
shipped as-is for transparency but should not be relied upon.

Full integration requires:
* a real grid generator,
* a stable consumer API in `train.py`/`toy_train.py`, and
* a reproducible P2PCLAW-backed knowledge source.

### Robust run-metrics parsing (PARTIAL)
The legacy `train.py` emits a `---`/`key: value` block on stdout and callers
parsed it with brittle regexes. v1.0.0 introduces a **JSON sidecar** (see
`autoresearch.metrics.write_metrics` / `read_metrics`) that the toy-train
path writes today. The legacy `train.py` has **not yet been migrated** to
also emit the sidecar — that's the next step. The regex parser
(`parse_stdout_summary`) is retained as a fallback.

### Railway P2PCLAW API (OPTIONAL, not bundled)
Scripts like `p2pclaw_publisher.py` and `train_with_p2pclaw.py` assume a live
Railway-hosted API. They are excluded from the installable package. The
core training loop, CLI, and tests work **without** network access.

## Done in v1.0.0

* Installable package layout (`autoresearch/`) with `pyproject.toml`.
* CPU-friendly nanoGPT-style model (`autoresearch.model.GPT`).
* Karpathy-style char tokenizer (`autoresearch.tokenizer.CharTokenizer`).
* Tiny training smoke-test (`autoresearch toy-train`) — no GPU required.
* CLI (`autoresearch info | config | toy-train`).
* Structured JSON metrics sidecar (replacement for stdout regex).
* Offline pytest suite (no CUDA, no network).
* Apache-2.0 license.
