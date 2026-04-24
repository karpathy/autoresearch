"""Tribunal evaluation — STUB.

The full Tribunal evaluation pipeline (multi-judge scoring of research-loop
outputs) is NOT IMPLEMENTED in the v1.0.0 release. This module exists so that
callers expecting ``autoresearch.tribunal`` can import it and get a clear,
explicit error instead of an obscure ImportError.

Tracked in ROADMAP.md.
"""
from __future__ import annotations


class TribunalNotImplementedError(NotImplementedError):
    """Raised whenever Tribunal evaluation is invoked."""


def evaluate(*args, **kwargs):
    """Placeholder. Always raises :class:`TribunalNotImplementedError`."""
    raise TribunalNotImplementedError(
        "Tribunal evaluation is not implemented in autoresearch v1.0.0. "
        "See ROADMAP.md."
    )


IMPLEMENTED = False
