"""Defines LabError, the error type used by the lab runtime."""

from __future__ import annotations






class LabError(Exception):
    """A configuration or experiment contract failure."""

    def __init__(self, message: str, code: str | None = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint
