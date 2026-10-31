"""Machine-readable result every harness command prints."""

from __future__ import annotations

import json
import sys
from typing import Any

_EXIT_BY_CODE: dict[str, int] = {
    "E_LOCKED": 5,
    "E_BUSY": 5,
    "input_tampered": 4,
    "E_DRIFT": 4,
    "E_CONFLICT": 4,
    "E_RESTORE": 4,
    "E_PREFIX_MISMATCH": 4,
    "E_SOURCE_READ": 4,
    "E_SOURCE_WRITE": 4,
    "E_HELP": 4,
    "E_MEASUREMENT": 3,
    "E_EVALUATOR": 3,
    "E_BUDGET": 3,
}


def exit_code_for(error_code: str | None, *, default: int = 2) -> int:
    if not error_code:
        return default
    if error_code in _EXIT_BY_CODE:
        return _EXIT_BY_CODE[error_code]
    if error_code.startswith("E_USAGE") or error_code.startswith("E_CONTRACT"):
        return 2
    return default


def emit(
    command: str,
    data: Any,
    *,
    warnings: list[str] | None = None,
    next_steps: list[str] | None = None,
    cockpit: dict[str, Any] | None = None,
) -> int:
    _utf8()
    payload: dict[str, Any] = {
        "ok": True, "command": command, "data": data,
        "warnings": warnings or [], "next": next_steps or [], "error": None,
    }
    if cockpit is not None:
        payload["cockpit"] = cockpit
    print(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
    return 0


def fail(command: str, error: BaseException, *, code: str = "E_RUNTIME", exit_code: int | None = None) -> int:
    _utf8()
    hint = getattr(error, "hint", None) or ""
    error_code = getattr(error, "code", None) or code
    if exit_code is None:
        exit_code = exit_code_for(error_code)
    payload = json.dumps({
        "ok": False, "command": command, "data": None, "warnings": [], "next": [],
        "error": {"code": error_code, "message": str(error), "hint": hint},
    }, ensure_ascii=True, indent=2, default=str)
    stream = sys.stderr
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        payload.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        payload = payload.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
    print(payload, file=stream)
    return exit_code


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
