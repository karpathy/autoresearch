"""Helpers for starting child processes without showing Windows consoles."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def hardened_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Keep evaluator output in UTF-8 and stop it writing bytecode into the project."""
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if base:
        env.update(base)
    return env


def hidden_process_kwargs() -> dict[str, Any]:
    """Return platform-specific subprocess options that keep child consoles hidden."""
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }
