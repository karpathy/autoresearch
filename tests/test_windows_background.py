from __future__ import annotations

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_runtime import runtime_env
from autoresearch_runtime.runtime_env import hidden_process_kwargs


class HiddenProcessKwargsTest(unittest.TestCase):
    def test_non_windows_options_are_empty(self):
        with patch.object(runtime_env, "os", SimpleNamespace(name="posix")):
            self.assertEqual(hidden_process_kwargs(), {})

    def test_windows_options_hide_console_without_detaching_process(self):
        class StartupInfo:
            dwFlags = 0
            wShowWindow = None

        with patch.object(runtime_env, "os", SimpleNamespace(name="nt")), \
             patch.object(runtime_env.subprocess, "STARTUPINFO", StartupInfo, create=True), \
             patch.object(runtime_env.subprocess, "STARTF_USESHOWWINDOW", 0x00000001, create=True), \
             patch.object(runtime_env.subprocess, "SW_HIDE", 0, create=True), \
             patch.object(runtime_env.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True), \
             patch.object(runtime_env.subprocess, "DETACHED_PROCESS", 0x00000008, create=True):
            options = hidden_process_kwargs()

        self.assertEqual(options["creationflags"], 0x08000000)
        self.assertFalse(options["creationflags"] & 0x00000008)
        self.assertEqual(options["startupinfo"].dwFlags, 0x00000001)
        self.assertEqual(options["startupinfo"].wShowWindow, 0)

    def test_hidden_child_runs_with_local_process_api(self):
        code = "print('ready')"
        if os.name == "nt":
            code = (
                "import ctypes; "
                "get_console = ctypes.windll.kernel32.GetConsoleWindow; "
                "get_console.restype = ctypes.c_void_p; "
                "print('ready'); "
                "print(f'console={int(get_console() or 0)}')"
            )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            **hidden_process_kwargs(),
        )
        lines = result.stdout.splitlines()
        if os.name == "nt":
            self.assertEqual(lines, ["ready", "console=0"])
        else:
            self.assertEqual(lines, ["ready"])


if __name__ == "__main__":
    unittest.main()
