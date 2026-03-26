"""Tests for daemon config (task.get_daemon_config, write_daemon_config) and daemon timeout helpers."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pdca_system.daemon as daemon_module
import pdca_system.task as task_module
from pdca_system.task import (
    DEFAULT_DAEMON_CONFIG,
    get_daemon_config,
    write_daemon_config,
)


class DaemonConfigTaskTests(unittest.TestCase):
    """Tests for get_daemon_config and write_daemon_config (history/state/daemon_config.json)."""

    def test_get_daemon_config_when_no_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "daemon_config.json"
            self.assertFalse(config_path.exists())
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path):
                out = get_daemon_config()
            self.assertEqual(out["stuck_check_timeout_seconds"], 120)
            self.assertEqual(out["default_timeouts"], {"pd": 900, "ca": 3600, "direct": 3600})

    def test_write_daemon_config_creates_file_and_get_returns_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "daemon_config.json"
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path), patch.object(
                task_module, "ensure_queue_layout"
            ):
                write_daemon_config({"stuck_check_timeout_seconds": 90, "default_timeouts": {"pd": 600, "ca": 1800}})
            self.assertTrue(config_path.exists())
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path):
                out = get_daemon_config()
            self.assertEqual(out["stuck_check_timeout_seconds"], 90)
            self.assertEqual(out["default_timeouts"]["pd"], 600)
            self.assertEqual(out["default_timeouts"]["ca"], 1800)
            self.assertEqual(out["default_timeouts"]["direct"], 3600, "unchanged key keeps default")

    def test_write_daemon_config_partial_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "daemon_config.json"
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path), patch.object(
                task_module, "ensure_queue_layout"
            ):
                write_daemon_config({"stuck_check_timeout_seconds": 60})
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path):
                out = get_daemon_config()
            self.assertEqual(out["stuck_check_timeout_seconds"], 60)
            self.assertEqual(out["default_timeouts"], DEFAULT_DAEMON_CONFIG["default_timeouts"])

    def test_write_daemon_config_ignores_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "daemon_config.json"
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path), patch.object(
                task_module, "ensure_queue_layout"
            ):
                write_daemon_config({
                    "stuck_check_timeout_seconds": 0,
                    "default_timeouts": {"pd": -1, "ca": 3600},
                })
            with patch.object(task_module, "DAEMON_CONFIG_PATH", config_path):
                out = get_daemon_config()
            self.assertEqual(out["stuck_check_timeout_seconds"], 120, "zero not written")
            self.assertEqual(out["default_timeouts"]["pd"], 900, "negative not written")
            self.assertEqual(out["default_timeouts"]["ca"], 3600)


class DaemonTimeoutHelperTests(unittest.TestCase):
    """Tests for _get_timeout and _stuck_check_timeout_seconds using config/env."""

    def test_get_timeout_uses_config_when_no_env(self) -> None:
        with patch.object(daemon_module, "get_daemon_config") as mock_get:
            mock_get.return_value = {"default_timeouts": {"pd": 100, "ca": 200, "direct": 300}}
            self.assertEqual(daemon_module._get_timeout("pd"), 100)
            self.assertEqual(daemon_module._get_timeout("ca"), 200)
            self.assertEqual(daemon_module._get_timeout("direct"), 300)

    def test_get_timeout_env_overrides_config(self) -> None:
        with patch.object(daemon_module, "get_daemon_config") as mock_get:
            mock_get.return_value = {"default_timeouts": {"pd": 100, "ca": 200, "direct": 3600}}
            with patch.dict(os.environ, {"PDCA_TIMEOUT_PD": "999"}, clear=False):
                self.assertEqual(daemon_module._get_timeout("pd"), 999)
            with patch.dict(os.environ, {"PDCA_TIMEOUT_CA": "1234"}, clear=False):
                self.assertEqual(daemon_module._get_timeout("ca"), 1234)
            # When env is set, get_daemon_config is not called (early return)

    def test_stuck_check_timeout_uses_config_when_no_env(self) -> None:
        with patch.object(daemon_module, "get_daemon_config") as mock_get:
            mock_get.return_value = {"stuck_check_timeout_seconds": 90}
            self.assertEqual(daemon_module._stuck_check_timeout_seconds(), 90)

    def test_stuck_check_timeout_env_overrides_config(self) -> None:
        with patch.object(daemon_module, "get_daemon_config") as mock_get:
            mock_get.return_value = {"stuck_check_timeout_seconds": 90}
            with patch.dict(os.environ, {"PDCA_STUCK_CHECK_TIMEOUT": "45"}, clear=False):
                self.assertEqual(daemon_module._stuck_check_timeout_seconds(), 45)
            # When env is set, get_daemon_config is not called (early return)


if __name__ == "__main__":
    unittest.main()
