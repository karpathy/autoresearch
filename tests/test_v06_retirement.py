from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from autoresearch_runtime.archive import archive_idle_labs
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.lab import main


class RetiredCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.source = self.home / "projects" / "legacy" / "lab-a"
        self.source.mkdir(parents=True)
        (self.source / "lab.json").write_text(
            json.dumps({"state_dir": str(self.source), "source_root": str(self.root)}),
            encoding="utf-8",
        )
        (self.source / "state.json").write_text('{"setup_pending": true, "runs": []}\n', encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "AUTORESEARCH_HOME": str(self.home),
            "AUTORESEARCH_NO_DASHBOARD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def _tree_digest(self, root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def invoke(self, *args: object) -> tuple[int, dict[str, object]]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--headless", *(str(value) for value in args)])
        output = stderr.getvalue() if code else stdout.getvalue()
        self.assertTrue(output.strip(), f"CLI emitted no JSON for {args!r}")
        return code, json.loads(output)

    def assert_removed_without_writes(self, *args: object, error_code: str = "E_OPERATION_REMOVED") -> None:
        before = self._tree_digest(self.home)
        code, payload = self.invoke(*args)
        self.assertEqual(code, 2, payload)
        self.assertIs(payload.get("ok"), False, payload)
        error = payload.get("error")
        self.assertIsInstance(error, dict, payload)
        self.assertEqual(error.get("code"), error_code, payload)
        hint = str(error.get("hint", "")).casefold()
        self.assertIn("export", hint, payload)
        self.assertRegex(hint, r"new.?start|prepare", payload)
        self.assertEqual(self._tree_digest(self.home), before, "retired commands must not mutate lab data")

    def test_storage_migrate_apply_is_retired_without_database_or_lab_changes(self) -> None:
        self.assert_removed_without_writes(
            "storage", "migrate", "--root", self.home, "--lab", "lab-a", "--apply",
        )

    def test_storage_cleanup_apply_is_retired_without_lab_changes(self) -> None:
        self.assert_removed_without_writes(
            "storage", "cleanup", "--root", self.home, "--lab", "lab-a", "--apply",
        )

    def test_history_import_write_path_is_retired_without_reading_or_writing_inputs(self) -> None:
        missing_history = self.root / "not-read.json"
        self.assert_removed_without_writes(
            "--config", self.source / "lab.json", "history-import", "--file", missing_history,
        )

    def test_compat_migrate_is_retired_without_resolving_or_mutating_source(self) -> None:
        self.assert_removed_without_writes(
            "--config", self.source / "lab.json", "compat", "--migrate", "--apply",
        )

    def test_legacy_archive_move_is_retired_without_moving_source(self) -> None:
        self.assert_removed_without_writes(
            "archive", "--root", self.home, "--apply", error_code="E_LEGACY_READ_ONLY",
        )

    def test_catalog_repair_apply_refuses_legacy_metadata_writes(self) -> None:
        before = self._tree_digest(self.home)
        code, payload = self.invoke("catalog", "repair", "--root", self.home, "--apply")
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_LEGACY_READ_ONLY", payload)
        hint = str(payload["error"].get("hint", "")).casefold()
        self.assertIn("export", hint, payload)
        self.assertRegex(hint, r"prepare|new.?start", payload)
        self.assertEqual(self._tree_digest(self.home), before)

    def test_catalog_repair_noop_apply_is_still_a_read_only_refusal(self) -> None:
        canonical = self.root / "canonical-catalog-root"
        canonical.mkdir()
        catalog_path = canonical / "catalog.json"
        catalog_path.write_text(json.dumps({"schema": 1, "projects": [], "archived": []}), encoding="utf-8")
        before = self._tree_digest(canonical)

        code, payload = self.invoke("catalog", "repair", "--root", canonical, "--apply")

        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_LEGACY_READ_ONLY", payload)
        self.assertEqual(self._tree_digest(canonical), before)

    def test_archive_helper_cannot_move_legacy_folders(self) -> None:
        before = self._tree_digest(self.home)
        with self.assertRaises(LabError) as raised:
            archive_idle_labs(self.home, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(self._tree_digest(self.home), before)
        self.assertTrue(self.source.is_dir())

    def test_unknown_lab_selector_has_actionable_typed_error(self) -> None:
        code, payload = self.invoke("--lab", "missing-lab", "status")
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_LAB_NOT_FOUND", payload)
        hint = str(payload["error"].get("hint", "")).casefold()
        self.assertIn("--config", hint)
        self.assertNotIn("labs list", hint)

    def test_invalid_legacy_config_json_has_typed_config_error(self) -> None:
        invalid = self.root / "invalid-lab.json"
        invalid.write_text("{not json", encoding="utf-8")
        code, payload = self.invoke("--config", invalid, "guide", "--json")
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_CONFIG_INVALID", payload)

    def test_export_rejects_competing_legacy_and_sqlite_selectors(self) -> None:
        code, payload = self.invoke(
            "--config", self.source / "lab.json", "storage", "export",
            "--lab", "sqlite-ref", "--output", self.root / "unused.zip",
        )
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_USAGE_SELECTOR", payload)
        self.assertFalse((self.root / "unused.zip").exists())

    def test_reconcile_rejects_competing_global_and_subcommand_lab_selectors(self) -> None:
        before = self._tree_digest(self.home)
        code, payload = self.invoke(
            "--lab", "global-ref", "storage", "reconcile", "--lab", "nested-ref", "--apply",
        )
        self.assertEqual(code, 2, payload)
        self.assertEqual(payload["error"]["code"], "E_USAGE_SELECTOR", payload)
        self.assertEqual(self._tree_digest(self.home), before)

    def test_legacy_status_and_guide_offer_an_executable_archive_export(self) -> None:
        missing_source = self.root / "missing-project"
        config_path = self.source / "lab.json"
        config_path.write_text(json.dumps({
            "state_dir": ".",
            "source_root": str(missing_source),
            "pins": {"program": {"sha256": "stale"}},
        }), encoding="utf-8")
        (self.source / "state.json").write_text(
            json.dumps({"setup_pending": False, "runs": [], "baseline_id": "old-baseline"}),
            encoding="utf-8",
        )

        status_code, status = self.invoke("--config", config_path, "status")
        self.assertEqual(status_code, 0, status)
        summary = status["data"]
        self.assertTrue(summary["read_only"])
        self.assertEqual(summary["goal_status"], "ARCHIVE_ONLY")
        self.assertIsNone(summary["current_epoch"])
        self.assertFalse(summary["baseline_available"])
        self.assertEqual(summary["runs"], [])

        path_selector_code, path_selector = self.invoke("--lab", config_path, "status")
        self.assertEqual(path_selector_code, 0, path_selector)
        self.assertTrue(path_selector["data"]["read_only"])

        guide_code, guide = self.invoke("--config", config_path, "guide", "--json")
        self.assertEqual(guide_code, 0, guide)
        next_steps = guide["data"]["next_steps"]
        export_step = next(step for step in next_steps if step["action"] == "EXPORT_LEGACY_ARCHIVE")
        argv = export_step["argv"]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[argv.index("storage") + 1], "export")
        self.assertNotIn("--confirmed", argv)
        self.assertFalse(Path(argv[-1]).is_relative_to(self.source))

        completed = subprocess.run(
            argv, capture_output=True, text=True, env=os.environ.copy(),
            **support.hidden_process_kwargs(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["file_count"], 2)
        with ZipFile(argv[-1]) as archive:
            self.assertIn("archive-index.json", archive.namelist())
            self.assertEqual(archive.read("lab/lab.json"), config_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
