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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.dash_api import archive_lab, delete_lab, restore_lab
from autoresearch_runtime.lab import main as lab_main
from tests.test_legacy_fixture import materialize


class LegacyMutatorBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root / "legacy"
        self.config_path, self.program, self.project = materialize(self.case)
        self.state_dir = self.config_path.parent
        self.home = self.config_path.parents[3]
        self.env_patch = patch.dict(os.environ, {
            "AUTORESEARCH_HOME": str(self.home),
            "AUTORESEARCH_NO_DASHBOARD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def make_dashboard_lab(self, root: Path, *, archived: bool = False) -> tuple[Path, Path]:
        project_id = "project-a"
        lab_id = "lab-a"
        active = root / "projects" / project_id / lab_id
        archived_path = root / "archive" / project_id / lab_id
        lab = archived_path if archived else active
        lab.mkdir(parents=True)
        (lab / "lab.json").write_text(json.dumps({
            "state_dir": str(active), "source_root": str(self.project),
        }), encoding="utf-8")
        (lab / "state.json").write_text(json.dumps({"setup_pending": True}), encoding="utf-8")
        (lab / "lab-meta.json").write_text(json.dumps({"archived": archived}), encoding="utf-8")
        catalog = {"schema": 1, "projects": [], "archived": []}
        if archived:
            catalog["archived"].append({
                "project": "Project A", "project_id": project_id,
                "path": "projects/project-a/lab-a", "origin": "projects/project-a/lab-a",
                "archived_path": "archive/project-a/lab-a", "name": lab_id,
            })
        else:
            catalog["projects"].append({
                "id": project_id, "name": "Project A", "source_root": str(self.project),
                "labs": [{"id": lab_id, "path": "projects/project-a/lab-a"}],
            })
        (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        return active, archived_path

    def invoke(self, *args: object) -> tuple[int, dict[str, object]]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = lab_main(["--headless", *(str(arg) for arg in args)])
        output = stderr.getvalue() if code else stdout.getvalue()
        if not output.strip():
            output = stdout.getvalue() or stderr.getvalue()
        if not output.strip():
            raise AssertionError(f"CLI emitted no JSON (code={code}); command={args!r}")
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise AssertionError(
                f"CLI emitted non-JSON (code={code}); command={args!r}; stdout={stdout.getvalue()!r}; "
                f"stderr={stderr.getvalue()!r}"
            ) from error
        return code, payload

    def assert_error_code(self, result: tuple[int, dict[str, object]], expected: str) -> None:
        code, payload = result
        self.assertNotEqual(code, 0, payload)
        self.assertIs(payload.get("ok"), False, payload)
        error = payload.get("error")
        self.assertIsInstance(error, dict, payload)
        self.assertEqual(error.get("code"), expected, payload)

    def test_ingest_legacy_report_is_blocked_with_structured_error(self) -> None:
        report = self.root / "report.json"
        report.write_text(json.dumps({"metrics": {"score": 3}}), encoding="utf-8")

        self.assert_error_code(
            self.invoke("--config", self.config_path, "ingest", "--source", report),
            "E_LEGACY_READ_ONLY",
        )

    def test_history_import_legacy_write_is_blocked_with_structured_error(self) -> None:
        history = self.root / "history.json"
        (self.project / "NOTES.md").write_text("Prior experiment notes\n", encoding="utf-8")
        history.write_text(json.dumps({"hypotheses": [{"title": "Old result", "source": "NOTES.md"}]}),
                           encoding="utf-8")

        self.assert_error_code(
            self.invoke("--config", self.config_path, "history-import", "--file", history),
            "E_OPERATION_REMOVED",
        )

    def test_archive_apply_is_blocked_but_preview_remains_available(self) -> None:
        root = self.root / "archive-root"
        lab = root / "projects" / "project-a" / "lab-a"
        lab.mkdir(parents=True)
        (lab / "lab.json").write_text(json.dumps({"state_dir": str(lab), "source_root": str(self.project)}),
                                       encoding="utf-8")
        (lab / "state.json").write_text(json.dumps({"setup_pending": True}), encoding="utf-8")

        preview_code, preview = self.invoke("archive", "--root", root)
        self.assertEqual(preview_code, 0, preview)
        self.assertEqual(len(preview["data"]["eligible"]), 1, preview)
        self.assertEqual(preview["data"]["archived"], [], preview)

        self.assert_error_code(self.invoke("archive", "--root", root, "--apply"), "E_LEGACY_READ_ONLY")

    def test_dashboard_legacy_lifecycle_actions_are_blocked_with_structured_error(self) -> None:
        actions = (archive_lab, delete_lab)
        for action in actions:
            with self.subTest(action=action.__name__):
                local_root = self.root / f"dashboard-{action.__name__}"
                local_active, _ = self.make_dashboard_lab(local_root)
                with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(local_root)}):
                    status, payload = action(local_root, "project-a", "lab-a")
                self.assertEqual(status, 409, payload)
                self.assertEqual(payload.get("code"), "E_LEGACY_READ_ONLY", payload)
                self.assertTrue(local_active.is_dir(), "blocked dashboard actions must leave the lab in place")

        restore_root = self.root / "dashboard-restore"
        restore_active, restore_archived = self.make_dashboard_lab(restore_root, archived=True)
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(restore_root)}):
            status, payload = restore_lab(restore_root, "project-a", "lab-a")
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload.get("code"), "E_LEGACY_READ_ONLY", payload)
        self.assertTrue(restore_archived.is_dir(), "blocked restore must leave the archived lab in place")
        self.assertFalse(restore_active.exists())

    def test_vault_restore_keeps_explicit_source_write_semantics(self) -> None:
        source_target = self.project / "protected-input.txt"
        source_target.write_bytes(b"changed")
        original = b"vaulted source input"
        digest = hashlib.sha256(original).hexdigest()
        vault = self.state_dir / "vault"
        (vault / "blobs").mkdir(parents=True, exist_ok=True)
        (vault / "blobs" / digest).write_bytes(original)
        (vault / "index.json").write_text(json.dumps({"files": [{
            "path": str(source_target.resolve()), "sha256": digest, "bytes": len(original), "stored": True,
        }]}), encoding="utf-8")

        blocked_code, blocked = self.invoke(
            "--config", self.config_path, "vault", "restore", "--path", source_target,
        )
        self.assertNotEqual(blocked_code, 0, blocked)
        self.assertEqual(blocked["error"]["code"], "E_SOURCE_WRITE", blocked)
        self.assertEqual(source_target.read_bytes(), b"changed")

        restored_code, restored = self.invoke(
            "--config", self.config_path, "vault", "restore", "--path", source_target,
            "--allow-source-write",
        )
        self.assertEqual(restored_code, 0, restored)
        self.assertEqual(source_target.read_bytes(), original)

    def test_promote_patch_export_remains_available_for_legacy_lab(self) -> None:
        code, result = self.invoke("--config", self.config_path, "promote")
        self.assertEqual(code, 0, result)
        patch_path = Path(result["data"]["patch"])
        self.assertTrue(patch_path.is_file())
        self.assertGreater(result["data"]["bytes"], 0, result)

    def test_cases_report_and_global_root_ops_remain_available(self) -> None:
        left = self.root / "left"
        right = self.root / "right"
        left.mkdir()
        right.mkdir()
        (left / "answer.txt").write_text("1\n", encoding="utf-8")
        (right / "answer.txt").write_text("2\n", encoding="utf-8")
        diff_out = self.root / "exports" / "diff.json"

        diff_code, diff = self.invoke("cases", "diff", "--left", left, "--right", right, "--out", diff_out)
        self.assertEqual(diff_code, 0, diff)
        self.assertTrue(diff_out.is_file())

        report_code, report = self.invoke("--config", self.config_path, "report")
        self.assertEqual(report_code, 0, report)
        self.assertTrue((self.state_dir / "dashboard.html").is_file())

        extra_root = self.root / "extra-labs"
        extra_root.mkdir()
        roots_code, roots = self.invoke("roots", "add", extra_root)
        self.assertEqual(roots_code, 0, roots)
        self.assertIn(str(extra_root.resolve()), {item["path"] for item in roots["data"]["roots"]})

    def test_storage_migrate_and_cleanup_are_removed_without_writing_legacy_labs(self) -> None:
        before = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        for command in (
            ("storage", "migrate", "--root", self.home, "--apply"),
            ("storage", "cleanup", "--root", self.home, "--apply"),
        ):
            with self.subTest(command=command[1]):
                self.assert_error_code(self.invoke(*command), "E_OPERATION_REMOVED")
        after = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
