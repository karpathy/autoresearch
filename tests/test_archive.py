try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.archive import archive_idle_labs, inspect_archive_candidates
from autoresearch_runtime.errors import LabError


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / ".labs"
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()

    def make_lab(self, name, *, pending=False, baseline=False, status=None):
        path = self.root / "projects" / "project-id" / name
        (path / "runs").mkdir(parents=True)
        (path / "lab.json").write_text(json.dumps({"source_root": str(self.project)}), encoding="utf-8")
        (path / "state.json").write_text(json.dumps({
            "setup_pending": pending, "baseline_id": "baseline" if baseline else None,
            "inputs": [],
        }), encoding="utf-8")
        if status:
            run = path / "runs" / "baseline"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "id": "baseline", "status": status, "runs": [],
            }), encoding="utf-8")
        return path

    def test_preview_reports_running_legacy_lab_and_apply_is_read_only(self):
        failed = self.make_lab("failed", pending=False, baseline=False, status="crash")
        self.make_lab("active", pending=True)
        run = self.root / "projects" / "project-id" / "active" / "runs" / "trial-0001"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"id": "trial-0001", "status": "running"}), encoding="utf-8")

        result = archive_idle_labs(self.root)
        self.assertEqual(result["blocked"], "A lab is still running")
        self.assertTrue(failed.exists())
        self.assertEqual(inspect_archive_candidates(self.root)["eligible"], [])
        with self.assertRaises(LabError) as raised:
            archive_idle_labs(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertTrue(failed.exists())

    def test_preview_lists_unreferenced_failed_labs_but_apply_does_not_move_them(self):
        failed = self.make_lab("failed", status="crash")
        kept = self.make_lab("has-baseline", baseline=True, status="accepted")
        result = archive_idle_labs(self.root)
        archived_path = self.root / "archive" / "project-id" / "failed"
        self.assertEqual([item["path"] for item in result["eligible"]], [str(failed.resolve())])
        self.assertEqual(result["archived"], [])
        with self.assertRaises(LabError) as raised:
            archive_idle_labs(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertTrue(failed.exists())
        self.assertFalse(archived_path.exists())
        self.assertTrue(kept.exists())

    def test_preview_and_apply_preserve_legacy_tree(self):
        failed = self.make_lab("failed", status="crash")

        preview = archive_idle_labs(self.root)

        self.assertEqual(preview["archived"], [])
        self.assertTrue(failed.exists())
        with self.assertRaises(LabError) as raised:
            archive_idle_labs(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertTrue(failed.exists())

    def test_cross_lab_input_reference_prevents_archival(self):
        failed = self.make_lab("failed", status="crash")
        referring = self.make_lab("referring", pending=True)
        raw = {"input_files": ["../failed/candidate/dataset.sql"]}
        (referring / "lab.json").write_text(json.dumps(raw), encoding="utf-8")
        result = archive_idle_labs(self.root)
        self.assertNotIn(str(failed.resolve()), [item["path"] for item in result["eligible"]])
        self.assertTrue(failed.exists())

    def test_catalog_repair_is_previewed_idempotent_and_does_not_move_lab_files(self):
        from autoresearch_runtime.catalog import repair_catalog

        active = self.make_lab("active", baseline=True, status="accepted")
        archived = self.root / "archive" / "project-id" / "old"
        archived.mkdir(parents=True)
        (archived / "lab.json").write_text(json.dumps({"source_root": str(self.project)}), encoding="utf-8")
        sentinel = active / "runs" / "result.bin"
        sentinel.write_bytes(b"preserve me")
        catalog = {
            "projects": [{"id": "project-id", "name": "Project", "source_root": str(self.project), "labs": [
                {"id": "active", "path": "active", "name": "Readable goal"},
                {"id": "old", "path": "old", "name": "Old duplicate"},
            ]}],
            "archived": [{"project": "Project", "path": "old", "archived_path": str(archived), "status": "failed"}],
        }
        (self.root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

        preview = repair_catalog(self.root)
        self.assertFalse(preview["applied"])
        self.assertEqual(preview["active_count"], 1)
        self.assertEqual(preview["archived_count"], 1)
        self.assertEqual(json.loads((self.root / "catalog.json").read_text())["projects"][0]["labs"][0]["path"], "active")

        before = support.tree_digest(self.root)
        with self.assertRaises(LabError) as raised:
            repair_catalog(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.root), before)
        self.assertTrue(sentinel.is_file())
        self.assertFalse((self.root / "internal" / "migrations").exists())

    def test_catalog_repair_refuses_to_apply_while_any_lab_is_running(self):
        from autoresearch_runtime.catalog import repair_catalog

        active = self.make_lab("active", pending=True)
        run = active / "runs" / "trial-0001"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"id": "trial-0001", "status": "running"}), encoding="utf-8")
        preview = repair_catalog(self.root)
        self.assertIn("running", preview["blocked"].lower())
        before = support.tree_digest(self.root)
        with self.assertRaises(LabError) as raised:
            repair_catalog(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.root), before)
        self.assertFalse((self.root / "internal" / "migrations").exists())

    def test_catalog_repair_noop_apply_refuses_existing_canonical_legacy_catalog(self):
        from autoresearch_runtime.catalog import repair_catalog

        canonical = self.root / "canonical"
        canonical.mkdir(parents=True)
        catalog_path = canonical / "catalog.json"
        catalog_path.write_text(json.dumps({"schema": 1, "projects": [], "archived": []}), encoding="utf-8")

        preview = repair_catalog(canonical)
        self.assertEqual(preview["changes"], [])
        before = support.tree_digest(canonical)
        with self.assertRaises(LabError) as raised:
            repair_catalog(canonical, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(canonical), before)

    def test_catalog_repair_restores_goal_from_saved_brief_instead_of_hash(self):
        from autoresearch_runtime.catalog import repair_catalog

        active = self.make_lab("contract-abc123", baseline=True, status="accepted")
        (active / "lab-meta.json").write_text(json.dumps({
            "goal": "contract-abc123", "contract_id": "contract-abc123", "primary_metric": "F1",
        }), encoding="utf-8")
        (active / "program.snapshot.md").write_text(
            "# Program\n\n## Цель\n\nЧто улучшить: `Raise F1 without reducing Accuracy`\n",
            encoding="utf-8",
        )
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": "project-id", "name": "Project", "source_root": str(self.project),
            "labs": [{"id": "contract-abc123", "path": "contract-abc123", "name": "contract-abc123"}],
        }], "archived": []}), encoding="utf-8")

        preview = repair_catalog(self.root)
        self.assertIn("restore_goal", [change["action"] for change in preview["changes"]])
        before = support.tree_digest(self.root)
        with self.assertRaises(LabError) as raised:
            repair_catalog(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.root), before)

    def test_catalog_repair_preview_preserves_registered_legacy_fingerprints(self):
        from autoresearch_runtime.catalog import repair_catalog

        project_dir = self.root / "projects" / "sample-id"
        for contract_id in ("0d05784e5355", "dbdd0e4bc512"):
            path = project_dir / contract_id
            (path / "runs" / "baseline").mkdir(parents=True)
            (path / "lab.json").write_text(json.dumps({"source_root": str(self.project)}), encoding="utf-8")
            (path / "state.json").write_text(json.dumps({"setup_pending": False, "baseline_id": "baseline"}), encoding="utf-8")
            manifest = path / "runs" / "baseline" / "manifest.json"
            manifest.write_text(json.dumps({"id": "baseline", "status": "accepted", "runs": []}), encoding="utf-8")
        catalog = {"projects": [{"id": "sample-id", "name": "sample-project", "labs": [
            {"id": "0d05784e5355", "path": "0d05784e5355"},
            {"id": "dbdd0e4bc512", "path": "dbdd0e4bc512"},
        ]}], "archived": []}
        (self.root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        manifests = [(self.root / "projects" / "sample-id" / lab / "runs" / "baseline" / "manifest.json")
                     for lab in ("0d05784e5355", "dbdd0e4bc512")]
        before = [path.read_bytes() for path in manifests]
        catalog_before = (self.root / "catalog.json").read_bytes()

        preview = repair_catalog(self.root)
        self.assertFalse(preview["applied"])
        self.assertNotIn("mark_incomparable", [change["action"] for change in preview["changes"]])
        tree_before_apply = support.tree_digest(self.root)
        with self.assertRaises(LabError) as raised:
            repair_catalog(self.root, apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.root), tree_before_apply)
        self.assertEqual((self.root / "catalog.json").read_bytes(), catalog_before)
        self.assertEqual([path.read_bytes() for path in manifests], before)


if __name__ == "__main__":
    unittest.main()
