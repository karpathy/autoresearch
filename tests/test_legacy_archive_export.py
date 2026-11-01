from __future__ import annotations

import contextlib

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_runtime import legacy_archive
from autoresearch_runtime.legacy_archive import (
    LegacyArchiveError,
    export_legacy_zip,
    read_legacy_lab,
)
from autoresearch_runtime.commands.common import legacy_read_only_status


class LegacyArchiveExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()  # the code under test works with resolved paths
        self.root = self.base / "legacy-lab"
        self.root.mkdir()
        self.config_path = self.root / "lab.json"
        self.state_dir = self.root
        self.source_root = self.base / "missing-source"
        self._write_config()

    def _write_config(self, *, state_dir: Path | None = None, raw: bytes | None = None) -> None:
        if raw is None:
            payload = {
                "state_dir": str(state_dir or self.state_dir),
                "source_root": str(self.source_root),
            }
            raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.config_path.write_bytes(raw)

    def _write_state_and_run(self) -> None:
        (self.state_dir / "state.json").write_text(
            json.dumps({"setup_pending": False, "baseline_id": "baseline"}),
            encoding="utf-8",
        )
        manifest = self.state_dir / "runs" / "trial-0007" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({
            "id": "trial-0007",
            "status": "accepted",
            "metrics": {"private_metric": "not part of the summary"},
        }), encoding="utf-8")

    def test_read_keeps_raw_config_and_returns_compact_runs_without_source_validation(self) -> None:
        raw = b'{ "state_dir" : ".", "source_root": "missing-project", "private": "hidden" }\n'
        self._write_config(raw=raw)
        self._write_state_and_run()

        lab = read_legacy_lab(self.config_path)

        self.assertEqual(lab["config_path"], self.config_path.resolve())
        self.assertEqual(lab["root"], self.root.resolve())
        self.assertEqual(lab["config_bytes"], raw)
        self.assertEqual(lab["config"]["private"], "hidden")
        self.assertEqual(lab["state_dir"], self.root.resolve())
        self.assertEqual(lab["state"]["baseline_id"], "baseline")
        self.assertEqual(lab["runs"], [{"id": "trial-0007", "status": "accepted", "kind": "trial"}])
        self.assertFalse(self.source_root.exists())

    def test_signature_ignores_ctime_only_where_stat_calls_disagree_on_it(self) -> None:
        common = {"st_mode": stat.S_IFREG, "st_size": 1, "st_mtime": 0, "st_ctime": 0,
                  "st_mtime_ns": 2, "st_dev": 4, "st_ino": 5}
        first = SimpleNamespace(st_ctime_ns=3, **common)
        second = SimpleNamespace(st_ctime_ns=9, **common)
        with patch.object(legacy_archive, "_SIGNATURE_USES_CTIME", False):
            self.assertEqual(legacy_archive._file_signature(first), legacy_archive._file_signature(second))
        with patch.object(legacy_archive, "_SIGNATURE_USES_CTIME", True):
            self.assertNotEqual(legacy_archive._file_signature(first), legacy_archive._file_signature(second))

    def test_run_summaries_preserve_budget_and_interrupted_producer_statuses(self) -> None:
        expected = {"budget": "baseline-budget", "interrupted": "trial-interrupted"}
        for status, run_id in expected.items():
            manifest = self.state_dir / "runs" / run_id / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"id": run_id, "status": status}), encoding="utf-8")

        lab = read_legacy_lab(self.config_path)

        self.assertEqual(
            {item["id"]: item["status"] for item in lab["runs"]},
            {run_id: status for status, run_id in expected.items()},
        )
        self.assertEqual(lab["run_manifest_warning_count"], 0)
        self.assertIsNone(lab["run_manifest_warning"])

    def test_run_summaries_and_status_are_bounded_but_keep_total_counts(self) -> None:
        run_total = 127
        running_indexes = {0, 60, 126}
        for index in range(run_total):
            run_id = f"trial-{index:04d}"
            manifest = self.state_dir / "runs" / run_id / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({
                "id": run_id,
                "status": "running" if index in running_indexes else "accepted",
            }), encoding="utf-8")

        lab = read_legacy_lab(self.config_path)

        self.assertEqual(len(lab["runs"]), 100)
        self.assertEqual(lab["run_count"], run_total)
        self.assertEqual(lab["trial_count"], run_total)
        self.assertEqual(lab["running_run_count"], len(running_indexes))
        self.assertEqual(lab["truncated_run_count"], run_total - 100)
        self.assertEqual(lab["runs"][0]["id"], "trial-0126")
        self.assertEqual(lab["runs"][-1]["id"], "trial-0027")

        with patch("autoresearch_runtime.commands.common.read_history", return_value={}):
            status = legacy_read_only_status(lab)
        self.assertEqual(status["run_count"], run_total)
        self.assertEqual(status["truncated_run_count"], run_total - 100)
        self.assertEqual(status["trials"], run_total)
        self.assertEqual(status["running_run_count"], len(running_indexes))
        self.assertEqual(len(status["runs"]), 100)

    def test_oversized_manifest_is_bounded_warned_and_still_exported_verbatim(self) -> None:
        manifest = self.state_dir / "runs" / "trial-oversized" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        raw = (json.dumps({"id": "trial-oversized", "status": "accepted", "private": "x" * (1024 * 1024)})
               + "\n").encode("utf-8")
        self.assertGreater(len(raw), 1024 * 1024)
        manifest.write_bytes(raw)
        original_read = os.read
        original_path_read = Path.read_bytes
        read_bytes = 0

        def count_manifest_reads(descriptor: int, size: int) -> bytes:
            nonlocal read_bytes
            block = original_read(descriptor, size)
            read_bytes += len(block)
            return block

        def count_path_reads(path: Path) -> bytes:
            nonlocal read_bytes
            data = original_path_read(path)
            if path == manifest:
                read_bytes += len(data)
            return data

        with patch.object(Path, "read_bytes", count_path_reads):
            with patch.object(legacy_archive.os, "read", count_manifest_reads):
                lab = read_legacy_lab(self.config_path)

        self.assertLessEqual(read_bytes, 1024 * 1024)
        self.assertEqual(lab["runs"], [{"id": "trial-oversized", "status": None, "kind": "trial"}])
        self.assertEqual(lab["run_manifest_warning_count"], 1)
        self.assertNotIn("x" * 128, lab["run_manifest_warning"])

        output = self.base / "oversized-manifest.zip"
        result = export_legacy_zip(self.config_path, output)
        with zipfile.ZipFile(output) as archive:
            archived = archive.read("lab/runs/trial-oversized/manifest.json")
            archive_index = json.loads(archive.read("archive-index.json"))
        self.assertEqual(archived, raw)
        payload = next(item for item in result["files"] if item["path"] == "runs/trial-oversized/manifest.json")
        indexed = next(item for item in archive_index["files"]
                       if item["path"] == "lab/runs/trial-oversized/manifest.json")
        self.assertEqual(payload["bytes"], len(raw))
        self.assertEqual(payload["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(indexed["bytes"], len(raw))
        self.assertEqual(indexed["sha256"], hashlib.sha256(raw).hexdigest())

    def test_bad_or_unknown_manifests_keep_safe_run_summaries_and_status_warning(self) -> None:
        malformed = self.state_dir / "runs" / "bad-json" / "manifest.json"
        malformed.parent.mkdir(parents=True)
        malformed.write_bytes(b'{"private-body":"do-not-echo-this-value"')

        unknown = self.state_dir / "runs" / "unknown-status" / "manifest.json"
        unknown.parent.mkdir(parents=True)
        unknown.write_text(json.dumps({
            "id": "safe-run-id",
            "status": "future-private-status",
            "reason": "do-not-echo-this-value",
        }), encoding="utf-8")

        unreadable = self.state_dir / "runs" / "permission-denied-folder" / "manifest.json"
        unreadable.parent.mkdir(parents=True)
        unreadable.write_text(json.dumps({"id": "hidden-run-id", "status": "accepted"}), encoding="utf-8")
        original_read_bytes = Path.read_bytes
        original_os_open = os.open

        def deny_unreadable_manifest(path: Path) -> bytes:
            if path == unreadable:
                raise PermissionError("private filesystem detail")
            return original_read_bytes(path)

        def deny_manifest_open(path, *args, **kwargs):
            if Path(path) == unreadable:
                raise PermissionError("private filesystem detail")
            return original_os_open(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", deny_unreadable_manifest):
            with patch.object(legacy_archive.os, "open", deny_manifest_open):
                lab = read_legacy_lab(self.config_path)

        summaries = {item["id"]: item for item in lab["runs"]}
        self.assertEqual(set(summaries), {"bad-json", "safe-run-id", "permission-denied-folder"})
        self.assertIsNone(summaries["bad-json"]["status"])
        self.assertIsNone(summaries["safe-run-id"]["status"])
        self.assertIsNone(summaries["permission-denied-folder"]["status"])
        self.assertEqual(lab["run_manifest_warning_count"], 3)
        warning = lab["run_manifest_warning"]
        self.assertIsInstance(warning, str)
        self.assertNotIn("bad-json", warning)
        self.assertNotIn("permission-denied-folder", warning)
        self.assertNotIn("do-not-echo-this-value", warning)
        self.assertNotIn("private filesystem detail", warning)

        with patch("autoresearch_runtime.commands.common.read_history", return_value={}):
            status = legacy_read_only_status(lab)
        self.assertEqual(status["run_manifest_warning_count"], 3)
        self.assertEqual(status["run_manifest_warning"], warning)

    def test_windows_absolute_state_paths_are_refused_on_posix(self) -> None:
        with patch.object(legacy_archive.os, "name", "posix"):
            for state_dir in (r"C:\outside\state", r"\\server\share\state"):
                with self.subTest(state_dir="drive" if state_dir.startswith("C:") else "unc"):
                    with self.assertRaises(LegacyArchiveError) as caught:
                        legacy_archive._state_directory({"state_dir": state_dir}, self.root)
                    self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_export_includes_hidden_git_and_candidate_files_with_full_hash_index(self) -> None:
        files = {
            "candidate/.git/HEAD": b"ref: refs/heads/main\n",
            "candidate/.git/objects/aa/blob": bytes(range(256)),
            "candidate/.hidden": b"hidden candidate data\n",
            ".private/notes.txt": b"private note\n",
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        output = self.base / "legacy-export.zip"

        result = export_legacy_zip(self.config_path, output)

        self.assertIsNone(read_legacy_lab(self.config_path)["state"])
        with zipfile.ZipFile(output) as archive:
            archived = {name: archive.read(name) for name in archive.namelist()}
        payload_paths = {f"lab/{name}": content for name, content in files.items()}
        payload_paths["lab/lab.json"] = self.config_path.read_bytes()
        self.assertEqual({name: archived[name] for name in payload_paths}, payload_paths)
        self.assertIn("archive-index.json", archived)
        self.assertFalse(self.source_root.exists())
        self.assertEqual(result["file_count"], len(payload_paths))
        self.assertEqual(result["byte_count"], sum(map(len, payload_paths.values())))
        self.assertEqual(result["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        index = {item["path"]: item for item in result["files"]}
        for name, content in {name.removeprefix("lab/"): content for name, content in payload_paths.items()}.items():
            self.assertEqual(index[name]["bytes"], len(content))
            self.assertEqual(index[name]["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["excluded_files"], [])

        archive_index = json.loads(archived["archive-index.json"])
        self.assertEqual(archive_index["format"], "autoresearch-legacy-archive-v1")
        self.assertEqual(set(archive_index), {"format", "files", "excluded_files"})
        archive_paths = {item["path"] for item in archive_index["files"]}
        self.assertEqual(archive_paths, set(payload_paths))
        self.assertTrue(all(not Path(path).is_absolute() for path in archive_paths))
        archive_files = {item["path"]: item for item in archive_index["files"]}
        for path, content in payload_paths.items():
            self.assertEqual(archive_files[path]["bytes"], len(content))
            self.assertEqual(archive_files[path]["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(archive_index["excluded_files"], [])

    def test_export_excludes_and_lists_only_exact_lock_artifacts(self) -> None:
        for name in (".lock", ".lock.guard", ".lock.owner.tmp"):
            (self.root / name).write_bytes(name.encode("ascii"))
        included = {"other.lock": b"keep", ".lockish": b"keep", ".lock.owner.json": b"keep"}
        for name, content in included.items():
            (self.root / name).write_bytes(content)
        output = self.base / "locks.zip"

        result = export_legacy_zip(self.config_path, output)

        with zipfile.ZipFile(output) as archive:
            names = set(archive.namelist())
            archive_index = json.loads(archive.read("archive-index.json"))
        self.assertTrue({f"lab/{name}" for name in included}.issubset(names))
        self.assertTrue({f"lab/{name}" for name in (".lock", ".lock.guard", ".lock.owner.tmp")}.isdisjoint(names))
        self.assertIn("archive-index.json", names)
        self.assertEqual(result["excluded_files"], [".lock", ".lock.guard", ".lock.owner.tmp"])
        self.assertEqual(archive_index["excluded_files"], [".lock", ".lock.guard", ".lock.owner.tmp"])
        self.assertEqual(
            {item["path"] for item in archive_index["files"]},
            {f"lab/{name}" for name in included} | {"lab/lab.json"},
        )

    def test_external_state_directory_is_refused(self) -> None:
        external = self.base / "external-state"
        external.mkdir()
        self._write_config(state_dir=external)

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(self.config_path)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_config_changed_after_metadata_read_cannot_be_archived_with_stale_config(self) -> None:
        external = self.base / "external-state"
        external.mkdir()
        changed_config = json.dumps({
            "state_dir": str(external),
            "source_root": str(self.source_root),
        }, indent=2).encode("utf-8") + b"\n"
        original_snapshot = legacy_archive._snapshot_tree
        changed = False

        def change_config_before_snapshot(root: Path):
            nonlocal changed
            if not changed:
                self.config_path.write_bytes(changed_config)
                changed = True
            return original_snapshot(root)

        output = self.base / "changed-config.zip"
        with patch.object(legacy_archive, "_snapshot_tree", side_effect=change_config_before_snapshot):
            with self.assertRaises(LegacyArchiveError) as caught:
                export_legacy_zip(self.config_path, output)

        self.assertIn(caught.exception.code, {"E_LEGACY_ARCHIVE_DRIFT", "E_LEGACY_ARCHIVE_UNSAFE"})
        self.assertFalse(output.exists())

    def test_late_nested_addition_during_final_scan_is_detected(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "present.txt").write_bytes(b"present before export")
        late_file = nested / "added-after-scan.txt"
        real_scandir = os.scandir
        real_snapshot = legacy_archive._snapshot_tree
        real_signature = legacy_archive._file_signature
        nested_scans = 0
        snapshot_number = 0

        def stable_directory_signature(info):
            signature = real_signature(info)
            if stat.S_ISDIR(info.st_mode):
                return signature[0], 0, 0, 0, signature[4], signature[5]
            return signature

        def count_snapshots(root: Path):
            nonlocal snapshot_number
            snapshot_number += 1
            return real_snapshot(root)

        @contextlib.contextmanager
        def scan_then_add(path):
            nonlocal nested_scans
            with real_scandir(path) as entries:
                names = list(entries)
            yield iter(names)
            if Path(path) == nested and snapshot_number == 2:
                nested_scans += 1
                if nested_scans == 1:
                    late_file.write_bytes(b"added after this directory was enumerated")

        output = self.base / "late-addition.zip"
        with patch.object(legacy_archive, "_snapshot_tree", side_effect=count_snapshots):
            with patch.object(legacy_archive, "_file_signature", side_effect=stable_directory_signature):
                with patch.object(legacy_archive.os, "scandir", side_effect=scan_then_add):
                    with self.assertRaises(LegacyArchiveError) as caught:
                        export_legacy_zip(self.config_path, output)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_DRIFT")
        self.assertTrue(late_file.is_file())
        self.assertFalse(output.exists())

    def test_invalid_config_error_does_not_echo_config_body(self) -> None:
        self._write_config(raw=b'{ "private": "do-not-echo-this-value"')

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(self.config_path)

        self.assertEqual(caught.exception.code, "E_CONFIG_INVALID")
        self.assertNotIn("do-not-echo-this-value", str(caught.exception))

    def test_symlink_inside_lab_is_refused(self) -> None:
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        linked = self.root / "linked.txt"
        try:
            linked.symlink_to(outside)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"symlink creation is not available: {type(error).__name__}")

        with self.assertRaises(LegacyArchiveError) as caught:
            export_legacy_zip(self.config_path, self.base / "linked.zip")

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_config_path_symlink_component_is_checked_before_resolution(self) -> None:
        alias = self.base / "lab-alias"
        self._create_directory_link(alias, self.root)

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(alias / self.config_path.name)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_reparse_runs_directory_is_refused_by_metadata_reader(self) -> None:
        external = self.base / "external-runs"
        external.mkdir()
        self._create_directory_link(self.state_dir / "runs", external)

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(self.config_path)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_reparse_run_directory_is_refused_by_metadata_reader(self) -> None:
        runs = self.state_dir / "runs"
        runs.mkdir()
        external = self.base / "external-run"
        external.mkdir()
        self._create_directory_link(runs / "trial-0001", external)

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(self.config_path)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_symlinked_manifest_is_refused_by_metadata_reader(self) -> None:
        manifest = self.state_dir / "runs" / "trial-0001" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        external = self.base / "external-manifest.json"
        external.write_text(json.dumps({"id": "trial-0001", "status": "accepted"}), encoding="utf-8")
        try:
            manifest.symlink_to(external)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"symlink creation is not available: {type(error).__name__}")

        with self.assertRaises(LegacyArchiveError) as caught:
            read_legacy_lab(self.config_path)

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_reparse_directory_inside_lab_is_refused_when_supported(self) -> None:
        outside = self.base / "external-folder"
        outside.mkdir()
        self._create_directory_link(self.root / "junction", outside)

        with self.assertRaises(LegacyArchiveError) as caught:
            export_legacy_zip(self.config_path, self.base / "junction.zip")

        self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_UNSAFE")

    def test_output_inside_lab_is_refused_and_existing_output_is_never_overwritten(self) -> None:
        contained = self.root / "inside.zip"
        with self.assertRaises(LegacyArchiveError) as caught:
            export_legacy_zip(self.config_path, contained)
        self.assertEqual(caught.exception.code, "E_OUTPUT_CONTAINED")
        self.assertFalse(contained.exists())

        existing = self.base / "already-there.zip"
        existing.write_bytes(b"keep this exact file")
        with self.assertRaises(LegacyArchiveError) as caught:
            export_legacy_zip(self.config_path, existing)
        self.assertEqual(caught.exception.code, "E_OUTPUT_EXISTS")
        self.assertEqual(existing.read_bytes(), b"keep this exact file")

    def test_dangling_output_symlink_is_refused_without_creating_its_target(self) -> None:
        output = self.base / "dangling.zip"
        target = self.base / "missing" / "target.zip"
        try:
            output.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is not available: {type(error).__name__}")

        with self.assertRaises(LegacyArchiveError) as caught:
            export_legacy_zip(self.config_path, output)

        self.assertEqual(caught.exception.code, "E_OUTPUT_EXISTS")
        self.assertTrue(output.is_symlink())
        self.assertFalse(target.exists())

    def test_output_created_during_publish_wins_the_no_overwrite_race(self) -> None:
        output = self.base / "raced.zip"
        real_link = os.link

        def create_racing_output(source, destination, *args, **kwargs):
            output.write_bytes(b"created by another process")
            return real_link(source, destination, *args, **kwargs)

        with patch.object(legacy_archive.os, "link", side_effect=create_racing_output):
            with self.assertRaises(LegacyArchiveError) as caught:
                export_legacy_zip(self.config_path, output)

        self.assertEqual(caught.exception.code, "E_OUTPUT_EXISTS")
        self.assertEqual(output.read_bytes(), b"created by another process")

    def test_export_rejects_source_tree_drift_before_publishing_output(self) -> None:
        cases = (
            ("added", lambda: (self.root / "zzz-added").write_bytes(b"new")),
            ("deleted", lambda: (self.root / "zzz-target").unlink()),
            ("type-changed", self._replace_target_with_directory),
            ("content-changed", lambda: (self.root / "zzz-target").write_bytes(b"changed")),
        )
        for label, mutate in cases:
            with self.subTest(change=label):
                for path in self.root.iterdir():
                    if path.name != "lab.json":
                        if path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                (self.root / "zzz-target").write_bytes(b"before")
                output = self.base / f"drift-{label}.zip"
                original_write = legacy_archive._write_zip_entry
                changed = False

                def write_then_mutate(*args, **kwargs):
                    nonlocal changed
                    result = original_write(*args, **kwargs)
                    if not changed:
                        changed = True
                        mutate()
                    return result

                with patch.object(legacy_archive, "_write_zip_entry", side_effect=write_then_mutate):
                    with self.assertRaises(LegacyArchiveError) as caught:
                        export_legacy_zip(self.config_path, output)
                self.assertEqual(caught.exception.code, "E_LEGACY_ARCHIVE_DRIFT")
                self.assertFalse(output.exists())

    def _replace_target_with_directory(self) -> None:
        target = self.root / "zzz-target"
        target.unlink()
        target.mkdir()

    def _create_directory_link(self, link: Path, target: Path) -> None:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                self.skipTest("directory junction creation is not available")
            self.addCleanup(os.rmdir, link)
            return
        try:
            link.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"directory symlink creation is not available: {type(error).__name__}")
        self.addCleanup(link.unlink)


if __name__ == "__main__":
    unittest.main()
