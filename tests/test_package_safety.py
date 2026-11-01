try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.integrity import check_skill, manifest_payload


SKILL = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch"


def _load_installer(source: Path):
    spec = importlib.util.spec_from_file_location("autoresearch_installer_test", source / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    return installer


def _package_copy(destination: Path) -> Path:
    shutil.copytree(SKILL, destination)
    return destination


class PackageSafetyTest(unittest.TestCase):
    def test_source_and_python_package_versions_match(self):
        version = (SKILL / "VERSION").read_text(encoding="utf-8").strip()
        project_metadata = (SKILL.parents[2] / "pyproject.toml").read_text(encoding="utf-8")

        self.assertEqual(version, "0.8.0")
        self.assertRegex(project_metadata, rf'(?m)^version = "{version}"$')

    def test_license_is_mit_and_shipped_with_the_skill(self):
        root_license = (SKILL.parents[2] / "LICENSE").read_bytes()
        skill_license = (SKILL / "LICENSE").read_bytes()
        skill_frontmatter = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
        project_metadata = (SKILL.parents[2] / "pyproject.toml").read_text(encoding="utf-8")

        self.assertEqual(root_license, skill_license)
        self.assertIn(b"MIT License", root_license)
        self.assertNotIn(b"Andrej Karpathy", root_license)
        self.assertIn(b"Copyright (c) 2026 nightwishrus", root_license)
        self.assertIn("LICENSE", manifest_payload(SKILL)["files"])
        self.assertRegex(skill_frontmatter, r"(?m)^license: MIT$")
        self.assertRegex(project_metadata, r'(?m)^license = "MIT"$')

    def test_build_requirement_supports_pep639_license_string(self):
        project_metadata = (SKILL.parents[2] / "pyproject.toml").read_text(encoding="utf-8")

        # The PEP 639 string form of `license` fails to build with setuptools older than 77.0.1.
        self.assertRegex(project_metadata, r'(?m)^requires = \["setuptools>=77\.0\.1"\]$')

    def test_readme_credits_upstream(self):
        readme = (SKILL.parents[2] / "README.md").read_text(encoding="utf-8")

        self.assertIn("https://github.com/karpathy/autoresearch", readme)
        self.assertIn("Andrej Karpathy", readme)

    def test_readme_points_to_current_migration_notes(self):
        readme = (SKILL.parents[2] / "README.md").read_text(encoding="utf-8")
        notes = SKILL.parents[2] / "docs" / "v08-migration.md"

        self.assertIn("docs/v08-migration.md", readme)
        self.assertTrue(notes.is_file())
        for marker in ("E_CONTRACT_SPEND_LIMIT", "E_HARNESS_POLICY_STALE", "does not limit spending"):
            self.assertIn(marker, notes.read_text(encoding="utf-8"))

    def test_readme_keeps_the_0_7_migration_notes(self):
        readme = (SKILL.parents[2] / "README.md").read_text(encoding="utf-8")
        notes = SKILL.parents[2] / "docs" / "v07-migration.md"

        self.assertIn("docs/v07-migration.md", readme)
        self.assertTrue(notes.is_file())
        for marker in ("E_CONFLICT", "direct_sync", "AUTORESEARCH_HOME/pins"):
            self.assertIn(marker, notes.read_text(encoding="utf-8"))

    def test_readme_drops_unsupported_fragment_token_claim(self):
        readme = (SKILL.parents[2] / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("fragment token", readme)

    def test_manifest_covers_reference_docs_and_installer_but_not_gitnexus(self):
        manifest = manifest_payload(SKILL)
        files = set(manifest["files"])

        self.assertIn("install.py", files)
        self.assertIn("references/safety.md", files)
        self.assertIn("scripts/autoresearch_runtime/integrity.py", files)
        self.assertFalse(any(path.startswith(".gitnexus/") for path in files))

    def test_relative_symlink_through_directory_alias_is_recognized_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            alias = root / "source-alias"
            install_parent = root / "installed"
            install_parent.mkdir()
            destination = install_parent / "autoresearch"
            try:
                alias.symlink_to(source, target_is_directory=True)
                destination.symlink_to(Path("..") / alias.name, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            foreign_cwd = root / "foreign"
            foreign_cwd.mkdir()
            installer = _load_installer(source)
            manifest, check_skill = installer._source_manifest(source)
            previous_cwd = Path.cwd()
            try:
                os.chdir(foreign_cwd)
                plan = installer._prepare_install(source, destination, manifest, check_skill, link=True)
            finally:
                os.chdir(previous_cwd)

            self.assertTrue(plan["already_linked"])
            self.assertIsNone(plan["stage"])

    def test_manifest_excludes_nested_gitnexus_and_cache_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "skill"
            _package_copy(root)
            generated = {
                "scripts/.gitnexus/local.json": "index",
                "scripts/.cache/temp.json": "cache",
                "references/__pycache__/page.pyc": "bytecode",
            }
            for name, content in generated.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            payload = __import__("autoresearch_runtime.integrity", fromlist=["manifest_payload"]).manifest_payload(root)

            self.assertFalse(set(payload["files"]) & set(generated))

    def test_install_generates_lf_manifest_from_source_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target

            installer.install("agents")

            result = check_skill(target)
            self.assertTrue(result["ok"])
            self.assertTrue(result["managed"])
            self.assertNotIn(b"\r", (target / "skill.manifest.json").read_bytes())

    def test_install_upgrades_existing_schema_2_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target
            installer.install("agents")

            (source / "SKILL.md").write_text("# Updated skill\n", encoding="utf-8")
            installer.install("agents")

            self.assertEqual((target / "SKILL.md").read_text(encoding="utf-8"), "# Updated skill\n")
            self.assertTrue(check_skill(target)["ok"])

    def test_install_refuses_truncated_source_before_touching_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target
            installer.install("agents")
            before = (target / "SKILL.md").read_bytes()
            required = (
                "SKILL.md",
                "VERSION",
                "scripts/autoresearch.py",
                "scripts/autoresearch_runtime/__init__.py",
                "scripts/autoresearch_runtime/integrity.py",
            )
            for relative in required:
                with self.subTest(missing=relative):
                    saved = (source / relative).read_bytes()
                    (source / relative).unlink()
                    try:
                        with self.assertRaises(installer.InstallError) as caught:
                            installer.install("agents")
                    finally:
                        (source / relative).write_bytes(saved)
                    self.assertIn(relative, str(caught.exception))
                    self.assertEqual((target / "SKILL.md").read_bytes(), before)

    def test_link_install_works_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            try:
                (root / "probe").symlink_to(source, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target

            installer.install("agents", link=True)

            self.assertTrue(target.is_symlink())
            self.assertTrue(check_skill(target)["ok"])

    def test_failed_activation_restores_previous_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target
            installer.install("agents")
            original_skill = (target / "SKILL.md").read_bytes()

            source_manifest = installer._source_manifest
            checks = 0

            def reject_after_staging(package_root):
                manifest, _ = source_manifest(package_root)

                def check(path):
                    nonlocal checks
                    checks += 1
                    if checks == 2:  # Stage verifies; activation must roll back on failure.
                        return {"ok": False, "error": "simulated post-install verification failure"}
                    return check_skill(path)

                return manifest, check

            installer._source_manifest = reject_after_staging

            with self.assertRaisesRegex(ValueError, "verification"):
                installer.install("agents")

            self.assertEqual((target / "SKILL.md").read_bytes(), original_skill)
            self.assertFalse(list(target.parent.glob(".autoresearch.autoresearch-failed-*")))

    def test_all_install_failure_does_not_move_an_already_linked_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            first = root / "installed" / "agents"
            first.mkdir(parents=True)
            marker = first / "existing-target.txt"
            marker.write_text("keep", encoding="utf-8")
            second = root / "installed" / "cursor"
            installer = _load_installer(source)
            installer.TARGETS = {"agents": first, "cursor": second}
            original_activate = installer._activate

            def prepare(source_path, destination, manifest, check_skill, *, link=False):
                if destination == first:
                    return {"destination": first, "stage": None, "backup": None, "already_linked": True}
                return {"destination": second, "stage": root / "unused-stage", "backup": None,
                        "already_linked": False, "link": link}

            def activate(plan, check_skill):
                if plan["destination"] == second:
                    raise installer.InstallError("simulated later target activation failure")
                return original_activate(plan, check_skill)

            with patch.object(installer, "_prepare_install", prepare), \
                 patch.object(installer, "_activate", activate):
                with self.assertRaisesRegex(ValueError, "later target activation failure"):
                    installer.install("all", link=True)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertFalse(list(first.parent.glob(".agents.autoresearch-failed-*")))

    def test_all_install_failure_removes_newly_activated_target_after_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            first = root / "installed" / "agents"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = first
            installer.install("agents")
            old_skill = (first / "SKILL.md").read_bytes()
            second = root / "installed" / "cursor"
            installer.TARGETS = {"agents": first, "cursor": second}
            original_prepare = installer._prepare_install
            original_activate = installer._activate

            def prepare(source_path, destination, manifest, check_skill, *, link=False):
                if destination == second:
                    return {"destination": second, "stage": root / "unused-stage", "backup": None,
                            "already_linked": False, "link": link}
                return original_prepare(source_path, destination, manifest, check_skill, link=link)

            def activate(plan, check_skill):
                if plan["destination"] == second:
                    raise installer.InstallError("simulated later target activation failure")
                return original_activate(plan, check_skill)

            with patch.object(installer, "_prepare_install", prepare), \
                 patch.object(installer, "_activate", activate):
                with self.assertRaisesRegex(ValueError, "later target activation failure"):
                    installer.install("all")

            self.assertEqual((first / "SKILL.md").read_bytes(), old_skill)
            self.assertFalse(list(first.parent.glob(".agents.autoresearch-failed-*")))
            self.assertFalse(list(first.parent.glob(".agents.autoresearch-backup-*")))

    def test_all_link_failure_preserves_real_link_to_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            first = root / "installed" / "agents"
            first.parent.mkdir(parents=True)
            try:
                first.symlink_to(source, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            second = root / "installed" / "cursor"
            installer = _load_installer(source)
            installer.TARGETS = {"agents": first, "cursor": second}
            original_prepare = installer._prepare_install
            original_activate = installer._activate

            def prepare(source_path, destination, manifest, check_skill, *, link=False):
                if destination == first:
                    return original_prepare(source_path, destination, manifest, check_skill, link=link)
                return {"destination": destination, "stage": root / "unused-stage", "backup": None,
                        "already_linked": False, "link": link}

            def activate(plan, check_skill):
                if plan["destination"] == second:
                    raise installer.InstallError("simulated later target activation failure")
                return original_activate(plan, check_skill)

            with patch.object(installer, "_prepare_install", prepare), \
                 patch.object(installer, "_activate", activate):
                with self.assertRaisesRegex(ValueError, "later target activation failure"):
                    installer.install("all", link=True)

            self.assertTrue(first.is_symlink())
            self.assertEqual(first.resolve(), source.resolve())
            self.assertTrue((source / "SKILL.md").is_file())

    def test_install_refuses_to_replace_its_verified_source_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _package_copy(root / "source")
            installer = _load_installer(source)
            installer.TARGETS = {"agents": source}
            before = {
                path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*") if path.is_file()
            }

            with self.assertRaisesRegex(ValueError, "source|overlap"):
                installer.install("agents")

            after = {
                path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*") if path.is_file()
            }
            self.assertEqual(after, before)

    def test_install_update_preserves_unowned_files_and_stops_on_modified_managed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target
            installer.install("agents")

            marker = target / "user-data.txt"
            marker.write_text("keep", encoding="utf-8")
            gitnexus = target / ".gitnexus" / "local-config.json"
            gitnexus.parent.mkdir()
            gitnexus.write_text('{"user":true}', encoding="utf-8")
            before = (target / "SKILL.md").read_bytes()
            with self.assertRaisesRegex(ValueError, "unmanaged|conflict"):
                installer.install("agents")
            self.assertEqual((target / "SKILL.md").read_bytes(), before)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(gitnexus.read_text(encoding="utf-8"), '{"user":true}')

            marker.unlink()
            gitnexus.unlink()
            gitnexus.parent.rmdir()

            managed = target / "SKILL.md"
            managed.write_text(managed.read_text(encoding="utf-8") + "\nuser edit\n", encoding="utf-8")
            modified_contents = managed.read_text(encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflict|modified|changed"):
                installer.install("agents")

            self.assertEqual(managed.read_text(encoding="utf-8"), modified_contents)
            self.assertFalse(marker.exists())

    def test_install_update_stops_on_unowned_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            target = root / "installed" / "autoresearch"
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target
            installer.install("agents")
            user_directory = target / "user-data"
            user_directory.mkdir()

            with self.assertRaisesRegex(ValueError, "unmanaged|conflict"):
                installer.install("agents")

            self.assertTrue(user_directory.is_dir())

    def test_install_rejects_legacy_manifest_without_reference_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _package_copy(source)
            target = root / "installed" / "autoresearch"
            target.mkdir(parents=True)
            legacy_files = {}
            legacy_digest = hashlib.sha256()
            for name in ("SKILL.md", "VERSION"):
                content = (source / name).read_bytes()
                (target / name).write_bytes(content)
                legacy_files[name] = hashlib.sha256(content).hexdigest()
                legacy_digest.update(Path(name).name.encode("utf-8"))
                legacy_digest.update(content)
            (target / "skill.manifest.json").write_text(json.dumps({
                "schema": 1,
                "version": "0.4.1",
                "sha256": legacy_digest.hexdigest(),
                "files": legacy_files,
            }), encoding="utf-8")
            before = {name: (target / name).read_bytes() for name in legacy_files}
            installer = _load_installer(source)
            installer.TARGETS["agents"] = target

            with self.assertRaisesRegex(ValueError, "schema 1|upgrade|resolve manually|reference"):
                installer.install("agents")

            self.assertEqual({name: (target / name).read_bytes() for name in legacy_files}, before)
            self.assertFalse((target / "references" / "harness-profile.md").exists())


if __name__ == "__main__":
    unittest.main()
