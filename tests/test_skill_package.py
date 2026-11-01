try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from autoresearch_runtime.integrity import check_skill, manifest_payload, skill_stamp

SKILL = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch"


def _write_minimal_skill(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (root / "VERSION").write_text("0.6.0\n", encoding="utf-8")
    (root / "install.py").write_text("# installer\n", encoding="utf-8")
    return root


class SkillPackageTest(unittest.TestCase):
    def test_source_checkout_without_manifest_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_minimal_skill(Path(directory) / "skill")

            result = check_skill(root)

            self.assertTrue(result["ok"])
            self.assertFalse(result["modified"])
            self.assertFalse(result["managed"])
            self.assertFalse((root / "skill.manifest.json").exists())

    def test_skill_stamp_ignores_stale_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_minimal_skill(Path(directory) / "skill")
            (root / "skill.manifest.json").write_text(
                '{"schema": 2, "version": "0", "sha256": "bad", "files": {}}', encoding="utf-8",
            )

            self.assertEqual(skill_stamp(root)[1], manifest_payload(root)["sha256"])
            self.assertTrue(check_skill(root)["modified"])

    def test_skill_stamp_changes_when_a_skill_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_minimal_skill(Path(directory) / "skill")
            before = skill_stamp(root)

            (root / "SKILL.md").write_text("# Changed skill\n", encoding="utf-8")

            self.assertNotEqual(skill_stamp(root), before)

    def test_egg_info_build_metadata_is_not_tracked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_minimal_skill(Path(directory) / "skill")
            (root / "scripts").mkdir()
            (root / "scripts" / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
            before = manifest_payload(root)

            egg_info = root / "scripts" / "x.egg-info"
            egg_info.mkdir()
            (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\n", encoding="utf-8")
            after = manifest_payload(root)

            self.assertNotIn("scripts/x.egg-info/PKG-INFO", after["files"])
            self.assertEqual(after["files"], before["files"])
            self.assertEqual(after["sha256"], before["sha256"])

    def test_manifest_is_not_tracked(self):
        repository = Path(__file__).resolve().parents[1]
        try:
            result = subprocess.run(
                ["git", "ls-files", "--", ".agents/skills/autoresearch/skill.manifest.json"],
                cwd=repository, capture_output=True, text=True, encoding="utf-8",
                **support.hidden_process_kwargs(),
            )
        except OSError:
            self.skipTest("git is not available")
        if result.returncode != 0:
            self.skipTest("not a git checkout")
        self.assertEqual(result.stdout.strip(), "")

    def test_stamp_is_stable_for_resume_of_same_build(self):
        from autoresearch_runtime.state import stamp_manifest

        stamped = stamp_manifest({})["skill_sha"]

        self.assertEqual(stamped, skill_stamp()[1])
        self.assertEqual(stamp_manifest({})["skill_sha"], stamped)

    def test_manifest_covers_subpackages_and_assets(self):
        manifest = manifest_payload(SKILL)
        names = set(manifest["files"])
        self.assertIn("scripts/autoresearch_runtime/commands/trial.py", names)
        self.assertIn("scripts/autoresearch_runtime/dashboard_assets/app.js", names)
        self.assertIn("scripts/autoresearch.py", names)

    def test_markdown_relative_links_resolve_in_skill_and_references(self):
        documents = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for destination in re.findall(r"!?\[[^\]]+\]\(([^)]+)\)", text):
                destination = destination.strip()
                if destination.startswith("<") and ">" in destination:
                    destination = destination[1:destination.index(">")]
                else:
                    destination = destination.split(maxsplit=1)[0]
                parsed = urlsplit(destination)
                if parsed.scheme or parsed.netloc or destination.startswith("/") or not parsed.path:
                    continue
                target = (document.parent / unquote(parsed.path)).resolve()
                self.assertTrue(
                    target.is_file() or target.is_dir(),
                    f"{document.relative_to(SKILL)} links to missing relative path {destination!r}",
                )

    def test_installed_copy_runs_doctor_from_a_foreign_directory_without_warnings(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("skill_install", SKILL / "install.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "skills" / "autoresearch"
            installer.TARGETS["agents"] = target
            installer.install("agents")
            foreign = Path(temp) / "some project"
            foreign.mkdir()
            env = dict(os.environ, AUTORESEARCH_HOME=str(Path(temp) / "home"), PYTHONDONTWRITEBYTECODE="1")
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, str(target / "scripts" / "autoresearch.py"), "doctor"],
                cwd=foreign, env=env, capture_output=True, text=True, encoding="utf-8",
                **support.hidden_process_kwargs(),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["warnings"], [])

    def test_skill_describes_intake_without_requiring_program_md(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        self.assertNotIn("described in program.md", frontmatter)
        self.assertIn("## Start or continue a lab", text)
        self.assertIn("Get confirmation before creating a lab or measuring", text)
        self.assertIn("Never create `program.md`", text)
        self.assertIn("path rules", text)
        brief = SKILL / "references" / "brief.md"
        self.assertTrue(brief.is_file())
        guide = brief.read_text(encoding="utf-8")
        for heading in ("Scout", "Questions", "Passport", "Paths", "Exact quote"):
            self.assertIn(heading, guide)
        self.assertIn("absolute paths", guide)

    def test_every_registered_command_is_documented_in_skill_or_references(self):
        from autoresearch_runtime.commands import COMMANDS

        documents = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
        text = "\n".join(document.read_text(encoding="utf-8") for document in documents)
        missing = []
        for module in COMMANDS:
            name = module.__name__.rsplit(".", 1)[-1]
            public = {"catalog_cmd": "catalog", "history_import": "history-import"}.get(name, name)
            if not re.search(rf"(?<![\w-]){re.escape(public)}(?![\w-])", text):
                missing.append(public)
        self.assertEqual(missing, [], f"SKILL.md and references do not mention: {missing}")


if __name__ == "__main__":
    unittest.main()
