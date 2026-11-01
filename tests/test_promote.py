try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import json
import subprocess
import tempfile
import unittest
from autoresearch_runtime.store import _write_json
from pathlib import Path

from autoresearch_runtime.lab import (
    LabError,
    export_best,
    init_lab,
    load_config,
    load_state,
    promote_in_place,
    promote_to_branch,
    run_baseline,
    run_trial,
)

use_home = support.use_home


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = use_home(self.root / "home")
        self.addCleanup(self._restore_home)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "value.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "add", "value.txt"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "init"],
            cwd=self.source,
            check=True,
            **support.hidden_process_kwargs(),
        )
        self.opaque = self.root / "sample.bin"
        self.opaque.write_bytes(b"\x00")
        self.config_path = self.root / "lab.json"
        self._write_config()

    def _restore_home(self):
        if self.previous is None:
            import os

            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            import os

            os.environ["AUTORESEARCH_HOME"] = self.previous

    def _write_config(self, **overrides):
        payload = {
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["value.txt"],
            "editable_files": ["value.txt"],
            "input_files": [str(self.opaque)],
            "evaluator": ["{python}", str(self._evaluator_path()), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 1,
            "timeout_seconds": 5,
            "budget": {"max_seconds": 60, "max_trials": 3, "max_requests": 0, "max_cost_usd": 0},
        }
        payload.update(overrides)
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

    def _evaluator_path(self) -> Path:
        evaluator = self.root / "evaluator.py"
        if not evaluator.is_file():
            evaluator.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                "candidate, output = map(Path, sys.argv[1:3])\n"
                "score = float((candidate / 'value.txt').read_text())\n"
                "output.mkdir(parents=True, exist_ok=True)\n"
                "(output / 'result.json').write_text(json.dumps({'metrics': {'score': score},"
                " 'usage': {'requests': 0, 'cost_usd': 0}}))\n",
                encoding="utf-8",
            )
        return evaluator

    def _ready_lab(self):
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "value.txt"
        candidate.write_text("2", encoding="utf-8")
        run_trial(
            config,
            {"title": "Better", "observation": "higher", "expected_effect": "score up"},
        )
        return config

    def test_dirty_tree_blocks_in_place(self):
        config = self._ready_lab()
        exported = export_best(config)
        self.assertGreater(exported["bytes"], 0)
        (config.source_root / "value.txt").write_text("dirty", encoding="utf-8")
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_DIRTY")

    def test_crlf_patch_applies_in_outside_worktree(self):
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        target = config.state_dir / "candidate" / "value.txt"
        target.write_bytes(b"2\r\n")
        run_trial(
            config,
            {"title": "CRLF", "observation": "crlf", "expected_effect": "score"},
        )
        result = promote_to_branch(config, "crlf-branch")
        self.assertTrue(result["applied"], result)
        work = Path(result["worktree"])
        self.assertTrue(work.is_dir())
        applied = work / "value.txt"
        if not applied.is_file():
            applied = work / result.get("project_prefix", "") / "value.txt"
        self.assertEqual(applied.read_bytes(), b"2\r\n")

    def test_project_subdirectory_uses_project_prefix(self):
        repo = self.root / "repo"
        repo.mkdir()
        project = repo / "pkg" / "app"
        project.mkdir(parents=True)
        (project / "value.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "init"],
            cwd=repo,
            check=True,
            **support.hidden_process_kwargs(),
        )
        self.source = project
        self.config_path = self.root / "lab-sub.json"
        self._write_config(
            source_root=str(project.relative_to(self.root)),
            state_dir="state-sub",
        )
        config = load_config(self.config_path)
        init_lab(config)
        state = load_state(config)
        self.assertEqual(state["project_prefix"], "pkg/app")
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("3", encoding="utf-8")
        run_trial(
            config,
            {"title": "Sub", "observation": "sub", "expected_effect": "up"},
        )
        exported = export_best(config)
        self.assertEqual(exported["project_prefix"], "pkg/app")
        self.assertEqual(exported["apply_check"], "ok")
        promoted = promote_to_branch(config, "sub-branch")
        self.assertTrue(promoted["applied"], promoted)
        work_file = Path(promoted["worktree"]) / "pkg" / "app" / "value.txt"
        self.assertEqual(work_file.read_text(encoding="utf-8"), "3")

    def test_head_drift_refused_without_allow_drift(self):
        config = self._ready_lab()
        state = load_state(config)
        state["source_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.source_root, text=True,
            **support.hidden_process_kwargs(),
        ).strip()
        _write_json(config.state_dir / "state.json", state)
        (config.source_root / "note.txt").write_text("drift", encoding="utf-8")
        subprocess.run(["git", "add", "note.txt"], cwd=config.source_root, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "drift"],
            cwd=config.source_root,
            check=True,
            **support.hidden_process_kwargs(),
        )
        with self.assertRaises(LabError) as ctx:
            promote_to_branch(config, "drift-branch")
        self.assertEqual(ctx.exception.code, "E_DRIFT")

    def _git_out(self, *args):
        completed = subprocess.run(
            ["git", *args], cwd=self.source, check=True, capture_output=True, **support.hidden_process_kwargs(),
        )
        return completed.stdout.decode("utf-8", errors="replace")

    def _commit_source(self, message):
        subprocess.run(["git", "add", "-A"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", message],
            cwd=self.source, check=True, **support.hidden_process_kwargs(),
        )

    def _user_state(self):
        files = support.tree_digest(self.source)
        return {
            "head": self._git_out("rev-parse", "HEAD"),
            "status": self._git_out("status", "--porcelain=v1", "-z", "--untracked-files=all"),
            "cached": self._git_out("diff", "--cached", "--binary"),
            "files": {name: digest for name, digest in files.items() if not name.startswith(".git/")},
        }

    def test_in_place_conflict_leaves_all_user_work_untouched(self):
        (self.source / "notes.txt").write_bytes(b"notes v1")
        self._commit_source("add notes")
        config = self._ready_lab()
        (self.source / "value.txt").write_bytes(b"upstream")
        self._commit_source("upstream edit to the promoted file")
        (self.source / "notes.txt").write_bytes(b"notes edited by user")
        (self.source / "staged.txt").write_bytes(b"staged by user")
        subprocess.run(["git", "add", "staged.txt"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        (self.source / "scratch.txt").write_bytes(b"untracked by user")
        before = self._user_state()
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config, allow_drift=True)
        self.assertEqual(ctx.exception.code, "E_CONFLICT")
        self.assertIn("value.txt", str(ctx.exception))
        self.assertEqual(self._user_state(), before)

    def test_in_place_conflict_removes_files_the_patch_created(self):
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        del payload["editable_files"]
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate"
        (candidate / "value.txt").write_text("2", encoding="utf-8")
        (candidate / "extra.txt").write_bytes(b"created by candidate")
        run_trial(config, {"title": "Create", "observation": "new file", "expected_effect": "score up"})
        (self.source / "value.txt").write_bytes(b"upstream")
        self._commit_source("upstream edit")
        before = self._user_state()
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config, allow_drift=True)
        self.assertEqual(ctx.exception.code, "E_CONFLICT")
        self.assertFalse((self.source / "extra.txt").exists())
        self.assertEqual(self._user_state(), before)

    def test_in_place_conflict_restores_exact_bytes_under_autocrlf(self):
        (self.source / "value.txt").write_bytes(b"1\n")
        self._commit_source("baseline with a real newline")
        subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        config = self._ready_lab()
        (self.source / "value.txt").write_bytes(b"upstream\n")
        self._commit_source("upstream edit")
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config, allow_drift=True)
        self.assertEqual(ctx.exception.code, "E_CONFLICT")
        self.assertEqual((self.source / "value.txt").read_bytes(), b"upstream\n")

    def _subdirectory_lab(self, name, *, commit):
        repo = self.root / name
        project = repo / "pkg" / "app"
        project.mkdir(parents=True)
        (project / "value.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, **support.hidden_process_kwargs())
        if commit:
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, **support.hidden_process_kwargs())
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "init"],
                cwd=repo, check=True, **support.hidden_process_kwargs(),
            )
        self.source = project
        self.config_path = self.root / f"lab-{name}.json"
        self._write_config(source_root=str(project.relative_to(self.root)), state_dir=f"state-{name}")
        return load_config(self.config_path)

    def test_unborn_head_project_prefix_and_dirty_guard(self):
        config = self._subdirectory_lab("unborn-repo", commit=False)
        init_lab(config)
        self.assertEqual(load_state(config)["project_prefix"], "pkg/app")
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("3", encoding="utf-8")
        run_trial(config, {"title": "Unborn", "observation": "unborn", "expected_effect": "up"})
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_DIRTY")
        self.assertIn("pkg/app/value.txt", str(ctx.exception))
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")

    def test_stored_prefix_mismatch_fails_closed(self):
        config = self._subdirectory_lab("prefix-mismatch-repo", commit=True)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("3", encoding="utf-8")
        run_trial(config, {"title": "Mismatch", "observation": "prefix", "expected_effect": "up"})
        state = load_state(config)
        state["project_prefix"] = ""
        _write_json(config.state_dir / "state.json", state)
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_PREFIX_MISMATCH")
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")

    def test_unrelated_unmerged_file_does_not_block_clean_apply(self):
        (self.source / "conflict.txt").write_bytes(b"base\n")
        self._commit_source("base conflict file")
        subprocess.run(["git", "checkout", "-q", "-b", "side"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        (self.source / "conflict.txt").write_bytes(b"side\n")
        self._commit_source("side edit")
        subprocess.run(["git", "checkout", "-q", "-"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        (self.source / "conflict.txt").write_bytes(b"main\n")
        self._commit_source("main edit")
        config = self._ready_lab()
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "merge", "--no-edit", "side"],
            cwd=self.source, capture_output=True, check=False, **support.hidden_process_kwargs(),
        )
        self.assertIn("UU conflict.txt", self._git_out("status", "--porcelain=v1"))
        conflict_bytes = (self.source / "conflict.txt").read_bytes()
        result = promote_in_place(config)
        self.assertTrue(result["applied"], result)
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "2")
        self.assertIn("UU conflict.txt", self._git_out("status", "--porcelain=v1"))
        self.assertEqual((self.source / "conflict.txt").read_bytes(), conflict_bytes)

    def test_staged_rename_marks_source_and_destination_dirty(self):
        from autoresearch_runtime import promote

        subprocess.run(["git", "mv", "value.txt", "moved.txt"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        self.assertEqual(promote._dirty_paths(self.source, {"value.txt"}, ""), ["value.txt"])
        self.assertEqual(promote._dirty_paths(self.source, {"moved.txt"}, ""), ["moved.txt"])

    def test_restore_puts_back_snapshot_and_removes_patch_created_entries(self):
        from autoresearch_runtime import promote

        (self.source / "keep.txt").write_bytes(b"line1\nline2\n")
        (self.source / "flip").write_bytes(b"f\n")
        (self.source / "dirtarget").mkdir()
        (self.source / "sub").mkdir()
        (self.source / "sub" / "other.txt").write_bytes(b"o")
        targets = ["keep.txt", "flip", "dirtarget", "sub/new.txt", "made/new.txt"]
        snapshot = promote._snapshot_targets(self.source, targets)
        # Simulate a failed apply: rewrite, replace a file with a directory, a directory with a file, and create entries.
        (self.source / "keep.txt").write_bytes(b"patched")
        (self.source / "flip").unlink()
        (self.source / "flip").mkdir()
        (self.source / "flip" / "inner.txt").write_bytes(b"inner")
        (self.source / "dirtarget").rmdir()
        (self.source / "dirtarget").write_bytes(b"now a file")
        (self.source / "sub" / "new.txt").write_bytes(b"created")
        (self.source / "made").mkdir()
        (self.source / "made" / "new.txt").write_bytes(b"created")
        promote._restore_applied_paths(self.source, targets, snapshot)
        self.assertEqual((self.source / "keep.txt").read_bytes(), b"line1\nline2\n")
        self.assertEqual((self.source / "flip").read_bytes(), b"f\n")
        self.assertTrue((self.source / "dirtarget").is_dir())
        self.assertFalse((self.source / "sub" / "new.txt").exists())
        self.assertEqual((self.source / "sub" / "other.txt").read_bytes(), b"o")
        self.assertFalse((self.source / "made").exists())

    def test_restore_skips_unchanged_read_only_file_and_restores_the_rest(self):
        import os
        import stat
        from autoresearch_runtime import promote

        (self.source / "a.txt").write_bytes(b"a-original")
        (self.source / "ro.txt").write_bytes(b"ro-original")
        (self.source / "z.txt").write_bytes(b"z-original")
        os.chmod(self.source / "ro.txt", stat.S_IREAD)
        self.addCleanup(os.chmod, self.source / "ro.txt", stat.S_IREAD | stat.S_IWRITE)
        targets = ["a.txt", "ro.txt", "z.txt"]
        snapshot = promote._snapshot_targets(self.source, targets)
        (self.source / "a.txt").write_bytes(b"a-patched")
        (self.source / "z.txt").write_bytes(b"z-patched")
        promote._restore_applied_paths(self.source, targets, snapshot)
        self.assertEqual((self.source / "a.txt").read_bytes(), b"a-original")
        self.assertEqual((self.source / "z.txt").read_bytes(), b"z-original")
        self.assertEqual((self.source / "ro.txt").read_bytes(), b"ro-original")

    def test_failed_restore_path_does_not_block_others_and_is_named_alone(self):
        from unittest.mock import patch
        from autoresearch_runtime import promote

        for name in ("a.txt", "bad.txt", "z.txt"):
            (self.source / name).write_bytes(f"{name}-original".encode("utf-8"))
        targets = ["a.txt", "bad.txt", "z.txt"]
        snapshot = promote._snapshot_targets(self.source, targets)
        for name in targets:
            (self.source / name).write_bytes(f"{name}-patched".encode("utf-8"))
        real_write = Path.write_bytes

        def failing_write(path, data):
            if path.name == "bad.txt":
                raise PermissionError("simulated write failure")
            return real_write(path, data)

        with patch.object(Path, "write_bytes", failing_write):
            with self.assertRaises(LabError) as ctx:
                promote._restore_applied_paths(self.source, targets, snapshot)
        self.assertEqual(ctx.exception.code, "E_RESTORE")
        self.assertIn("bad.txt", str(ctx.exception))
        self.assertNotIn("a.txt", str(ctx.exception))
        self.assertNotIn("z.txt", str(ctx.exception))
        self.assertEqual((self.source / "a.txt").read_bytes(), b"a.txt-original")
        self.assertEqual((self.source / "z.txt").read_bytes(), b"z.txt-original")
        self.assertEqual((self.source / "bad.txt").read_bytes(), b"bad.txt-patched")

    def test_unreadable_target_fails_before_apply_with_e_source_read(self):
        from unittest.mock import patch
        from autoresearch_runtime import promote

        config = self._ready_lab()
        real_entry_kind = promote._entry_kind

        def unreadable(path):
            if path.name == "value.txt":
                raise PermissionError("simulated read failure")
            return real_entry_kind(path)

        with patch("autoresearch_runtime.promote._entry_kind", side_effect=unreadable):
            with self.assertRaises(LabError) as ctx:
                promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_SOURCE_READ")
        self.assertIn("value.txt", str(ctx.exception))
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")

    def test_index_reset_failure_still_restores_working_tree_and_names_index(self):
        from autoresearch_runtime import promote

        (self.source / "other.txt").write_bytes(b"o-original")
        targets = ["value.txt", "other.txt"]
        snapshot = promote._snapshot_targets(self.source, targets)
        (self.source / "value.txt").write_bytes(b"patched")
        (self.source / "other.txt").write_bytes(b"o-patched")
        subprocess.run(["git", "add", "value.txt"], cwd=self.source, check=True, **support.hidden_process_kwargs())
        lock = self.source / ".git" / "index.lock"
        lock.write_bytes(b"")
        self.addCleanup(lambda: lock.unlink(missing_ok=True))
        with self.assertRaises(LabError) as ctx:
            promote._restore_applied_paths(self.source, targets, snapshot)
        self.assertEqual(ctx.exception.code, "E_RESTORE")
        self.assertIn("index", str(ctx.exception))
        self.assertEqual((self.source / "value.txt").read_bytes(), b"1")
        self.assertEqual((self.source / "other.txt").read_bytes(), b"o-original")

    def test_unchanged_symlink_is_not_recreated(self):
        from unittest.mock import patch
        from autoresearch_runtime import promote

        with patch("autoresearch_runtime.promote._current_kind", return_value="link"), \
                patch("autoresearch_runtime.promote.os.readlink", return_value="value.txt"), \
                patch("autoresearch_runtime.promote._remove") as remove, \
                patch("autoresearch_runtime.promote.os.symlink") as symlink:
            promote._write_entry(self.source / "link.txt", ("link", "value.txt"))
        remove.assert_not_called()
        symlink.assert_not_called()

    def test_real_unchanged_symlink_is_not_recreated(self):
        import os
        from unittest.mock import patch
        from autoresearch_runtime import promote

        path = self.source / "link.txt"
        try:
            os.symlink("value.txt", path)
        except OSError:
            self.skipTest("symlink creation needs extra privilege on this platform")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        with patch("autoresearch_runtime.promote.os.symlink", wraps=os.symlink) as symlink:
            promote._write_entry(path, ("link", "value.txt"))
        symlink.assert_not_called()
        self.assertEqual(os.readlink(path), "value.txt")

    def test_in_place_success_never_creates_a_commit(self):
        config = self._ready_lab()
        self.assertEqual(export_best(config)["paths"], ["value.txt"])
        head = self._git_out("rev-parse", "HEAD")
        result = promote_in_place(config)
        self.assertTrue(result["applied"], result)
        self.assertEqual(self._git_out("rev-parse", "HEAD"), head)
        self.assertEqual(self._git_out("rev-list", "--count", "HEAD").strip(), "1")
        self.assertEqual(self._git_out("diff", "--cached", "--name-only").split(), ["value.txt"])
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "2")

    def test_dirty_target_detected_for_spaces_cyrillic_and_nested_untracked(self):
        (self.source / "my file.txt").write_bytes(b"1")
        (self.source / "файл.txt").write_bytes(b"1")
        self._commit_source("tracked files with spaces and Cyrillic names")
        (self.source / "newdir").mkdir()
        # Untracked and alone in its directory, so a plain status line shows only "?? newdir/".
        (self.source / "newdir" / "x.txt").write_bytes(b"1")
        targets = ["value.txt", "my file.txt", "файл.txt", "newdir/x.txt"]
        self._write_config(source_paths=targets, editable_files=targets)
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate"
        for name in targets:
            (candidate / name).write_bytes(b"2")
        run_trial(config, {"title": "Names", "observation": "names", "expected_effect": "score up"})
        (self.source / "my file.txt").write_bytes(b"user edit")
        (self.source / "файл.txt").write_bytes(b"user edit")
        (self.source / "newdir" / "x.txt").write_bytes(b"user edit")
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_DIRTY")
        for name in ("my file.txt", "файл.txt", "newdir/x.txt"):
            self.assertIn(name, str(ctx.exception))
        self.assertEqual((self.source / "newdir" / "x.txt").read_bytes(), b"user edit")

    def test_dirty_target_detected_with_project_prefix(self):
        repo = self.root / "prefix-repo"
        project = repo / "pkg" / "app"
        project.mkdir(parents=True)
        (project / "value.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "init"],
            cwd=repo, check=True, **support.hidden_process_kwargs(),
        )
        self.source = project
        self.config_path = self.root / "lab-prefix.json"
        self._write_config(source_root=str(project.relative_to(self.root)), state_dir="state-prefix")
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("3", encoding="utf-8")
        run_trial(config, {"title": "Prefix", "observation": "prefix", "expected_effect": "up"})
        (project / "value.txt").write_text("user edit", encoding="utf-8")
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "E_DIRTY")
        self.assertIn("pkg/app/value.txt", str(ctx.exception))
        self.assertEqual((project / "value.txt").read_text(encoding="utf-8"), "user edit")

    def test_git_status_failure_fails_closed(self):
        from unittest.mock import patch
        from autoresearch_runtime import promote

        config = self._ready_lab()
        real_source_git = promote.source_git

        def failing_status(project, *args, **kwargs):
            if args and args[0] == "status":
                raise LabError("git status: simulated failure")
            return real_source_git(project, *args, **kwargs)

        with patch("autoresearch_runtime.promote.source_git", side_effect=failing_status), \
                patch("autoresearch_runtime.promote._run_apply") as run_apply:
            with self.assertRaises(LabError) as ctx:
                promote_in_place(config)
            run_apply.assert_not_called()
        self.assertIn("simulated failure", str(ctx.exception))
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")

    def test_in_place_on_non_git_source_fails_closed(self):
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "value.txt").write_text("1", encoding="utf-8")
        self._write_config(source_root="plain")
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")
        run_trial(config, {"title": "Plain", "observation": "plain", "expected_effect": "up"})
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertIn("rev-parse", str(ctx.exception))
        self.assertEqual((plain / "value.txt").read_text(encoding="utf-8"), "1")

    def test_head_lookup_failure_fails_closed(self):
        from autoresearch_runtime.promote import _ensure_drift

        repo = self.root / "unborn"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, **support.hidden_process_kwargs())
        with self.assertRaises(LabError) as ctx:
            _ensure_drift({"source_head": "a" * 40}, repo, allow_drift=False)
        self.assertIn("rev-parse", str(ctx.exception))

    def test_to_branch_leaves_shared_git_config_unchanged(self):
        config = self._ready_lab()

        def local_config(key):
            completed = subprocess.run(
                ["git", "config", "--local", "--get", key], cwd=self.source, capture_output=True,
                check=False, **support.hidden_process_kwargs(),
            )
            return completed.returncode, completed.stdout

        before = local_config("core.autocrlf")
        result = promote_to_branch(config, "config-branch")
        self.assertTrue(result["applied"], result)
        self.assertEqual(local_config("core.autocrlf"), before)

    def test_non_editable_target_rejected_before_apply_on_both_paths(self):
        config = self._ready_lab()
        candidate = config.state_dir / "candidate"
        (candidate / "extra.txt").write_bytes(b"not editable")
        subprocess.run(["git", "add", "-A"], cwd=candidate, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=t@local", "commit", "-q", "-m", "touch non-editable"],
            cwd=candidate, check=True, **support.hidden_process_kwargs(),
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=candidate, capture_output=True, check=True, **support.hidden_process_kwargs(),
        ).stdout.decode("utf-8").strip()
        state = load_state(config)
        state["best_commit"] = head
        _write_json(config.state_dir / "state.json", state)
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "frozen_tampered")
        with self.assertRaises(LabError) as ctx:
            promote_to_branch(config, "frozen-branch")
        self.assertEqual(ctx.exception.code, "frozen_tampered")
        self.assertFalse((self.source / "extra.txt").exists())
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")
        self.assertEqual(self._git_out("branch", "--list", "frozen-branch").strip(), "")

    def test_tampered_input_rejected_before_apply_on_both_paths(self):
        config = self._ready_lab()
        self.opaque.write_bytes(b"\x01")
        with self.assertRaises(LabError) as ctx:
            promote_in_place(config)
        self.assertEqual(ctx.exception.code, "input_tampered")
        with self.assertRaises(LabError) as ctx:
            promote_to_branch(config, "tampered-branch")
        self.assertEqual(ctx.exception.code, "input_tampered")
        self.assertEqual((self.source / "value.txt").read_text(encoding="utf-8"), "1")
        self.assertEqual(self._git_out("branch", "--list", "tampered-branch").strip(), "")

    def test_restore_failure_raises_e_restore(self):
        from unittest.mock import patch
        from autoresearch_runtime import promote

        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"fatal: unable to write index")
        with patch("autoresearch_runtime.promote.subprocess.run", return_value=failed):
            with self.assertRaises(LabError) as ctx:
                promote._restore_applied_paths(self.source, ["value.txt"], {"value.txt": ("file", b"1", 0o644)})
        self.assertEqual(ctx.exception.code, "E_RESTORE")
        self.assertIn("value.txt", str(ctx.exception))

    def test_conflict_and_restore_errors_exit_with_four(self):
        from autoresearch_runtime.envelope import exit_code_for

        self.assertEqual(exit_code_for("E_CONFLICT"), 4)
        self.assertEqual(exit_code_for("E_RESTORE"), 4)
        self.assertEqual(exit_code_for("E_PREFIX_MISMATCH"), 4)
        self.assertEqual(exit_code_for("E_SOURCE_READ"), 4)


if __name__ == "__main__":
    unittest.main()
