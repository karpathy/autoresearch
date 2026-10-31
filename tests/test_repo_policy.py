"""Repository policy: generic rules for staged content, configured project rules, and tracked paths that clone on Windows."""

import contextlib
import io
import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import repo_policy
except ImportError:
    from tests import repo_policy

REPO = Path(__file__).resolve().parents[1]
SAMPLE_TOKEN = "forbidden-token"
SAMPLE_RULES = [("sample rule", re.compile(SAMPLE_TOKEN, re.IGNORECASE))]


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _run_cli(argv, repo):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = repo_policy.main(argv, repo=repo)
    return code, out.getvalue(), err.getvalue()


class RepoPolicyTest(unittest.TestCase):
    def test_tracked_paths_fit_windows_clone_roots(self):
        self.assertEqual(repo_policy.long_paths(repo_policy.tracked_paths(REPO)), [])

    def test_fixtures_are_stored_as_exact_bytes(self):
        fixtures = [
            path for path in repo_policy.tracked_paths(REPO)
            if path.startswith("tests/fixtures/")
        ]
        self.assertTrue(fixtures)
        result = subprocess.run(
            ["git", "check-attr", "text", "--", *fixtures],
            cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8",
        )
        expected = {f"{path}: text: unset" for path in fixtures}
        self.assertEqual(set(result.stdout.splitlines()), expected)

    def test_staged_files_contain_no_generic_or_configured_private_data(self):
        rules = repo_policy.default_private_rules(REPO)
        self.assertEqual(repo_policy.tracked_violations(REPO, rules), [])

    def test_stub_account_names_are_allowed_and_real_profiles_are_flagged(self):
        profile = "\\".join(["C:", "Users", "alice", "notes"])
        unix_profile = "/".join(["", "Users", "alice", "notes"])
        self.assertEqual(repo_policy.scan_bytes(b"C:\\Users\\user\\notes"), [])
        self.assertEqual(repo_policy.scan_bytes(b"/home/runner/work"), [])
        self.assertEqual(repo_policy.scan_bytes(profile.encode()), ["windows user profile"])
        self.assertEqual(repo_policy.scan_bytes(unix_profile.encode()), ["unix home directory"])

    def test_private_key_block_is_flagged(self):
        self.assertEqual(repo_policy.scan_bytes(("PRIVATE" + " KEY").encode()), ["private key"])

    def test_secret_tokens_are_flagged_by_shape_not_by_bare_prefix(self):
        self.assertEqual(repo_policy.scan_bytes(("sk-" + "a" * 24).encode()), ["openai-style API key"])
        self.assertEqual(repo_policy.scan_bytes(("ghp_" + "b" * 36).encode()), ["GitHub token"])
        self.assertEqual(repo_policy.scan_bytes(b"task-manager and sk-secret"), [])

    def test_configured_rules_match_case_insensitively_and_only_their_pattern(self):
        self.assertEqual(repo_policy.scan_bytes(SAMPLE_TOKEN.upper().encode(), SAMPLE_RULES), ["sample rule"])
        self.assertEqual(repo_policy.scan_bytes(b"task-manager", SAMPLE_RULES), [])

    def test_configured_rules_skip_binary_content_but_generic_rules_do_not(self):
        token = SAMPLE_TOKEN.encode()
        self.assertEqual(repo_policy.scan_bytes(b"\x89PNG\0" + token, SAMPLE_RULES), [])
        secret = ("PRIVATE" + " KEY").encode()
        self.assertEqual(repo_policy.scan_bytes(b"\0" + secret, SAMPLE_RULES), ["private key"])

    def test_private_rules_file_is_parsed_strictly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rules.txt"
            path.write_text("# comment\n\nsample rule\t" + SAMPLE_TOKEN + "\n", encoding="utf-8")
            rules = repo_policy.load_private_rules(path)
            self.assertEqual([name for name, _ in rules], ["sample rule"])
            self.assertEqual(repo_policy.scan_bytes(SAMPLE_TOKEN.upper().encode(), rules), ["sample rule"])
            for text, line in (("no tab here\n", 1), ("ok\tfine\nbad line\n", 2), ("name\t(unclosed\n", 1)):
                path.write_text(text, encoding="utf-8")
                with self.subTest(line=line):
                    with self.assertRaisesRegex(ValueError, f"line {line}"):
                        repo_policy.load_private_rules(path)
            path.write_text("name\t(unclosed\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                repo_policy.load_private_rules(path)
            self.assertNotIn("unclosed", str(caught.exception))

    def test_archives_are_scanned_by_member_name_and_content(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("ok/readme.txt", "nothing here")
            archive.writestr("data/notes.txt", "\\".join(["C:", "Users", "alice", "notes"]))
            archive.writestr(f"{SAMPLE_TOKEN}/member.txt", "clean")
        self.assertEqual(
            repo_policy.scan_member("fixture.zip", buffer.getvalue(), SAMPLE_RULES),
            ["sample rule", "windows user profile"],
        )
        self.assertEqual(repo_policy.scan_member("broken.zip", b"not a zip"), ["unreadable archive"])

    def test_redaction_never_echoes_the_matched_text(self):
        self.assertEqual(
            repo_policy.redact_matches(f"docs/{SAMPLE_TOKEN}/a.md", SAMPLE_RULES),
            "docs/[redacted]/a.md",
        )
        profile = "\\".join(["C:", "Users", "alice", "x"])
        self.assertNotIn("alice", repo_policy.redact_matches(profile))

    def test_history_mode_flags_blobs_that_only_exist_in_past_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            _git(repo, "init", "-q")
            (repo / "notes.txt").write_text(f"{SAMPLE_TOKEN}\n", encoding="utf-8")
            _git(repo, "add", "notes.txt")
            _git(repo, "commit", "-q", "-m", "add notes")
            _git(repo, "rm", "-q", "notes.txt")
            _git(repo, "commit", "-q", "-m", "remove notes")
            self.assertEqual(repo_policy.tracked_violations(repo, SAMPLE_RULES), [])
            self.assertEqual(repo_policy.history_violations(repo, SAMPLE_RULES), [("notes.txt", "sample rule")])

    def test_staged_content_is_scanned_not_the_working_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            _git(repo, "init", "-q")
            (repo / "notes.txt").write_text(f"{SAMPLE_TOKEN}\n", encoding="utf-8")
            _git(repo, "add", "notes.txt")
            (repo / "notes.txt").write_text("clean\n", encoding="utf-8")
            (repo / "gone.txt").write_text(f"{SAMPLE_TOKEN}\n", encoding="utf-8")
            _git(repo, "add", "gone.txt")
            (repo / "gone.txt").unlink()
            self.assertEqual(
                repo_policy.tracked_violations(repo, SAMPLE_RULES),
                [("gone.txt", "sample rule"), ("notes.txt", "sample rule")],
            )

    def test_cli_exit_codes_and_rule_status(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as rules_dir:
            repo = Path(temp)
            _git(repo, "init", "-q")
            (repo / "notes.txt").write_text(f"{SAMPLE_TOKEN}\n", encoding="utf-8")
            _git(repo, "add", "notes.txt")
            rules = Path(rules_dir) / "rules.txt"
            rules.write_text("sample rule\t" + SAMPLE_TOKEN + "\n", encoding="utf-8")

            code, out, err = _run_cli([], repo)
            self.assertEqual((code, out, err.strip()), (0, "", "private rules: none"))

            code, out, err = _run_cli(["--history"], repo)
            self.assertEqual(code, 2)
            self.assertIn("private rules: none", err)

            code, out, err = _run_cli(["--private-rules", str(rules)], repo)
            self.assertEqual(code, 1)
            self.assertEqual(err.strip(), "private rules: 1 loaded")
            self.assertEqual(out.strip(), "notes.txt: sample rule")
            self.assertNotIn(SAMPLE_TOKEN, out)

            code, _, _ = _run_cli(["--private-rules", str(Path(rules_dir) / "missing.txt")], repo)
            self.assertEqual(code, 2)

            malformed = Path(rules_dir) / "malformed.txt"
            malformed.write_text("no tab here\n", encoding="utf-8")
            code, _, err = _run_cli(["--private-rules", str(malformed)], repo)
            self.assertEqual(code, 2)
            self.assertIn("line 1", err)

    def test_cli_loads_the_default_rules_file_inside_the_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            _git(repo, "init", "-q")
            (repo / "notes.txt").write_text(f"{SAMPLE_TOKEN}\n", encoding="utf-8")
            _git(repo, "add", "notes.txt")
            (repo / ".private").mkdir()
            (repo / ".private" / "policy-rules.txt").write_text("sample rule\t" + SAMPLE_TOKEN + "\n", encoding="utf-8")
            code, out, err = _run_cli([], repo)
            self.assertEqual((code, err.strip()), (1, "private rules: 1 loaded"))
            self.assertEqual(out.strip(), "notes.txt: sample rule")


if __name__ == "__main__":
    unittest.main()
