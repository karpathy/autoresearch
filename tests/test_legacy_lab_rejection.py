try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime import config as config_module
from autoresearch_runtime.cli import main
from autoresearch_runtime.errors import LabError

DRAFT = Path(__file__).resolve().parent / "fixtures" / "draft-config"


class LegacyLabRejectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name) / "home"
        self.lab = Path(self.temp.name) / "old-lab"
        shutil.copytree(DRAFT, self.lab)
        self.previous = support.use_home(self.home)
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        import os
        if self.previous is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.previous

    def test_a_real_lab_json_without_a_sqlite_lab_is_rejected_and_left_untouched(self):
        before = support.tree_digest(self.lab)
        # The shim in support.py loads drafts; call the runtime loader directly.
        loader = config_module._load_config
        with self.assertRaises(LabError) as caught:
            loader(self.lab / "lab.json")
        self.assertEqual(caught.exception.code, "E_LEGACY_LAB")
        message = str(caught.exception)
        self.assertIn("no longer supported", message)
        self.assertIn("v0.8.0", message)
        self.assertEqual(support.tree_digest(self.lab), before)

    def test_cli_reports_e_legacy_lab_for_every_command_that_loads_the_lab(self):
        before = support.tree_digest(self.lab)
        for command in (["status"], ["guide", "--json"], ["baseline"]):
            with self.subTest(command=command):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--config", str(self.lab / "lab.json"), "--headless", *command])
                self.assertNotEqual(code, 0)
                self.assertIn("E_LEGACY_LAB", out.getvalue() + err.getvalue())
        self.assertEqual(support.tree_digest(self.lab), before)

    def test_a_missing_lab_json_is_not_reported_as_a_legacy_lab(self):
        with self.assertRaises(LabError) as caught:
            config_module._load_config(self.lab / "absent" / "lab.json")
        self.assertEqual(caught.exception.code, "E_LAB_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
