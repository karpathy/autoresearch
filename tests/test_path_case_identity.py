try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.paths import _path_key, home_root_id, project_id


class PathCaseIdentityTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows legacy keys used Unicode casefold")
    def test_windows_path_keys_preserve_existing_casefold_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td) / "straße", Path(td) / "strasse"]
            old_keys = [str(path.resolve()).casefold() for path in paths]

            self.assertEqual(old_keys[0], old_keys[1])
            for path, old_key in zip(paths, old_keys):
                self.assertEqual(_path_key(path), old_key)
                self.assertEqual(
                    project_id(path).rsplit("-", 1)[-1],
                    hashlib.sha256(old_key.encode("utf-8")).hexdigest()[:10],
                )
            self.assertEqual(
                home_root_id(paths[0]),
                hashlib.sha256(old_keys[0].encode("utf-8")).hexdigest()[:16],
            )

    def test_project_ids_follow_current_platform_path_case_rules(self):
        with tempfile.TemporaryDirectory() as td:
            upper = Path(td) / "Project"
            lower = Path(td) / "project"
            expected_distinct = os.path.normcase(str(upper.resolve())) != os.path.normcase(str(lower.resolve()))
            self.assertEqual(project_id(upper) != project_id(lower), expected_distinct)

    def test_project_ids_use_platform_normalization_instead_of_casefold(self):
        # Simulate a case-sensitive platform even when this suite runs on Windows.
        path_type = type(Path())
        upper_path = Path("root/Project")
        lower_path = Path("root/project")
        with patch.object(Path, "resolve", autospec=True, side_effect=lambda path: path):
            with patch("autoresearch_runtime.paths.Path", path_type):
                with patch("autoresearch_runtime.paths.os.name", "posix"):
                    with patch("autoresearch_runtime.paths.os.path.normcase", side_effect=lambda value: value):
                        upper = project_id(upper_path)
                        lower = project_id(lower_path)
        self.assertNotEqual(upper, lower)



if __name__ == "__main__":
    unittest.main()
