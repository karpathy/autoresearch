from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.archive import inspect_archive_candidates
from autoresearch_runtime.commands.archive import run as run_archive
from autoresearch_runtime.errors import LabError


class LegacyArchiveBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "empty-root"
        self.root.mkdir()

    def test_archive_apply_fails_before_scanning_even_when_root_has_no_candidates(self) -> None:
        args = argparse.Namespace(root=self.root, apply=True)

        with patch(
            "autoresearch_runtime.archive.inspect_archive_candidates",
            wraps=inspect_archive_candidates,
        ) as inspect:
            with self.assertRaises(LabError) as caught:
                run_archive(args, None)

        self.assertEqual(caught.exception.code, "E_LEGACY_READ_ONLY")
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
