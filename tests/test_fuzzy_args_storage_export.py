from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import unittest

from autoresearch_runtime.fuzzy_args import normalize_cli_args


class StorageExportArgumentsTest(unittest.TestCase):
    def test_export_lab_option_stays_with_storage_export_subcommand(self):
        args = ["storage", "export", "--lab", "lab-id", "--output", "snapshot.json"]

        self.assertEqual(normalize_cli_args(args), args)

    def test_global_config_is_hoisted_but_export_lab_option_is_not(self):
        args = ["storage", "export", "--lab", "lab-id", "--output", "snapshot.json"]

        self.assertEqual(
            normalize_cli_args(["--config", "lab.json", *args]),
            ["--config", "lab.json", *args],
        )

    def test_reconcile_lab_option_stays_with_storage_reconcile_subcommand(self):
        args = ["storage", "reconcile", "--lab", "lab-id", "--apply"]

        self.assertEqual(normalize_cli_args(args), args)
        self.assertEqual(
            normalize_cli_args(["--lab", "global-id", *args]),
            ["--lab", "global-id", "storage", "reconcile", "--lab", "lab-id", "--apply"],
        )


if __name__ == "__main__":
    unittest.main()
