"""Tests for analysis.py."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd

from analysis import compute_stats, load_results, print_text_report, save_plot


def _write_tsv(rows):
    """Write rows to a temp TSV file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    header = "commit\tdescription\tval_bpb\tmemory_gb\tstatus\n"
    with open(path, "w") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    return path


BASELINE_ROWS = [
    ("abc001", "baseline",                 0.9979, 4.2, "KEEP"),
    ("abc002", "tweak lr",                 1.0050, 4.2, "DISCARD"),
    ("abc003", "increase LR to 0.04",      0.9932, 4.2, "KEEP"),
    ("abc004", "bad experiment",           1.0100, 4.2, "DISCARD"),
    ("abc005", "add warmup ratio 0.05",    0.9901, 4.2, "KEEP"),
    ("abc006", "OOM attempt",              "NaN",  4.2, "CRASH"),
    ("abc007", "increase batch size 2**20", 0.9885, 4.2, "KEEP"),
]


class LoadResultsTests(unittest.TestCase):
    def test_parses_tsv_and_normalizes_status(self):
        path = _write_tsv([
            ("abc", "desc", 0.99, 4.0, " keep "),
            ("def", "desc", 0.98, 4.0, "discard"),
        ])
        try:
            df = load_results(path)
        finally:
            os.remove(path)
        self.assertEqual(list(df["status"]), ["KEEP", "DISCARD"])
        self.assertAlmostEqual(df.loc[0, "val_bpb"], 0.99)

    def test_non_numeric_val_bpb_becomes_nan(self):
        path = _write_tsv([("abc", "crashed", "NaN", "NaN", "CRASH")])
        try:
            df = load_results(path)
        finally:
            os.remove(path)
        self.assertTrue(pd.isna(df.loc[0, "val_bpb"]))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_results("/nonexistent/path/results.tsv")


class ComputeStatsTests(unittest.TestCase):
    def setUp(self):
        self.path = _write_tsv(BASELINE_ROWS)
        self.df = load_results(self.path)
        self.stats = compute_stats(self.df)

    def tearDown(self):
        os.remove(self.path)

    def test_counts(self):
        self.assertEqual(self.stats["total_experiments"], 7)
        self.assertEqual(self.stats["kept"], 4)
        self.assertEqual(self.stats["discarded"], 2)
        self.assertEqual(self.stats["crashed"], 1)

    def test_keep_rate_excludes_crashes(self):
        # 4 kept / (4 kept + 2 discarded) = 0.6667
        self.assertAlmostEqual(self.stats["keep_rate"], 0.6667, places=4)

    def test_baseline_and_best(self):
        self.assertAlmostEqual(self.stats["baseline_bpb"], 0.9979)
        self.assertAlmostEqual(self.stats["best_bpb"], 0.9885)
        self.assertAlmostEqual(self.stats["improvement"], 0.0094, places=6)
        self.assertEqual(self.stats["best_experiment"], "increase batch size 2**20")

    def test_top_hits_sorted_by_delta_desc(self):
        hits = self.stats["top_hits"]
        self.assertEqual(len(hits), 3)  # one per kept beyond baseline
        deltas = [h["delta"] for h in hits]
        self.assertEqual(deltas, sorted(deltas, reverse=True))
        # Biggest single-step improvement is baseline (0.9979) -> LR 0.04 (0.9932)
        self.assertEqual(hits[0]["description"], "increase LR to 0.04")

    def test_trajectory_improving(self):
        # baseline case has steady improvement across recent kept
        self.assertEqual(self.stats["trajectory"], "improving")


class TrajectoryTests(unittest.TestCase):
    def _trajectory(self, kept_bpbs):
        rows = [
            (f"c{i:03d}", f"exp{i}", bpb, 4.0, "KEEP")
            for i, bpb in enumerate(kept_bpbs)
        ]
        path = _write_tsv(rows)
        try:
            return compute_stats(load_results(path))["trajectory"]
        finally:
            os.remove(path)

    def test_early_when_fewer_than_three_kept(self):
        self.assertEqual(self._trajectory([1.00, 0.99]), "early")

    def test_plateauing_on_tiny_positive_deltas(self):
        # recent avg delta in (0, 0.001] -> plateauing
        self.assertEqual(
            self._trajectory([1.0000, 0.9998, 0.9996, 0.9995]),
            "plateauing",
        )

    def test_stuck_when_no_recent_improvement(self):
        # recent avg delta <= 0 -> stuck
        self.assertEqual(
            self._trajectory([0.9900, 0.9900, 0.9900, 0.9900]),
            "stuck",
        )

    def test_no_data_when_no_keeps(self):
        path = _write_tsv([
            ("abc", "x", 1.0, 4.0, "DISCARD"),
            ("def", "y", "NaN", 4.0, "CRASH"),
        ])
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        self.assertEqual(stats["trajectory"], "no_data")
        self.assertIsNone(stats["baseline_bpb"])
        self.assertIsNone(stats["best_bpb"])
        self.assertEqual(stats["top_hits"], [])


class EdgeCaseTests(unittest.TestCase):
    def test_keep_rate_none_when_only_crashes(self):
        path = _write_tsv([("abc", "x", "NaN", "NaN", "CRASH")])
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        self.assertIsNone(stats["keep_rate"])

    def test_single_keep_has_no_top_hits(self):
        path = _write_tsv([("abc", "only", 0.99, 4.0, "KEEP")])
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        self.assertEqual(stats["top_hits"], [])
        self.assertAlmostEqual(stats["improvement"], 0.0)
        self.assertEqual(stats["trajectory"], "early")


class TextReportTests(unittest.TestCase):
    def test_report_runs_and_mentions_key_fields(self):
        path = _write_tsv(BASELINE_ROWS)
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_text_report(stats)
        out = buf.getvalue()
        self.assertIn("AUTORESEARCH EXPERIMENT REPORT", out)
        self.assertIn("Total experiments:", out)
        self.assertIn("Best experiment:", out)
        self.assertIn("Trajectory:", out)

    def test_report_handles_no_keeps(self):
        path = _write_tsv([("abc", "x", 1.0, 4.0, "DISCARD")])
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_text_report(stats)
        self.assertIn("No kept experiments yet.", buf.getvalue())

    def test_stats_are_json_serializable(self):
        path = _write_tsv(BASELINE_ROWS)
        try:
            stats = compute_stats(load_results(path))
        finally:
            os.remove(path)
        # Should not raise.
        json.dumps(stats)


class SavePlotTests(unittest.TestCase):
    def test_save_plot_creates_file(self):
        path = _write_tsv(BASELINE_ROWS)
        df = load_results(path)
        fd, out = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            with redirect_stderr(io.StringIO()):
                save_plot(df, out)
            self.assertTrue(os.path.getsize(out) > 0)
        finally:
            os.remove(path)
            if os.path.exists(out):
                os.remove(out)

    def test_save_plot_noop_when_no_keeps(self):
        path = _write_tsv([("abc", "x", 1.0, 4.0, "DISCARD")])
        df = load_results(path)
        fd, out = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        os.remove(out)  # so we can check it wasn't created
        buf = io.StringIO()
        try:
            with redirect_stderr(buf):
                save_plot(df, out)
            self.assertIn("No kept experiments to plot.", buf.getvalue())
            self.assertFalse(os.path.exists(out))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
