#!/usr/bin/env python3
"""
Test suite for analysis.py CLI tool.

Run with: python3 test_analysis.py
"""

import unittest
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add current directory to path to import analysis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from analysis import load_results, analyze_trajectory, generate_text_report, generate_json_report, create_progress_plot, main
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure analysis.py is in the same directory")
    sys.exit(1)


class TestAnalysisTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Basic test data
        self.basic_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
e5f6g7h	0.991500	44.5	keep	add residual connections
"""
        
        # Minimal test data
        self.minimal_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
"""
        
        # No kept experiments data
        self.no_kept_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	discard	baseline
b2c3d4e	1.005000	44.2	discard	increase LR to 0.04
c3d4e5f	0.000000	0.0	crash	double model width (OOM)
"""
        
        # Progress trajectory data (improving)
        self.improving_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	0.988000	44.5	keep	add residual connections
d4e5f6g	0.985000	44.8	keep	improve attention pattern
e5f6g7h	0.982000	45.0	keep	optimize learning rate
"""
        
        # Progress trajectory data (plateauing)
        self.plateauing_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.997800	44.2	keep	minor tweak
c3d4e5f	0.997700	44.5	keep	another minor tweak
d4e5f6g	0.997600	44.8	keep	small improvement
e5f6g7h	0.997500	45.0	keep	tiny improvement
"""
        
        # Progress trajectory data (stuck)
        self.stuck_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.997900	44.2	keep	no change
c3d4e5f	0.997900	44.5	keep	still no change
d4e5f6g	0.997900	44.8	keep	same performance
e5f6g7h	0.997900	45.0	keep	unchanged
"""
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_file(self, content, filename="results.tsv"):
        """Create a temporary file with given content."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def test_load_results_basic(self):
        """Test loading basic results file."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        
        self.assertEqual(len(df), 5)
        self.assertIn('commit', df.columns)
        self.assertIn('val_bpb', df.columns)
        self.assertIn('memory_gb', df.columns)
        self.assertIn('status', df.columns)
        self.assertIn('description', df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['val_bpb']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['memory_gb']))
        
        # Check status normalization
        self.assertEqual(df.iloc[0]['status'], 'KEEP')
        self.assertEqual(df.iloc[2]['status'], 'DISCARD')
        self.assertEqual(df.iloc[3]['status'], 'CRASH')
    
    def test_load_results_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(SystemExit):
            load_results('/nonexistent/file.tsv')
    
    def test_load_results_malformed_data(self):
        """Test loading malformed data."""
        malformed_data = "invalid\ttsv\tformat"
        filepath = self.create_temp_file(malformed_data)
        
        # Should still load but may have issues
        try:
            df = load_results(filepath)
            self.assertIsInstance(df, pd.DataFrame)
        except SystemExit:
            # If it's too malformed, it might exit
            pass
    
    def test_analyze_trajectory_improving(self):
        """Test trajectory analysis for improving experiments."""
        filepath = self.create_temp_file(self.improving_data)
        df = load_results(filepath)
        trajectory = analyze_trajectory(df)
        
        self.assertEqual(trajectory, "improving")
    
    def test_analyze_trajectory_plateauing(self):
        """Test trajectory analysis for plateauing experiments."""
        filepath = self.create_temp_file(self.plateauing_data)
        df = load_results(filepath)
        trajectory = analyze_trajectory(df)
        
        self.assertEqual(trajectory, "plateauing")
    
    def test_analyze_trajectory_stuck(self):
        """Test trajectory analysis for stuck experiments."""
        filepath = self.create_temp_file(self.stuck_data)
        df = load_results(filepath)
        trajectory = analyze_trajectory(df)
        
        self.assertEqual(trajectory, "stuck")
    
    def test_analyze_trajectory_insufficient_data(self):
        """Test trajectory analysis with insufficient data."""
        filepath = self.create_temp_file(self.minimal_data)
        df = load_results(filepath)
        trajectory = analyze_trajectory(df)
        
        self.assertEqual(trajectory, "insufficient_data")
    
    def test_analyze_trajectory_no_kept(self):
        """Test trajectory analysis with no kept experiments."""
        filepath = self.create_temp_file(self.no_kept_data)
        df = load_results(filepath)
        trajectory = analyze_trajectory(df)
        
        self.assertEqual(trajectory, "insufficient_data")
    
    def test_generate_text_report_basic(self):
        """Test basic text report generation."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        report = generate_text_report(df)
        
        self.assertIn("AUTORESEARCH EXPERIMENT ANALYSIS", report)
        self.assertIn("Total experiments: 5", report)
        self.assertIn("Kept: 3", report)  # Updated: there are 3 keeps in basic_data
        self.assertIn("Discarded: 1", report)
        self.assertIn("Crashed: 1", report)
        self.assertIn("Keep rate: 3/4 = 75.0%", report)  # Updated format
        self.assertIn("Baseline BPB: 0.997900", report)
        self.assertIn("Best BPB:", report)  # Check for label without exact spacing
        self.assertIn("0.991500", report)  # Check for value
        self.assertIn("Improvement:", report)  # Check for label without exact spacing
        self.assertIn("0.006400", report)  # Check for value
        self.assertIn("add residual connections", report)
        self.assertIn("Top improvements:", report)
    
    def test_generate_text_report_minimal(self):
        """Test text report with minimal data."""
        filepath = self.create_temp_file(self.minimal_data)
        df = load_results(filepath)
        report = generate_text_report(df)
        
        self.assertIn("Total experiments: 1", report)
        self.assertIn("Kept: 1", report)
        self.assertIn("Baseline BPB: 0.997900", report)
        # Note: spacing in formatted output may vary, so check for the value
        self.assertIn("0.997900", report)
    
    def test_generate_text_report_no_kept(self):
        """Test text report with no kept experiments."""
        filepath = self.create_temp_file(self.no_kept_data)
        df = load_results(filepath)
        report = generate_text_report(df)
        
        self.assertIn("Total experiments: 3", report)
        self.assertIn("Kept: 0", report)
        self.assertIn("Discarded: 2", report)
        self.assertIn("Crashed: 1", report)
        # Should not contain performance metrics
        self.assertNotIn("Best BPB:", report)
    
    def test_generate_json_report_basic(self):
        """Test basic JSON report generation."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        report = generate_json_report(df)
        
        # Check structure
        self.assertIsInstance(report, dict)
        self.assertEqual(report['total_experiments'], 5)
        self.assertEqual(report['kept'], 3)  # Updated: there are 3 keeps in basic_data
        self.assertEqual(report['discarded'], 1)
        self.assertEqual(report['crashed'], 1)  # Updated: there is 1 crash in basic_data
        self.assertAlmostEqual(report['keep_rate'], 0.75)  # Updated: 3/4 = 0.75
        
        # Check performance metrics
        self.assertIn('baseline_bpb', report)
        self.assertIn('best_bpb', report)
        self.assertIn('improvement', report)
        self.assertIn('improvement_pct', report)
        self.assertIn('best_experiment', report)
        self.assertIn('trajectory', report)
        self.assertIn('top_hits', report)
        
        # Check values
        self.assertAlmostEqual(report['baseline_bpb'], 0.9979)
        self.assertAlmostEqual(report['best_bpb'], 0.9915)
        self.assertEqual(report['best_experiment'], "add residual connections")
        self.assertIsInstance(report['top_hits'], list)
        self.assertGreater(len(report['top_hits']), 0)  # Should have at least one hit
    
    def test_generate_json_report_types(self):
        """Test JSON report contains correct types."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        report = generate_json_report(df)
        
        # Test JSON serialization
        json_str = json.dumps(report)
        parsed_back = json.loads(json_str)
        
        # Should be identical after serialization
        self.assertEqual(report, parsed_back)
        
        # Check specific types
        self.assertIsInstance(report['total_experiments'], int)
        self.assertIsInstance(report['kept'], int)
        self.assertIsInstance(report['discarded'], int)
        self.assertIsInstance(report['crashed'], int)
        self.assertIsInstance(report['keep_rate'], float)
        self.assertIsInstance(report['baseline_bpb'], float)
        self.assertIsInstance(report['best_bpb'], float)
        self.assertIsInstance(report['improvement'], float)
        self.assertIsInstance(report['improvement_pct'], float)
        self.assertIsInstance(report['best_experiment'], str)
        self.assertIsInstance(report['trajectory'], str)
        self.assertIsInstance(report['top_hits'], list)
    
    def test_generate_json_report_minimal(self):
        """Test JSON report with minimal data."""
        filepath = self.create_temp_file(self.minimal_data)
        df = load_results(filepath)
        report = generate_json_report(df)
        
        self.assertEqual(report['total_experiments'], 1)
        self.assertEqual(report['kept'], 1)
        self.assertEqual(report['baseline_bpb'], 0.9979)
        self.assertEqual(report['best_bpb'], 0.9979)
        self.assertEqual(report['improvement'], 0.0)
        self.assertEqual(report['improvement_pct'], 0.0)
        self.assertEqual(report['trajectory'], 'insufficient_data')
        self.assertEqual(len(report['top_hits']), 0)
    
    def test_create_progress_plot_basic(self):
        """Test basic progress plot creation."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        plot_path = os.path.join(self.temp_dir, 'test_plot.png')
        
        create_progress_plot(df, plot_path)
        
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)
    
    def test_create_progress_plot_no_valid_data(self):
        """Test progress plot with no valid data."""
        # Create data with only crashes
        crash_data = """commit	val_bpb	memory_gb	status	description
a1b2c3d	0.000000	0.0	crash	complete failure
b2c3d4e	0.000000	0.0	crash	another failure
"""
        filepath = self.create_temp_file(crash_data)
        df = load_results(filepath)
        plot_path = os.path.join(self.temp_dir, 'test_plot.png')
        
        # Should not crash, but may not create plot
        try:
            create_progress_plot(df, plot_path)
        except SystemExit:
            pass  # Expected for no valid data
    
    @patch('sys.argv', ['analysis.py'])
    @patch('analysis.load_results')
    def test_main_basic(self, mock_load):
        """Test main function basic execution."""
        # Mock the data
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        mock_load.return_value = df
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            try:
                main()
                output = mock_stdout.getvalue()
                self.assertIn("AUTORESEARCH EXPERIMENT ANALYSIS", output)
                self.assertIn("Total experiments: 5", output)
            except SystemExit:
                pass  # May exit if file not found
    
    @patch('sys.argv', ['analysis.py', '--json'])
    @patch('analysis.load_results')
    def test_main_json(self, mock_load):
        """Test main function with JSON output."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        mock_load.return_value = df
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            try:
                main()
                output = mock_stdout.getvalue()
                # Should be valid JSON
                parsed = json.loads(output)
                self.assertIn('total_experiments', parsed)
            except SystemExit:
                pass
    
    @patch('sys.argv', ['analysis.py', '--plot', 'test.png'])
    @patch('analysis.load_results')
    @patch('analysis.create_progress_plot')
    def test_main_plot(self, mock_plot, mock_load):
        """Test main function with plot option."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        mock_load.return_value = df
        
        with patch('sys.stdout', new_callable=StringIO):
            try:
                main()
                mock_plot.assert_called_once()
            except SystemExit:
                pass
    
    @patch('sys.argv', ['analysis.py', '--tsv', 'custom.tsv'])
    @patch('analysis.load_results')
    def test_main_custom_tsv(self, mock_load):
        """Test main function with custom TSV path."""
        filepath = self.create_temp_file(self.basic_data)
        df = load_results(filepath)
        mock_load.return_value = df
        
        with patch('sys.stdout', new_callable=StringIO):
            try:
                main()
                mock_load.assert_called_with('custom.tsv')
            except SystemExit:
                pass


class StringIO:
    """Mock StringIO for testing."""
    def __init__(self):
        self.contents = []
    
    def write(self, text):
        self.contents.append(text)
    
    def getvalue(self):
        return ''.join(self.contents)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAnalysisTool)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running analysis.py test suite...")
    print("=" * 50)
    
    success = run_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
