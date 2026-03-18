import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosaas.results_logger import RESULTS_TSV_HEADER, ensure_results_file


def test_ensure_results_file_writes_header(tmp_path):
    out = tmp_path / "autosaas-results.tsv"
    ensure_results_file(out)
    assert out.read_text() == RESULTS_TSV_HEADER
