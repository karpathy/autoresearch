from pathlib import Path


def test_runtime_layout_exists_after_migration():
    root = Path(__file__).resolve().parents[1]
    assert (root / "legacy").exists()
    assert (root / "autosaas").exists()
    assert (root / "program.md").read_text().startswith("# autoresearch-saas")
