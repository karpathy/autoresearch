import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosaas.config import load_target_config


def test_runtime_layout_exists_after_migration():
    root = Path(__file__).resolve().parents[1]
    assert (root / "legacy").exists()
    assert (root / "autosaas").exists()
    assert (root / "program.md").read_text().startswith("# autoresearch-saas")


def test_load_target_config_reads_required_commands(tmp_path):
    config_path = tmp_path / "project.autosaas.yaml"
    config_path.write_text(
        """
commands:
  lint: pnpm lint
  typecheck: pnpm typecheck
  test: pnpm test
  dev: pnpm dev
  smoke: pnpm playwright test
"""
    )
    cfg = load_target_config(config_path)
    assert cfg.commands.lint == "pnpm lint"


def test_load_target_config_reads_optional_app_boot(tmp_path):
    config_path = tmp_path / "project.autosaas.yaml"
    config_path.write_text(
        """
commands:
  lint: pnpm lint
  typecheck: pnpm typecheck
  test: pnpm test
  dev: pnpm dev
  smoke: pnpm playwright test
app_boot_url: http://localhost
"""
    )

    cfg = load_target_config(config_path)
    assert cfg.app_boot_url == "http://localhost"


def test_load_target_config_rejects_non_mapping_root(tmp_path):
    config_path = tmp_path / "project.autosaas.yaml"
    config_path.write_text("[]")

    with pytest.raises(ValueError):
        load_target_config(config_path)
