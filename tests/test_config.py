import json
from pathlib import Path

import pytest

from autoresearch.config import Config, load_config


def test_defaults_valid():
    cfg = Config()
    cfg.validate()
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.max_steps > 0


def test_load_from_dict_overrides():
    cfg = load_config({"n_layer": 3, "max_steps": 42})
    assert cfg.n_layer == 3
    assert cfg.max_steps == 42
    # Unspecified keys keep defaults:
    assert cfg.n_head == Config().n_head


def test_load_from_json(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_steps": 7, "learning_rate": 1e-3, "unknown_key": "x"}))
    cfg = load_config(p)
    assert cfg.max_steps == 7
    assert cfg.learning_rate == 1e-3


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_validation_errors():
    with pytest.raises(ValueError):
        load_config({"n_embd": 65, "n_head": 2})  # not divisible
    with pytest.raises(ValueError):
        load_config({"dropout": 1.5})
    with pytest.raises(ValueError):
        load_config({"device": "tpu"})
    with pytest.raises(ValueError):
        load_config({"max_steps": 0})
