import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from autoresearch.config import Config
from autoresearch.toy_train import toy_train


def test_forward_backward_loss_decreases(tmp_path: Path):
    cfg = Config(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        batch_size=4,
        max_steps=5,
        learning_rate=3e-3,
        device="cpu",
        metrics_path=str(tmp_path / "run_metrics.json"),
    )
    metrics = toy_train(cfg)
    losses = metrics["losses"]
    assert len(losses) == 5
    # Across 5 steps, last should be <= first (strict decrease not guaranteed
    # at tiny scale, but mean of last 2 should be < mean of first 2).
    import statistics
    assert statistics.mean(losses[-2:]) < statistics.mean(losses[:2])

    # Sidecar was written and round-trips:
    p = Path(cfg.metrics_path)
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["num_steps"] == 5
    assert data["loss_last"] == losses[-1]


def test_metrics_sidecar_helpers(tmp_path: Path):
    from autoresearch.metrics import write_metrics, read_metrics, parse_stdout_summary

    p = tmp_path / "m.json"
    write_metrics(p, {"val_bpb": 0.9, "num_steps": 10})
    assert read_metrics(p) == {"val_bpb": 0.9, "num_steps": 10}
    assert read_metrics(tmp_path / "missing.json") is None

    # Legacy fallback parser
    stdout = (
        "some preamble\n"
        "---\n"
        "val_bpb:          0.997900\n"
        "num_steps:        953\n"
        "depth:            8\n"
    )
    parsed = parse_stdout_summary(stdout)
    assert parsed["val_bpb"] == pytest.approx(0.9979)
    assert parsed["num_steps"] == 953
    assert parsed["depth"] == 8
