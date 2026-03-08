"""
Tests for director.py — queue management and orchestration.

Uses tmp_path fixture for isolated knowledge directories so tests
never touch the real knowledge/ folder.
"""

import json
from pathlib import Path

import pytest

from director import (
    add_to_queue,
    claim_next_experiment,
    complete_experiment,
    load_queue,
    save_queue,
    plan_experiments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a minimal knowledge directory structure."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir(parents=True)
    (kdir / "cards").mkdir()
    (kdir / "synthesis").mkdir()
    return kdir


# ---------------------------------------------------------------------------
# load_queue
# ---------------------------------------------------------------------------


class TestLoadQueue:
    def test_load_empty_queue_no_file(self, knowledge_dir: Path):
        """Loading from a dir without queue.json returns an empty queue."""
        queue = load_queue(knowledge_dir)
        assert queue["version"] == 1
        assert queue["last_updated"] is None
        assert queue["experiments"] == []

    def test_load_existing_queue(self, knowledge_dir: Path):
        """Loading reads back a previously saved queue."""
        queue = {
            "version": 1,
            "last_updated": "2026-01-01T00:00:00Z",
            "experiments": [
                {
                    "id": "exp-0001",
                    "hypothesis": "Test",
                    "category": "lr",
                    "priority": 1,
                    "status": "pending",
                    "assigned_to": None,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        }
        (knowledge_dir / "queue.json").write_text(
            json.dumps(queue), encoding="utf-8"
        )
        loaded = load_queue(knowledge_dir)
        assert len(loaded["experiments"]) == 1
        assert loaded["experiments"][0]["id"] == "exp-0001"

    def test_load_corrupt_file_returns_empty(self, knowledge_dir: Path):
        """A corrupt queue.json should return an empty queue, not crash."""
        (knowledge_dir / "queue.json").write_text("NOT JSON", encoding="utf-8")
        queue = load_queue(knowledge_dir)
        assert queue["experiments"] == []


# ---------------------------------------------------------------------------
# add_to_queue
# ---------------------------------------------------------------------------


class TestAddToQueue:
    def test_add_single_experiment(self, knowledge_dir: Path):
        exp = add_to_queue(
            knowledge_dir,
            hypothesis="Try larger LR",
            category="learning_rate",
            priority=2,
        )
        assert exp["id"] == "exp-0001"
        assert exp["hypothesis"] == "Try larger LR"
        assert exp["category"] == "learning_rate"
        assert exp["priority"] == 2
        assert exp["status"] == "pending"
        assert exp["assigned_to"] is None
        assert exp["created_at"] is not None

    def test_add_default_priority(self, knowledge_dir: Path):
        exp = add_to_queue(
            knowledge_dir,
            hypothesis="Default priority test",
            category="misc",
        )
        assert exp["priority"] == 5

    def test_add_multiple_sequential_ids(self, knowledge_dir: Path):
        e1 = add_to_queue(knowledge_dir, hypothesis="First", category="a")
        e2 = add_to_queue(knowledge_dir, hypothesis="Second", category="b")
        e3 = add_to_queue(knowledge_dir, hypothesis="Third", category="c")
        assert e1["id"] == "exp-0001"
        assert e2["id"] == "exp-0002"
        assert e3["id"] == "exp-0003"

    def test_add_persists_to_disk(self, knowledge_dir: Path):
        add_to_queue(knowledge_dir, hypothesis="Persisted", category="test")
        queue = load_queue(knowledge_dir)
        assert len(queue["experiments"]) == 1
        assert queue["last_updated"] is not None


# ---------------------------------------------------------------------------
# claim_next_experiment
# ---------------------------------------------------------------------------


class TestClaimNextExperiment:
    def test_claim_returns_highest_priority(self, knowledge_dir: Path):
        """Priority 1 should be claimed before priority 3."""
        add_to_queue(knowledge_dir, hypothesis="Low priority", category="a", priority=3)
        add_to_queue(knowledge_dir, hypothesis="High priority", category="b", priority=1)
        add_to_queue(knowledge_dir, hypothesis="Medium priority", category="c", priority=2)

        exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert exp is not None
        assert exp["hypothesis"] == "High priority"
        assert exp["priority"] == 1
        assert exp["status"] == "in_progress"
        assert exp["assigned_to"] == "worker-1"

    def test_claim_skips_assigned(self, knowledge_dir: Path):
        """Already claimed (in_progress) experiments should be skipped."""
        add_to_queue(knowledge_dir, hypothesis="First", category="a", priority=1)
        add_to_queue(knowledge_dir, hypothesis="Second", category="b", priority=1)

        # Claim the first
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        # Second claim should get the other experiment
        exp = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert exp is not None
        assert exp["hypothesis"] == "Second"
        assert exp["assigned_to"] == "worker-2"

    def test_claim_empty_queue_returns_none(self, knowledge_dir: Path):
        result = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert result is None

    def test_claim_all_assigned_returns_none(self, knowledge_dir: Path):
        add_to_queue(knowledge_dir, hypothesis="Only one", category="a", priority=1)
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        result = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert result is None

    def test_claim_persists_status(self, knowledge_dir: Path):
        """After claiming, the queue on disk should reflect in_progress status."""
        add_to_queue(knowledge_dir, hypothesis="Track this", category="a")
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        queue = load_queue(knowledge_dir)
        exp = queue["experiments"][0]
        assert exp["status"] == "in_progress"
        assert exp["assigned_to"] == "worker-1"

    def test_claim_fifo_within_same_priority(self, knowledge_dir: Path):
        """Among same-priority experiments, oldest (first added) wins."""
        add_to_queue(knowledge_dir, hypothesis="Oldest", category="a", priority=2)
        add_to_queue(knowledge_dir, hypothesis="Newest", category="b", priority=2)

        exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert exp is not None
        assert exp["hypothesis"] == "Oldest"


# ---------------------------------------------------------------------------
# complete_experiment
# ---------------------------------------------------------------------------


class TestCompleteExperiment:
    def test_complete_marks_status(self, knowledge_dir: Path):
        exp = add_to_queue(knowledge_dir, hypothesis="To complete", category="a")
        claim_next_experiment(knowledge_dir, worker_id="worker-1")
        complete_experiment(knowledge_dir, exp["id"])

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "completed"

    def test_complete_custom_status(self, knowledge_dir: Path):
        exp = add_to_queue(knowledge_dir, hypothesis="To fail", category="a")
        complete_experiment(knowledge_dir, exp["id"], status="failed")

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "failed"

    def test_complete_nonexistent_id_is_noop(self, knowledge_dir: Path):
        """Completing a non-existent id should not raise."""
        add_to_queue(knowledge_dir, hypothesis="Exists", category="a")
        complete_experiment(knowledge_dir, "exp-9999")

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "pending"


# ---------------------------------------------------------------------------
# Stale claims
# ---------------------------------------------------------------------------


class TestStaleClaims:
    def test_stale_claim_released(self, knowledge_dir: Path):
        from datetime import datetime, timezone, timedelta
        exp = add_to_queue(knowledge_dir, hypothesis="Will stall", category="a")
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        # Manually backdate the claimed_at timestamp
        queue = load_queue(knowledge_dir)
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        queue["experiments"][0]["claimed_at"] = stale_time
        save_queue(knowledge_dir, queue)

        # Next claim should reclaim the stale experiment
        reclaimed = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert reclaimed is not None
        assert reclaimed["id"] == exp["id"]
        assert reclaimed["assigned_to"] == "worker-2"

    def test_fresh_claim_not_released(self, knowledge_dir: Path):
        add_to_queue(knowledge_dir, hypothesis="First", category="a", priority=1)
        add_to_queue(knowledge_dir, hypothesis="Second", category="b", priority=1)
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        exp = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert exp is not None
        assert exp["hypothesis"] == "Second"


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class TestFileLocking:
    def test_rapid_sequential_adds(self, knowledge_dir: Path):
        for i in range(10):
            add_to_queue(knowledge_dir, hypothesis=f"Exp {i}", category="test")
        queue = load_queue(knowledge_dir)
        assert len(queue["experiments"]) == 10
        for exp in queue["experiments"]:
            assert "id" in exp
            assert "status" in exp


# ---------------------------------------------------------------------------
# Hypothesis deduplication
# ---------------------------------------------------------------------------


class TestHypothesisDedup:
    def test_no_duplicate_category_priority_pairs(self, knowledge_dir: Path):
        """plan_experiments should not produce two ideas with the same (category, priority)."""
        from commons import create_card
        create_card(
            knowledge_dir, commit_id="c1",
            hypothesis="Test", config_diff={},
            results={"val_bpb": 0.99, "delta": -0.01},
            status="keep", lesson="test", tags=["opt"],
        )
        ideas = plan_experiments(knowledge_dir)
        seen = set()
        for idea in ideas:
            key = (idea["category"], idea["priority"])
            assert key not in seen, f"Duplicate idea: {key}"
            seen.add(key)

    def test_saturated_tag_not_under_explored(self, knowledge_dir: Path):
        """A tag with 5+ experiments should NOT get priority 1 (under-explored)."""
        from commons import create_card
        for i in range(5):
            create_card(
                knowledge_dir, commit_id=f"c{i}",
                hypothesis=f"LR experiment {i}", config_diff={},
                results={"val_bpb": 0.99, "delta": -0.01},
                status="keep", lesson="test", tags=["lr"],
            )
        ideas = plan_experiments(knowledge_dir)
        lr_p1 = [i for i in ideas if i["category"] == "lr" and i["priority"] == 1]
        assert len(lr_p1) == 0

    def test_empty_knowledge_returns_empty(self, knowledge_dir: Path):
        ideas = plan_experiments(knowledge_dir)
        assert ideas == []


# ---------------------------------------------------------------------------
# Director CLI tests
# ---------------------------------------------------------------------------


class TestDirectorCLI:
    def test_plan_command(self, knowledge_dir: Path):
        import subprocess
        from commons import create_card
        create_card(
            knowledge_dir, commit_id="c1",
            hypothesis="Test", config_diff={},
            results={"val_bpb": 0.99, "delta": -0.01},
            status="keep", lesson="test", tags=["lr"],
        )
        result = subprocess.run(
            ["python", "director.py", "plan"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "experiment" in result.stdout.lower() or "planned" in result.stdout.lower()

    def test_status_command_empty(self, knowledge_dir: Path):
        import subprocess
        result = subprocess.run(
            ["python", "director.py", "status"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Queue status" in result.stdout

    def test_add_command(self, knowledge_dir: Path):
        import subprocess
        result = subprocess.run(
            [
                "python", "director.py", "add",
                "--hypothesis", "Test from CLI",
                "--category", "test",
                "--priority", "2",
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Added" in result.stdout

    def test_status_after_add(self, knowledge_dir: Path):
        import subprocess
        env = {**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)}
        cwd = str(Path(__file__).resolve().parent)
        # Add an experiment
        subprocess.run(
            ["python", "director.py", "add", "--hypothesis", "Test", "--category", "test"],
            capture_output=True, text=True, cwd=cwd, env=env,
        )
        # Check status
        result = subprocess.run(
            ["python", "director.py", "status"],
            capture_output=True, text=True, cwd=cwd, env=env,
        )
        assert result.returncode == 0
        assert "Pending:     1" in result.stdout or "Pending: 1" in result.stdout or "pending" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Integration tests (commons + director together)
# ---------------------------------------------------------------------------


class TestCommonsDirectorIntegration:
    def test_plan_from_cards(self, knowledge_dir: Path):
        """Director can plan experiments based on cards created by commons."""
        from commons import create_card
        create_card(
            knowledge_dir, commit_id="c1",
            hypothesis="Baseline", config_diff={},
            results={"val_bpb": 1.0, "delta": 0.0},
            status="keep", lesson="baseline", tags=["baseline"],
        )
        create_card(
            knowledge_dir, commit_id="c2",
            hypothesis="Halve batch", config_diff={},
            results={"val_bpb": 0.99, "delta": -0.01},
            status="keep", lesson="helped", tags=["batch_size"],
        )
        ideas = plan_experiments(knowledge_dir)
        assert len(ideas) >= 2
        categories = {i["category"] for i in ideas}
        assert "baseline" in categories
        assert "batch_size" in categories

    def test_full_lifecycle(self, knowledge_dir: Path):
        """End-to-end: create card -> plan -> enqueue -> claim -> complete."""
        from commons import create_card
        create_card(
            knowledge_dir, commit_id="c1",
            hypothesis="Test", config_diff={},
            results={"val_bpb": 0.99, "delta": -0.01},
            status="keep", lesson="test", tags=["lr"],
        )
        ideas = plan_experiments(knowledge_dir)
        assert len(ideas) > 0

        exp = add_to_queue(
            knowledge_dir,
            hypothesis=ideas[0]["hypothesis"],
            category=ideas[0]["category"],
            priority=ideas[0]["priority"],
        )
        claimed = claim_next_experiment(knowledge_dir, worker_id="w1")
        assert claimed is not None
        assert claimed["id"] == exp["id"]

        complete_experiment(knowledge_dir, exp["id"])
        queue = load_queue(knowledge_dir)
        completed = [e for e in queue["experiments"] if e["status"] == "completed"]
        assert len(completed) == 1
