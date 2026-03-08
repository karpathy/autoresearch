"""
Comprehensive tests for commons.py — the knowledge base interface.

Uses tmp_path fixture for isolated knowledge directories so tests
never touch the real knowledge/ folder.
"""

import json
from pathlib import Path

import pytest

from commons import (
    create_card,
    generate_session_synthesis,
    get_cards_by_tag,
    get_coverage_map,
    get_meta_synthesis,
    get_open_questions,
    get_platform_findings,
    get_recent_cards,
    load_cards,
    load_index,
    read_brief,
    retract_card,
    update_index,
    update_meta_synthesis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a minimal knowledge directory structure."""
    kdir = tmp_path / "knowledge"
    (kdir / "cards").mkdir(parents=True)
    (kdir / "synthesis").mkdir(parents=True)
    # Seed an empty index
    (kdir / "index.json").write_text(
        json.dumps({"version": 1, "last_updated": None, "cards": []}),
        encoding="utf-8",
    )
    return kdir


def _make_card(knowledge_dir: Path, **overrides) -> dict:
    """Helper to create a card with sensible defaults, allowing overrides."""
    defaults = {
        "knowledge_dir": knowledge_dir,
        "commit_id": "abc1234",
        "hypothesis": "Test hypothesis",
        "config_diff": {"TOTAL_BATCH_SIZE": {"from": 524288, "to": 262144}},
        "results": {
            "val_bpb": 0.99,
            "delta": -0.01,
            "peak_vram_mb": 1000,
            "training_seconds": 300,
            "num_steps": 100,
        },
        "status": "keep",
        "lesson": "Test lesson",
        "tags": ["batch_size", "optimization"],
    }
    defaults.update(overrides)
    return create_card(**defaults)


# ---------------------------------------------------------------------------
# Card creation tests
# ---------------------------------------------------------------------------


class TestCreateCard:
    def test_creates_card_file(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)

        card_files = list((knowledge_dir / "cards").glob("*.json"))
        assert len(card_files) == 1

    def test_card_has_required_fields(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)

        required = [
            "id",
            "timestamp",
            "platform",
            "parent_commit",
            "hypothesis",
            "config_diff",
            "results",
            "status",
            "lesson",
            "tags",
            "prior_knowledge_used",
        ]
        for field in required:
            assert field in card, f"Missing field: {field}"

    def test_card_id_is_7_chars(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        assert len(card["id"]) == 7

    def test_card_timestamp_format(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        # Should be ISO 8601 with Z suffix
        assert card["timestamp"].endswith("Z")
        assert "T" in card["timestamp"]

    def test_card_preserves_commit_id(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, commit_id="deadbeef")
        assert card["parent_commit"] == "deadbeef"

    def test_card_preserves_hypothesis(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, hypothesis="Larger LR converges faster")
        assert card["hypothesis"] == "Larger LR converges faster"

    def test_card_preserves_tags(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, tags=["lr", "convergence"])
        assert card["tags"] == ["lr", "convergence"]

    def test_card_default_prior_knowledge(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        assert card["prior_knowledge_used"] == []

    def test_card_custom_prior_knowledge(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, prior_knowledge_used=["card_aaa"])
        assert card["prior_knowledge_used"] == ["card_aaa"]

    def test_platform_is_dict(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        assert isinstance(card["platform"], dict)
        # Fallback platform has these keys
        assert "gpu" in card["platform"]
        assert "ram_gb" in card["platform"]
        assert "framework" in card["platform"]

    def test_card_filename_format(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        card_files = list((knowledge_dir / "cards").glob("*.json"))
        filename = card_files[0].name
        # Should contain commit id and no colons
        assert "abc1234" in filename
        assert ":" not in filename

    def test_auto_updates_index(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        index = load_index(knowledge_dir)
        assert len(index["cards"]) == 1


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------


class TestLoadCards:
    def test_load_empty(self, knowledge_dir: Path):
        cards = load_cards(knowledge_dir)
        assert cards == []

    def test_load_single(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        cards = load_cards(knowledge_dir)
        assert len(cards) == 1

    def test_load_multiple_sorted_newest_first(self, knowledge_dir: Path):
        # Create cards with different commits to get different ids/timestamps
        import time

        c1 = _make_card(knowledge_dir, commit_id="commit_aaa")
        time.sleep(1.1)  # Ensure different timestamps
        c2 = _make_card(knowledge_dir, commit_id="commit_bbb")

        cards = load_cards(knowledge_dir)
        assert len(cards) == 2
        # Newest first
        assert cards[0]["timestamp"] >= cards[1]["timestamp"]

    def test_load_nonexistent_dir(self, tmp_path: Path):
        cards = load_cards(tmp_path / "nonexistent")
        assert cards == []


# ---------------------------------------------------------------------------
# Recent cards
# ---------------------------------------------------------------------------


class TestGetRecentCards:
    def test_returns_limited_count(self, knowledge_dir: Path):
        for i in range(5):
            _make_card(knowledge_dir, commit_id=f"commit_{i}")

        recent = get_recent_cards(knowledge_dir, n=3)
        assert len(recent) == 3

    def test_returns_all_if_fewer_than_n(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        recent = get_recent_cards(knowledge_dir, n=20)
        assert len(recent) == 1


# ---------------------------------------------------------------------------
# Tag filtering
# ---------------------------------------------------------------------------


class TestGetCardsByTag:
    def test_filter_by_tag(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="c1", tags=["lr", "convergence"])
        _make_card(knowledge_dir, commit_id="c2", tags=["batch_size"])
        _make_card(knowledge_dir, commit_id="c3", tags=["lr", "warmup"])

        lr_cards = get_cards_by_tag(knowledge_dir, "lr")
        assert len(lr_cards) == 2

        batch_cards = get_cards_by_tag(knowledge_dir, "batch_size")
        assert len(batch_cards) == 1

    def test_no_matching_tag(self, knowledge_dir: Path):
        _make_card(knowledge_dir, tags=["lr"])
        assert get_cards_by_tag(knowledge_dir, "nonexistent") == []


# ---------------------------------------------------------------------------
# Platform findings
# ---------------------------------------------------------------------------


class TestGetPlatformFindings:
    def test_finds_by_gpu_substring(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        # Read back the card to find the actual platform gpu string
        cards = load_cards(knowledge_dir)
        gpu_name = cards[0].get("platform", {}).get("gpu", "")
        # Search for a substring of the actual gpu name
        substring = gpu_name[:5] if gpu_name else "cpu"
        findings = get_platform_findings(knowledge_dir, substring)
        assert len(findings) == 1

    def test_case_insensitive(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        cards = load_cards(knowledge_dir)
        gpu_name = cards[0].get("platform", {}).get("gpu", "")
        substring = gpu_name[:5].upper() if gpu_name else "CPU"
        findings = get_platform_findings(knowledge_dir, substring)
        assert len(findings) == 1

    def test_no_match(self, knowledge_dir: Path):
        _make_card(knowledge_dir)
        findings = get_platform_findings(knowledge_dir, "A100")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


class TestIndex:
    def test_update_index_creates_file(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="idx1")
        index = load_index(knowledge_dir)
        assert index["version"] == 1
        assert len(index["cards"]) == 1
        assert index["last_updated"] is not None

    def test_index_card_entry_fields(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="idx2", tags=["test_tag"])
        index = load_index(knowledge_dir)
        entry = index["cards"][0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "status" in entry
        assert "tags" in entry
        assert "parent_commit" in entry

    def test_update_index_reflects_all_cards(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="a1")
        _make_card(knowledge_dir, commit_id="a2")
        _make_card(knowledge_dir, commit_id="a3")
        index = load_index(knowledge_dir)
        assert len(index["cards"]) == 3

    def test_load_index_empty(self, tmp_path: Path):
        """Loading index from dir without index.json returns empty structure."""
        index = load_index(tmp_path)
        assert index == {"version": 1, "last_updated": None, "cards": []}


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------


class TestCoverageMap:
    def test_empty_coverage(self, knowledge_dir: Path):
        coverage = get_coverage_map(knowledge_dir)
        assert coverage == {}

    def test_single_card_coverage(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            tags=["lr"],
            status="keep",
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        coverage = get_coverage_map(knowledge_dir)
        assert "lr" in coverage
        assert coverage["lr"]["count"] == 1
        assert coverage["lr"]["kept"] == 1
        assert coverage["lr"]["best_delta"] == -0.05
        assert coverage["lr"]["best_bpb"] == 0.95

    def test_multi_tag_coverage(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="c1",
            tags=["lr", "warmup"],
            status="keep",
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        _make_card(
            knowledge_dir,
            commit_id="c2",
            tags=["lr"],
            status="revert",
            results={"val_bpb": 1.00, "delta": 0.00, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )

        coverage = get_coverage_map(knowledge_dir)
        assert coverage["lr"]["count"] == 2
        assert coverage["lr"]["kept"] == 1
        assert coverage["lr"]["best_delta"] == -0.05
        assert coverage["lr"]["best_bpb"] == 0.95

        assert coverage["warmup"]["count"] == 1
        assert coverage["warmup"]["kept"] == 1

    def test_coverage_best_values(self, knowledge_dir: Path):
        """Best delta is most negative, best bpb is lowest."""
        _make_card(
            knowledge_dir,
            commit_id="c1",
            tags=["opt"],
            results={"val_bpb": 0.99, "delta": -0.01, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        _make_card(
            knowledge_dir,
            commit_id="c2",
            tags=["opt"],
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )

        coverage = get_coverage_map(knowledge_dir)
        assert coverage["opt"]["best_delta"] == -0.05
        assert coverage["opt"]["best_bpb"] == 0.95


# ---------------------------------------------------------------------------
# Meta-synthesis
# ---------------------------------------------------------------------------


class TestMetaSynthesis:
    def test_no_file_returns_default(self, knowledge_dir: Path):
        result = get_meta_synthesis(knowledge_dir)
        assert result == "No meta-synthesis available yet."

    def test_reads_file(self, knowledge_dir: Path):
        content = "# Meta Synthesis\n\nSome findings here.\n"
        (knowledge_dir / "synthesis" / "meta-synthesis.md").write_text(content, encoding="utf-8")
        assert get_meta_synthesis(knowledge_dir) == content


# ---------------------------------------------------------------------------
# Open questions extraction
# ---------------------------------------------------------------------------


class TestOpenQuestions:
    def test_no_synthesis(self, knowledge_dir: Path):
        assert get_open_questions(knowledge_dir) == []

    def test_extracts_questions(self, knowledge_dir: Path):
        content = """# Meta Synthesis

Some analysis text.

## Open Questions

- What is the optimal batch size for M3 Max?
- Does warmup length matter at this scale?
- Should we try cosine annealing?

## Next Steps

- Run more experiments
"""
        (knowledge_dir / "synthesis" / "meta-synthesis.md").write_text(content, encoding="utf-8")

        questions = get_open_questions(knowledge_dir)
        assert len(questions) == 3
        assert "optimal batch size" in questions[0]
        assert "warmup length" in questions[1]
        assert "cosine annealing" in questions[2]

    def test_asterisk_bullets(self, knowledge_dir: Path):
        content = """# Open Questions

* Question one?
* Question two?
"""
        (knowledge_dir / "synthesis" / "meta-synthesis.md").write_text(content, encoding="utf-8")

        questions = get_open_questions(knowledge_dir)
        assert len(questions) == 2


# ---------------------------------------------------------------------------
# Read brief
# ---------------------------------------------------------------------------


class TestReadBrief:
    def test_empty_brief(self, knowledge_dir: Path):
        brief = read_brief(knowledge_dir)
        assert "Coverage Map" in brief
        assert "No experiments recorded yet." in brief
        assert "Recent Experiments" in brief
        assert "Open Questions" in brief

    def test_brief_with_cards(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="brief1",
            hypothesis="Test hypothesis for brief",
            tags=["lr"],
            status="keep",
        )
        brief = read_brief(knowledge_dir)
        assert "lr" in brief
        assert "Test hypothesis for brief" in brief
        assert "keep" in brief

    def test_brief_includes_open_questions(self, knowledge_dir: Path):
        content = """# Open Questions

- Is gradient accumulation beneficial?
"""
        (knowledge_dir / "synthesis" / "meta-synthesis.md").write_text(content, encoding="utf-8")
        brief = read_brief(knowledge_dir)
        assert "gradient accumulation" in brief


# ---------------------------------------------------------------------------
# Session synthesis
# ---------------------------------------------------------------------------


class TestGenerateSessionSynthesis:
    def test_creates_file(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="s1", status="keep", tags=["lr"])
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        assert Path(out_path).exists()
        assert Path(out_path).name == "mar8.md"

    def test_includes_header(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="s1", status="keep", tags=["lr"])
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "# Session Synthesis: mar8" in content
        assert "**Platform:**" in content
        assert "**Experiments:** 1" in content

    def test_includes_confirmed_findings(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="s1",
            status="keep",
            hypothesis="LR warmup helps",
            tags=["lr"],
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Confirmed Findings" in content
        assert "LR warmup helps" in content

    def test_includes_dead_ends(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="s1",
            status="revert",
            hypothesis="Bigger batch is worse",
            tags=["batch"],
        )
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Dead Ends" in content
        assert "Bigger batch is worse" in content

    def test_includes_crashes_section(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="s1",
            status="crash",
            hypothesis="OOM with large model",
            tags=["scaling"],
        )
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Crashes" in content
        assert "OOM with large model" in content

    def test_includes_open_questions(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="s1", status="keep", tags=["lr"])
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Open Questions" in content
        # 'lr' has only 1 experiment so should be noted as under-explored
        assert "under-explored" in content

    def test_best_bpb_in_header(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="s1",
            status="keep",
            tags=["lr"],
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        out_path = generate_session_synthesis(knowledge_dir, "mar8")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "**Best val_bpb:** 0.950000" in content

    def test_empty_cards(self, knowledge_dir: Path):
        out_path = generate_session_synthesis(knowledge_dir, "empty_session")
        content = Path(out_path).read_text(encoding="utf-8")
        assert "# Session Synthesis: empty_session" in content
        assert "**Experiments:** 0" in content
        assert "No confirmed findings yet." in content


# ---------------------------------------------------------------------------
# Meta-synthesis generation
# ---------------------------------------------------------------------------


class TestUpdateMetaSynthesis:
    def test_creates_file(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="m1", status="keep", tags=["lr"])
        out_path = update_meta_synthesis(knowledge_dir)
        assert Path(out_path).exists()
        assert Path(out_path).name == "meta-synthesis.md"

    def test_includes_header(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="m1", tags=["lr"])
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "# Meta-Synthesis" in content
        assert "**Last updated:**" in content
        assert "**Total experiments:** 1" in content

    def test_includes_key_findings(self, knowledge_dir: Path):
        _make_card(
            knowledge_dir,
            commit_id="m1",
            status="keep",
            hypothesis="LR warmup beneficial",
            tags=["lr"],
            results={"val_bpb": 0.95, "delta": -0.05, "peak_vram_mb": 1000, "training_seconds": 300, "num_steps": 100},
        )
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Key Findings" in content
        assert "LR warmup beneficial" in content

    def test_includes_coverage_map_table(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="m1", tags=["lr"], status="keep")
        _make_card(knowledge_dir, commit_id="m2", tags=["batch"], status="revert")
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Experiment Coverage Map" in content
        assert "| Tag | Count | Kept | Best Delta | Saturated? |" in content
        assert "| lr |" in content
        assert "| batch |" in content

    def test_includes_open_questions(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="m1", tags=["lr"])
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Open Questions" in content

    def test_empty_cards(self, knowledge_dir: Path):
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "# Meta-Synthesis" in content
        assert "**Total experiments:** 0" in content
        assert "No confirmed findings yet." in content

    def test_no_platform_section_for_single_platform(self, knowledge_dir: Path):
        _make_card(knowledge_dir, commit_id="m1", tags=["lr"])
        out_path = update_meta_synthesis(knowledge_dir)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "## Platform-Specific Findings" not in content

    def test_open_questions_readable_by_get_open_questions(self, knowledge_dir: Path):
        """Meta-synthesis open questions should be parseable by get_open_questions."""
        _make_card(knowledge_dir, commit_id="m1", tags=["lr"])
        update_meta_synthesis(knowledge_dir)
        questions = get_open_questions(knowledge_dir)
        # Should find at least the under-explored tag question
        assert len(questions) >= 1
        assert any("under-explored" in q for q in questions)


# ---------------------------------------------------------------------------
# CLI tests (via subprocess to avoid sys.exit issues)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_read_brief_command(self, knowledge_dir: Path):
        import subprocess

        result = subprocess.run(
            ["python", "commons.py", "read-brief"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Coverage Map" in result.stdout

    def test_write_card_command(self, knowledge_dir: Path):
        import subprocess

        result = subprocess.run(
            [
                "python",
                "commons.py",
                "write-card",
                "--commit",
                "cli_test_commit",
                "--hypothesis",
                "CLI test hypothesis",
                "--result",
                "0.99",
                "--delta",
                "-0.01",
                "--peak-memory",
                "2000",
                "--training-seconds",
                "300",
                "--num-steps",
                "150",
                "--status",
                "keep",
                "--lesson",
                "CLI works",
                "--tags",
                "cli,test",
                "--config-diff",
                '{"LR": {"from": 0.001, "to": 0.002}}',
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Card created" in result.stdout

        # Verify card was actually written
        cards = load_cards(knowledge_dir)
        assert len(cards) == 1
        assert cards[0]["hypothesis"] == "CLI test hypothesis"

    def test_coverage_command(self, knowledge_dir: Path):
        import subprocess

        _make_card(knowledge_dir, tags=["test_tag"])
        result = subprocess.run(
            ["python", "commons.py", "coverage"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "test_tag" in result.stdout

    def test_update_index_command(self, knowledge_dir: Path):
        import subprocess

        _make_card(knowledge_dir)
        result = subprocess.run(
            ["python", "commons.py", "update-index"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Index updated" in result.stdout

    def test_synthesize_command(self, knowledge_dir: Path):
        import subprocess

        _make_card(knowledge_dir, tags=["lr"])
        result = subprocess.run(
            ["python", "commons.py", "synthesize", "--session", "mar8"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Session synthesis written to" in result.stdout
        assert (knowledge_dir / "synthesis" / "mar8.md").exists()

    def test_update_meta_command(self, knowledge_dir: Path):
        import subprocess

        _make_card(knowledge_dir, tags=["lr"])
        result = subprocess.run(
            ["python", "commons.py", "update-meta"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
            env={**__import__("os").environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        )
        assert result.returncode == 0
        assert "Meta-synthesis written to" in result.stdout
        assert (knowledge_dir / "synthesis" / "meta-synthesis.md").exists()


# ---------------------------------------------------------------------------
# Card validation
# ---------------------------------------------------------------------------


class TestCardValidation:
    def test_rejects_empty_commit_id(self, knowledge_dir: Path):
        with pytest.raises(ValueError, match="commit_id"):
            _make_card(knowledge_dir, commit_id="")

    def test_rejects_empty_hypothesis(self, knowledge_dir: Path):
        with pytest.raises(ValueError, match="hypothesis"):
            _make_card(knowledge_dir, hypothesis="")

    def test_rejects_empty_tags(self, knowledge_dir: Path):
        with pytest.raises(ValueError, match="tags"):
            _make_card(knowledge_dir, tags=[])

    def test_rejects_non_numeric_val_bpb(self, knowledge_dir: Path):
        with pytest.raises(ValueError, match="val_bpb"):
            _make_card(knowledge_dir, results={"val_bpb": "not_a_number", "delta": 0})

    def test_rejects_invalid_status(self, knowledge_dir: Path):
        with pytest.raises(ValueError, match="status"):
            _make_card(knowledge_dir, status="invalid_status")

    def test_accepts_all_valid_statuses(self, knowledge_dir: Path):
        for i, status in enumerate(["keep", "revert", "inconclusive", "crash"]):
            card = _make_card(knowledge_dir, commit_id=f"c_{status}_{i}", status=status)
            assert card["status"] == status


# ---------------------------------------------------------------------------
# Card retraction
# ---------------------------------------------------------------------------


class TestRetractCard:
    def test_retract_changes_status(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        retract_card(knowledge_dir, card["id"], reason="Buggy measurement")
        cards = load_cards(knowledge_dir)
        assert cards[0]["status"] == "retracted"

    def test_retract_adds_reason(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir)
        retract_card(knowledge_dir, card["id"], reason="Bad GPU readings")
        cards = load_cards(knowledge_dir)
        assert cards[0]["retraction_reason"] == "Bad GPU readings"

    def test_retract_nonexistent_raises(self, knowledge_dir: Path):
        with pytest.raises(FileNotFoundError):
            retract_card(knowledge_dir, "nonexistent_id", reason="test")

    def test_retracted_excluded_from_coverage(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, tags=["lr"])
        retract_card(knowledge_dir, card["id"], reason="bad")
        coverage = get_coverage_map(knowledge_dir)
        assert coverage.get("lr", {}).get("count", 0) == 0

    def test_retracted_excluded_from_brief(self, knowledge_dir: Path):
        card = _make_card(knowledge_dir, hypothesis="Retracted experiment")
        retract_card(knowledge_dir, card["id"], reason="bad")
        brief = read_brief(knowledge_dir)
        assert "Retracted experiment" not in brief
