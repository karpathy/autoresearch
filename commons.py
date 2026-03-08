"""
commons.py — Knowledge base read/write interface for autoresearch agents.

Both a Python library and a CLI tool. Experiment cards are stored as JSON files
in knowledge/cards/ and indexed in knowledge/index.json.

Usage (CLI):
    uv run commons.py read-brief
    uv run commons.py read-meta
    uv run commons.py write-card --commit abc1234 --hypothesis "..." ...
    uv run commons.py update-index
    uv run commons.py coverage
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Default knowledge directory: knowledge/ relative to this script
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

VALID_STATUSES = {"keep", "revert", "inconclusive", "crash", "retracted"}

# ---------------------------------------------------------------------------
# Platform detection — try to import from platform_utils, fall back gracefully
# ---------------------------------------------------------------------------

try:
    from platform_utils import get_device_info
except ImportError:

    def get_device_info() -> dict:
        """Fallback when platform_utils is not available."""
        return {"gpu": "unknown", "ram_gb": 0, "framework": "unknown"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs(knowledge_dir: Path) -> None:
    """Make sure the cards/ directory exists."""
    (knowledge_dir / "cards").mkdir(parents=True, exist_ok=True)


def _safe_timestamp(ts: str) -> str:
    """Replace colons with dashes so the timestamp is filename-safe."""
    return ts.replace(":", "-")


def _short_id(commit_id: str, timestamp: str) -> str:
    """Generate a 7-char hex id from commit + timestamp."""
    raw = f"{commit_id}-{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:7]


def _card_filename(timestamp: str, commit_id: str) -> str:
    """Build the card filename: {safe_timestamp}_{commit_id}.json"""
    safe_ts = _safe_timestamp(timestamp)
    return f"{safe_ts}_{commit_id}.json"


def _validate_card_inputs(commit_id, hypothesis, results, status, tags):
    if not commit_id or not commit_id.strip():
        raise ValueError("commit_id must be a non-empty string")
    if not hypothesis or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty string")
    if not tags:
        raise ValueError("tags must be a non-empty list")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got '{status}'")
    val_bpb = results.get("val_bpb")
    if val_bpb is not None and not isinstance(val_bpb, (int, float)):
        raise ValueError(f"val_bpb must be numeric, got {type(val_bpb).__name__}")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to temp file in same dir, then rename."""
    content = json.dumps(data, indent=2) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Library functions
# ---------------------------------------------------------------------------


def create_card(
    knowledge_dir: Path | str,
    commit_id: str,
    hypothesis: str,
    config_diff: dict,
    results: dict,
    status: str,
    lesson: str,
    tags: list[str],
    prior_knowledge_used: list[str] | None = None,
) -> dict:
    """Create a new experiment card and write it to disk.

    Returns the card dict that was written.
    """
    knowledge_dir = Path(knowledge_dir)
    _validate_card_inputs(commit_id, hypothesis, results, status, tags)
    _ensure_dirs(knowledge_dir)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    card_id = _short_id(commit_id, timestamp)
    platform_info = get_device_info()

    card = {
        "id": card_id,
        "timestamp": timestamp,
        "platform": platform_info,
        "parent_commit": commit_id,
        "hypothesis": hypothesis,
        "config_diff": config_diff,
        "results": results,
        "status": status,
        "lesson": lesson,
        "tags": tags,
        "prior_knowledge_used": prior_knowledge_used or [],
    }

    filename = _card_filename(timestamp, commit_id)
    card_path = knowledge_dir / "cards" / filename
    _atomic_write_json(card_path, card)

    # Auto-update the index after creating a card
    update_index(knowledge_dir)

    return card


def load_cards(knowledge_dir: Path | str) -> list[dict]:
    """Load all experiment cards, sorted by timestamp newest first."""
    knowledge_dir = Path(knowledge_dir)
    cards_dir = knowledge_dir / "cards"
    if not cards_dir.exists():
        return []

    cards = []
    for fpath in cards_dir.glob("*.json"):
        try:
            card = json.loads(fpath.read_text(encoding="utf-8"))
            cards.append(card)
        except (json.JSONDecodeError, OSError):
            continue

    cards.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
    return cards


def get_recent_cards(knowledge_dir: Path | str, n: int = 20) -> list[dict]:
    """Return the *n* most recent experiment cards."""
    return load_cards(knowledge_dir)[:n]


def get_cards_by_tag(knowledge_dir: Path | str, tag: str) -> list[dict]:
    """Return all cards that include *tag* in their tags list."""
    return [c for c in load_cards(knowledge_dir) if tag in c.get("tags", [])]


def get_platform_findings(knowledge_dir: Path | str, gpu_substring: str) -> list[dict]:
    """Return cards whose platform gpu field contains *gpu_substring*."""
    sub = gpu_substring.lower()
    return [
        c
        for c in load_cards(knowledge_dir)
        if sub in c.get("platform", {}).get("gpu", "").lower()
    ]


def update_index(knowledge_dir: Path | str) -> None:
    """Regenerate index.json from the card files on disk."""
    knowledge_dir = Path(knowledge_dir)
    cards = load_cards(knowledge_dir)

    index = {
        "version": 1,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cards": [
            {
                "id": c["id"],
                "timestamp": c["timestamp"],
                "status": c.get("status", "unknown"),
                "tags": c.get("tags", []),
                "parent_commit": c.get("parent_commit", ""),
            }
            for c in cards
        ],
    }

    index_path = knowledge_dir / "index.json"
    _atomic_write_json(index_path, index)


def retract_card(
    knowledge_dir: Path | str,
    card_id: str,
    reason: str,
) -> dict:
    """Mark a card as retracted. Modifies the JSON file in place.

    Raises FileNotFoundError if no card with that ID exists.
    """
    knowledge_dir = Path(knowledge_dir)
    cards_dir = knowledge_dir / "cards"

    for fpath in cards_dir.glob("*.json"):
        try:
            card = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if card.get("id") == card_id:
            card["status"] = "retracted"
            card["retraction_reason"] = reason
            _atomic_write_json(fpath, card)
            update_index(knowledge_dir)
            return card

    raise FileNotFoundError(f"No card found with id '{card_id}'")


def load_index(knowledge_dir: Path | str) -> dict:
    """Load and return the index.json contents."""
    knowledge_dir = Path(knowledge_dir)
    index_path = knowledge_dir / "index.json"
    if not index_path.exists():
        return {"version": 1, "last_updated": None, "cards": []}
    return json.loads(index_path.read_text(encoding="utf-8"))


def get_coverage_map(knowledge_dir: Path | str) -> dict:
    """Build a coverage map: {tag: {count, kept, best_delta, best_bpb}}.

    Aggregates across all cards grouped by tag.
    """
    cards = load_cards(knowledge_dir)
    coverage: dict[str, dict] = {}

    for card in cards:
        if card.get("status") == "retracted":
            continue
        results = card.get("results", {})
        delta = results.get("delta")
        bpb = results.get("val_bpb")

        for tag in card.get("tags", []):
            if tag not in coverage:
                coverage[tag] = {
                    "count": 0,
                    "kept": 0,
                    "best_delta": None,
                    "best_bpb": None,
                }

            entry = coverage[tag]
            entry["count"] += 1

            if card.get("status") == "keep":
                entry["kept"] += 1

            # best_delta: most negative (biggest improvement) wins
            if delta is not None:
                if entry["best_delta"] is None or delta < entry["best_delta"]:
                    entry["best_delta"] = delta

            # best_bpb: lowest is best
            if bpb is not None:
                if entry["best_bpb"] is None or bpb < entry["best_bpb"]:
                    entry["best_bpb"] = bpb

    return coverage


def get_meta_synthesis(knowledge_dir: Path | str) -> str:
    """Read and return the meta-synthesis.md file content."""
    knowledge_dir = Path(knowledge_dir)
    meta_path = knowledge_dir / "synthesis" / "meta-synthesis.md"
    if not meta_path.exists():
        return "No meta-synthesis available yet."
    return meta_path.read_text(encoding="utf-8")


def get_open_questions(knowledge_dir: Path | str) -> list[str]:
    """Extract open questions from meta-synthesis.md.

    Looks for lines starting with '- ' or '* ' under a heading containing
    'open question' (case-insensitive), or lines starting with '?' or
    containing a question mark after a bullet.
    """
    content = get_meta_synthesis(knowledge_dir)
    if content == "No meta-synthesis available yet.":
        return []

    questions: list[str] = []
    in_open_questions_section = False

    for line in content.splitlines():
        stripped = line.strip()

        # Detect open-questions heading
        if re.match(r"^#{1,6}\s+.*open.question", stripped, re.IGNORECASE):
            in_open_questions_section = True
            continue

        # Another heading ends the section
        if in_open_questions_section and re.match(r"^#{1,6}\s+", stripped):
            in_open_questions_section = False
            continue

        # Collect bullet items in the open questions section
        if in_open_questions_section and re.match(r"^[-*]\s+", stripped):
            question_text = re.sub(r"^[-*]\s+", "", stripped)
            questions.append(question_text)

    return questions


def read_brief(knowledge_dir: Path | str) -> str:
    """Produce an agent-facing summary of the knowledge base.

    Includes: coverage map, recent cards, and open questions.
    """
    knowledge_dir = Path(knowledge_dir)
    parts: list[str] = []

    # --- Coverage map ---
    coverage = get_coverage_map(knowledge_dir)
    parts.append("## Coverage Map")
    if not coverage:
        parts.append("No experiments recorded yet.")
    else:
        for tag, stats in sorted(coverage.items()):
            delta_str = f"{stats['best_delta']:.6f}" if stats["best_delta"] is not None else "n/a"
            bpb_str = f"{stats['best_bpb']:.6f}" if stats["best_bpb"] is not None else "n/a"
            parts.append(
                f"- {tag}: {stats['count']} experiments, "
                f"{stats['kept']} kept, "
                f"best_delta={delta_str}, "
                f"best_bpb={bpb_str}"
            )

    # --- Recent cards ---
    recent = [c for c in get_recent_cards(knowledge_dir, n=10) if c.get("status") != "retracted"][:5]
    parts.append("")
    parts.append("## Recent Experiments")
    if not recent:
        parts.append("No recent experiments.")
    else:
        for card in recent:
            results = card.get("results", {})
            bpb = results.get("val_bpb", "n/a")
            delta = results.get("delta", "n/a")
            parts.append(
                f"- [{card['id']}] {card['hypothesis'][:80]} "
                f"| status={card['status']} bpb={bpb} delta={delta}"
            )

    # --- Open questions ---
    questions = get_open_questions(knowledge_dir)
    parts.append("")
    parts.append("## Open Questions")
    if not questions:
        parts.append("No open questions identified yet.")
    else:
        for q in questions:
            parts.append(f"- {q}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synthesis generation
# ---------------------------------------------------------------------------


def generate_session_synthesis(knowledge_dir: Path | str, session_tag: str) -> str:
    """Generate a session synthesis report from all current cards.

    Saves the report to knowledge/synthesis/{session_tag}.md and returns the
    file path as a string.
    """
    knowledge_dir = Path(knowledge_dir)
    cards = load_cards(knowledge_dir)
    coverage = get_coverage_map(knowledge_dir)

    # Ensure synthesis directory exists
    synthesis_dir = knowledge_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    # Gather platform info from the most recent card (or fallback)
    platform_info = get_device_info()
    if cards:
        platform_info = cards[0].get("platform", platform_info)
    gpu = platform_info.get("gpu", "unknown")

    # Categorize cards
    kept = [c for c in cards if c.get("status") == "keep"]
    discarded = [c for c in cards if c.get("status") == "revert"]
    crashed = [c for c in cards if c.get("status") == "crash"]

    # Sort kept by delta (most negative = best improvement first)
    kept.sort(key=lambda c: c.get("results", {}).get("delta", 0))

    # Best val_bpb across all cards
    all_bpbs = [
        c.get("results", {}).get("val_bpb")
        for c in cards
        if c.get("results", {}).get("val_bpb") is not None
    ]
    best_bpb = min(all_bpbs) if all_bpbs else None

    # Baseline: the val_bpb of a card with delta == 0 or closest to 0
    baseline_cards = [
        c for c in cards
        if c.get("results", {}).get("delta") is not None
        and c.get("results", {}).get("val_bpb") is not None
    ]
    if baseline_cards:
        baseline_card = min(
            baseline_cards,
            key=lambda c: abs(c["results"]["delta"]),
        )
        baseline_bpb = baseline_card["results"]["val_bpb"] + abs(baseline_card["results"]["delta"])
    else:
        baseline_bpb = None

    # Build the report
    lines: list[str] = []
    lines.append(f"# Session Synthesis: {session_tag}")
    lines.append("")
    lines.append(f"- **Platform:** {gpu}")
    lines.append(f"- **Experiments:** {len(cards)}")
    if best_bpb is not None:
        lines.append(f"- **Best val_bpb:** {best_bpb:.6f}")
    if baseline_bpb is not None:
        lines.append(f"- **Baseline val_bpb:** {baseline_bpb:.6f}")
    lines.append("")

    # Confirmed Findings
    lines.append("## Confirmed Findings")
    lines.append("")
    if kept:
        for card in kept:
            results = card.get("results", {})
            delta = results.get("delta")
            bpb = results.get("val_bpb")
            delta_str = f"{delta:+.6f}" if delta is not None else "n/a"
            bpb_str = f"{bpb:.6f}" if bpb is not None else "n/a"
            lines.append(
                f"- [{card['id']}] {card['hypothesis'][:80]} "
                f"| delta={delta_str}, bpb={bpb_str}"
            )
            lines.append(f"  - Lesson: {card.get('lesson', 'n/a')}")
    else:
        lines.append("No confirmed findings yet.")
    lines.append("")

    # Dead Ends
    lines.append("## Dead Ends")
    lines.append("")
    if discarded:
        for card in discarded:
            results = card.get("results", {})
            delta = results.get("delta")
            delta_str = f"{delta:+.6f}" if delta is not None else "n/a"
            lines.append(
                f"- [{card['id']}] {card['hypothesis'][:80]} "
                f"| delta={delta_str}"
            )
            lines.append(f"  - Lesson: {card.get('lesson', 'n/a')}")
    else:
        lines.append("No dead ends recorded.")
    lines.append("")

    # Crashes
    if crashed:
        lines.append("## Crashes")
        lines.append("")
        for card in crashed:
            lines.append(
                f"- [{card['id']}] {card['hypothesis'][:80]}"
            )
            lines.append(f"  - Lesson: {card.get('lesson', 'n/a')}")
        lines.append("")

    # Open Questions
    lines.append("## Open Questions")
    lines.append("")
    open_q = get_open_questions(knowledge_dir)
    # Also identify under-explored tags (count <= 2 and not saturated)
    under_explored = [
        tag for tag, stats in coverage.items()
        if stats["count"] <= 2
    ]
    if open_q:
        for q in open_q:
            lines.append(f"- {q}")
    if under_explored:
        for tag in sorted(under_explored):
            lines.append(f"- Tag '{tag}' is under-explored ({coverage[tag]['count']} experiments)")
    if not open_q and not under_explored:
        lines.append("No open questions identified.")
    lines.append("")

    content = "\n".join(lines)
    out_path = synthesis_dir / f"{session_tag}.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


def update_meta_synthesis(knowledge_dir: Path | str) -> str:
    """Regenerate knowledge/synthesis/meta-synthesis.md from ALL cards.

    Returns the file path as a string.
    """
    knowledge_dir = Path(knowledge_dir)
    cards = load_cards(knowledge_dir)
    coverage = get_coverage_map(knowledge_dir)

    synthesis_dir = knowledge_dir / "synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Categorize
    kept = [c for c in cards if c.get("status") == "keep"]
    kept.sort(key=lambda c: c.get("results", {}).get("delta", 0))

    # Collect unique platforms
    platforms: dict[str, list[dict]] = {}
    for card in cards:
        gpu = card.get("platform", {}).get("gpu", "unknown")
        platforms.setdefault(gpu, []).append(card)

    # Build meta-synthesis
    lines: list[str] = []
    lines.append("# Meta-Synthesis")
    lines.append("")
    lines.append(f"- **Last updated:** {now}")
    lines.append(f"- **Total experiments:** {len(cards)}")
    lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    if kept:
        for card in kept:
            results = card.get("results", {})
            delta = results.get("delta")
            bpb = results.get("val_bpb")
            gpu = card.get("platform", {}).get("gpu", "unknown")
            delta_str = f"{delta:+.6f}" if delta is not None else "n/a"
            bpb_str = f"{bpb:.6f}" if bpb is not None else "n/a"
            lines.append(
                f"- [{card['id']}] {card['hypothesis'][:80]} "
                f"| delta={delta_str}, bpb={bpb_str} (platform: {gpu})"
            )
    else:
        lines.append("No confirmed findings yet.")
    lines.append("")

    # Platform-Specific sections (only if multiple platforms)
    if len(platforms) > 1:
        lines.append("## Platform-Specific Findings")
        lines.append("")
        for gpu_name, gpu_cards in sorted(platforms.items()):
            gpu_kept = [c for c in gpu_cards if c.get("status") == "keep"]
            lines.append(f"### {gpu_name}")
            lines.append(f"- Experiments: {len(gpu_cards)}, Kept: {len(gpu_kept)}")
            if gpu_kept:
                best = min(gpu_kept, key=lambda c: c.get("results", {}).get("delta", 0))
                delta = best.get("results", {}).get("delta")
                if delta is not None:
                    lines.append(f"- Best delta: {delta:+.6f}")
            lines.append("")

    # Experiment Coverage Map table
    lines.append("## Experiment Coverage Map")
    lines.append("")
    lines.append("| Tag | Count | Kept | Best Delta | Saturated? |")
    lines.append("|-----|-------|------|------------|------------|")
    for tag, stats in sorted(coverage.items()):
        delta_str = f"{stats['best_delta']:+.6f}" if stats["best_delta"] is not None else "n/a"
        # A tag is "saturated" if it has 5+ experiments and keeping rate is low
        saturated = "Yes" if stats["count"] >= 5 and stats["kept"] <= 1 else "No"
        lines.append(
            f"| {tag} | {stats['count']} | {stats['kept']} | {delta_str} | {saturated} |"
        )
    lines.append("")

    # Open Questions
    lines.append("## Open Questions")
    lines.append("")
    under_explored = [
        tag for tag, stats in coverage.items()
        if stats["count"] <= 2
    ]
    high_potential = [
        tag for tag, stats in coverage.items()
        if stats["best_delta"] is not None and stats["best_delta"] < -0.01 and stats["count"] <= 3
    ]
    questions_added = False
    if under_explored:
        for tag in sorted(under_explored):
            lines.append(f"- Tag '{tag}' is under-explored ({coverage[tag]['count']} experiments)")
        questions_added = True
    if high_potential:
        for tag in sorted(high_potential):
            if tag not in under_explored:
                delta_str = f"{coverage[tag]['best_delta']:+.6f}"
                lines.append(
                    f"- Tag '{tag}' shows promise (best_delta={delta_str}) — worth more exploration"
                )
                questions_added = True
    if not questions_added:
        lines.append("No open questions identified.")
    lines.append("")

    content = "\n".join(lines)
    out_path = synthesis_dir / "meta-synthesis.md"
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge base interface for autoresearch agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # read-brief
    subparsers.add_parser("read-brief", help="Print knowledge summary")

    # read-meta
    subparsers.add_parser("read-meta", help="Print full meta-synthesis")

    # write-card
    wc = subparsers.add_parser("write-card", help="Create a new experiment card")
    wc.add_argument("--commit", required=True, help="Parent commit id")
    wc.add_argument("--hypothesis", required=True, help="Experiment hypothesis")
    wc.add_argument("--result", required=True, type=float, help="val_bpb result")
    wc.add_argument("--delta", required=True, type=float, help="Delta from baseline")
    wc.add_argument("--peak-memory", required=True, type=float, help="Peak VRAM in MB")
    wc.add_argument("--training-seconds", required=True, type=float, help="Training time in seconds")
    wc.add_argument("--num-steps", required=True, type=int, help="Number of training steps")
    wc.add_argument("--status", required=True, choices=["keep", "revert", "inconclusive", "crash"], help="Experiment status")
    wc.add_argument("--lesson", required=True, help="Lesson learned")
    wc.add_argument("--tags", required=True, help="Comma-separated tags")
    wc.add_argument("--config-diff", default="{}", help="JSON string of config diff")

    # retract
    rt = subparsers.add_parser("retract", help="Mark a card as retracted")
    rt.add_argument("--id", required=True, help="Card ID to retract")
    rt.add_argument("--reason", required=True, help="Reason for retraction")

    # update-index
    subparsers.add_parser("update-index", help="Regenerate index.json")

    # coverage
    subparsers.add_parser("coverage", help="Print coverage map")

    # synthesize
    syn = subparsers.add_parser("synthesize", help="Generate session synthesis")
    syn.add_argument("--session", required=True, help="Session tag (e.g. mar8)")

    # update-meta
    subparsers.add_parser("update-meta", help="Regenerate meta-synthesis.md")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    knowledge_dir = Path(os.environ.get("KNOWLEDGE_DIR", str(KNOWLEDGE_DIR)))

    if args.command == "read-brief":
        print(read_brief(knowledge_dir))

    elif args.command == "read-meta":
        print(get_meta_synthesis(knowledge_dir))

    elif args.command == "write-card":
        try:
            config_diff = json.loads(args.config_diff)
        except json.JSONDecodeError:
            print("Error: --config-diff must be valid JSON", file=sys.stderr)
            sys.exit(1)

        results = {
            "val_bpb": args.result,
            "delta": args.delta,
            "peak_vram_mb": args.peak_memory,
            "training_seconds": args.training_seconds,
            "num_steps": args.num_steps,
        }

        card = create_card(
            knowledge_dir=knowledge_dir,
            commit_id=args.commit,
            hypothesis=args.hypothesis,
            config_diff=config_diff,
            results=results,
            status=args.status,
            lesson=args.lesson,
            tags=[t.strip() for t in args.tags.split(",")],
        )
        print(f"Card created: {card['id']}")

    elif args.command == "retract":
        try:
            card = retract_card(knowledge_dir, args.id, args.reason)
            print(f"Card {card['id']} retracted: {args.reason}")
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    elif args.command == "update-index":
        update_index(knowledge_dir)
        index = load_index(knowledge_dir)
        print(f"Index updated: {len(index['cards'])} cards")

    elif args.command == "coverage":
        coverage = get_coverage_map(knowledge_dir)
        if not coverage:
            print("No experiments recorded yet.")
        else:
            for tag, stats in sorted(coverage.items()):
                delta_str = f"{stats['best_delta']:.6f}" if stats["best_delta"] is not None else "n/a"
                bpb_str = f"{stats['best_bpb']:.6f}" if stats["best_bpb"] is not None else "n/a"
                print(
                    f"{tag}: {stats['count']} experiments, "
                    f"{stats['kept']} kept, "
                    f"best_delta={delta_str}, "
                    f"best_bpb={bpb_str}"
                )

    elif args.command == "synthesize":
        out_path = generate_session_synthesis(knowledge_dir, args.session)
        print(f"Session synthesis written to: {out_path}")

    elif args.command == "update-meta":
        out_path = update_meta_synthesis(knowledge_dir)
        print(f"Meta-synthesis written to: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
