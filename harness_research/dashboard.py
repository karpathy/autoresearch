"""CLI dashboard for harness research pipeline state."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

OPS_REPO = Path(os.environ.get("OPENCASTOR_OPS_DIR", Path.home() / "opencastor-ops"))
HARNESS_DIR = OPS_REPO / "harness-research"
CHAMPION_PATH = HARNESS_DIR / "champion.yaml"
PROFILES_DIR = HARNESS_DIR / "profiles"
CANDIDATES_DIR = HARNESS_DIR / "candidates"

HARDWARE_TIERS = [
    "pi5-hailo8l",
    "pi5-8gb",
    "pi4-8gb",
    "server",
    "waveshare-alpha",
    "unitree-go2",
]


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _recent_winners(n: int = 5) -> list[tuple[str, float]]:
    """Return list of (filename, score) for the most recent winner yaml files."""
    if not CANDIDATES_DIR.exists():
        return []
    files = sorted(CANDIDATES_DIR.glob("*-winner.yaml"), reverse=True)[:n]
    results = []
    for f in files:
        data = _load_yaml(f)
        score = data.get("score", 0.0)
        results.append((f.name, score))
    return results


def main() -> int:
    pdt = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz=pdt)
    timestamp = now.strftime("%Y-%m-%d %H:%M %Z")

    width = 55
    sep = "═" * width

    print()
    print("OpenCastor Harness Research Dashboard")
    print(sep)
    print(f"Last updated: {timestamp}")
    print()

    # Global champion
    print("CHAMPION (global)")
    champion = _load_yaml(CHAMPION_PATH)
    if champion:
        cid = champion.get("candidate_id", "unknown")
        score = champion.get("score", 0.0)
        config = champion.get("config", {})
        tunables = ", ".join(f"{k}={v}" for k, v in config.items()
                             if k in ("cost_gate_usd", "thinking_budget"))
        print(f"  Config:  {cid} | score: {score:.4f}")
        if tunables:
            print(f"  Tunables: {tunables}")
    else:
        print("  (no champion data found)")
    print()

    # Per-tier champions
    print("PER-TIER CHAMPIONS")
    for tier in HARDWARE_TIERS:
        profile_path = PROFILES_DIR / f"{tier}.yaml"
        if profile_path.exists():
            data = _load_yaml(profile_path)
            score = data.get("score", 0.0)
            if score > 0:
                cid = data.get("candidate_id", "unknown")
                print(f"  {tier:<22} score: {score:.4f}  ({cid})")
            else:
                print(f"  {tier:<22} score: {score:.4f}  (no data yet)")
        else:
            print(f"  {tier:<22} score: 0.0000  (no data yet)")
    print()

    # Recent reports
    print("RECENT REPORTS  (last 5 in harness-research/candidates/)")
    winners = _recent_winners(5)
    if winners:
        for name, score in winners:
            print(f"  {name}  score={score:.4f}")
    else:
        print("  (no winner reports found)")
    print()

    # Queue status
    print("QUEUE STATUS")
    print(f"  Reading from: {OPS_REPO / 'harness-research'}")
    print("  Candidates in queue: N/A (requires Firestore)")
    print()

    print("To refresh: python -m harness_research.dashboard")
    print("To run research: python -m harness_research.run --dry-run")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
