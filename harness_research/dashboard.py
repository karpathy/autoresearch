"""CLI dashboard for harness research pipeline state."""

import os
import sys
from datetime import datetime, timezone, timedelta
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


def _queue_depth() -> dict:
    """Return pending candidate count per tier from Firestore.

    Connects to Firestore if GOOGLE_APPLICATION_CREDENTIALS is set.
    Returns {} gracefully if unavailable.
    """
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return {}
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        queue_col = db.collection("harness_research_queue")
        docs = queue_col.where("status", "==", "pending").stream()

        counts: dict[str, int] = {}
        for doc in docs:
            data = doc.to_dict() or {}
            tier = data.get("hardware_tier", "unknown")
            counts[tier] = counts.get(tier, 0) + 1
        return counts
    except Exception:
        return {}


def _per_tier_champions() -> dict:
    """Read profiles/{tier}/*.yaml and return dict of tier -> {id, score}.

    Handles both flat profiles/{tier}.yaml and profiles/{tier}/*.yaml layouts.
    """
    result: dict[str, dict] = {}
    if not PROFILES_DIR.exists():
        return result

    for tier in HARDWARE_TIERS:
        # Try flat file first: profiles/pi5-hailo8l.yaml
        flat = PROFILES_DIR / f"{tier}.yaml"
        if flat.exists():
            data = _load_yaml(flat)
            cid = data.get("candidate_id") or data.get("id", "unknown")
            score = data.get("score", 0.0)
            result[tier] = {"id": cid, "score": score}
            continue

        # Try directory: profiles/pi5-hailo8l/*.yaml — pick highest score
        tier_dir = PROFILES_DIR / tier
        if tier_dir.is_dir():
            best: dict | None = None
            for f in tier_dir.glob("*.yaml"):
                data = _load_yaml(f)
                score = data.get("score", 0.0)
                if best is None or score > best["score"]:
                    cid = data.get("candidate_id") or data.get("id", "unknown")
                    best = {"id": cid, "score": score}
            if best:
                result[tier] = best

    return result


def main() -> int:
    pdt = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz=pdt)
    timestamp = now.strftime("%Y-%m-%d %H:%M %Z")

    width = 57
    sep = "═" * width

    print()
    print(sep)
    print(f"  Harness Research Dashboard — {timestamp}")
    print(sep)

    # Global champion
    champion = _load_yaml(CHAMPION_PATH)
    if champion:
        cid = champion.get("candidate_id", "unknown")
        score = champion.get("score", 0.0)
        date = champion.get("date", "unknown")
        print(f"  Global Champion  {cid:<20} score={score:.4f}  ({date})")
    else:
        print("  Global Champion  (no champion data found)")
    print()

    # Per-tier champions
    print("  Per-Tier Champions:")
    tier_champs = _per_tier_champions()
    for tier in HARDWARE_TIERS:
        info = tier_champs.get(tier)
        if info and info["score"] > 0:
            print(f"    {tier:<18} {info['id']:<20} score={info['score']:.4f}")
        else:
            print(f"    {tier:<18} {'(no data yet)':<20}")
    print()

    # Firestore queue
    print("  Firestore Queue:")
    queue = _queue_depth()
    if queue:
        for tier in HARDWARE_TIERS:
            n = queue.get(tier, 0)
            print(f"    {tier:<18} {n} candidates pending")
    else:
        print("    (Firestore unavailable — set GOOGLE_APPLICATION_CREDENTIALS)")
    print()

    # Recent runs
    print("  Recent Runs (last 5):")
    winners = _recent_winners(5)
    if winners:
        # Load champion score to determine if each run improved champion
        champ_score = champion.get("score", 0.0) if champion else 0.0
        for name, score in winners:
            # Extract date from filename e.g. 2026-03-20-winner.yaml
            date_part = name.replace("-winner.yaml", "")
            improved = score > champ_score
            flag = "✅ champion" if improved else "no improvement"
            print(f"    {date_part}  {score:.4f}  {flag}")
    else:
        print("    (no winner reports found)")
    print()

    # Search space progress
    print("  Search Space Progress:")
    try:
        from .search_space import SEARCH_SPACE_SIZE, explored_count
        count = explored_count()
        pct = count / SEARCH_SPACE_SIZE * 100
        bar_filled = int(pct / 2)  # 50-char bar = 100%
        bar = "█" * bar_filled + "░" * (50 - bar_filled)
        print(f"    Explored: {count:,} / {SEARCH_SPACE_SIZE:,} configs")
        print(f"    [{bar}] {pct:.2f}%")
    except Exception as exc:
        print(f"    (search space unavailable: {exc})")
    print()

    # Next scheduled run
    # Cron: 0 8 * * * UTC = 1 AM Pacific (PST) / midnight PDT
    now_utc = datetime.now(tz=timezone.utc)
    next_run_utc = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_utc.hour >= 8:
        next_run_utc += timedelta(days=1)
    next_run_pdt = next_run_utc.astimezone(pdt)
    print(f"  Next scheduled run: {next_run_pdt.strftime('%Y-%m-%d %H:%M %Z')} daily (cron 0 8 * * * UTC)")
    print(sep)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
