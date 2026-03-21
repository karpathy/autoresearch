"""Promote winning harness configs to OpenCastor and Firestore."""

import logging
import os
import subprocess
from datetime import date
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Allow CI to override paths via env vars
OPS_REPO = Path(os.environ.get("OPENCASTOR_OPS_DIR", Path.home() / "opencastor-ops"))
CHAMPION_PATH = OPS_REPO / "harness-research" / "champion.yaml"
PROFILES_DIR = OPS_REPO / "harness-research" / "profiles"
OPENCASTOR_REPO = Path(os.environ.get("OPENCASTOR_REPO_DIR", Path.home() / "OpenCastor"))
TARGET_HARNESS = OPENCASTOR_REPO / "castor" / "harness" / "default_harness.yaml"

# Tunables the research pipeline is allowed to update automatically
_TUNABLE_KEYS = {
    "max_iterations", "thinking_budget", "context_budget",
    "p66_consent_threshold", "retry_on_error", "drift_detection",
    "cost_gate_usd", "enabled",
}


def _run(args: list[str], cwd: Path) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _merge_tunables(existing: dict, new_config: dict) -> dict:
    """Merge tunables into existing harness config, preserving layers and other keys."""
    harness_section = existing.get("harness", existing)
    for key, value in new_config.items():
        if key in _TUNABLE_KEYS:
            harness_section[key] = value
    if "harness" in existing:
        existing["harness"] = harness_section
        return existing
    return harness_section


def _write_harness(config: dict, score: float, champion_date: str, candidate_id: str):
    """Write updated harness YAML to OpenCastor repo."""
    TARGET_HARNESS.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if TARGET_HARNESS.exists():
        existing = yaml.safe_load(TARGET_HARNESS.read_text()) or {}
    out = _merge_tunables(existing, config)
    header = (
        f"# castor/harness/default_harness.yaml\n"
        f"# Canonical default agent harness for OpenCastor.\n"
        f"# Auto-updated by harness-research pipeline — "
        f"champion from {champion_date}, score {score:.4f}\n"
        f"# Keep in sync with: opencastor-client HarnessConfig.defaults() "
        f"and website/src/pages/docs/harness.astro\n\n"
    )
    TARGET_HARNESS.write_text(header + yaml.dump(out, default_flow_style=False))
    log.info("Harness config written to %s", TARGET_HARNESS)


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model_id for use in Firestore field names (replace / with _)."""
    return model_id.replace("/", "_")


def _update_firestore_robots(
    config: dict,
    hardware_tier: str | None = None,
    model_id: str | None = None,
    dry_run: bool = False,
):
    """Update matching robots in Firestore with the winning harness config.

    If hardware_tier is specified, only updates robots with that tier.
    If model_id is specified, only updates robots whose agent.model matches,
    and writes to harness_tunables_{model_id_sanitized} instead of harness_tunables.
    Otherwise updates all robots (generic champion).
    """
    try:
        from google.cloud import firestore as _firestore
        import os
        creds_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            str(Path.home() / ".config/opencastor/firebase-sa-key.json"),
        )
        db = _firestore.Client(project="opencastor", credentials=(
            _load_service_account_credentials(creds_path)
        ))

        robots_ref = db.collection("robots")
        if hardware_tier:
            query = robots_ref.where("hardware_tier", "==", hardware_tier)
            log.info("Querying robots with hardware_tier=%s", hardware_tier)
        else:
            query = robots_ref
            log.info("Querying all robots (generic champion)")

        robots = list(query.stream())

        # Filter by model_id if provided (match agent.model or agent.provider/model)
        if model_id:
            filtered = []
            for robot in robots:
                data = robot.to_dict() or {}
                agent = data.get("agent", {})
                robot_model = agent.get("model", "")
                robot_provider = agent.get("provider", "")
                full_model = f"{robot_provider}/{robot_model}" if robot_provider else robot_model
                if robot_model == model_id or full_model == model_id:
                    filtered.append(robot)
            log.info(
                "Filtered to %d robot(s) with model_id=%s (from %d total)",
                len(filtered), model_id, len(robots),
            )
            robots = filtered
        else:
            log.info("Found %d robot(s) to update", len(robots))

        if dry_run:
            for robot in robots:
                data = robot.to_dict() or {}
                log.info(
                    "[dry-run] Would update robot %s (%s)",
                    robot.id, data.get("hardware_tier", "unknown"),
                )
            return

        harness_update = {k: v for k, v in config.items() if k in _TUNABLE_KEYS}
        field_name = (
            f"harness_tunables_{_sanitize_model_id(model_id)}" if model_id else "harness_tunables"
        )
        for robot in robots:
            robot.reference.update({field_name: harness_update})
            log.info("Updated robot %s %s", robot.id, field_name)

    except Exception as e:
        log.warning("Firestore update skipped: %s", e)


def _load_service_account_credentials(path: str):
    """Load Google service account credentials from file."""
    try:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(
            path,
            scopes=["https://www.googleapis.com/auth/datastore"],
        )
    except Exception:
        import google.auth
        creds, _ = google.auth.default()
        return creds


def promote(
    dry_run: bool = False,
    hardware_tier: str | None = None,
    model_id: str | None = None,
) -> bool:
    """Promote winning champion config to OpenCastor and Firestore.

    Args:
        dry_run: Skip git and Firestore operations.
        hardware_tier: If set, promote the profile champion for that tier.
            If None, promote the generic champion.yaml.
        model_id: If set, promote from profiles/{tier}/{model_id_sanitized}.yaml
            and only update robots whose agent.model matches model_id.

    Returns True if promotion was successful.
    """
    if hardware_tier and model_id:
        champion_path = PROFILES_DIR / hardware_tier / f"{_sanitize_model_id(model_id)}.yaml"
        log.info("Promoting profile champion for tier=%s model=%s", hardware_tier, model_id)
    elif hardware_tier:
        champion_path = PROFILES_DIR / f"{hardware_tier}.yaml"
        log.info("Promoting profile champion for tier: %s", hardware_tier)
    else:
        champion_path = CHAMPION_PATH
        log.info("Promoting generic champion")

    if not champion_path.exists():
        log.error("No champion found at %s", champion_path)
        return False

    champion = yaml.safe_load(champion_path.read_text())
    if not champion or not champion.get("config"):
        log.error("Champion file is empty or missing config")
        return False

    config = champion["config"]
    score = champion.get("score", 0.0)
    champion_date = champion.get("date", date.today().isoformat())
    today = date.today().isoformat()
    candidate_id = champion.get("candidate_id", "unknown")

    log.info(
        "Champion: tier=%s score=%.4f date=%s candidate=%s",
        hardware_tier or "generic", score, champion_date, candidate_id,
    )

    # For generic champion: update default_harness.yaml in OpenCastor
    # For per-tier: skip (harness is tier-specific, not a global default)
    if not hardware_tier:
        if dry_run:
            log.info("[dry-run] Would write generic champion to %s", TARGET_HARNESS)
            log.info("[dry-run] Config: %s", config)
        else:
            try:
                _run(["git", "checkout", "main"], cwd=OPENCASTOR_REPO)
                _run(["git", "pull", "--ff-only"], cwd=OPENCASTOR_REPO)
                _write_harness(config, score, champion_date, candidate_id)
                _run(["git", "add", str(TARGET_HARNESS)], cwd=OPENCASTOR_REPO)
                _run(
                    ["git", "commit", "-m",
                     f"feat(harness): update default agent harness {today} "
                     f"[{candidate_id}, score={score:.4f}]"],
                    cwd=OPENCASTOR_REPO,
                )
                _run(["git", "push", "origin", "main"], cwd=OPENCASTOR_REPO)
                log.info("Pushed harness update to OpenCastor main")
            except subprocess.CalledProcessError as e:
                log.error("Promotion to OpenCastor failed: %s\n%s", e, e.stderr)
                raise

    # Update Firestore (scoped by hardware_tier and/or model_id when set)
    _update_firestore_robots(config, hardware_tier=hardware_tier, model_id=model_id, dry_run=dry_run)

    return True


def promote_all_profiles(dry_run: bool = False) -> dict[str, bool]:
    """Promote all existing profile champions. Returns dict of tier → success."""
    results = {}
    if not PROFILES_DIR.exists():
        log.info("No profiles directory found — nothing to promote")
        return results
    for profile_file in PROFILES_DIR.glob("*.yaml"):
        tier = profile_file.stem
        log.info("Promoting profile: %s", tier)
        try:
            results[tier] = promote(dry_run=dry_run, hardware_tier=tier)
        except Exception as e:
            log.error("Failed to promote %s: %s", tier, e)
            results[tier] = False
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dry = "--dry-run" in sys.argv
    tier_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tier=")), None)
    if "--all-profiles" in sys.argv:
        results = promote_all_profiles(dry_run=dry)
        for t, ok in results.items():
            log.info("%s: %s", t, "OK" if ok else "FAILED")
    else:
        promote(dry_run=dry, hardware_tier=tier_arg)
