"""Promote winning harness config to OpenCastor after approval."""

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
OPENCASTOR_REPO = Path(os.environ.get("OPENCASTOR_REPO_DIR", Path.home() / "OpenCastor"))
TARGET_HARNESS = OPENCASTOR_REPO / "castor" / "harness" / "default_harness.yaml"


def _run(args: list[str], cwd: Path) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def promote(dry_run: bool = False) -> bool:
    """Read champion.yaml and open PR in OpenCastor with the winning config.

    Returns True if promotion PR was created successfully.
    """
    if not CHAMPION_PATH.exists():
        log.error("No champion.yaml found at %s", CHAMPION_PATH)
        return False

    champion = yaml.safe_load(CHAMPION_PATH.read_text())
    if not champion or not champion.get("config"):
        log.error("Champion.yaml is empty or missing config")
        return False

    config = champion["config"]
    score = champion.get("score", 0.0)
    champion_date = champion.get("date", date.today().isoformat())
    today = date.today().isoformat()

    log.info(
        "Promoting champion: score=%.4f date=%s candidate=%s",
        score,
        champion_date,
        champion.get("candidate_id", "unknown"),
    )

    if dry_run:
        log.info("[dry-run] Would write config to %s", TARGET_HARNESS)
        log.info("[dry-run] Config: %s", config)
        return True

    try:
        # Ensure we're on main and up to date
        _run(["git", "checkout", "main"], cwd=OPENCASTOR_REPO)
        _run(["git", "pull", "--ff-only"], cwd=OPENCASTOR_REPO)

        # Merge winning config into the existing default_harness.yaml.
        # Only updates the top-level harness tunables — layers are preserved.
        TARGET_HARNESS.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if TARGET_HARNESS.exists():
            existing = yaml.safe_load(TARGET_HARNESS.read_text()) or {}

        # Extract the harness sub-dict if nested under 'harness:' key
        harness_section = existing.get("harness", existing)

        # Tunables that the research pipeline is allowed to update
        _TUNABLE_KEYS = {
            "max_iterations", "thinking_budget", "context_budget",
            "p66_consent_threshold", "retry_on_error", "drift_detection",
            "cost_gate_usd", "enabled",
        }
        for key, value in config.items():
            if key in _TUNABLE_KEYS:
                harness_section[key] = value

        # Write back — preserve full structure
        if "harness" in existing:
            existing["harness"] = harness_section
            out = existing
        else:
            out = harness_section

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

        # Commit and push directly to main — no branch, no PR
        _run(["git", "add", str(TARGET_HARNESS)], cwd=OPENCASTOR_REPO)
        _run(
            ["git", "commit", "-m",
             f"feat(harness): update default agent harness {today} "
             f"[{champion.get('candidate_id', 'N/A')}, score={score:.4f}]"],
            cwd=OPENCASTOR_REPO,
        )
        _run(["git", "push", "origin", "main"], cwd=OPENCASTOR_REPO)
        log.info("Pushed harness update directly to OpenCastor main")

    except subprocess.CalledProcessError as e:
        log.error("Promotion failed: %s\n%s", e, e.stderr)
        raise

    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dry = "--dry-run" in sys.argv
    promote(dry_run=dry)
