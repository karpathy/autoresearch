"""Offline toy run of the SQLite lab flow, in throwaway directories.

Run from any directory:  python examples/toy/run_demo.py

Copies examples/toy into a temp project, uses a temp AUTORESEARCH_HOME and a temp
host profile, runs prepare, guide, baseline and one trial, then prints a summary.
The temp directories are removed afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOY = Path(__file__).resolve().parent
REPO = TOY.parents[1]
SCRIPTS = REPO / ".agents" / "skills" / "autoresearch" / "scripts"
PREPARE = SCRIPTS / "prepare.py"
LAB = SCRIPTS / "lab.py"
BASELINE_METRICS = {"mean_error": 2.0, "coverage": 3.0}  # rule.txt = 1 against cases.json
CANDIDATE_RULE = "2\n"  # the one-line edit under test
TRIAL_FLAGS = [
    "--title", "Double the rule factor",
    "--observation", "Factor 1 undershoots every case by its input",
    "--expected-effect", "mean_error falls from 2 to 0",
]


class DemoError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_known_case(project: Path) -> None:
    """Write the known case for this copy; its hashes depend on the checked-out bytes."""
    known_case = project / "known_case.json"
    known_case.write_text(json.dumps({
        "schema": 1,
        "id": "toy-baseline-v1",
        "reproducible": True,
        "evaluator_sha256": _sha256(project / "evaluator.py"),
        "input_hashes": {},
        "expected_metrics": BASELINE_METRICS,
        "observed_metrics": BASELINE_METRICS,
        "tolerance": 1e-9,
    }), encoding="utf-8")
    spec_path = project / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["baseline_validation"]["known_case"]["sha256"] = _sha256(known_case)
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def _metric(data: Any, name: str) -> float | None:
    """Return the first metrics mapping that carries `name`, searching nested dicts and lists."""
    if isinstance(data, dict):
        metrics = data.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get(name), (int, float)):
            return float(metrics[name])
        children = list(data.values())
    elif isinstance(data, list):
        children = data
    else:
        return None
    for child in children:
        found = _metric(child, name)
        if found is not None:
            return found
    return None


class Lab:
    """Runs CLI commands with the demo environment and records shortened command lines."""

    def __init__(self, project: Path, home: Path, host: Path, base: Path):
        self.project = project
        self.env = {key: value for key, value in os.environ.items()
                    if key != "AUTORESEARCH_HARNESS_PROFILE"}
        self.env.update({
            "AUTORESEARCH_HOME": str(home),
            "AUTORESEARCH_NO_DASHBOARD": "1",
            "AUTORESEARCH_NO_BROWSER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(SCRIPTS),
        })
        pairs = [(str(project), "<project>"), (str(home), "<lab>"), (str(host), "<host>"),
                 (str(base), "<tmp>"), (str(REPO), "<repo>"), (sys.executable, "<python>"),
                 (str(Path.home()), "~")]
        self.replacements = sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)
        self.shown: list[str] = []

    def scrub(self, text: str) -> str:
        for raw, label in self.replacements:
            for variant in {raw, raw.replace("\\", "/")}:
                text = text.replace(variant, label)
        return text

    def run(self, script: Path, args: list[str]) -> dict[str, Any]:
        shown = self.scrub(" ".join([script.name, *args]))
        self.shown.append(shown)
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(script), *args],
                cwd=self.project, env=self.env, capture_output=True, text=True,
                encoding="utf-8", timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise DemoError(f"{shown}\ntimed out after {error.timeout} seconds") from error
        output = proc.stdout.strip() or proc.stderr.strip()
        try:
            payload = json.loads(output)
        except ValueError:
            payload = None
        if proc.returncode != 0 or not isinstance(payload, dict) or payload.get("ok") is not True:
            raise DemoError(f"{shown}\nexit {proc.returncode}\n{self.scrub(output)}")
        return payload["data"]

    def lab(self, config: str, *args: str) -> dict[str, Any]:
        return self.run(LAB, ["--config", config, "--headless", *args])


def run_flow(base: Path, host: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    project = base / "toy"
    home = base / "lab-home"
    shutil.copytree(TOY, project)
    _pin_known_case(project)
    lab = Lab(project, home, host, base)
    prepared = lab.run(PREPARE, [
        "--program", str(project / "brief.md"), "--spec", str(project / "spec.json"),
        "--confirmed", "--no-dashboard", "--no-browser",
    ])
    config = str(prepared["config"])
    fingerprint = lab.lab(config, "guide", "--json")["contract_fingerprint"]

    profile = host / "harness-profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "harness": {"id": "toy-offline-demo", "version": "1"},
        "approval": {"approved": True, "contract_fingerprint": fingerprint},
        "sandbox": {"enabled": True, "scope": "workspace"},
        "network": {"mode": "disabled"},
    }), encoding="utf-8")
    lab.env["AUTORESEARCH_HARNESS_PROFILE"] = str(profile)

    baseline = lab.lab(config, "--confirmed", "baseline")
    (Path(prepared["candidate"]) / "rule.txt").write_text(CANDIDATE_RULE, encoding="utf-8")
    trial = lab.lab(config, "--confirmed", "trial", *TRIAL_FLAGS)
    lab.lab(config, "guide", "--json")
    return baseline, trial, lab.shown


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="autoresearch-toy-"))
    host = Path(tempfile.mkdtemp(prefix="autoresearch-toy-host-"))
    try:
        baseline, trial, shown = run_flow(base, host)
        base_value = _metric(baseline.get("runs", baseline), "mean_error")
        trial_value = _metric(trial.get("runs", trial), "mean_error")
        if base_value is None or trial_value is None:
            raise DemoError("mean_error is missing from the baseline or trial result")
    except (DemoError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        for path in (base, host):
            shutil.rmtree(path, ignore_errors=True)

    print("toy demo: offline evaluator; temp project and lab, removed on exit")
    print(f"baseline  mean_error={base_value:.3f}  status={baseline.get('status')}")
    print(f"trial     mean_error={trial_value:.3f}  id={trial.get('id')}")
    print(f"decision  {trial.get('status')}  (mean_error {base_value:.3f} -> {trial_value:.3f}, min)")
    print("commands:")
    for line in shown:
        print(f"  $ {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
