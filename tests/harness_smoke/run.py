"""Two-phase end-to-end smoke test of the autoresearch harness with a real Codex or Claude Code run.

Usage: python tests/harness_smoke/run.py {codex|claude} [--model MODEL] [--binary PATH] [--timeout 900] [--keep]

Phase 1: the agent gets the toy project and evaluator, but no spec, brief or known case. From the
skill's documentation it writes the spec, runs preflight and prepares the lab with confirmation.
The host then finds the lab config itself (never from the agent's text), reads its contract
fingerprint and writes the offline harness profile.
Phase 2: the same agent session resumes with the profile in its environment and runs the baseline
and one trial.
The host checks the outcome itself with `lab.py guide --json`, never from the agent's own report.
Output goes to .tests/harness_smoke/<agent>-<UTC timestamp>/ under the repo root.
Exit codes: 0 PASS, 1 FAIL, 2 usage error (codex without --model) or agent binary not found.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKILL = REPO / ".agents" / "skills" / "autoresearch"
RESULTS_ROOT = REPO / ".tests" / "harness_smoke"
AGENTS = ("codex", "claude")
WITHHELD = frozenset({"spec.json", "brief.md", "known_case.json", "README.md", "run_demo.py"})
PHASE1_PROMPT = (
    "Use the autoresearch skill in this workspace. The user has confirmed this passport. "
    "Project: toy/project. Goal: lower mean_error. Metric: mean_error, direction min. "
    "Guard: coverage must not fall. Evaluator: toy/evaluator.py, offline, makes no network calls. "
    "Limits: at most 3 trials, no paid calls. Constraints: never edit toy/project directly. "
    "Create what the lab needs outside the project, run preflight, and prepare the lab with "
    "confirmation. Then stop: do not run a baseline. Print the path of the created lab.json."
)
PHASE2_PROMPT = (
    "The host has approved the contract and set the harness profile in the environment; you have "
    "approval for confirmed operations. Run the baseline, then make one change in the lab's "
    "candidate directory that you expect to lower mean_error, and run one trial. Do not edit "
    "toy/project. Do not promote. Finish by printing the guide's best_id."
)
IGNORE_CACHE = shutil.ignore_patterns("__pycache__")


def _load_demo() -> Any:
    """Import examples/toy/run_demo.py by path, so its lab helpers are reused rather than copied."""
    path = REPO / "examples" / "toy" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("toy_run_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEMO = _load_demo()


def toy_ignore(directory: str, names: list[str]) -> set[str]:
    """Copy the toy without caches and without the files the agent must write itself."""
    ignored = {"__pycache__"}
    if Path(directory).resolve() == DEMO.TOY.resolve():
        ignored |= WITHHELD
    return ignored.intersection(names)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-phase end-to-end smoke test of the autoresearch harness with a real Codex or Claude Code run.")
    parser.add_argument("agent", choices=AGENTS)
    parser.add_argument("--binary", help="agent executable (default: the one found on PATH)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds per phase before the agent process tree is killed (default: %(default)s)")
    parser.add_argument("--model", help="agent model; required for codex (no default), claude defaults to haiku")
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = parser.parse_args(argv)
    if args.agent == "codex" and not args.model:
        parser.error("codex needs --model: there is no default model for codex")
    return args


def resolve_binary(agent: str, requested: str | None) -> str | None:
    if requested and os.path.isfile(requested):
        return os.path.abspath(requested)
    return shutil.which(requested or agent)


def agent_command(agent: str, binary: str, prompt: str, workspace: Path, model: str | None,
                  resume: bool) -> list[str]:
    """Phase 1 starts a session; phase 2 resumes the most recent session in the workspace."""
    if agent == "codex":
        if resume:
            return [binary, "exec", "resume", "--last", "--model", str(model), "--skip-git-repo-check",
                    "-c", 'sandbox_mode="workspace-write"', prompt]
        return [binary, "exec", "--model", str(model), "--sandbox", "workspace-write",
                "--skip-git-repo-check", "-C", str(workspace), prompt]
    return [binary, "-p", prompt, *(["--continue"] if resume else []), "--model", model or "haiku",
            "--max-budget-usd", "1", "--allowedTools", "Bash", "Read", "Edit", "Write", "Glob", "Grep",
            "--permission-mode", "acceptEdits"]


def agent_env(home: Path, profile: Path | None) -> dict[str, str]:
    """Phase 1 gets no harness profile; phase 2 gets the one the host wrote."""
    env = {key: value for key, value in os.environ.items() if key != "AUTORESEARCH_HARNESS_PROFILE"}
    env.update({"AUTORESEARCH_HOME": str(home), "AUTORESEARCH_NO_DASHBOARD": "1"})
    if profile is not None:
        env["AUTORESEARCH_HARNESS_PROFILE"] = str(profile)
    return env


def scrub(text: str, workspace: Path | None = None) -> str:
    """Replace absolute locations with labels, so printed output names no user paths."""
    pairs = [(str(REPO), "<repo>"), (str(Path.home()), "~")]
    if workspace is not None:
        pairs.append((str(workspace), "<workspace>"))
    for raw, label in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
        for variant in {raw, raw.replace("\\", "/")}:
            text = text.replace(variant, label)
    return text


def agent_version(binary: str, workspace: Path) -> str:
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return scrub(lines[0], workspace) if lines else "unknown"


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def launch(cmd: list[str], cwd: Path, env: dict[str, str], out_path: Path, err_path: Path,
           timeout: int) -> tuple[int | None, bool]:
    """Run the agent with stdout and stderr in files. On timeout, kill its whole process tree."""
    extra: dict[str, Any] = {}
    if os.name == "nt":
        extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        extra["start_new_session"] = True
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, **extra)
        except OSError as error:
            err.write(str(error).encode("utf-8", "replace"))
            return None, False
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            proc.wait()
            return None, True


def run_phase(name: str, args: argparse.Namespace, binary: str, workspace: Path, run_dir: Path,
              prompt: str, env: dict[str, str], resume: bool) -> dict[str, Any]:
    began = time.monotonic()
    exit_code, timed_out = launch(
        agent_command(args.agent, binary, prompt, workspace, args.model, resume), workspace, env,
        run_dir / f"{name}.stdout.txt", run_dir / f"{name}.stderr.txt", args.timeout)
    return {"exit_code": exit_code, "timed_out": timed_out,
            "seconds": round(time.monotonic() - began, 1)}


def find_lab_configs(home: Path) -> list[Path]:
    """Every lab.json the agent left under the lab home. The host trusts these, not the agent's text."""
    return sorted(path for path in home.glob("**/lab.json") if path.is_file())


def tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and "__pycache__" not in rel.parts:
            hashes[rel.as_posix()] = DEMO._sha256(path)
    return hashes


def _find_first(node: Any, key: str) -> Any:
    """Return the first value stored under `key` anywhere in a nested JSON payload."""
    if isinstance(node, dict):
        if key in node and node[key] is not None:
            return node[key]
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_first(child, key)
        if found is not None:
            return found
    return None


def _collect_runs(node: Any, runs: dict[str, dict[str, Any]]) -> None:
    """Gather run-like dicts (an id and a status) in document order, keyed by id."""
    if isinstance(node, dict):
        run_id = node.get("id", node.get("run_id"))
        if isinstance(run_id, str) and isinstance(node.get("status"), str):
            runs.setdefault(run_id, node)
        for child in node.values():
            _collect_runs(child, runs)
    elif isinstance(node, list):
        for child in node:
            _collect_runs(child, runs)


def _value(entry: Any, metric: str) -> float | None:
    if not isinstance(entry, dict):
        return None
    found = DEMO._metric(entry, metric)
    mean = entry.get("mean")
    if found is None and isinstance(mean, (int, float)) and not isinstance(mean, bool):
        found = float(mean)
    return found


def guide_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the baseline, trials and metrics from a `guide --json` payload, whatever its nesting."""
    baseline_card = _find_first(payload, "baseline")
    baseline_card = baseline_card if isinstance(baseline_card, dict) else {}
    runs: dict[str, dict[str, Any]] = {}
    _collect_runs(payload, runs)
    baseline_id = baseline_card.get("id")
    if not isinstance(baseline_id, str) or not baseline_id:
        baseline_id = next((rid for rid in runs if rid.startswith("baseline")), None)
    trials = [(rid, entry) for rid, entry in runs.items()
              if rid != baseline_id and not rid.startswith("baseline")]
    metric = _find_first(payload, "primary_metric") or "mean_error"
    baseline_entry = runs.get(baseline_id, baseline_card) if baseline_id else baseline_card
    trial_id, trial_entry = trials[-1] if trials else (None, {})
    return {
        "baseline_id": baseline_id,
        "trial_count": len(trials),
        "trial_id": trial_id,
        "decision": trial_entry.get("status"),
        "metric": metric,
        "baseline_value": _value(baseline_entry, metric),
        "trial_value": _value(trial_entry, metric),
    }


def run_smoke(args: argparse.Namespace, binary: str, workspace: Path, run_dir: Path,
              started: datetime) -> dict[str, Any]:
    toy = workspace / "toy"
    project = toy / "project"
    rule = project / "rule.txt"  # the measured rule, inside the toy project
    home = workspace / ".home"
    host = workspace / "host"

    shutil.copytree(DEMO.TOY, toy, ignore=toy_ignore)
    for skill in (".agents/skills/autoresearch", ".claude/skills/autoresearch"):
        shutil.copytree(SKILL, workspace / skill, ignore=IGNORE_CACHE)
    home.mkdir()
    host.mkdir()
    try:
        subprocess.run(["git", "init", "-q"], cwd=workspace, capture_output=True, check=False)
    except OSError:
        pass  # best effort; codex runs with --skip-git-repo-check
    rule_before = DEMO._sha256(rule)
    tree_before = tree_hashes(project)
    version = agent_version(binary, workspace)

    # Phase 1: the agent writes the spec and prepares the lab. No harness profile is set.
    phases: dict[str, dict[str, Any] | None] = {}
    phases["phase1"] = run_phase("phase1", args, binary, workspace, run_dir, PHASE1_PROMPT,
                                 agent_env(home, None), resume=False)

    # Host: find the lab config itself and read its contract fingerprint.
    found = find_lab_configs(home)
    config = found[0] if len(found) == 1 else None
    lab = DEMO.Lab(project, home, host, workspace)
    fingerprint: Any = None
    guide_error = None
    if config is not None:
        try:
            payload = lab.lab(str(config), "guide", "--json")
            fingerprint = payload.get("contract_fingerprint") if isinstance(payload, dict) else None
        except DEMO.DemoError as error:
            guide_error = str(error)
    lab_ready = isinstance(fingerprint, str) and bool(fingerprint)
    checks: dict[str, bool] = {
        "agent_wrote_spec": len(found) == 1,  # one lab found: prepare ran
        "agent_prepared_lab": lab_ready,
    }

    # Phase 2: the host approves the contract, then the same agent session resumes.
    if lab_ready:
        profile = host / "profile.json"
        profile.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "harness-smoke", "version": "1"},
            "approval": {"approved": True, "contract_fingerprint": fingerprint},
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
        }), encoding="utf-8")
        lab.env["AUTORESEARCH_HARNESS_PROFILE"] = str(profile)
        phases["phase2"] = run_phase("phase2", args, binary, workspace, run_dir, PHASE2_PROMPT,
                                     agent_env(home, profile), resume=True)
    else:
        phases["phase2"] = None  # skipped: no usable lab to run

    # Host verification: the guide and the project files, not the agent's report.
    rule_after = DEMO._sha256(rule) if rule.is_file() else None
    tree_after = tree_hashes(project)
    changed = [name for name in sorted(set(tree_before) | set(tree_after))
               if tree_before.get(name) != tree_after.get(name)]
    checks["agent_exit_zero"] = all(
        phase is not None and phase["exit_code"] == 0 and not phase["timed_out"]
        for phase in phases.values())
    facts = guide_facts({})
    if lab_ready:
        try:
            payload = lab.lab(str(config), "guide", "--json")
        except DEMO.DemoError as error:
            payload, guide_error = {}, str(error)
        facts = guide_facts(payload)
        checks.update({
            "guide_readable": guide_error is None,
            "baseline_present": bool(facts["baseline_id"]),
            "trial_present": facts["trial_count"] >= 1,
            "metrics_recorded": facts["baseline_value"] is not None and facts["trial_value"] is not None,
            "project_rule_untouched": rule_after is not None and rule_after == rule_before,
        })
    failing = [name for name, ok in checks.items() if not ok]
    return {
        "agent": args.agent,
        "agent_version": version,
        "binary": Path(binary).name,
        "model": args.model or "haiku",
        "date_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompts": {"phase1": PHASE1_PROMPT, "phase2": PHASE2_PROMPT},
        "phases": phases,
        "lab_configs_found": len(found),
        "lab_config": config.relative_to(workspace).as_posix() if config is not None else None,
        "baseline_id": facts["baseline_id"],
        "trial_count": facts["trial_count"],
        "trial_id": facts["trial_id"],
        "decision": facts["decision"],
        "metric": facts["metric"],
        "baseline_value": facts["baseline_value"],
        "trial_value": facts["trial_value"],
        "rule_sha256_before": rule_before,
        "rule_sha256_after": rule_after,
        "project_rule_untouched": rule_after is not None and rule_after == rule_before,
        "project_files_changed": changed,
        "guide_error": guide_error,
        "checks": checks,
        "failing_checks": failing,
        "overall": "FAIL" if failing else "PASS",
        "results_dir": run_dir.relative_to(REPO).as_posix(),
    }


def print_summary(report: dict[str, Any]) -> None:
    def fmt(value: float | None) -> str:
        return "missing" if value is None else f"{value:.3f}"

    def phase_text(phase: dict[str, Any] | None) -> str:
        if phase is None:
            return "skipped (no usable lab)"
        exit_text = "timed out" if phase["timed_out"] else str(phase["exit_code"])
        return f"exit {exit_text}  ({phase['seconds']} s)"

    lab_text = report["lab_config"] or f"none usable ({report['lab_configs_found']} lab.json found)"
    print(f"agent          {report['agent']}")
    print(f"agent version  {report['agent_version']}")
    print(f"date           {report['date_utc']}")
    print(f"phase 1        {phase_text(report['phases'].get('phase1'))}")
    print(f"phase 2        {phase_text(report['phases'].get('phase2'))}")
    print(f"lab config     {lab_text}")
    print(f"baseline       {'present' if report['baseline_id'] else 'missing'}")
    print(f"trials         {report['trial_count']}")
    print(f"decision       {report['decision'] or 'none'}")
    print(f"metrics        {report['metric']}: baseline={fmt(report['baseline_value'])} "
          f"trial={fmt(report['trial_value'])}")
    print(f"project        rule.txt {'byte-identical' if report['project_rule_untouched'] else 'CHANGED'}; "
          f"files changed under toy/project: {len(report['project_files_changed'])}")
    print(f"results        {report['results_dir']}/")
    if report.get("workspace_kept"):
        print(f"workspace kept {report['workspace_kept']}")
    if report["overall"] == "PASS":
        print("overall        PASS")
    else:
        print(f"overall        FAIL (failing: {', '.join(report['failing_checks'])})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = resolve_binary(args.agent, args.binary)
    if binary is None:
        print(f"{args.agent}: agent binary not found (use --binary PATH)", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    run_dir = RESULTS_ROOT / f"{args.agent}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="harness-smoke-")).resolve()
    try:
        report = run_smoke(args, binary, workspace, run_dir, started)
    except (DEMO.DemoError, OSError, ValueError) as error:
        print(f"host setup failed: {scrub(str(error), workspace)}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)
    if args.keep:
        report["workspace_kept"] = f"<tmp>/{workspace.name}"
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
