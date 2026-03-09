from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

from ttt_autoresearch.config import BootstrapContext, TTTAutoResearchConfig


VAL_BPB_RE = re.compile(r"^val_bpb:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", re.MULTILINE)
ALLOWED_CANDIDATE_KEYS = {"summary", "rationale", "train_py"}


@dataclass(slots=True)
class PatchCandidate:
    summary: str
    rationale: str
    train_py: str


@dataclass(slots=True)
class RunResult:
    status: str
    val_bpb: float | None
    stdout_path: Path
    stderr_path: Path
    elapsed_sec: float
    workspace_path: Path
    metrics_path: Path | None
    command: list[str]
    returncode: int | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stdout_path"] = str(self.stdout_path)
        data["stderr_path"] = str(self.stderr_path)
        data["workspace_path"] = str(self.workspace_path)
        data["metrics_path"] = str(self.metrics_path) if self.metrics_path else None
        return data


def parse_patch_candidate(candidate_json: str) -> PatchCandidate:
    try:
        payload = json.loads(candidate_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Candidate must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Candidate payload must be a JSON object.")
    unknown_keys = set(payload) - ALLOWED_CANDIDATE_KEYS
    if unknown_keys:
        raise ValueError(f"Candidate may only contain {sorted(ALLOWED_CANDIDATE_KEYS)}. Found {sorted(unknown_keys)}.")
    missing = [key for key in ("summary", "rationale", "train_py") if key not in payload]
    if missing:
        raise ValueError(f"Candidate is missing required keys: {missing}.")
    summary = payload["summary"]
    rationale = payload["rationale"]
    train_py = payload["train_py"]
    if not all(isinstance(value, str) for value in (summary, rationale, train_py)):
        raise ValueError("Candidate fields summary, rationale, and train_py must all be strings.")
    if not train_py.strip():
        raise ValueError("train_py must contain the full replacement file.")
    return PatchCandidate(summary=summary.strip(), rationale=rationale.strip(), train_py=train_py)


def parse_val_bpb(stdout: str) -> float | None:
    match = VAL_BPB_RE.search(stdout)
    if not match:
        return None
    return float(match.group(1))


class AutoResearchRunner:
    def __init__(self, repo_root: Path, config: TTTAutoResearchConfig, run_dir: Path) -> None:
        self.repo_root = repo_root
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "baseline").mkdir(exist_ok=True)
        (self.run_dir / "candidates").mkdir(exist_ok=True)
        (self.run_dir / "best").mkdir(exist_ok=True)

    def build_bootstrap(self, baseline_val_bpb: float) -> BootstrapContext:
        program_text = (self.repo_root / "program.md").read_text(encoding="utf-8")
        baseline_train_py = (self.repo_root / "train.py").read_text(encoding="utf-8")
        return BootstrapContext(
            repo_root=self.repo_root,
            run_dir=self.run_dir,
            config=self.config,
            program_text=program_text,
            baseline_train_py=baseline_train_py,
            baseline_val_bpb=baseline_val_bpb,
        )

    def run_baseline(self, bootstrap: BootstrapContext | None = None) -> RunResult:
        workspace = self.run_dir / "baseline" / "workspace"
        self._copy_repo(workspace)
        result = self._execute_workspace(
            workspace=workspace,
            command_template=self.config.baseline_command_override,
            bootstrap=bootstrap,
            label="baseline",
        )
        self._write_json(self.run_dir / "baseline.json", result.to_dict())
        return result

    def run_candidate(
        self,
        bootstrap: BootstrapContext,
        candidate: PatchCandidate,
        step: int,
        state_id: str,
    ) -> RunResult:
        workspace = self.run_dir / "candidates" / f"{step:04d}_{uuid.uuid4().hex[:8]}"
        self._copy_repo(workspace)
        (workspace / "train.py").write_text(candidate.train_py, encoding="utf-8")
        result = self._execute_workspace(
            workspace=workspace,
            command_template=self.config.candidate_command_override,
            bootstrap=bootstrap,
            label=f"candidate-{step:04d}",
            state_id=state_id,
        )
        return result

    def initialize_best_from_baseline(self, baseline_result: RunResult, train_py_text: str) -> None:
        if baseline_result.val_bpb is None:
            return
        self.update_best(train_py_text, baseline_result, summary="baseline", rationale="seed baseline")

    def update_best(self, train_py_text: str, result: RunResult, summary: str, rationale: str) -> bool:
        if result.val_bpb is None:
            return False
        best_metrics_path = self.run_dir / "best" / "metrics.json"
        current_best = self.read_best_val_bpb()
        if current_best is not None and result.val_bpb >= current_best:
            return False
        (self.run_dir / "best" / "train.py").write_text(train_py_text, encoding="utf-8")
        self._write_json(
            best_metrics_path,
            {
                "summary": summary,
                "rationale": rationale,
                "val_bpb": result.val_bpb,
                "status": result.status,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "workspace_path": str(result.workspace_path),
                "elapsed_sec": result.elapsed_sec,
            },
        )
        return True

    def read_best_val_bpb(self) -> float | None:
        best_metrics_path = self.run_dir / "best" / "metrics.json"
        if not best_metrics_path.exists():
            return None
        data = json.loads(best_metrics_path.read_text(encoding="utf-8"))
        value = data.get("val_bpb")
        return float(value) if value is not None else None

    def append_history(self, entry: dict[str, Any]) -> None:
        history_path = self.run_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _copy_repo(self, workspace: Path) -> None:
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(
            self.repo_root,
            workspace,
            ignore=shutil.ignore_patterns(".git", "runs", "__pycache__", ".pytest_cache", ".venv", "*.pyc", "*.pyo"),
        )

    def _execute_workspace(
        self,
        workspace: Path,
        command_template: list[str] | None,
        bootstrap: BootstrapContext | None,
        label: str,
        state_id: str | None = None,
    ) -> RunResult:
        command = self._resolve_command(command_template, workspace, bootstrap, label, state_id)
        env = bootstrap.subprocess_env() if bootstrap else dict(os.environ)
        stdout_path = workspace / "stdout.log"
        stderr_path = workspace / "stderr.log"
        metrics_path = workspace / "metrics.json"
        start = time.time()
        try:
            proc = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                timeout=self.config.timeout_sec,
                text=True,
                capture_output=True,
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
            status = "success" if returncode == 0 else "crash"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = None
            status = "timeout"
        elapsed_sec = time.time() - start
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        val_bpb = self._read_val_bpb(stdout, metrics_path)
        if status == "success" and val_bpb is None:
            status = "missing_metric"

        if val_bpb is not None and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = {"val_bpb": val_bpb}
        self._write_json(metrics_path, metrics)

        return RunResult(
            status=status,
            val_bpb=val_bpb,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            elapsed_sec=elapsed_sec,
            workspace_path=workspace,
            metrics_path=metrics_path,
            command=command,
            returncode=returncode,
        )

    def _read_val_bpb(self, stdout: str, metrics_path: Path) -> float | None:
        direct = parse_val_bpb(stdout)
        if direct is not None:
            return direct
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            value = payload.get("val_bpb")
            return float(value) if value is not None else None
        return None

    def _resolve_command(
        self,
        command_template: list[str] | None,
        workspace: Path,
        bootstrap: BootstrapContext | None,
        label: str,
        state_id: str | None,
    ) -> list[str]:
        template = command_template or [sys.executable, "train.py"]
        values = {
            "workspace": str(workspace),
            "repo_root": str(self.repo_root),
            "run_dir": str(self.run_dir),
            "label": label,
            "state_id": state_id or "",
            "data_path": bootstrap.config.data_path if bootstrap and bootstrap.config.data_path else "",
        }
        resolved = []
        for part in template:
            for key, val in values.items():
                part = part.replace("{" + key + "}", val)
            resolved.append(part)
        return resolved

    @staticmethod
    def read_text(path: Path, max_chars: int = 4000) -> str:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...(truncated)...\n"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
