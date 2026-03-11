from __future__ import annotations

import ast
import builtins
from dataclasses import asdict, dataclass
import difflib
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
from ttt_autoresearch.hyperbolic import HyperbolicPool
from ttt_autoresearch.runpod import RunPodPool


VAL_BPB_RE = re.compile(r"^val_bpb:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", re.MULTILINE)
SEARCH_REPLACE_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)
VAL_BPB_PRINT_RE = re.compile(r"print\(\s*f?[\"']val_bpb:\s*", re.MULTILINE)
FORWARD_WITH_REDUCTION_RE = re.compile(r"def\s+forward\s*\([^)]*\breduction\s*=", re.MULTILINE)
_KNOWN_PREPARE_CONSTANTS = {"MAX_SEQ_LEN": 2048}


@dataclass(slots=True)
class PatchCandidate:
    summary: str
    rationale: str
    train_py: str
    candidate_format: str
    patch_block_count: int
    lines_changed: int


@dataclass(slots=True)
class PreflightResult:
    ok: bool
    stage: str
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "reason": self.reason,
            "details": self.details,
        }


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
    return parse_patch_candidate_for_state(candidate_json, "")


def parse_patch_candidate_for_state(candidate_json: str, current_train_py: str) -> PatchCandidate:
    stripped = candidate_json.strip()
    if not stripped:
        raise ValueError("Candidate must not be empty.")

    updated_train_py, patch_block_count, extracted = apply_search_replace_patch(stripped, current_train_py)
    lines_changed = count_lines_changed(current_train_py, updated_train_py)
    if lines_changed == 0:
        raise ValueError("Patch did not change train.py.")
    return PatchCandidate(
        summary="search_replace_patch_candidate",
        rationale="model returned search/replace patch",
        train_py=updated_train_py,
        candidate_format="search_replace_patch_extracted" if extracted else "search_replace_patch",
        patch_block_count=patch_block_count,
        lines_changed=lines_changed,
    )


def apply_search_replace_patch(patch_text: str, current_train_py: str) -> tuple[str, int, bool]:
    blocks = list(SEARCH_REPLACE_BLOCK_RE.finditer(patch_text))
    if not blocks:
        raise ValueError("Candidate must contain one or more SEARCH/REPLACE patch blocks.")

    updated = current_train_py
    for match in blocks:
        search_text = match.group(1)
        replace_text = match.group(2)
        if not search_text:
            raise ValueError("SEARCH block must not be empty.")
        occurrences = updated.count(search_text)
        if occurrences == 0:
            raise ValueError("SEARCH block did not match the current train.py.")
        if occurrences > 1:
            raise ValueError("SEARCH block matched multiple locations. Make the patch more specific.")
        updated = updated.replace(search_text, replace_text, 1)

    extracted = _has_non_block_wrapper_text(patch_text, blocks)
    return updated, len(blocks), extracted


def count_lines_changed(previous_text: str, updated_text: str) -> int:
    changed = 0
    for line in difflib.unified_diff(
        previous_text.splitlines(),
        updated_text.splitlines(),
        lineterm="",
    ):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith(("+", "-")):
            changed += 1
    return changed


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
        self._hyperbolic_pool: HyperbolicPool | None = None
        self._runpod_pool: RunPodPool | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "baseline").mkdir(exist_ok=True)
        (self.run_dir / "candidates").mkdir(exist_ok=True)
        (self.run_dir / "best").mkdir(exist_ok=True)

    def build_bootstrap(self, baseline_val_bpb: float) -> BootstrapContext:
        program_text = (self.repo_root / "program.md").read_text(encoding="utf-8")
        baseline_train_py = self._load_baseline_train_py()
        return BootstrapContext(
            repo_root=self.repo_root,
            run_dir=self.run_dir,
            config=self.config,
            program_text=program_text,
            baseline_train_py=baseline_train_py,
            baseline_val_bpb=baseline_val_bpb,
        )

    def load_existing_baseline_result(self) -> RunResult | None:
        baseline_path = self.run_dir / "baseline.json"
        if not baseline_path.exists():
            return None
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        try:
            return RunResult(
                status=str(payload["status"]),
                val_bpb=float(payload["val_bpb"]) if payload.get("val_bpb") is not None else None,
                stdout_path=Path(payload["stdout_path"]),
                stderr_path=Path(payload["stderr_path"]),
                elapsed_sec=float(payload["elapsed_sec"]),
                workspace_path=Path(payload["workspace_path"]),
                metrics_path=Path(payload["metrics_path"]) if payload.get("metrics_path") else None,
                command=[str(part) for part in payload.get("command", [])],
                returncode=int(payload["returncode"]) if payload.get("returncode") is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def run_baseline(self, bootstrap: BootstrapContext | None = None) -> RunResult:
        workspace = self.run_dir / "baseline" / "workspace"
        self._copy_repo(workspace)
        (self.run_dir / "baseline" / "train.py").write_text((workspace / "train.py").read_text(encoding="utf-8"), encoding="utf-8")
        result = self._execute_workspace(
            workspace=workspace,
            command_template=self.config.baseline_command_override,
            bootstrap=bootstrap,
            label="baseline",
        )
        self._write_json(self.run_dir / "baseline.json", result.to_dict())
        return result

    def prepare_candidate_workspace(
        self,
        candidate: PatchCandidate,
        step: int,
        *,
        prefix: str = "candidate",
    ) -> Path:
        workspace = self.run_dir / "candidates" / f"{step:04d}_{prefix}_{uuid.uuid4().hex[:8]}"
        self._copy_repo(workspace)
        (workspace / "train.py").write_text(candidate.train_py, encoding="utf-8")
        (workspace / "applied_train.py").write_text(candidate.train_py, encoding="utf-8")
        return workspace

    def preflight_candidate(self, workspace: Path, candidate: PatchCandidate) -> PreflightResult:
        train_path = workspace / "train.py"
        source = train_path.read_text(encoding="utf-8")
        try:
            module = ast.parse(source, filename=str(train_path))
        except SyntaxError as exc:
            return PreflightResult(
                ok=False,
                stage="syntax",
                reason="train.py does not parse as Python",
                details={
                    "lineno": exc.lineno,
                    "offset": exc.offset,
                    "text": exc.text,
                    "message": exc.msg,
                },
            )

        undefined_name = _find_top_level_undefined_name(module)
        if undefined_name is not None:
            return PreflightResult(
                ok=False,
                stage="top_level_names",
                reason=f"Top-level code references undefined name {undefined_name!r}.",
                details={"name": undefined_name},
            )

        if not VAL_BPB_PRINT_RE.search(source):
            return PreflightResult(
                ok=False,
                stage="summary_output",
                reason="train.py no longer prints a final val_bpb summary line.",
                details={"required_pattern": "print(... val_bpb: ...)"},
            )

        if not FORWARD_WITH_REDUCTION_RE.search(source):
            return PreflightResult(
                ok=False,
                stage="forward_signature",
                reason="train.py no longer defines a forward(...) with a reduction parameter.",
                details={"required_pattern": "def forward(... reduction=...)"},
            )

        divisibility = _check_batch_divisibility(module)
        if not divisibility.ok:
            return divisibility

        compiled = subprocess.run(
            [sys.executable, "-m", "py_compile", str(train_path)],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if compiled.returncode != 0:
            return PreflightResult(
                ok=False,
                stage="py_compile",
                reason="python -m py_compile failed.",
                details={"stdout": compiled.stdout, "stderr": compiled.stderr},
            )

        return PreflightResult(
            ok=True,
            stage="ok",
            reason="Preflight checks passed.",
            details=divisibility.details,
        )

    def run_candidate(
        self,
        bootstrap: BootstrapContext,
        workspace: Path,
        step: int,
        state_id: str,
        gpu_device: str | None = None,
    ) -> RunResult:
        return self._execute_workspace(
            workspace=workspace,
            command_template=self.config.candidate_command_override,
            bootstrap=bootstrap,
            label=f"candidate-{step:04d}",
            state_id=state_id,
            gpu_device=gpu_device,
        )

    def create_candidate_artifact_dir(self, step: int, prefix: str = "candidate") -> Path:
        label = prefix.replace(" ", "_")
        workspace = self.run_dir / "candidates" / f"{step:04d}_{label}_{uuid.uuid4().hex[:8]}"
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

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

    def write_rollout_manifest(self, workspace: Path, payload: dict[str, Any]) -> Path:
        path = workspace / "rollout_manifest.json"
        self._write_json(path, payload)
        return path

    def write_json_artifact(self, path: Path, payload: dict[str, Any]) -> Path:
        self._write_json(path, payload)
        return path

    def close(self) -> None:
        if self._hyperbolic_pool is not None:
            self._hyperbolic_pool.close()
            self._hyperbolic_pool = None
        if self._runpod_pool is not None:
            self._runpod_pool.close()
            self._runpod_pool = None

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
        gpu_device: str | None = None,
    ) -> RunResult:
        command = self._resolve_command(command_template, workspace, bootstrap, label, state_id)
        env = bootstrap.subprocess_env() if bootstrap else dict(os.environ)
        if gpu_device is not None and self.config.execution_backend in {"local", "hyperbolic"}:
            env["CUDA_VISIBLE_DEVICES"] = gpu_device
        stdout_path = workspace / "stdout.log"
        stderr_path = workspace / "stderr.log"
        metrics_path = workspace / "metrics.json"
        start = time.time()
        if self.config.execution_backend == "runpod":
            pool = self._get_runpod_pool()
            remote_result = pool.execute_workspace(
                workspace=workspace,
                command=command,
                env=env,
                timeout_sec=self.config.timeout_sec,
                label=label,
            )
            stdout = remote_result.stdout
            stderr = remote_result.stderr
            returncode = remote_result.returncode
            elapsed_sec = remote_result.elapsed_sec
            if returncode == 124:
                status = "timeout"
                returncode = None
            else:
                status = "success" if returncode == 0 else "crash"
        elif self.config.execution_backend == "hyperbolic":
            pool = self._get_hyperbolic_pool()
            remote_result = pool.execute_workspace(
                workspace=workspace,
                command=command,
                env=env,
                timeout_sec=self.config.timeout_sec,
                label=label,
            )
            stdout = remote_result.stdout
            stderr = remote_result.stderr
            returncode = remote_result.returncode
            elapsed_sec = remote_result.elapsed_sec
            if returncode == 124:
                status = "timeout"
                returncode = None
            else:
                status = "success" if returncode == 0 else "crash"
        else:
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

        metrics = {"val_bpb": val_bpb}
        if val_bpb is not None and metrics_path.exists():
            try:
                loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                metrics["metrics_json_error"] = str(exc)
            else:
                if isinstance(loaded_metrics, dict):
                    metrics = loaded_metrics
                    metrics.setdefault("val_bpb", val_bpb)
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

    def _get_runpod_pool(self) -> RunPodPool:
        if self._runpod_pool is None:
            self._runpod_pool = RunPodPool(repo_root=self.repo_root, run_dir=self.run_dir, config=self.config)
        return self._runpod_pool

    def _get_hyperbolic_pool(self) -> HyperbolicPool:
        if self._hyperbolic_pool is None:
            self._hyperbolic_pool = HyperbolicPool(repo_root=self.repo_root, run_dir=self.run_dir, config=self.config)
        return self._hyperbolic_pool

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

    def _load_baseline_train_py(self) -> str:
        for candidate in (
            self.run_dir / "baseline" / "train.py",
            self.run_dir / "baseline" / "workspace" / "train.py",
            self.repo_root / "train.py",
        ):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError("Could not locate baseline train.py in either the run directory or repo root.")


def _check_batch_divisibility(module: ast.Module) -> PreflightResult:
    env: dict[str, int] = dict(_KNOWN_PREPARE_CONSTANTS)
    tracked = {"TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE", "MAX_SEQ_LEN", "tokens_per_fwdbwd"}

    for stmt in module.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or target.id not in tracked:
            continue
        value = _safe_eval_int_expr(stmt.value, env)
        if value is not None:
            env[target.id] = value

    missing = [name for name in ("TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE", "MAX_SEQ_LEN") if name not in env]
    if missing:
        return PreflightResult(
            ok=False,
            stage="batch_divisibility",
            reason=f"Could not statically resolve required batch constants: {missing}.",
            details={"resolved": env},
        )

    tokens_per_fwdbwd = env.get("tokens_per_fwdbwd", env["DEVICE_BATCH_SIZE"] * env["MAX_SEQ_LEN"])
    if tokens_per_fwdbwd <= 0:
        return PreflightResult(
            ok=False,
            stage="batch_divisibility",
            reason="tokens_per_fwdbwd must be positive.",
            details={"resolved": env},
        )
    if env["TOTAL_BATCH_SIZE"] % tokens_per_fwdbwd != 0:
        return PreflightResult(
            ok=False,
            stage="batch_divisibility",
            reason="TOTAL_BATCH_SIZE is not divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN.",
            details={
                "TOTAL_BATCH_SIZE": env["TOTAL_BATCH_SIZE"],
                "DEVICE_BATCH_SIZE": env["DEVICE_BATCH_SIZE"],
                "MAX_SEQ_LEN": env["MAX_SEQ_LEN"],
                "tokens_per_fwdbwd": tokens_per_fwdbwd,
            },
        )

    return PreflightResult(
        ok=True,
        stage="batch_divisibility",
        reason="Batch-size divisibility check passed.",
        details={
            "TOTAL_BATCH_SIZE": env["TOTAL_BATCH_SIZE"],
            "DEVICE_BATCH_SIZE": env["DEVICE_BATCH_SIZE"],
            "MAX_SEQ_LEN": env["MAX_SEQ_LEN"],
            "tokens_per_fwdbwd": tokens_per_fwdbwd,
            "grad_accum_steps": env["TOTAL_BATCH_SIZE"] // tokens_per_fwdbwd,
        },
    )


def _safe_eval_int_expr(node: ast.AST, env: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_eval_int_expr(node.operand, env)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _safe_eval_int_expr(node.left, env)
        right = _safe_eval_int_expr(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right if right != 0 else None
        if isinstance(node.op, ast.Div):
            return left // right if right != 0 and left % right == 0 else None
        if isinstance(node.op, ast.Mod):
            return left % right if right != 0 else None
        if isinstance(node.op, ast.Pow):
            return left ** right
    return None


def _find_top_level_undefined_name(module: ast.Module) -> str | None:
    defined = set(dir(builtins)) | {
        "__name__",
        "__file__",
        "__package__",
        "__spec__",
        "__builtins__",
    }
    defined.update(_collect_defined_names(module.body))

    for stmt in module.body:
        for name in _top_level_loaded_names(stmt):
            if name not in defined:
                return name
    return None


def _top_level_loaded_names(stmt: ast.stmt) -> set[str]:
    loaded: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                visit(decorator)
            for default in node.args.defaults:
                visit(default)
            for default in node.args.kw_defaults:
                if default is not None:
                    visit(default)
            if node.returns is not None:
                visit(node.returns)
            return
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                visit(decorator)
            for base in node.bases:
                visit(base)
            for keyword in node.keywords:
                visit(keyword.value)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(stmt)
    return loaded


def _names_defined_by_stmt(stmt: ast.stmt) -> set[str]:
    names: set[str] = set()

    def add_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                add_target(elt)

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(stmt.name)
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            add_target(target)
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        add_target(stmt.target)
    elif isinstance(stmt, ast.With):
        for item in stmt.items:
            if item.optional_vars is not None:
                add_target(item.optional_vars)
    elif isinstance(stmt, ast.Import):
        for alias in stmt.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            names.add(alias.asname or alias.name)
    return names


def _collect_defined_names(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        names.update(_names_defined_by_stmt(stmt))
    return names


def _has_non_block_wrapper_text(response_text: str, blocks: list[re.Match[str]]) -> bool:
    pieces: list[str] = []
    cursor = 0
    for match in blocks:
        pieces.append(response_text[cursor:match.start()])
        cursor = match.end()
    pieces.append(response_text[cursor:])
    wrapper = "".join(pieces).strip()
    if not wrapper:
        return False
    wrapper = wrapper.replace("```", "").strip()
    return bool(wrapper)
