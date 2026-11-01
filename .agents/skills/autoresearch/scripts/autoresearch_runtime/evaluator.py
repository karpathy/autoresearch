"""Runs evaluator jobs: builds their environment, validates results, and tracks budgets and usage."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


from .config import LabConfig
from .errors import LabError
from .state import _runs
from .store import _read_json, _sha256, _write_json
from .runtime_env import hidden_process_kwargs


def _validate_result(config: LabConfig, run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    try:
        result = _read_json(result_path)
        if not isinstance(result, dict):
            raise ValueError("Result must be a JSON object")
        metrics = result["metrics"]
        required = (config.primary, *(name for name, _ in config.guards))
        if not isinstance(metrics, dict) or any(name not in metrics for name in required):
            raise ValueError("Required metric missing")
        for name, value in metrics.items():
            if not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Metrics must be finite numbers")
        usage = result.setdefault("usage", {"requests": 0, "cost_usd": 0.0})
        if not isinstance(usage, dict):
            raise ValueError("Invalid usage")
        usage.setdefault("requests", 0)
        requests = usage["requests"]
        if not isinstance(requests, int) or isinstance(requests, bool) or requests < 0:
            raise ValueError("Invalid usage")
        if "cost_usd" not in usage:
            usage["cost_usd"] = None if requests else 0.0
        unknown_requests = usage.get("unknown_requests", 0)
        if (
            not isinstance(unknown_requests, int)
            or isinstance(unknown_requests, bool)
            or unknown_requests < 0
        ):
            raise ValueError("Invalid usage.unknown_requests")
        cost = usage["cost_usd"]
        unknown_costs = usage.get("unknown_costs", 0)
        if isinstance(unknown_costs, bool) or not isinstance(unknown_costs, int) or unknown_costs < 0:
            raise ValueError("Invalid usage.unknown_costs")
        unknown_cost_usd = usage.get("unknown_cost_usd", 0)
        if (
            isinstance(unknown_cost_usd, bool)
            or not isinstance(unknown_cost_usd, (int, float))
            or not math.isfinite(unknown_cost_usd)
            or unknown_cost_usd < 0
        ):
            raise ValueError("Invalid usage.unknown_cost_usd")
        if requests > 0 and usage["cost_usd"] is None and not (
            unknown_requests or unknown_costs or unknown_cost_usd
        ):
            raise ValueError("Invalid usage: unknown cost requires explicit unknown evidence")
        if requests > 0 and (unknown_requests or unknown_costs or unknown_cost_usd):
            if cost is not None:
                usage.setdefault("confirmed_cost_usd", cost)
                usage["cost_usd"] = None
                cost = None
        cost_is_unknown = cost is None and requests > 0
        confirmed_cost = usage.get("confirmed_cost_usd")
        if confirmed_cost is not None and (
            isinstance(confirmed_cost, bool)
            or not isinstance(confirmed_cost, (int, float))
            or not math.isfinite(confirmed_cost)
            or confirmed_cost < 0
        ):
            raise ValueError("Invalid usage.confirmed_cost_usd")
        if (
            not cost_is_unknown
            and (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            )
        ):
            raise ValueError("Invalid usage")
        if "details" in result and not isinstance(result["details"], dict):
            raise ValueError("details must be an object")
        details = result.get("details")
        if isinstance(details, dict):
            origin = details.get("evidence_origin")
            if origin is not None and origin not in {"fresh", "offline_recompute", "historical", "imported"}:
                raise ValueError("Invalid details.evidence_origin")
            if "diagnostics" in details:
                diagnostics = details["diagnostics"]
                if not isinstance(diagnostics, list):
                    raise ValueError("details.diagnostics must be an array")
                for item in diagnostics:
                    if not isinstance(item, dict) or any(
                        not isinstance(item.get(key), str) or not item[key].strip()
                        for key in ("case_id", "slice", "failure_kind")
                    ):
                        raise ValueError("diagnostics need case_id, slice, and failure_kind")
        expected = config.expected_attestations
        if expected:
            observed = result.get("attestations")
            if not isinstance(observed, dict):
                raise ValueError("attestations missing")
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise ValueError(f"attestation mismatch: {key}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LabError(f"Invalid result.json: {error}") from error
    return result

def _spent(manifests: list[dict[str, Any]], state_dir: Path | None = None) -> dict[str, Any]:
    """Count each repeat once, merging evaluator, usage-file, and durable HTTP evidence."""
    seconds = requests = cost = unknown_requests = unknown_cost = unknown_cost_count = 0.0
    for manifest in manifests:
        run_items = manifest.get("runs", [])
        seconds += sum(float(run.get("duration_seconds", 0.0) or 0.0) for run in run_items)
        run_id = str(manifest.get("id", ""))
        repeat_indexes = set(range(1, len(run_items) + 1))
        if state_dir is not None:
            run_dir = state_dir / "runs" / run_id
            for repeat_dir in run_dir.glob("repeat_*"):
                try:
                    repeat_indexes.add(int(repeat_dir.name.removeprefix("repeat_")))
                except ValueError:
                    continue
        for index in repeat_indexes:
            run = run_items[index - 1] if index <= len(run_items) and isinstance(run_items[index - 1], dict) else {}
            run_dir = state_dir / "runs" / run_id / f"repeat_{index:02d}" if state_dir else None
            sources = [run.get("usage") if isinstance(run.get("usage"), dict) else {}]
            if run_dir is not None:
                usage_file = _read_json(run_dir / "usage.json", {})
                if isinstance(usage_file, dict):
                    sources.append(usage_file)
                attempts_path = run_dir / "attempts.jsonl"
                from .store import _sqlite_document
                attempts_target = _sqlite_document(attempts_path)
                if attempts_path.is_file() or (attempts_target is not None and attempts_target[2] == "attempts"):
                    from .case_checkpoint import summarize_usage

                    journal = summarize_usage(attempts_path)
                    if journal.get("requests") or attempts_path.is_file():
                        sources.append(journal)  # an empty journal says nothing about pricing
            usage = _max_metered_usage(*sources)
            requests += float(usage.get("requests", 0) or 0)
            cost += float(usage.get("confirmed_cost_usd", usage.get("cost_usd", 0)) or 0)
            unknown_requests += float(usage.get("unknown_requests", 0) or 0)
            unknown_cost += float(usage.get("unknown_cost_usd", 0) or 0)
            unknown_cost_count += float(usage.get("unknown_costs", 0) or 0)
            if (usage.get("requests", 0) and usage.get("cost_usd") is None
                    and not usage.get("unknown_requests") and not usage.get("unknown_costs")):
                unknown_cost_count += 1
    return {
        "seconds": seconds,
        "requests": requests,
        "cost_usd": None if unknown_requests or unknown_cost or unknown_cost_count else cost,
        "confirmed_cost_usd": cost,
        "unknown_requests": unknown_requests,
        "unknown_cost_usd": unknown_cost,
        "unknown_costs": unknown_cost_count,
    }


def _max_metered_usage(*sources: dict[str, Any]) -> dict[str, Any]:
    valid = [source for source in sources if isinstance(source, dict)]

    def number(source: dict[str, Any], key: str) -> float | None:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return float(value)

    requests = max((int(number(source, "requests") or 0) for source in valid), default=0)
    unknown_values = [int(number(source, "unknown_requests") or 0) for source in valid]
    unknown_requests = max(unknown_values, default=0)
    # A result that explicitly prices every request is authoritative over the
    # transport journal, whose per-request cost may be unavailable.
    fully_priced = any(
        int(number(source, "requests") or 0) == requests
        and int(number(source, "unknown_requests") or 0) == 0
        and int(number(source, "unknown_costs") or 0) == 0
        and (number(source, "cost_usd") is not None or number(source, "confirmed_cost_usd") is not None)
        for source in valid
    )
    if fully_priced:
        unknown_requests = 0

    confirmed_values = [
        value
        for source in valid
        for value in (number(source, "confirmed_cost_usd"), number(source, "cost_usd"))
        if value is not None
    ]
    confirmed = max(confirmed_values, default=0.0)
    unknown_cost_values = [number(source, "unknown_cost_usd") for source in valid]
    unknown_cost = max((value for value in unknown_cost_values if value is not None), default=0.0)
    unknown_cost_counts = [int(number(source, "unknown_costs") or 0) for source in valid]
    unknown_cost_count = max(unknown_cost_counts, default=0)
    inferred_unknown = not fully_priced and any(
        int(number(source, "requests") or 0) > 0 and number(source, "cost_usd") is None
        for source in valid
    )
    cost_is_unknown = bool(unknown_requests or unknown_cost_count or unknown_cost > 0 or inferred_unknown)
    merged = {
        "requests": requests,
        "unknown_requests": unknown_requests,
        "cost_usd": None if cost_is_unknown else confirmed,
    }
    if any(number(source, "confirmed_cost_usd") is not None or number(source, "cost_usd") is not None
           for source in valid):
        merged["confirmed_cost_usd"] = confirmed
    if any(number(source, "unknown_cost_usd") is not None for source in valid):
        merged["unknown_cost_usd"] = unknown_cost
    if unknown_cost_count:
        merged["unknown_costs"] = unknown_cost_count
    return merged


def _merge_attempt_usage(run_dir: Path, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist durable attempt totals and merge them with evaluator-reported totals."""
    from .case_checkpoint import summarize_usage

    run_dir = Path(run_dir)
    usage_path = run_dir / "usage.json"
    file_usage = _read_json(usage_path, {})
    if not isinstance(file_usage, dict):
        file_usage = {}
    attempts_path = run_dir / "attempts.jsonl"
    from .store import _sqlite_document
    attempts_target = _sqlite_document(attempts_path)
    usage_target = _sqlite_document(usage_path)
    has_attempts = attempts_path.is_file() or (attempts_target is not None and attempts_target[2] == "attempts")
    attempt_usage = summarize_usage(attempts_path) if has_attempts else {}
    if not attempt_usage.get("requests") and not attempts_path.is_file():
        attempt_usage = {}  # No journal rows yet: nothing was metered.
    metered = _max_metered_usage(file_usage, _disk_usage(run_dir), attempt_usage, base or {})
    result_usage = dict(base) if isinstance(base, dict) else {}
    result_usage.update(metered)
    if (has_attempts or usage_path.is_file()) and not (
        usage_target is not None and usage_target[2] == "usage"
    ):
        file_usage.update(metered)
        _write_json(usage_path, file_usage)
    return result_usage

def _remaining(config: LabConfig) -> float:
    """Return remaining time. Unknown spend and request/cost caps never stop a run."""
    spent = _spent(_runs(config), config.state_dir)
    limit = config.budget.get("max_seconds")
    if limit is not None and spent["seconds"] >= limit:
        raise LabError("Time budget exhausted")
    return float("inf") if limit is None else float(limit) - spent["seconds"]

def _budget_overrun(config: LabConfig) -> str | None:
    spent = _spent(_runs(config), config.state_dir)
    limit = config.budget.get("max_seconds")
    if limit is not None and spent["seconds"] > limit:
        return "Time budget exceeded"
    return None

def build_eval_env(config: LabConfig, run_dir: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Layers added later: hardened runtime env, protected-input map, caller overrides."""
    from .runtime_env import hardened_env
    from .vault import input_map_env

    env = hardened_env()
    env.update(input_map_env(config))
    if extra_env:
        env.update(extra_env)
    return env


def _evaluator_job_handle() -> int | None:
    """Read the durable worker Job handle before evaluator descendants can write."""
    if os.name != "nt":
        return None
    from .runtimeops import current_operation_id

    operation_id = current_operation_id()
    if not operation_id:
        return None
    from .sqlite_store import SQLiteStore

    operation = SQLiteStore().get_operation(operation_id)
    progress = operation.get("progress")
    handle = progress.get("worker_job_handle") if isinstance(progress, dict) else None
    if isinstance(handle, int) and not isinstance(handle, bool) and handle > 0:
        return handle
    if operation.get("pid") == os.getpid():
        # Direct in-process invocation has no isolated worker Job Object.
        return None
    raise LabError(
        "Windows worker cannot verify the evaluator process tree",
        code="worker_tree_unavailable",
    )


def _extract_error_tail(
    stdout: str, stderr: str, max_lines: int = 10, max_chars: int = 600,
) -> str:
    source = stderr if (stderr and stderr.strip()) else (stdout or "")
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = "..." + tail[-(max_chars - 3):]
    return tail


def _evaluate_once(config: LabConfig, run_dir: Path, timeout: float, *,
                   candidate: Path | None = None, extra_env: dict[str, str] | None = None,
                   evaluator: tuple[str, ...] | None = None, resume_offline: bool = False,
                   resume_id: str | None = None) -> tuple[str, dict[str, Any], str]:
    candidate = (candidate or (config.state_dir / "candidate")).resolve()
    replacements = {
        "{python}": sys.executable, "{candidate}": str(candidate),
        "{run_dir}": str(run_dir), "{config_dir}": str(config.path.parent),
    }
    command = [replacements.get(arg, arg) for arg in (evaluator or config.evaluator)]
    run_dir.mkdir(parents=True, exist_ok=True)
    repeat_row: tuple[Any, Any, int] | None = None
    from .sqlite_store import SQLiteStore
    from .store import _sqlite_document
    target = _sqlite_document(run_dir / "result.json")
    if target is not None and target[2] == "result":
        store, lab_id, _, run_id, repeat_index = target
        store.ensure_repeat_started(lab_id, run_id, repeat_index)
        repeat_row = (store, run_id, repeat_index)
    environment = os.environ.copy()
    skill_scripts = str(Path(__file__).resolve().parents[1])
    hooks_dir = str(Path(skill_scripts) / "hooks")
    path_items = [hooks_dir, skill_scripts, environment.get("PYTHONPATH", "")]
    environment["AUTORESEARCH_CASE_LOG"] = str(run_dir / "cases.jsonl")
    environment["AUTORESEARCH_ATTEMPTS_LOG"] = str(run_dir / "attempts.jsonl")
    environment.pop("AUTORESEARCH_HTTP_RETRY", None)
    environment.pop("AUTORESEARCH_NET_STATS", None)
    if config.network_retry:
        environment["AUTORESEARCH_NET_STATS"] = str(run_dir / "net-stats.json")
        environment["AUTORESEARCH_USAGE_PATH"] = str(run_dir / "usage.json")
    environment["PYTHONPATH"] = os.pathsep.join(item for item in path_items if item)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.update(build_eval_env(config, run_dir, extra_env=extra_env))
    # A measured evaluator must not inherit the host's harness-profile path.
    environment.pop("AUTORESEARCH_HARNESS_PROFILE", None)
    # Evaluator overrides cannot re-enable the old unbounded hook retry mode.
    environment.pop("AUTORESEARCH_HTTP_RETRY", None)
    capabilities = getattr(config, "evaluator_capabilities", {}) or {}
    if capabilities.get("case_resume") is True:
        environment["AUTORESEARCH_CASE_SCOPE_REQUIRED"] = "1"
    else:
        environment.pop("AUTORESEARCH_CASE_SCOPE_REQUIRED", None)
    if resume_offline:
        if capabilities.get("case_resume") is not True or capabilities.get("offline_verified") is not True:
            return "invalid", {}, "Evaluator is not verified for offline case resume"
        environment["AUTORESEARCH_RESUME_OFFLINE"] = "1"
    else:
        environment.pop("AUTORESEARCH_RESUME_OFFLINE", None)
    if resume_id:
        environment["AUTORESEARCH_RESUME_ID"] = resume_id
    else:
        environment.pop("AUTORESEARCH_RESUME_ID", None)
    evaluator_job_handle = _evaluator_job_handle()
    started = time.perf_counter()
    try:
        try:
            completed = subprocess.run(
                command, cwd=candidate, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout, check=False, env=environment,
                **hidden_process_kwargs(),
            )
        finally:
            # Do not validate, persist or finalize a result while an evaluator child
            # can still mutate the candidate or its evidence directories.
            if evaluator_job_handle is not None:
                from .runtimeops import wait_for_job_worker_only
                wait_for_job_worker_only(evaluator_job_handle)
        duration = time.perf_counter() - started
        (run_dir / "run.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        _apply_net_stats(run_dir)
        _merge_attempt_usage(run_dir)
        if completed.returncode:
            error_tail = _extract_error_tail(completed.stdout or "", completed.stderr or "")
            last_line = error_tail.splitlines()[-1][:140] if error_tail else ""
            reason = f"Evaluator exited {completed.returncode}: {last_line}" if last_line else f"Evaluator exited {completed.returncode}"
            crash_attempt: dict[str, Any] = {"duration_seconds": duration, **_usage_from_file(run_dir)}
            if error_tail:
                crash_attempt["error_tail"] = error_tail
            return "crash", crash_attempt, reason
        result = _validate_result(config, run_dir)
        result.setdefault("details", {}).setdefault("evidence_origin", "fresh")
        result["duration_seconds"] = duration
        result["result_sha256"] = _sha256(run_dir / "result.json")
        result["usage"] = _merge_attempt_usage(run_dir, result.get("usage"))
        _attach_net_stats(run_dir, result)
        _attach_case_log(config, run_dir, result, repeat_row)
        if repeat_row is not None:
            store, run_id, repeat_index = repeat_row
            store.append_result(config.lab_id, run_id, repeat_index, result)
            # Preserve the on-disk result until its validated value is durable.
            (run_dir / "result.json").unlink(missing_ok=True)
        return "ok", result, ""
    except subprocess.TimeoutExpired:
        _apply_net_stats(run_dir)
        _merge_attempt_usage(run_dir)
        return "timeout", {"duration_seconds": time.perf_counter() - started, **_usage_from_file(run_dir)}, "Evaluator timed out"
    except OSError as error:
        _merge_attempt_usage(run_dir)
        return "crash", {"duration_seconds": time.perf_counter() - started, **_usage_from_file(run_dir)}, f"Evaluator could not start: {error}"
    except LabError as error:
        _apply_net_stats(run_dir)
        _merge_attempt_usage(run_dir)
        return "invalid", {"duration_seconds": time.perf_counter() - started, **_usage_from_file(run_dir)}, str(error)


def _apply_net_stats(run_dir: Path) -> None:
    """Copy retry counts onto usage.json without replacing a confirmed cost."""
    stats = _read_json(run_dir / "net-stats.json")
    if not isinstance(stats, dict):
        return
    limited = stats.get("rate_limited_responses") or 0
    if isinstance(limited, bool) or not isinstance(limited, int) or limited < 0:
        return
    usage_path = run_dir / "usage.json"
    from .store import _sqlite_document
    usage_target = _sqlite_document(usage_path)
    if usage_target is not None and usage_target[2] == "usage":
        # The hook already records each rate-limit as a unique SQLite usage event.
        return
    usage = _read_json(usage_path, {})
    if not isinstance(usage, dict):
        usage = {}
    previous = usage.get("unknown_requests") or 0
    if not isinstance(previous, (int, float)) or isinstance(previous, bool):
        previous = 0
    usage["unknown_requests"] = max(int(previous), limited)
    usage.setdefault("requests", int(usage.get("requests") or 0))
    usage.setdefault("cost_usd", float(usage.get("cost_usd") or 0))
    _write_json(usage_path, usage)

def _attach_net_stats(run_dir: Path, result: dict[str, Any]) -> None:
    stats = _read_json(run_dir / "net-stats.json")
    if not isinstance(stats, dict):
        return
    limited = int(stats.get("rate_limited_responses") or 0)
    usage = result.setdefault("usage", {})
    if isinstance(usage, dict) and limited:
        previous = usage.get("unknown_requests") or 0
        usage["unknown_requests"] = max(int(previous or 0), limited)
        if usage.get("cost_usd") is not None:
            usage.setdefault("confirmed_cost_usd", usage["cost_usd"])
        usage["cost_usd"] = None
    result["net"] = {
        "rate_limited_responses": limited,
        "retry_wait_seconds": stats.get("retry_wait_seconds") or 0,
        "attempts": stats.get("attempts") or 0,
    }

def _disk_usage(run_dir: Path) -> dict[str, Any]:
    """Usage the evaluator wrote to AUTORESEARCH_USAGE_PATH; SQLite labs would otherwise ignore it."""
    path = Path(run_dir) / "usage.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _usage_from_file(run_dir: Path) -> dict[str, Any]:
    usage = _read_json(run_dir / "usage.json")
    if not isinstance(usage, dict):
        usage = {}
    disk = _disk_usage(run_dir)
    if disk:
        usage = _max_metered_usage(usage, disk)
    if not usage:
        return {}
    # Keep only metering fields; evaluator logs and API responses are never copied here.
    return {"usage": {key: usage[key] for key in
                       ("requests", "cost_usd", "confirmed_cost_usd", "unknown_requests", "unknown_cost_usd",
                        "unknown_costs")
                       if key in usage}}


def _attach_case_log(
    config: LabConfig,
    run_dir: Path,
    result: dict[str, Any],
    repeat_row: tuple[Any, Any, int] | None = None,
) -> None:
    if not isinstance(result, dict):
        return
    details = result.get("details")
    if result.get("cases") or (isinstance(details, dict) and details.get("cases")):
        return
    cases: list[dict[str, Any]] = []
    case_path = Path(run_dir) / "cases.jsonl"
    if case_path.is_file():
        try:
            for line in case_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        cases.append(parsed)
                except ValueError:
                    continue
                if len(cases) >= 500:
                    break
        except OSError:
            pass
    if not cases and repeat_row is not None:
        store, run_id, repeat_index = repeat_row
        try:
            loaded = store.load_checkpoints(config.lab_id, run_id, repeat_index=repeat_index)
            if isinstance(loaded, list):
                cases = [item for item in loaded if isinstance(item, dict)][:500]
        except Exception:
            pass
    if cases:
        result.setdefault("details", {})["cases"] = cases
