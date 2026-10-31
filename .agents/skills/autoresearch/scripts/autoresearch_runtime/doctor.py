"""Preflight a machine and, when asked, one project."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .integrity import check_skill, read_version
from .locking import _pid_alive
from .paths import lab_storage_roots, resolve_home

STALE_SECONDS = 300


def _stale_running_manifests(home: Path, project: Path | None) -> list[str]:
    stale: list[str] = []
    manifests = []
    for storage in lab_storage_roots(home):
        manifests.extend(storage.rglob("runs/*/manifest.json"))
    for manifest_path in manifests:
        if "archive" in manifest_path.parts:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("status") != "running":
            continue
        state_dir = manifest_path.parent.parent.parent
        if project is not None:
            meta = state_dir / "lab-meta.json"
            if meta.is_file():
                try:
                    payload = json.loads(meta.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = {}
                if payload.get("project_path") and Path(str(payload["project_path"])).resolve() != project.resolve():
                    continue
        worker_path = state_dir / "worker.json"
        worker = None
        if worker_path.is_file():
            try:
                worker = json.loads(worker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                worker = None
        lock_path = state_dir / ".lock"
        lock_pid = 0
        if lock_path.is_file():
            try:
                lock_pid = int(json.loads(lock_path.read_text(encoding="utf-8")).get("pid") or 0)
            except (OSError, ValueError, TypeError):
                lock_pid = 0
        worker_pid = int((worker or {}).get("pid") or 0)
        worker_hb = float((worker or {}).get("heartbeat") or 0)
        lock_alive = _pid_alive(lock_pid)
        worker_alive = _pid_alive(worker_pid) and (time.time() - worker_hb) < STALE_SECONDS
        if not lock_alive and not worker_alive:
            stale.append(str(manifest_path))

    try:
        from .sqlite_store import SQLiteStore

        store = SQLiteStore(home)
        if store.is_initialized():
            for row in store.list_active_operations():
                if project is not None:
                    matched = False
                    if row["source_root"]:
                        try:
                            if Path(str(row["source_root"])).resolve() == project.resolve():
                                matched = True
                        except Exception:
                            pass
                    if not matched and row["state_dir"]:
                        try:
                            meta_path = Path(str(row["state_dir"])) / "lab-meta.json"
                            if meta_path.is_file():
                                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                                if meta_payload.get("project_path") and Path(str(meta_payload["project_path"])).resolve() == project.resolve():
                                    matched = True
                        except Exception:
                            pass
                    if not matched:
                        continue
                pid = int(row["pid"] or 0)
                alive = _pid_alive(pid) if pid > 0 else False
                if alive and row["heartbeat_at"]:
                    try:
                        from datetime import datetime
                        hb_ts = datetime.fromisoformat(str(row["heartbeat_at"]).replace("Z", "+00:00")).timestamp()
                        if (time.time() - hb_ts) >= STALE_SECONDS:
                            alive = False
                    except Exception:
                        pass
                lock_alive = False
                if row["state_dir"]:
                    try:
                        lock_path = Path(str(row["state_dir"])) / ".lock"
                        if lock_path.is_file():
                            lock_pid = int(json.loads(lock_path.read_text(encoding="utf-8")).get("pid") or 0)
                            lock_alive = _pid_alive(lock_pid) if lock_pid > 0 else False
                    except Exception:
                        pass
                if not alive and not lock_alive:
                    stale.append(f"operation:{row['operation_id']}")
    except Exception:
        pass

    return stale


def run_doctor(project: Path | None = None) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    if shutil.which("git") is None:
        problems.append("git is not on PATH")
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required")
    home = resolve_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        problems.append(f"AUTORESEARCH_HOME is not writable: {error}")

    skill = check_skill()
    if skill.get("error") or (skill.get("ok") is False and not skill.get("modified")):
        detail = skill.get("error") or "integrity check failed"
        problems.append(
            f"Skill manifest is missing or invalid: {detail}. Reinstall with install.py into a clean destination, then rerun doctor."
        )
    elif skill.get("modified"):
        warnings.append("W_SKILL_MODIFIED")

    stale_manifests = _stale_running_manifests(home, project)
    if stale_manifests:
        problems.append(f"Stale running manifest(s): {len(stale_manifests)}")

    dashboard_url = None
    dashboard_running = False
    try:
        from .dashboard_server import dashboard_server_status

        status = dashboard_server_status(home)
        dashboard_running = bool(status.get("running"))
        dashboard_url = status.get("url")
    except Exception:
        pass

    next_steps = []
    if not problems:
        next_steps = [
            "python SKILL_DIR/scripts/autoresearch.py doctor",
            "python SKILL_DIR/scripts/prepare.py --program program.md --spec spec.json",
            "python SKILL_DIR/scripts/lab.py --config <config> baseline",
            "python SKILL_DIR/scripts/lab.py --config <config> trial start --hypothesis hypothesis.json",
            "python SKILL_DIR/scripts/lab.py --config <config> trial wait",
            "python SKILL_DIR/scripts/lab.py --config <config> status",
            "python SKILL_DIR/scripts/lab.py --config <config> promote",
        ]

    payload: dict[str, Any] = {
        "ok": not problems,
        "python": sys.version.split()[0],
        "git": shutil.which("git"),
        "home": str(home),
        "project": str(project.resolve()) if project else None,
        "skill_version": read_version(),
        "skill_integrity": skill,
        "stale_running_manifests": stale_manifests,
        "dashboard_running": dashboard_running,
        "dashboard_url": dashboard_url,
        "problems": problems,
        "next": next_steps,
    }
    if warnings:
        payload["_warnings"] = warnings
    return payload
