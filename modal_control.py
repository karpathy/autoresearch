#!/usr/bin/env python3
"""Local control layer for Modal-backed autoresearch runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Any
from uuid import uuid4

from modal_contract import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VOLUME_NAME,
    EVENT_PREFIX,
    RESULT_PREFIX,
)
from parse_run import parse_log

REPO_ROOT = Path(__file__).resolve().parent
STATE_DIR = REPO_ROOT / ".modal-control"
LOG_DIR = STATE_DIR / "logs"
STATE_PATH = STATE_DIR / "runs.json"
APP_URL_PATTERN = re.compile(r"(https://modal\.com/apps/[^\s]+/(ap-[A-Za-z0-9]+))")


@dataclass
class RunRecord:
    run_id: str
    pid: int
    created_at: str
    status: str
    command: list[str]
    log_path: str
    gpu: str
    volume_name: str
    timeout_minutes: int
    skip_prepare: bool
    app_id: str | None = None
    run_url: str | None = None
    function_call_id: str | None = None
    stopped_at: str | None = None
    exit_status: int | None = None


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_state_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, dict[str, Any]]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, dict[str, Any]]) -> None:
    ensure_state_dirs()
    temp_path = STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    temp_path.replace(STATE_PATH)


def build_command(*, skip_prepare: bool, gpu: str, volume_name: str, timeout_minutes: int) -> list[str]:
    command = [
        "modal",
        "run",
        "modal_train.py",
        "--gpu",
        gpu,
        "--volume-name",
        volume_name,
        "--timeout-minutes",
        str(timeout_minutes),
    ]
    if skip_prepare:
        command.append("--skip-prepare")
    return command


def parse_app_info(text: str) -> tuple[str | None, str | None]:
    matches = APP_URL_PATTERN.findall(text)
    if not matches:
        return None, None
    run_url, app_id = matches[-1]
    return app_id, run_url


def parse_modal_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.startswith(EVENT_PREFIX):
            continue
        payload = raw_line[len(EVENT_PREFIX) :].strip()
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_log_text(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text()


def result_available(text: str) -> bool:
    return RESULT_PREFIX in text


def refresh_record(record: RunRecord) -> RunRecord:
    log_path = Path(record.log_path)
    text = read_log_text(log_path)
    app_id, run_url = parse_app_info(text)
    if app_id:
        record.app_id = app_id
    if run_url:
        record.run_url = run_url
    for event in parse_modal_events(text):
        if event.get("event") == "call_spawned" and isinstance(event.get("function_call_id"), str):
            record.function_call_id = event["function_call_id"]

    running = process_is_running(record.pid)
    if result_available(text):
        result = parse_log(log_path)
        record.exit_status = result.get("exit_status")
        if record.status != "stopped":
            record.status = "failed" if result.get("crashed") else "completed"
    elif record.status != "stopped":
        record.status = "running" if running else "exited"
    return record


def save_record(record: RunRecord) -> RunRecord:
    state = load_state()
    state[record.run_id] = asdict(record)
    save_state(state)
    return record


def get_record(run_id: str) -> RunRecord:
    state = load_state()
    payload = state.get(run_id)
    if payload is None:
        raise SystemExit(f"Unknown run_id: {run_id}")
    return RunRecord(**payload)


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_start(args: argparse.Namespace) -> None:
    ensure_state_dirs()
    run_id = uuid4().hex[:12]
    log_path = LOG_DIR / f"{run_id}.log"
    command = build_command(
        skip_prepare=args.skip_prepare,
        gpu=args.gpu,
        volume_name=args.volume_name,
        timeout_minutes=args.timeout_minutes,
    )

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record = RunRecord(
        run_id=run_id,
        pid=proc.pid,
        created_at=now_iso(),
        status="starting",
        command=command,
        log_path=str(log_path),
        gpu=args.gpu,
        volume_name=args.volume_name,
        timeout_minutes=args.timeout_minutes,
        skip_prepare=args.skip_prepare,
    )
    save_record(record)
    print_json(asdict(record))


def cmd_status(args: argparse.Namespace) -> None:
    record = refresh_record(get_record(args.run_id))
    save_record(record)
    payload = asdict(record)
    if result_available(read_log_text(Path(record.log_path))):
        payload["result"] = parse_log(Path(record.log_path))
    print_json(payload)


def cmd_logs(args: argparse.Namespace) -> None:
    record = refresh_record(get_record(args.run_id))
    save_record(record)
    lines = read_log_text(Path(record.log_path)).splitlines()
    if args.tail is not None:
        lines = lines[-args.tail :]
    print("\n".join(lines))


def cmd_result(args: argparse.Namespace) -> None:
    record = refresh_record(get_record(args.run_id))
    save_record(record)
    log_path = Path(record.log_path)
    if not result_available(read_log_text(log_path)):
        raise SystemExit(f"Run {args.run_id} has not emitted a final result yet")
    print_json(parse_log(log_path))


def stop_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def stop_remote_app(app_id: str) -> None:
    subprocess.run(
        ["modal", "app", "stop", app_id],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def cmd_stop(args: argparse.Namespace) -> None:
    record = refresh_record(get_record(args.run_id))
    if record.app_id:
        stop_remote_app(record.app_id)
    stop_process_group(record.pid)
    record.status = "stopped"
    record.stopped_at = now_iso()
    save_record(record)
    print_json(asdict(record))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start a Modal-backed run in the background")
    start_parser.add_argument("--skip-prepare", action="store_true")
    start_parser.add_argument("--gpu", default=DEFAULT_GPU)
    start_parser.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    start_parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    start_parser.set_defaults(func=cmd_start)

    status_parser = subparsers.add_parser("status", help="Show current run status")
    status_parser.add_argument("run_id")
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Show run logs")
    logs_parser.add_argument("run_id")
    logs_parser.add_argument("--tail", type=int, default=80)
    logs_parser.set_defaults(func=cmd_logs)

    result_parser = subparsers.add_parser("result", help="Show parsed final result")
    result_parser.add_argument("run_id")
    result_parser.set_defaults(func=cmd_result)

    stop_parser = subparsers.add_parser("stop", help="Stop a running Modal app")
    stop_parser.add_argument("run_id")
    stop_parser.set_defaults(func=cmd_stop)

    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
