#!/usr/bin/env python3
"""Parse a Modal-backed autoresearch run log into deterministic JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

RESULT_PREFIX = "AUTORESEARCH_MODAL_RESULT "
EVENT_PREFIX = "AUTORESEARCH_MODAL "
ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "Error:",
    "RemoteError",
)
SUMMARY_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9.]+)\s*$"),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)\s*$"),
    "total_seconds": re.compile(r"^total_seconds:\s+([0-9.]+)\s*$"),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9.]+)\s*$"),
}


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    for raw_line in reversed(text.splitlines()):
        if raw_line.startswith(RESULT_PREFIX):
            try:
                payload = json.loads(raw_line[len(RESULT_PREFIX) :].strip())
            except json.JSONDecodeError:
                break
            payload["path"] = str(path)
            return payload
    return parse_legacy_log(path, text)


def parse_legacy_log(path: Path, text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "val_bpb": None,
        "peak_vram_mb": None,
        "training_seconds": None,
        "total_seconds": None,
        "prepare_seconds": None,
        "remote_train_seconds": None,
        "wrapper_total_seconds": None,
        "skip_prepare": None,
        "exit_status": 1,
        "crashed": True,
    }

    modal_events: dict[str, dict[str, Any]] = {}

    for raw_line in text.splitlines():
        for key, pattern in SUMMARY_PATTERNS.items():
            match = pattern.match(raw_line)
            if match:
                result[key] = float(match.group(1))

        if raw_line.startswith(EVENT_PREFIX):
            payload = raw_line[len(EVENT_PREFIX) :].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            name = event.get("event")
            if isinstance(name, str):
                modal_events[name] = event

    run_complete = modal_events.get("run_complete", {})
    local_result = modal_events.get("local_result", {})

    if isinstance(run_complete.get("prepare_seconds"), (int, float)):
        result["prepare_seconds"] = float(run_complete["prepare_seconds"])
    if isinstance(run_complete.get("train_seconds"), (int, float)):
        result["remote_train_seconds"] = float(run_complete["train_seconds"])
    if isinstance(run_complete.get("total_seconds"), (int, float)):
        result["wrapper_total_seconds"] = float(run_complete["total_seconds"])
    if isinstance(local_result.get("skip_prepare"), bool):
        result["skip_prepare"] = bool(local_result["skip_prepare"])
    elif "prepare_skipped" in modal_events:
        result["skip_prepare"] = True
    elif "prepare_start" in modal_events:
        result["skip_prepare"] = False

    if isinstance(local_result.get("exit_status"), int):
        result["exit_status"] = int(local_result["exit_status"])
    elif isinstance(run_complete.get("exit_code"), int):
        result["exit_status"] = int(run_complete["exit_code"])

    has_metrics = result["val_bpb"] is not None and result["peak_vram_mb"] is not None
    saw_error = any(marker in text for marker in ERROR_MARKERS)
    if has_metrics and result["exit_status"] == 1 and not saw_error:
        result["exit_status"] = 0

    result["crashed"] = bool(
        result["exit_status"] != 0 or not has_metrics or run_complete.get("crashed") is True
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(parse_log(args.log_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
