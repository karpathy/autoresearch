#!/usr/bin/env python3
"""Run autoresearch train.py on a Tempo-paid remote sandbox."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from modal_contract import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_MINUTES,
    EVENT_PREFIX,
    RESULT_PREFIX,
)

REPO_ROOT = Path(__file__).resolve().parent
TEMPO_BIN = "tempo"
DEFAULT_ENDPOINT = "https://modal.mpp.tempo.xyz"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
SUMMARY_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9.]+)\s*$"),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)\s*$"),
    "total_seconds": re.compile(r"^total_seconds:\s+([0-9.]+)\s*$"),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9.]+)\s*$"),
}
TRAIN_CONFIG_PATTERNS = {
    "embedding_lr": re.compile(r"^EMBEDDING_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "unembedding_lr": re.compile(r"^UNEMBEDDING_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "matrix_lr": re.compile(r"^MATRIX_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "scalar_lr": re.compile(r"^SCALAR_LR\s*=\s*([0-9.]+)", re.MULTILINE),
}
FILES_TO_UPLOAD = [
    "train.py",
    "prepare.py",
    "pyproject.toml",
    "uv.lock",
]


def emit(event: str, **data: Any) -> None:
    print(f"{EVENT_PREFIX}{json.dumps({'event': event, **data}, sort_keys=True)}", flush=True)


def emit_result(data: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(data, sort_keys=True)}", flush=True)


def tempo_request(
    endpoint: str,
    path: str,
    data: dict[str, Any],
    *,
    quiet: bool = False,
    retries: int = 3,
) -> dict[str, str]:
    for attempt in range(retries):
        if not quiet:
            emit("tempo_request", attempt=attempt + 1, path=path)

        result = subprocess.run(
            [
                TEMPO_BIN,
                "request",
                "-t",
                "-X",
                "POST",
                "--json",
                json.dumps(data),
                f"{endpoint}{path}",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
        stdout = result.stdout.strip()
        if result.returncode != 0:
            is_payment_error = "E_PAYMENT" in stdout or "payment" in stdout.lower()
            if is_payment_error and attempt < retries - 1:
                wait_seconds = 5 * (attempt + 1)
                emit("payment_retry", attempt=attempt + 2, wait_seconds=wait_seconds, path=path)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(
                f"tempo request failed (rc={result.returncode}): "
                f"{result.stderr.strip()}\n{stdout[:500]}"
            )

        parsed: dict[str, str] = {}
        for line in stdout.splitlines():
            normalized = line.strip()
            if not normalized or ":" not in normalized:
                continue
            key, _, value = normalized.partition(":")
            key = key.strip()
            if key in {"sandbox_id", "stdout", "stderr", "returncode", "status"}:
                parsed[key] = value.strip().strip('"').replace("\\n", "\n")
        return parsed

    raise RuntimeError("tempo request retries exhausted")


def exec_remote(
    endpoint: str,
    sandbox_id: str,
    command: list[str],
    *,
    quiet: bool = False,
) -> dict[str, str]:
    return tempo_request(
        endpoint,
        "/sandbox/exec",
        {"sandbox_id": sandbox_id, "command": command},
        quiet=quiet,
    )


def encode_workspace_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def read_train_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    train_config: dict[str, float] = {}
    for key, pattern in TRAIN_CONFIG_PATTERNS.items():
        match = pattern.search(text)
        if match:
            train_config[key] = float(match.group(1))
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "config": train_config,
    }


def build_remote_script(args: argparse.Namespace) -> str:
    parts = [
        "set -e",
        "apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv git curl >/dev/null 2>&1",
        "ln -sf $(command -v python3) /usr/local/bin/python || true",
        "python3 -m pip install -q --break-system-packages uv",
        "mkdir -p /workspace",
        "cd /workspace",
    ]

    for file_name in FILES_TO_UPLOAD:
        local_path = REPO_ROOT / file_name
        if not local_path.exists():
            continue
        encoded = encode_workspace_file(local_path)
        parts.append(f'echo "{encoded}" | base64 -d > /workspace/{file_name}')

    parts.append("uv sync --extra train")
    if not args.skip_prepare:
        parts.append("uv run prepare.py")
    else:
        parts.append("echo 'prepare skipped'")
    parts.append("uv run train.py")
    return "\n".join(parts)


def parse_summary_metrics(log_content: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "val_bpb": None,
        "training_seconds": None,
        "total_seconds": None,
        "peak_vram_mb": None,
    }
    for raw_line in log_content.splitlines():
        line = raw_line.strip()
        for key, pattern in SUMMARY_PATTERNS.items():
            match = pattern.match(line)
            if match:
                metrics[key] = float(match.group(1))
    return metrics


def build_result_payload(
    args: argparse.Namespace,
    *,
    sandbox_id: str | None,
    elapsed: float | None,
    log_content: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    train_manifest = read_train_manifest(REPO_ROOT / "train.py")
    metrics = parse_summary_metrics(log_content)
    crashed = error is not None or metrics["val_bpb"] is None
    exit_status = 0 if not crashed else 1
    return {
        "schema_version": 1,
        "provider": "tempo-mpp",
        "tempo_endpoint": args.endpoint,
        "tempo_sandbox_id": sandbox_id,
        "image": args.image,
        "gpu": args.gpu,
        "timeout_minutes": args.timeout_minutes,
        "skip_prepare": args.skip_prepare,
        "train_sha256": train_manifest["sha256"],
        "train_bytes": train_manifest["bytes"],
        "train_config": train_manifest["config"],
        "val_bpb": metrics["val_bpb"],
        "peak_vram_mb": metrics["peak_vram_mb"],
        "training_seconds": metrics["training_seconds"],
        "total_seconds": metrics["total_seconds"],
        "prepare_seconds": None,
        "remote_train_seconds": metrics["total_seconds"],
        "wrapper_total_seconds": elapsed,
        "exit_status": exit_status,
        "crashed": crashed,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    sandbox_id: str | None = None
    started_at = time.time()
    log_content = ""
    try:
        emit(
            "run_start",
            endpoint=args.endpoint,
            gpu=args.gpu,
            timeout_minutes=args.timeout_minutes,
            skip_prepare=args.skip_prepare,
        )
        create_response = tempo_request(
            args.endpoint,
            "/sandbox/create",
            {
                "gpu": args.gpu,
                "timeout": args.timeout_minutes * 60,
                "image": args.image,
            },
            quiet=args.quiet,
        )
        sandbox_id = create_response.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Tempo create response did not include sandbox_id")
        emit("sandbox_created", sandbox_id=sandbox_id)

        script = build_remote_script(args)
        launch_command = (
            f"({script}) > /tmp/run.log 2>&1 && echo DONE > /tmp/ok &\n"
            "echo started"
        )
        exec_remote(
            args.endpoint,
            sandbox_id,
            ["bash", "-lc", launch_command],
            quiet=args.quiet,
        )

        poll_interval = 15
        while time.time() - started_at < args.timeout_minutes * 60:
            time.sleep(poll_interval)
            status = exec_remote(
                args.endpoint,
                sandbox_id,
                ["bash", "-lc", "cat /tmp/ok 2>/dev/null || echo WAIT"],
                quiet=True,
            )
            done = "DONE" in (status.get("stdout", "") or "")
            tail_response = exec_remote(
                args.endpoint,
                sandbox_id,
                ["bash", "-lc", "tail -1 /tmp/run.log 2>/dev/null"],
                quiet=True,
            )
            tail = (tail_response.get("stdout", "") or "").strip().split("\n")[-1].strip()
            emit(
                "poll",
                sandbox_id=sandbox_id,
                elapsed_seconds=round(time.time() - started_at, 1),
                done=done,
                tail=tail,
            )
            if done:
                break

        log_response = exec_remote(
            args.endpoint,
            sandbox_id,
            ["cat", "/tmp/run.log"],
            quiet=True,
        )
        log_content = log_response.get("stdout", "") or ""
        if log_content:
            print(log_content, end="" if log_content.endswith("\n") else "\n")
        return build_result_payload(
            args,
            sandbox_id=sandbox_id,
            elapsed=round(time.time() - started_at, 3),
            log_content=log_content,
        )
    except Exception as error:
        if sandbox_id and not log_content:
            try:
                log_response = exec_remote(
                    args.endpoint,
                    sandbox_id,
                    ["cat", "/tmp/run.log"],
                    quiet=True,
                )
                log_content = log_response.get("stdout", "") or ""
                if log_content:
                    print(log_content, end="" if log_content.endswith("\n") else "\n")
            except Exception:
                pass
        return build_result_payload(
            args,
            sandbox_id=sandbox_id,
            elapsed=round(time.time() - started_at, 3),
            log_content=log_content,
            error=error,
        )
    finally:
        if sandbox_id:
            try:
                emit("sandbox_terminating", sandbox_id=sandbox_id)
                tempo_request(
                    args.endpoint,
                    "/sandbox/terminate",
                    {"sandbox_id": sandbox_id},
                    quiet=True,
                )
                emit("sandbox_terminated", sandbox_id=sandbox_id)
            except Exception as terminate_error:
                emit("sandbox_terminate_failed", sandbox_id=sandbox_id, error=str(terminate_error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_experiment(args)
    emit_result(payload)
    raise SystemExit(0 if payload["exit_status"] == 0 else 1)


if __name__ == "__main__":
    main()
