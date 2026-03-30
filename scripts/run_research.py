from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRAM = REPO_ROOT / "program.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"
DEFAULT_QUEUE_ROOT = REPO_ROOT / "queue"
PENDING_DIR = DEFAULT_QUEUE_ROOT / "pending"
COMPLETED_DIR = DEFAULT_QUEUE_ROOT / "completed"
FAILED_DIR = DEFAULT_QUEUE_ROOT / "failed"
VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


@dataclass
class RunConfig:
    name: str
    spec_path: Path
    vars: dict[str, str]
    program_path: Path | None
    output_root: Path
    model: str | None
    search: bool
    tag: str | None
    extra_instructions: str | None


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "research-job"


def parse_key_value(items: Iterable[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing key in assignment: {item}")
        pairs[key] = value
    return pairs


def substitute_vars(text: str, values: dict[str, str]) -> str:
    missing = sorted({match.group(1) for match in VAR_PATTERN.finditer(text)} - set(values))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing template variables: {missing_list}")
    return VAR_PATTERN.sub(lambda match: values[match.group(1)], text)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(program_text: str | None, spec_text: str, extra_instructions: str | None) -> str:
    parts = [
        "You are running an autonomous research job through Codex CLI.",
        "Use web search when needed, follow the research spec exactly, and return only the final deliverable markdown.",
        "The prompt already includes the local instructions and the research specification, so do not waste time inspecting the repo unless the prompt explicitly asks for local files beyond what is already pasted here.",
    ]
    if program_text:
        parts.append("## Program Instructions")
        parts.append(program_text.strip())
    if extra_instructions:
        parts.append("## Run-Specific Instructions")
        parts.append(extra_instructions.strip())
    parts.append("## Research Specification")
    parts.append(spec_text.strip())
    return "\n\n".join(parts) + "\n"


def build_output_dir(output_root: Path, name: str, tag: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(tag or name)
    output_dir = output_root / f"{timestamp}-{slug}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def which_or_raise(executable: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        raise FileNotFoundError(f"Could not find `{executable}` on PATH.")
    return resolved


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    stdin_text: str | None = None,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if stdin_text is not None and process.stdin is not None:
            process.stdin.write(stdin_text)
            process.stdin.close()

        assert process.stdout is not None
        for line in process.stdout:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
                sys.stdout.flush()
            else:
                sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def run_codex_job(config: RunConfig) -> Path:
    spec_text = substitute_vars(load_text(config.spec_path), config.vars)
    program_text = None
    if config.program_path is not None:
        program_text = substitute_vars(load_text(config.program_path), config.vars)
    prompt_text = render_prompt(program_text, spec_text, config.extra_instructions)

    output_dir = build_output_dir(config.output_root, config.name, config.tag)
    final_path = output_dir / "final_report.md"
    prompt_path = output_dir / "prompt.txt"
    spec_copy_path = output_dir / "resolved_spec.md"
    log_path = output_dir / "codex.log"
    metadata_path = output_dir / "run.json"

    prompt_path.write_text(prompt_text, encoding="utf-8")
    spec_copy_path.write_text(spec_text, encoding="utf-8")

    command = [
        which_or_raise("codex"),
    ]
    if config.search:
        command.append("--search")
    if config.model:
        command.extend(["-m", config.model])
    command.extend(
        [
            "-a",
            "never",
            "-s",
            "workspace-write",
            "exec",
            "-C",
            str(REPO_ROOT),
            "--skip-git-repo-check",
            "-o",
            str(final_path),
            "-",
        ]
    )

    metadata = {
        "name": config.name,
        "spec_path": str(config.spec_path),
        "program_path": str(config.program_path) if config.program_path else None,
        "vars": config.vars,
        "model": config.model,
        "search": config.search,
        "tag": config.tag,
        "command": command,
        "started_at": datetime.now().isoformat(),
    }
    write_json(metadata_path, metadata)

    exit_code = run_command(command, cwd=REPO_ROOT, log_path=log_path, stdin_text=prompt_text)

    metadata["finished_at"] = datetime.now().isoformat()
    metadata["exit_code"] = exit_code
    metadata["final_report"] = str(final_path)
    metadata["log_path"] = str(log_path)
    write_json(metadata_path, metadata)

    if exit_code != 0:
        raise RuntimeError(f"Codex exited with code {exit_code}. See {log_path}")
    if not final_path.exists():
        raise RuntimeError(f"Codex finished without writing {final_path}")
    return output_dir


def load_job_file(job_path: Path, output_root: Path, default_program: Path | None) -> RunConfig:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    spec_path = (REPO_ROOT / payload["spec"]).resolve()
    program_path = default_program
    if "program" in payload:
        program_path = (REPO_ROOT / payload["program"]).resolve() if payload["program"] else None
    vars_payload = {str(key): str(value) for key, value in payload.get("vars", {}).items()}
    return RunConfig(
        name=str(payload.get("name") or job_path.stem),
        spec_path=spec_path,
        vars=vars_payload,
        program_path=program_path,
        output_root=output_root,
        model=payload.get("model"),
        search=bool(payload.get("search", True)),
        tag=payload.get("tag"),
        extra_instructions=payload.get("extra_instructions"),
    )


def run_doctor(search: bool) -> int:
    print(f"repo_root: {REPO_ROOT}")
    print(f"python: {sys.version.split()[0]}")
    print(f"codex: {which_or_raise('codex')}")
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    smoke_dir = output_root / "doctor"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    final_path = smoke_dir / ("doctor_search.txt" if search else "doctor.txt")
    log_path = smoke_dir / ("doctor_search.log" if search else "doctor.log")

    command = [which_or_raise("codex")]
    if search:
        command.append("--search")
    command.extend(
        [
            "-a",
            "never",
            "-s",
            "workspace-write",
            "exec",
            "-C",
            str(REPO_ROOT),
            "--skip-git-repo-check",
            "-o",
            str(final_path),
            "Reply with exactly DOCTOR_OK.",
        ]
    )
    print("running codex smoke...")
    exit_code = run_command(command, cwd=REPO_ROOT, log_path=log_path)
    if exit_code != 0:
        print(f"doctor failed: codex exited with {exit_code}")
        return exit_code
    if final_path.exists():
        print(f"doctor output: {final_path}")
        print(final_path.read_text(encoding="utf-8").strip())
    else:
        print(f"doctor failed: missing output file {final_path}")
        return 1
    return 0


def run_queue(args: argparse.Namespace) -> int:
    pending_dir = args.pending_dir.resolve()
    completed_dir = args.completed_dir.resolve()
    failed_dir = args.failed_dir.resolve()
    pending_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    while True:
        job_files = sorted(pending_dir.glob("*.json"))
        if not job_files:
            if args.watch:
                time.sleep(args.poll_seconds)
                continue
            break

        for job_path in job_files:
            print(f"processing job: {job_path.name}")
            try:
                config = load_job_file(job_path, args.output_root, args.program)
                output_dir = run_codex_job(config)
                destination = completed_dir / job_path.name
                shutil.move(str(job_path), str(destination))
                print(f"completed: {destination}")
                print(f"result: {output_dir}")
            except Exception as exc:  # noqa: BLE001
                destination = failed_dir / job_path.name
                shutil.move(str(job_path), str(destination))
                print(f"failed: {destination}")
                print(str(exc), file=sys.stderr)
            processed += 1
            if args.limit is not None and processed >= args.limit:
                return 0

        if not args.watch:
            break
    return 0


def build_run_config(args: argparse.Namespace) -> RunConfig:
    vars_payload = parse_key_value(args.var or [])
    if args.phase is not None:
        vars_payload["phase"] = args.phase
    if args.depth is not None:
        vars_payload["depth"] = args.depth
    return RunConfig(
        name=args.name or Path(args.spec).stem,
        spec_path=Path(args.spec).resolve(),
        vars=vars_payload,
        program_path=None if args.no_program or not args.program else args.program.resolve(),
        output_root=args.output_root.resolve(),
        model=args.model,
        search=args.search,
        tag=args.tag,
        extra_instructions=args.instructions,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Codex-powered research jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Verify Python, Codex CLI, and a minimal Codex run.")
    doctor.add_argument("--search", action="store_true", help="Also verify web-enabled Codex execution.")

    run = subparsers.add_parser("run", help="Run one research specification now.")
    run.add_argument("--spec", required=True, help="Path to the research specification markdown/text file.")
    run.add_argument("--name", help="Logical job name used in metadata and output folder naming.")
    run.add_argument("--tag", help="Optional output folder tag.")
    run.add_argument("--phase", help="Convenience value for ${phase}.")
    run.add_argument("--depth", help="Convenience value for ${depth}.")
    run.add_argument("--var", action="append", default=[], help="Additional template values as KEY=VALUE.")
    run.add_argument(
        "--program",
        type=Path,
        default=DEFAULT_PROGRAM,
        help="Path to the reusable research program markdown.",
    )
    run.add_argument("--no-program", action="store_true", help="Skip program.md and use only the spec.")
    run.add_argument("--instructions", help="Extra run-specific instructions appended after the program.")
    run.add_argument("--model", default="gpt-5.4", help="Codex model to use.")
    run.add_argument("--no-search", action="store_false", dest="search", help="Disable Codex web search.")
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for run artifacts.")
    run.set_defaults(search=True)

    queue = subparsers.add_parser("queue", help="Process queued JSON jobs from queue/pending.")
    queue.add_argument("--program", type=Path, default=DEFAULT_PROGRAM, help="Default program file for queued jobs.")
    queue.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for run artifacts.")
    queue.add_argument("--pending-dir", type=Path, default=PENDING_DIR, help="Queue input directory.")
    queue.add_argument("--completed-dir", type=Path, default=COMPLETED_DIR, help="Completed job archive directory.")
    queue.add_argument("--failed-dir", type=Path, default=FAILED_DIR, help="Failed job archive directory.")
    queue.add_argument("--watch", action="store_true", help="Keep polling the queue for more jobs.")
    queue.add_argument("--poll-seconds", type=int, default=15, help="Polling interval when --watch is enabled.")
    queue.add_argument("--limit", type=int, help="Stop after processing this many jobs.")

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        return run_doctor(search=args.search)
    if args.command == "run":
        config = build_run_config(args)
        config.output_root.mkdir(parents=True, exist_ok=True)
        output_dir = run_codex_job(config)
        print(f"final output: {output_dir}")
        return 0
    if args.command == "queue":
        return run_queue(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
