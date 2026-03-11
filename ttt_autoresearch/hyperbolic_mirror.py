from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mirror a remote Hyperbolic run directory back to the local machine.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--identity-file")
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-launch-dir", required=True)
    parser.add_argument("--remote-exitcode-path", required=True)
    parser.add_argument("--local-dest", required=True)
    parser.add_argument("--interval-sec", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    local_dest = Path(args.local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)
    local_launch_dir = local_dest / "_remote_launch"
    local_launch_dir.mkdir(parents=True, exist_ok=True)

    while True:
        _sync_remote_dir(
            host=args.host,
            port=args.port,
            user=args.user,
            identity_file=args.identity_file,
            remote_dir=args.remote_run_dir,
            local_dir=local_dest,
        )
        _sync_remote_dir(
            host=args.host,
            port=args.port,
            user=args.user,
            identity_file=args.identity_file,
            remote_dir=args.remote_launch_dir,
            local_dir=local_launch_dir,
        )
        if _remote_file_exists(
            host=args.host,
            port=args.port,
            user=args.user,
            identity_file=args.identity_file,
            remote_path=args.remote_exitcode_path,
        ):
            _sync_remote_dir(
                host=args.host,
                port=args.port,
                user=args.user,
                identity_file=args.identity_file,
                remote_dir=args.remote_run_dir,
                local_dir=local_dest,
            )
            _sync_remote_dir(
                host=args.host,
                port=args.port,
                user=args.user,
                identity_file=args.identity_file,
                remote_dir=args.remote_launch_dir,
                local_dir=local_launch_dir,
            )
            return 0
        time.sleep(max(5, args.interval_sec))


def _sync_remote_dir(
    *,
    host: str,
    port: int,
    user: str,
    identity_file: str | None,
    remote_dir: str,
    local_dir: Path,
) -> None:
    rsync = shutil.which("rsync")
    if rsync:
        cmd = [
            rsync,
            "-az",
            "--delete",
            "-e",
            _ssh_command(port=port, identity_file=identity_file),
            f"{user}@{host}:{remote_dir.rstrip('/')}/",
            str(local_dir),
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    cmd = _ssh_base_args(port=port, identity_file=identity_file) + [
        f"{user}@{host}",
        f"bash -lc {shlex_quote(f'cd {remote_dir} && tar -cf - .')}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0:
        return
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(local_dir)],
        input=proc.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if extract.returncode != 0:
        return


def _remote_file_exists(*, host: str, port: int, user: str, identity_file: str | None, remote_path: str) -> bool:
    cmd = _ssh_base_args(port=port, identity_file=identity_file) + [
        f"{user}@{host}",
        f"bash -lc {shlex_quote(f'test -f {remote_path}')}",
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def _ssh_command(*, port: int, identity_file: str | None) -> str:
    parts = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(port),
    ]
    if identity_file:
        parts.extend(["-i", identity_file])
    return " ".join(parts)


def _ssh_base_args(*, port: int, identity_file: str | None) -> list[str]:
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(port),
    ]
    if identity_file:
        args.extend(["-i", identity_file])
    return args


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
