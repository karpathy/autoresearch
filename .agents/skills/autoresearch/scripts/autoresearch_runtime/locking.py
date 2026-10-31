"""One mutating command per lab. A dead owner can be replaced."""

from __future__ import annotations

import errno
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import LabError

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            if ctypes.windll.kernel32.GetLastError() == 5:
                return True
            return False
        try:
            res = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
            return res == 0x00000102
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_os_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_metadata(path: Path, token: str) -> None:
    temporary = path.with_name(f"{path.name}.{token}.tmp")
    try:
        temporary.write_text(json.dumps({
            "pid": os.getpid(),
            "heartbeat": time.time(),
            "owner_token": token,
        }), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_metadata_if_owned(path: Path, token: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("owner_token") == token:
            path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass


@contextmanager
def lab_lock(state_dir: Path) -> Iterator[None]:
    """Exclusive lock. Raises E_LOCKED when another live process holds it."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / ".lock"
    guard = state_dir / ".lock.guard"
    handle = guard.open("a+b")
    try:
        _acquire_os_lock(handle)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise LabError(
                "Lab is locked by another process",
                code="E_LOCKED",
                hint="Wait for the other command or stop that process",
            ) from exc
        raise

    token = uuid.uuid4().hex
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(30):
            try:
                _write_metadata(path, token)
            except OSError:
                return

    thread = threading.Thread(target=beat, daemon=True)
    try:
        _write_metadata(path, token)
        thread.start()
        yield
    finally:
        stop.set()
        if thread.is_alive():
            thread.join()
        _remove_metadata_if_owned(path, token)
        try:
            _release_os_lock(handle)
        except OSError:
            pass
        handle.close()
