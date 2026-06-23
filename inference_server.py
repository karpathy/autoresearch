"""
inference_server.py — Generic GPU inference server lifecycle manager.

Reads config.yaml and manages stop/start of any inference server
(llama.cpp, vLLM, Ollama, TGI, TabbyML, LMDeploy, …) around training.

CLI usage (recommended):
    python inference_server.py --stop
    python inference_server.py --start
    python inference_server.py --restart

Python API usage (in train.py):
    from inference_server import InferenceServer
    server = InferenceServer.from_config()
    server.stop()      # call before training
    try:
        ...training loop...
    finally:
        server.start() # always restarts, even on crash
"""

import subprocess
import time
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading (no extra deps — uses stdlib only if PyYAML not present)
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """Load YAML config. Falls back to empty dict if file missing."""
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML not available — try a minimal key:value parser for simple cases
        log.warning("PyYAML not installed; falling back to minimal YAML parser. "
                    "Nested config may not parse correctly.")
        return _minimal_yaml(path)


def _minimal_yaml(path: str) -> dict:
    """
    Bare-minimum YAML parser for flat/one-level-deep config.
    Handles only the simple scalar values we need.
    """
    result: dict = {}
    current_section: Optional[str] = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.startswith(" ") or line.startswith("\t"):
                # indented key under current_section
                stripped = line.strip()
                if ":" in stripped and current_section is not None:
                    k, _, v = stripped.partition(":")
                    v = v.strip().strip('"').strip("'")
                    # inline comment
                    v = v.split("#")[0].strip()
                    if v.lower() in ("true", "yes"):
                        v = True
                    elif v.lower() in ("false", "no"):
                        v = False
                    else:
                        try:
                            v = int(v)
                        except (ValueError, TypeError):
                            pass
                    result.setdefault(current_section, {})[k.strip()] = v
            else:
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip()
                    if not v:
                        current_section = k.strip()
                    else:
                        current_section = None
                        result[k.strip()] = v
    return result


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------

@dataclass
class InferenceServerConfig:
    enabled: bool = False
    stop_before_training: bool = True
    restart_after_training: bool = True
    stop_command: str = ""
    start_command: str = ""
    stop_timeout: int = 30
    start_timeout: int = 120   # max seconds to wait for the shell script to return;
                               # the SERVER PROCESS itself keeps running after Python exits
    fail_safe: bool = True


class InferenceServer:
    """
    Generic lifecycle manager for any GPU inference server.

    Works with:
      - llama.cpp  (llama-server)
      - vLLM       (python -m vllm …)
      - Ollama     (systemctl / ollama serve)
      - TGI        (text-generation-launcher)
      - TabbyML, LMDeploy, LiteLLM proxy, …
      - Anything with a shell stop/start command
    """

    def __init__(self, cfg: InferenceServerConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config.yaml", section: str = "inference_server") -> "InferenceServer":
        """Load from config.yaml (or any YAML file)."""
        raw = _load_yaml(config_path)
        sec = raw.get(section, {})
        cfg = InferenceServerConfig(
            enabled=bool(sec.get("enabled", False)),
            stop_before_training=bool(sec.get("stop_before_training", True)),
            restart_after_training=bool(sec.get("restart_after_training", True)),
            stop_command=str(sec.get("stop_command", "")),
            start_command=str(sec.get("start_command", "")),
            stop_timeout=int(sec.get("stop_timeout", 30)),
            start_timeout=int(sec.get("start_timeout", 120)),
            fail_safe=bool(sec.get("fail_safe", True)),
        )
        server = cls(cfg)
        if cfg.enabled:
            log.info("[inference_server] Lifecycle management ENABLED")
            log.info("[inference_server]   stop_before_training  = %s", cfg.stop_before_training)
            log.info("[inference_server]   restart_after_training = %s", cfg.restart_after_training)
            log.info("[inference_server]   stop_command  = %s", cfg.stop_command or "<none>")
            log.info("[inference_server]   start_command = %s", cfg.start_command or "<none>")
        else:
            log.info("[inference_server] Lifecycle management disabled (enabled: false)")
        return server

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> bool:
        """
        Stop the inference server (if enabled and stop_before_training=true).
        Returns True on success, False on failure.
        Call this BEFORE training starts.
        """
        if not self.cfg.enabled or not self.cfg.stop_before_training:
            return True
        if not self.cfg.stop_command:
            log.warning("[inference_server] stop_command is empty — skipping stop")
            return True
        return self._run(self.cfg.stop_command, label="stop", timeout=self.cfg.stop_timeout)

    def start(self) -> bool:
        """
        Restart the inference server (if enabled and restart_after_training=true).
        Returns True on success, False on failure.
        Call this AFTER training (in a finally block so it always runs).

        NOTE: The server process launched by start_command runs INDEPENDENTLY of
        Python — it is backgrounded by the shell script (`&`) and gets reparented
        to PID 1 when the shell exits. It will keep running after train.py exits.
        `start_timeout` only limits how long we wait for the shell *script* to
        return (e.g. the health-check loop), not the server process lifetime.
        """
        if not self.cfg.enabled or not self.cfg.restart_after_training:
            return True
        if not self.cfg.start_command:
            log.warning("[inference_server] start_command is empty — skipping start")
            return True
        return self._run(self.cfg.start_command, label="start", timeout=self.cfg.start_timeout)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, command: str, label: str, timeout: Optional[int]) -> bool:
        print(f"[inference_server] Running {label}: {command}", flush=True)
        t0 = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"[inference_server] {label} OK ({elapsed:.1f}s)", flush=True)
                if result.stdout.strip():
                    log.debug("[inference_server] stdout: %s", result.stdout.strip())
                return True
            else:
                msg = (f"[inference_server] {label} exited with code {result.returncode} "
                       f"({elapsed:.1f}s)")
                if result.stderr.strip():
                    msg += f"\n  stderr: {result.stderr.strip()}"
                if self.cfg.fail_safe:
                    print(f"[inference_server] WARNING: {msg} — continuing (fail_safe=true)",
                          flush=True)
                    return False
                else:
                    raise RuntimeError(msg)
        except subprocess.TimeoutExpired:
            msg = f"[inference_server] {label} timed out after {timeout}s"
            if self.cfg.fail_safe:
                print(f"[inference_server] WARNING: {msg} — continuing (fail_safe=true)", flush=True)
                return False
            else:
                raise RuntimeError(msg)
        except Exception as exc:
            msg = f"[inference_server] {label} failed: {exc}"
            if self.cfg.fail_safe:
                print(f"[inference_server] WARNING: {msg} — continuing (fail_safe=true)", flush=True)
                return False
            else:
                raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Manage the GPU inference server lifecycle.",
        epilog="Reads stop_command / start_command from config.yaml.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stop",    action="store_true", help="Stop the inference server")
    group.add_argument("--start",   action="store_true", help="Start (restart) the inference server")
    group.add_argument("--restart", action="store_true", help="Stop then start the inference server")
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    server = InferenceServer.from_config(config_path=args.config)

    ok = True
    if args.stop or args.restart:
        ok = server.stop() and ok
    if args.start or args.restart:
        ok = server.start() and ok

    sys.exit(0 if ok else 1)
