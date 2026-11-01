"""Helpers for labs that must not touch the project under test."""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Tests must never touch the real lab home: a store opened without an explicit home uses this one.
_TEST_HOME = tempfile.mkdtemp(prefix="autoresearch-tests-")
os.environ["AUTORESEARCH_HOME"] = _TEST_HOME
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoresearch_runtime.runtime_env import hidden_process_kwargs

# CLI tests must not start persistent background dashboard processes by default.
# Lifecycle tests explicitly clear this flag when they exercise auto-start.
os.environ.setdefault("AUTORESEARCH_NO_DASHBOARD", "1")


def tree_digest(root: Path) -> dict[str, str]:
    """Hash every file under root, including git metadata."""
    digest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256()
        file_hash.update(path.read_bytes())
        digest[relative] = file_hash.hexdigest()
    return digest


def git_project(root: Path, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {"value.txt": "1"}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, **hidden_process_kwargs())
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, **hidden_process_kwargs())
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "initial"],
        cwd=root, check=True,
        **hidden_process_kwargs(),
    )
    return root


def use_home(path: Path) -> str | None:
    previous = os.environ.get("AUTORESEARCH_HOME")
    os.environ["AUTORESEARCH_HOME"] = str(path)
    return previous


# ---------------------------------------------------------------------------
# Draft lab.json files
#
# Many tests describe a lab as a lab.json file in a temp directory and then call
# `load_config(path)` / `init_lab(config)`. Runtime labs now live only in SQLite,
# so these two entry points are wrapped for the tests: a lab.json that is not
# registered loads as a validated draft, and `init_lab(draft)` registers it in
# SQLite (absolute paths, virtual `state_dir/lab.json`) and updates the draft
# object in place so it behaves like the loaded lab. Registered labs and
# virtual paths go straight to the runtime.
# ---------------------------------------------------------------------------
_DRAFT_ALIASES: dict[str, str] = {}


def _absolute_config_bytes(path: Path) -> bytes:
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    raw["source_root"] = str((base / raw.get("source_root", ".")).resolve())
    raw["state_dir"] = str((base / raw.get("state_dir", "../AutoResearch-state")).resolve())
    raw["input_files"] = [str((base / name).resolve()) for name in raw.get("input_files", [])]
    evaluator = list(raw["evaluator"])
    index = 1 if evaluator and (evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python")) else 0
    if index < len(evaluator) and not evaluator[index].startswith("{") and (base / evaluator[index]).is_file():
        evaluator[index] = str((base / evaluator[index]).resolve())
    raw["evaluator"] = evaluator
    return (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _install_draft_lab_shims() -> None:
    import dataclasses
    import json
    from dataclasses import replace

    import autoresearch_runtime.config as config_module
    import autoresearch_runtime.lab as lab_module
    import autoresearch_runtime.snapshot as snapshot_module
    from autoresearch_runtime.errors import LabError

    if getattr(config_module, "_DRAFT_SHIMS_INSTALLED", False):
        return  # support can be imported as `support` and as `tests.support`
    config_module._DRAFT_SHIMS_INSTALLED = True
    runtime_load = config_module.load_config
    runtime_init = snapshot_module.init_lab

    def load_config(path, *, allow_pending=False, read_only=False):
        try:
            return runtime_load(path, allow_pending=allow_pending)
        except LabError as error:
            if error.code != "E_LEGACY_LAB":
                raise
        alias = _DRAFT_ALIASES.get(str(Path(path).expanduser().resolve()))
        if alias is not None:
            return runtime_load(alias, allow_pending=allow_pending)
        return config_module.validate_config_file(Path(path), allow_pending=allow_pending)

    def init_lab(config):
        if getattr(config, "lab_id", None) or getattr(config, "passport", None) is not None:
            return runtime_init(config)  # already a registered-style config (prepare)
        original = Path(config.path).resolve()
        raw_bytes = _absolute_config_bytes(original)
        parsed = config_module.parse_config(original, raw_bytes, allow_pending=config.pending)
        raw = json.loads(raw_bytes.decode("utf-8"))
        # A draft built with dataclasses.replace(...) carries overrides: persist them in the stored contract.
        changed = False
        if config.unit != parsed.unit:
            raw["metrics"] = {**raw["metrics"], "unit": config.unit}; changed = True
        if config.goal_criteria != parsed.goal_criteria:
            raw["goal_criteria"] = config.goal_criteria; changed = True
        if config.budget != parsed.budget:
            raw["budget"] = dict(config.budget); changed = True
        if tuple(config.guards) != tuple(parsed.guards):
            raw["metrics"] = {**raw["metrics"], "guards": [{"name": n, "direction": d} for n, d in config.guards]}; changed = True
        if changed:
            raw_bytes = (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            parsed = config_module.parse_config(original, raw_bytes, allow_pending=config.pending)
        passport = {
            "confirmed": True, "goal": "test lab", "project": str(parsed.source_root),
            "metrics": raw["metrics"], "constraints": raw.get("budget", {}),
            "pins": [{"name": "config", "path": str(original), "sha256": hashlib.sha256(raw_bytes).hexdigest()}],
        }
        registered = replace(parsed, path=parsed.state_dir / "lab.json", config_bytes=raw_bytes, passport=passport)
        state = runtime_init(registered)
        final = runtime_load(registered.path, allow_pending=config.pending)
        for field in dataclasses.fields(final):
            object.__setattr__(config, field.name, getattr(final, field.name))
        _DRAFT_ALIASES[str(original)] = str(registered.path)
        # `{config_dir}` resolves to the lab directory: bring the draft's input files along.
        for name in raw.get("input_files", []):
            source = Path(name)
            if source.is_file() and source.parent == original.parent:
                shutil.copy2(source, parsed.state_dir / source.name)
        return state

    try:
        import durable_workflow_helpers as durable
    except ImportError:
        from tests import durable_workflow_helpers as durable
    lab_module.run_baseline = durable.run_baseline
    lab_module.run_trial = durable.run_trial
    config_module.load_config = load_config
    lab_module.load_config = load_config
    lab_module.init_lab = init_lab
    snapshot_module.init_lab = init_lab


_install_draft_lab_shims()


def save_manifest(config, manifest):
    """Write a run manifest into the lab's SQLite store, opening the epoch it names if needed."""
    from autoresearch_runtime.measurement import open_epoch
    from autoresearch_runtime.sqlite_store import SQLiteStore
    from autoresearch_runtime.state import _save_manifest

    store = SQLiteStore()
    state = store.load_state(config.lab_id)
    epoch = int(manifest.get("epoch") or 1)
    opened = False
    while int(state.get("current_epoch") or 1) < epoch:
        open_epoch(state, config)
        opened = True
    if opened:
        store.save_state(config.lab_id, state)
    _save_manifest(config, manifest)
