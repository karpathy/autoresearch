"""Validate host-provided approval and isolation attestations for durable runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .errors import LabError


_PROFILE_ENV = "AUTORESEARCH_HARNESS_PROFILE"
_MAX_PROFILE_BYTES = 64 * 1024
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")
_FINGERPRINT = re.compile(r"^[a-f0-9]{64}$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}(?::[0-9]{1,5})?$")
_SANDBOX_SCOPES = {"workspace", "container", "vm", "microvm"}


def contract_fingerprint(config: Any) -> str:
    """Hash the active measurement epoch and its evaluator contract for approval binding."""
    from .measurement import epoch_fingerprint
    from .state import load_state

    state = load_state(config)
    epoch_id = state.get("current_epoch")
    if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 1:
        raise LabError("SQLite lab has no current contract epoch", code="E_HARNESS_POLICY_REJECTED")
    material = {
        "schema_version": 1,
        "epoch_id": epoch_id,
        "epoch": epoch_fingerprint(config, state),
        "primary": config.primary,
        "direction": config.direction,
        "guards": list(config.guards),
        "repeats": config.repeats,
        "evaluator_capabilities": getattr(config, "evaluator_capabilities", {}) or {},
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fail(message: str, code: str = "E_HARNESS_PROFILE_INVALID") -> None:
    raise LabError(message, code=code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        _fail(f"Harness profile {name} must be a non-empty string no longer than 256 characters")
    return value.strip()


def _inside(path: Path, root: Path) -> bool:
    """Whether path is spelled through root. Each ancestor is resolved, so short (8.3) names cannot hide it."""
    try:
        resolved_root = root.resolve()
        return any(candidate.resolve() == resolved_root for candidate in (path, *path.parents))
    except (OSError, RuntimeError):
        return False


def _profile_path(config: Any) -> Path:
    raw = os.environ.get(_PROFILE_ENV)
    if not isinstance(raw, str) or not raw.strip():
        _fail(f"Set {_PROFILE_ENV} to a trusted host profile outside the measured workspace",
              "E_HARNESS_PROFILE_REQUIRED")
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        _fail(f"{_PROFILE_ENV} must be an absolute path")
    protected_roots = [getattr(config, "source_root", None), getattr(config, "state_dir", None)]
    lexical_path = Path(os.path.abspath(supplied))
    for root in protected_roots:
        if root is not None and _inside(lexical_path, Path(root)):
            _fail("Harness profile must be outside the measured project and lab state")
    if supplied.is_symlink():
        _fail("Harness profile must not be a symbolic link")
    try:
        path = supplied.resolve(strict=True)
    except OSError as error:
        _fail(f"Harness profile cannot be read: {error}", "E_HARNESS_PROFILE_REQUIRED")
    if not path.is_file():
        _fail("Harness profile path must name a regular file")
    for root in protected_roots:
        if root is not None and _inside(path, Path(root)):
            _fail("Harness profile must be outside the measured project and lab state")
    return path


def _normalize_profile(value: Any, fingerprint: str) -> dict[str, Any]:
    # "spend" was removed in 0.8.0. An older profile may still carry it; the block is ignored
    # and never enters the normalized, attested material.
    required = {"schema_version", "harness", "approval", "sandbox", "network"}
    if not isinstance(value, dict) or set(value) - required - {"spend"} or not required.issubset(value):
        _fail("Harness profile must contain exactly schema_version, harness, approval, sandbox, and network")
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        _fail("Harness profile schema_version must be 1")

    harness = value.get("harness")
    if not isinstance(harness, dict) or set(harness) != {"id", "version"}:
        _fail("Harness profile harness must contain exactly id and version")
    harness_id = _text(harness.get("id"), "harness.id")
    version = _text(harness.get("version"), "harness.version")
    if not _ID.fullmatch(harness_id) or not _ID.fullmatch(version):
        _fail("Harness profile harness id and version contain unsupported characters")

    approval = value.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"approved", "contract_fingerprint"}:
        _fail("Harness profile approval must contain exactly approved and contract_fingerprint")
    if approval.get("approved") is not True:
        _fail("Harness profile must attest approval for this run")
    approved_fingerprint = approval.get("contract_fingerprint")
    if not isinstance(approved_fingerprint, str) or not _FINGERPRINT.fullmatch(approved_fingerprint):
        _fail("Harness profile approval.contract_fingerprint must be a SHA-256 hex digest")
    if approved_fingerprint != fingerprint:
        _fail("Harness profile approval is stale for the current contract", "E_HARNESS_POLICY_REJECTED")

    sandbox = value.get("sandbox")
    if not isinstance(sandbox, dict) or set(sandbox) != {"enabled", "scope"}:
        _fail("Harness profile sandbox must contain exactly enabled and scope")
    sandbox_scope = sandbox.get("scope")
    if (sandbox.get("enabled") is not True or not isinstance(sandbox_scope, str)
            or sandbox_scope not in _SANDBOX_SCOPES):
        _fail("Harness profile must attest an enabled workspace, container, vm, or microvm sandbox",
              "E_HARNESS_POLICY_REJECTED")

    network = value.get("network")
    if not isinstance(network, dict) or set(network) - {"mode", "allowed_hosts"} or "mode" not in network:
        _fail("Harness profile network must contain mode and optional allowed_hosts")
    mode = network.get("mode")
    if not isinstance(mode, str) or mode not in {"disabled", "allowlisted", "unrestricted", "unknown"}:
        _fail("Harness profile network.mode is invalid")
    allowed_hosts: list[str] = []
    if mode == "allowlisted":
        hosts = network.get("allowed_hosts")
        if not isinstance(hosts, list) or not hosts or len(hosts) > 100:
            _fail("Allowlisted network requires 1–100 allowed_hosts", "E_HARNESS_POLICY_REJECTED")
        for host in hosts:
            item = _text(host, "network.allowed_hosts item")
            if not _HOST.fullmatch(item):
                _fail("Harness profile contains an invalid allowed host")
            allowed_hosts.append(item.lower())
        if len(set(allowed_hosts)) != len(allowed_hosts):
            _fail("Harness profile allowed_hosts must be unique")
    elif "allowed_hosts" in network:
        if network.get("allowed_hosts") != []:
            _fail("network.allowed_hosts is only valid for allowlisted network")
    if mode in {"unknown", "unrestricted"}:
        _fail(f"Harness profile network mode {mode!r} is blocked", "E_HARNESS_POLICY_REJECTED")

    normalized_network: dict[str, Any] = {"mode": mode}
    if mode == "allowlisted":
        normalized_network["allowed_hosts"] = allowed_hosts
    return {
        "schema_version": 1,
        "harness": {"id": harness_id, "version": version},
        "approval": {"approved": True, "contract_fingerprint": fingerprint},
        "sandbox": {"enabled": True, "scope": sandbox_scope},
        "network": normalized_network,
    }


def authorize_operation(config: Any, command: str, *, confirmed: Any) -> dict[str, Any]:
    """Capture an operation-scoped policy snapshot before durable enqueue."""
    if confirmed is not True:
        _fail("This evaluator/write operation requires explicit confirmed=true",
              "E_HARNESS_CONFIRMATION_REQUIRED")
    fingerprint = contract_fingerprint(config)
    path = _profile_path(config)
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"Harness profile cannot be read: {error}", "E_HARNESS_PROFILE_REQUIRED")
    if len(raw) > _MAX_PROFILE_BYTES:
        _fail("Harness profile exceeds the 64 KiB size limit")
    try:
        parsed = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError) as error:
        _fail(f"Harness profile is not strict UTF-8 JSON: {error}")
    profile = _normalize_profile(parsed, fingerprint)
    normalized = json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        "schema_version": 1,
        "decision": "allow",
        "operation": command,
        "confirmed": True,
        "contract_fingerprint": fingerprint,
        "profile_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "profile": profile,
    }


def recheck_operation(config: Any, command: str, stored: Any) -> dict[str, Any]:
    """Re-read the host profile immediately before dispatch and require an exact match."""
    if not isinstance(stored, dict) or stored.get("decision") != "allow" or stored.get("confirmed") is not True:
        _fail("Stored operation has no valid harness policy snapshot; explicit recovery is required",
              "E_HARNESS_POLICY_STALE")
    current = authorize_operation(config, command, confirmed=stored.get("confirmed"))
    if current != stored:
        _fail("Harness policy changed after enqueue; operation was not dispatched",
              "E_HARNESS_POLICY_STALE")
    return current
