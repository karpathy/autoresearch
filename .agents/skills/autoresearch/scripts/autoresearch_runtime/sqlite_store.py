"""SQLite source of truth for AutoResearch 0.4 labs.

The module deliberately has no import-time or constructor-time I/O. Only
``create_lab`` may create the shared database, after the caller has confirmed
the passport and prepared the candidate snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import statistics
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

from .paths import resolve_home


SCHEMA_VERSION = 2
INLINE_ARTIFACT_LIMIT = 256 * 1024
MAX_PAGE_SIZE = 100
_NIL_UUID = "00000000-0000-0000-0000-000000000000"
_ACTIVE_OPERATION_STATES = ("queued", "running", "finalizing", "recovery_required")
_FINAL_OPERATION_STATES = frozenset({"succeeded", "failed"})


class StoreError(RuntimeError):
    """Base class for explicit storage failures."""

    code = "store_error"


class StoreNotInitializedError(StoreError):
    code = "store_not_initialized"


class StoreCorruptError(StoreError):
    code = "store_corrupt"


class StoreVersionError(StoreError):
    code = "store_version_unsupported"


class StoreSchemaError(StoreError):
    code = "store_schema_invalid"


class StoreBusyError(StoreError):
    code = "store_busy"


class StoreIntegrityError(StoreError):
    code = "store_integrity_error"


class StoreConflictError(StoreError):
    code = "store_conflict"


class RevisionConflictError(StoreConflictError):
    code = "revision_conflict"


class IdempotencyConflictError(StoreConflictError):
    code = "idempotency_conflict"


class ActiveOperationError(StoreConflictError):
    code = "operation_already_active"


class OperationStateError(StoreConflictError):
    code = "operation_state_conflict"


class AmbiguousAliasError(StoreError):
    code = "ambiguous_alias"


class EntityNotFoundError(StoreError):
    code = "not_found"


class StoreGateError(StoreError, ValueError):
    code = "lab_confirmation_required"


class ArtifactIntegrityError(StoreError):
    code = "artifact_integrity_error"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not strict JSON: {exc}") from exc


def _loads(value: str | bytes | None, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    return json.loads(value)


def _uuid_text(value: uuid.UUID | str, *, label: str = "UUID") -> str:
    try:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    text = str(parsed)
    if not isinstance(value, uuid.UUID) and str(value) != text:
        raise ValueError(f"{label} must be a canonical lowercase UUID")
    return text


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            flags = getattr(path.lstat(), "st_file_attributes", 0)
            return bool(flags & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        except OSError:
            return True
    return False


def _path_alias(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _safe_page(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return min(limit, MAX_PAGE_SIZE), offset


@contextmanager
def _global_init_lock(path: Path) -> Iterator[None]:
    """Cross-process lock whose ownership is released automatically on exit/crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


_SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE labs (
    lab_id TEXT PRIMARY KEY CHECK(length(lab_id)=36 AND lab_id=lower(lab_id)
        AND substr(lab_id,9,1)='-' AND substr(lab_id,14,1)='-'
        AND substr(lab_id,19,1)='-' AND substr(lab_id,24,1)='-'
        AND lab_id NOT GLOB '*[^0-9a-f-]*'),
    state_dir TEXT NOT NULL,
    config_path TEXT NOT NULL,
    source_root TEXT NOT NULL,
    candidate_path TEXT NOT NULL,
    candidate_commit TEXT NOT NULL,
    goal TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('min','max')),
    status TEXT NOT NULL DEFAULT 'ready',
    lifecycle TEXT NOT NULL DEFAULT 'active' CHECK(lifecycle IN ('active','archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
    UNIQUE(state_dir),
    UNIQUE(config_path)
) STRICT;

CREATE TABLE epochs (
    lab_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL CHECK(epoch_id > 0),
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    superseded_at TEXT,
    reason TEXT,
    contract_hash TEXT,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(lab_id, epoch_id),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE metric_definitions (
    lab_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL CHECK(length(metric_name) > 0),
    role TEXT NOT NULL DEFAULT 'additional' CHECK(role IN ('primary','guard','additional')),
    direction TEXT CHECK(direction IS NULL OR direction IN ('min','max')),
    unit TEXT,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(lab_id, epoch_id, metric_name),
    FOREIGN KEY(lab_id, epoch_id) REFERENCES epochs(lab_id, epoch_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE aliases (
    alias_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('lab','run','operation')),
    scope_id TEXT NOT NULL,
    alias TEXT NOT NULL CHECK(length(alias) > 0),
    lab_id TEXT,
    run_id TEXT,
    operation_id TEXT,
    CHECK(
        (kind='lab' AND lab_id IS NOT NULL AND run_id IS NULL AND operation_id IS NULL)
        OR (kind='run' AND scope_id <> '""" + _NIL_UUID + """' AND lab_id IS NULL AND run_id IS NOT NULL AND operation_id IS NULL)
        OR (kind='operation' AND scope_id <> '""" + _NIL_UUID + """' AND lab_id IS NULL AND run_id IS NULL AND operation_id IS NOT NULL)
    ),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
    FOREIGN KEY(scope_id, run_id) REFERENCES runs(lab_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY(scope_id, operation_id) REFERENCES operations(lab_id, operation_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE runs (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL CHECK(length(run_id)=36 AND run_id=lower(run_id)
        AND substr(run_id,9,1)='-' AND substr(run_id,14,1)='-'
        AND substr(run_id,19,1)='-' AND substr(run_id,24,1)='-'
        AND run_id NOT GLOB '*[^0-9a-f-]*'),
    epoch_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    alias TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    parent_run_id TEXT,
    candidate_commit TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(lab_id, run_id),
    UNIQUE(run_id),
    UNIQUE(lab_id, run_id, epoch_id),
    UNIQUE(lab_id, alias),
    FOREIGN KEY(lab_id, epoch_id) REFERENCES epochs(lab_id, epoch_id),
    FOREIGN KEY(lab_id, parent_run_id) REFERENCES runs(lab_id, run_id)
) STRICT;

CREATE TABLE projections (
    lab_id TEXT NOT NULL,
    view TEXT NOT NULL CHECK(view IN ('config','state','candidate')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    payload_artifact_id TEXT,
    source_bytes BLOB,
    source_sha256 TEXT,
    source_artifact_id TEXT,
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(lab_id, view),
    CHECK((source_bytes IS NULL AND source_sha256 IS NULL AND source_artifact_id IS NULL) OR
          (source_bytes IS NOT NULL AND source_artifact_id IS NULL AND source_sha256 IS NOT NULL AND length(source_sha256)=64) OR
          (source_bytes IS NULL AND source_artifact_id IS NOT NULL AND source_sha256 IS NOT NULL AND length(source_sha256)=64)),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, payload_artifact_id) REFERENCES artifacts(lab_id, artifact_id),
    FOREIGN KEY(lab_id, source_artifact_id) REFERENCES artifacts(lab_id, artifact_id)
) STRICT;

CREATE TABLE lab_documents (
    lab_id TEXT NOT NULL,
    document_key TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(lab_id, document_key),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE history (
    lab_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK(sequence > 0),
    run_id TEXT,
    kind TEXT NOT NULL DEFAULT 'record',
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(lab_id, sequence),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, run_id) REFERENCES runs(lab_id, run_id)
) STRICT;

CREATE TABLE repeats (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    repeat_index INTEGER NOT NULL CHECK(repeat_index > 0),
    status TEXT NOT NULL,
    cost_usd REAL CHECK(cost_usd IS NULL OR (cost_usd >= 0 AND cost_usd < 1.0e308)),
    requests INTEGER CHECK(requests IS NULL OR requests >= 0),
    started_at TEXT,
    finished_at TEXT,
    result_sha256 TEXT NOT NULL CHECK(length(result_sha256)=64),
    result_json TEXT NOT NULL CHECK(json_valid(result_json)),
    result_artifact_id TEXT,
    PRIMARY KEY(lab_id, run_id, repeat_index),
    UNIQUE(lab_id, run_id, epoch_id, repeat_index),
    FOREIGN KEY(lab_id, run_id, epoch_id) REFERENCES runs(lab_id, run_id, epoch_id) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, result_artifact_id) REFERENCES artifacts(lab_id, artifact_id)
) STRICT;

CREATE TABLE metric_values (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    repeat_index INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL CHECK(value > -1.0e308 AND value < 1.0e308),
    PRIMARY KEY(lab_id, run_id, repeat_index, metric_name),
    FOREIGN KEY(lab_id, run_id, epoch_id, repeat_index)
        REFERENCES repeats(lab_id, run_id, epoch_id, repeat_index) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, epoch_id, metric_name)
        REFERENCES metric_definitions(lab_id, epoch_id, metric_name)
) STRICT;

CREATE TABLE cases (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    repeat_index INTEGER NOT NULL DEFAULT 1 CHECK(repeat_index > 0),
    case_id TEXT NOT NULL CHECK(length(case_id) > 0),
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    PRIMARY KEY(lab_id, run_id, repeat_index, case_id),
    FOREIGN KEY(lab_id, run_id, epoch_id, repeat_index)
        REFERENCES repeats(lab_id, run_id, epoch_id, repeat_index) ON DELETE CASCADE
) STRICT;

CREATE TABLE attempts (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    repeat_index INTEGER NOT NULL DEFAULT 1 CHECK(repeat_index > 0),
    operation_id TEXT,
    attempt_id TEXT NOT NULL CHECK(length(attempt_id) > 0),
    case_id TEXT,
    event TEXT NOT NULL CHECK(event IN ('started','finished','usage','attempt','wire')),
    created_at TEXT NOT NULL,
    cost_usd REAL CHECK(cost_usd IS NULL OR (cost_usd >= 0 AND cost_usd < 1.0e308)),
    requests INTEGER CHECK(requests IS NULL OR requests >= 0),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    UNIQUE(lab_id, run_id, repeat_index, attempt_id, event),
    FOREIGN KEY(lab_id, run_id, epoch_id, repeat_index)
        REFERENCES repeats(lab_id, run_id, epoch_id, repeat_index) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, operation_id) REFERENCES operations(lab_id, operation_id)
) STRICT;

CREATE UNIQUE INDEX attempts_operation_identity
    ON attempts(operation_id, attempt_id, event) WHERE operation_id IS NOT NULL;

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY CHECK(length(artifact_id)=36 AND artifact_id=lower(artifact_id)),
    lab_id TEXT NOT NULL,
    run_id TEXT,
    name TEXT NOT NULL CHECK(length(name) > 0),
    evidence_key TEXT,
    media_type TEXT,
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    inline_data BLOB,
    object_path TEXT,
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    created_at TEXT NOT NULL,
    CHECK((inline_data IS NOT NULL AND object_path IS NULL) OR
          (inline_data IS NULL AND object_path IS NOT NULL)),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE,
    FOREIGN KEY(lab_id, run_id) REFERENCES runs(lab_id, run_id) ON DELETE CASCADE,
    UNIQUE(lab_id, artifact_id)
) STRICT;

CREATE TABLE operations (
    lab_id TEXT NOT NULL,
    operation_id TEXT NOT NULL CHECK(length(operation_id)=36 AND operation_id=lower(operation_id)),
    kind TEXT NOT NULL CHECK(length(kind) > 0),
    idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) > 0),
    payload_hash TEXT NOT NULL CHECK(length(payload_hash)=64),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    status TEXT NOT NULL CHECK(status IN ('queued','running','finalizing','succeeded','failed','recovery_required')),
    pid INTEGER CHECK(pid IS NULL OR pid > 0),
    heartbeat_at TEXT,
    progress_json TEXT CHECK(progress_json IS NULL OR json_valid(progress_json)),
    result_json TEXT CHECK(result_json IS NULL OR json_valid(result_json)),
    error_json TEXT CHECK(error_json IS NULL OR json_valid(error_json)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(operation_id),
    UNIQUE(lab_id, operation_id),
    UNIQUE(lab_id, idempotency_key),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
) STRICT;

CREATE UNIQUE INDEX one_active_operation_per_lab ON operations(lab_id)
    WHERE status IN ('queued','running','finalizing','recovery_required');

CREATE TABLE pins (
    lab_id TEXT NOT NULL,
    name TEXT NOT NULL CHECK(length(name) > 0),
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    source_path TEXT,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    created_at TEXT NOT NULL,
    PRIMARY KEY(lab_id, name),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE roots (
    root_ref TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json))
) STRICT;

CREATE TABLE catalog_metadata (
    root_ref TEXT NOT NULL,
    project_ref TEXT NOT NULL,
    lab_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(root_ref, project_ref),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE SET NULL
) STRICT;

CREATE TABLE active_labs (
    root_ref TEXT NOT NULL,
    project_ref TEXT NOT NULL,
    lab_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(root_ref, project_ref),
    FOREIGN KEY(lab_id) REFERENCES labs(lab_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE run_documents (
    lab_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    document_kind TEXT NOT NULL CHECK(document_kind IN ('manifest')),
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(lab_id, run_id, document_kind),
    FOREIGN KEY(lab_id, run_id) REFERENCES runs(lab_id, run_id) ON DELETE CASCADE
) STRICT;

CREATE INDEX runs_by_lab_created ON runs(lab_id, created_at DESC, run_id DESC);
CREATE INDEX runs_by_lab_epoch ON runs(lab_id, epoch_id, created_at DESC, run_id DESC);
CREATE INDEX runs_by_lab_status ON runs(lab_id, status, created_at DESC);
CREATE INDEX repeats_by_lab_run ON repeats(lab_id, run_id, repeat_index);
CREATE INDEX metrics_by_lab_epoch ON metric_values(lab_id, epoch_id, metric_name, run_id);
CREATE INDEX cases_by_lab_run ON cases(lab_id, run_id, updated_at);
CREATE INDEX attempts_by_lab_run ON attempts(lab_id, run_id, repeat_index, created_at);
CREATE INDEX artifacts_by_evidence ON artifacts(lab_id, evidence_key, created_at DESC);
CREATE INDEX aliases_by_name ON aliases(kind, alias, scope_id);
CREATE UNIQUE INDEX unique_run_alias ON aliases(kind, scope_id, alias) WHERE kind='run';
CREATE UNIQUE INDEX unique_operation_alias ON aliases(kind, scope_id, alias) WHERE kind='operation';
CREATE INDEX operations_by_lab_status ON operations(lab_id, status, created_at DESC);
CREATE INDEX catalog_by_root_lab ON catalog_metadata(root_ref, lab_id, project_ref);
CREATE UNIQUE INDEX one_primary_metric_per_epoch ON metric_definitions(lab_id,epoch_id) WHERE role='primary';
"""


class SQLiteStore:
    """One SQLite database shared by every registered lab root."""

    def __init__(self, home: str | Path | None = None):
        # Path resolution alone is read-only: neither the parent nor database is created.
        self.home = Path(home).expanduser().resolve() if home is not None else resolve_home()
        self.db_path = self.home / "autoresearch.sqlite3"
        self._object_root = self.home / "objects" / "sha256"

    def is_initialized(self) -> bool:
        """Return whether a database file exists; never creates the home directory."""
        return self.db_path.is_file()

    def get_revision(self) -> int:
        """Return the database-wide mutation revision, or zero before initialization."""
        if not self.is_initialized():
            return 0
        with self._reader() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()
        if row is None:
            raise StoreSchemaError("SQLite metadata table is missing its revision record")
        return int(row[0])

    def _write_object(self, content: bytes, *, conn: sqlite3.Connection) -> tuple[str, str]:
        """Publish a content object while its caller holds the SQLite writer lock."""
        if not conn.in_transaction:
            raise AssertionError("Content objects must be published inside a writer transaction")
        digest = _sha256(content)
        directory = self._object_root / digest[:2]
        probe = self.home
        for part in directory.relative_to(self.home).parts:
            probe = probe / part
            if probe.exists() and _is_link_or_reparse(probe):
                raise ArtifactIntegrityError("Content store path contains a symlink or reparse point")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / digest
        if target.exists():
            if _is_link_or_reparse(target) or not target.is_file() or _sha256(target.read_bytes()) != digest:
                raise ArtifactIntegrityError("Existing content-addressed object failed integrity verification")
        else:
            temporary = directory / (digest + ".tmp-" + uuid.uuid4().hex)
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target.relative_to(self.home).as_posix(), digest

    def _insert_large_payload_artifact(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        name: str,
        content: bytes,
        *,
        run_id: str | None = None,
        prepared_object_path: str | None = None,
    ) -> str:
        digest = _sha256(content)
        if prepared_object_path is None:
            object_path, written_digest = self._write_object(content, conn=conn)
            if written_digest != digest:
                raise ArtifactIntegrityError("Prepared object hash changed")
        else:
            relative = Path(prepared_object_path)
            if relative.is_absolute() or ".." in relative.parts or relative.name != digest:
                raise ArtifactIntegrityError("Prepared object path is unsafe")
            unresolved = self.home / relative
            probe = self.home
            for part in relative.parts:
                probe = probe / part
                if _is_link_or_reparse(probe):
                    raise ArtifactIntegrityError("Prepared object path contains a symlink or reparse point")
            if not unresolved.is_file():
                raise ArtifactIntegrityError("Prepared content-addressed object is missing")
            object_path = relative.as_posix()
        artifact_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO artifacts(artifact_id,lab_id,run_id,name,evidence_key,media_type,sha256,size_bytes,"
            "inline_data,object_path,metadata_json,created_at) VALUES(?,?,?,?,NULL,'application/json',?,?,NULL,?,?,?)",
            (artifact_id, lab_id, run_id, name, digest, len(content), object_path,
             _json({"stored_as": "large_payload"}), _now()),
        )
        return artifact_id

    def _projection_storage_prepared(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        view: str,
        encoded: str,
        prepared_object_path: str | None,
    ) -> tuple[str, str | None]:
        raw = encoded.encode("utf-8")
        if len(raw) <= INLINE_ARTIFACT_LIMIT:
            return encoded, None
        artifact_id = self._insert_large_payload_artifact(
            conn, lab_id, f"{view}.json", raw, prepared_object_path=prepared_object_path
        )
        return "{}", artifact_id

    def _projection_storage(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        view: str,
        encoded: str,
    ) -> tuple[str, str | None]:
        raw = encoded.encode("utf-8")
        if len(raw) <= INLINE_ARTIFACT_LIMIT:
            return encoded, None
        artifact_id = self._insert_large_payload_artifact(conn, lab_id, f"{view}.json", raw)
        return "{}", artifact_id

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise StoreNotInitializedError(f"SQLite store is absent: {self.db_path}")
        mode = "rw" if write else "ro"
        uri = self.db_path.as_uri() + f"?mode={mode}"
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=15.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            journal = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == 1:
                conn.close()
                conn = None
                self._ensure_schema()
                return self._connect(write=write)
            self._check_schema(conn, version=version)
            if journal not in ("delete", "wal"):
                conn.close()
                conn = None
                raise StoreSchemaError("Database must use journal_mode=DELETE or WAL")
            return conn
        except StoreError:
            if conn is not None:
                conn.close()
            raise
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            self._raise_database_error(exc)
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            self._raise_database_error(exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_database_error(exc: sqlite3.DatabaseError) -> None:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            raise StoreBusyError("SQLite store remained busy for 15 seconds") from exc
        if any(word in message for word in ("malformed", "corrupt", "not a database", "disk image")):
            raise StoreCorruptError(f"SQLite store is corrupt: {exc}") from exc
        if "no such table" in message or "no such column" in message:
            raise StoreSchemaError(f"SQLite schema is incomplete: {exc}") from exc
        raise StoreError(f"SQLite operation failed: {exc}") from exc

    @staticmethod
    def _check_schema(conn: sqlite3.Connection, *, version: int | None = None) -> None:
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0]) if version is None else version
            if version == 0:
                raise StoreNotInitializedError("SQLite schema has not been initialized")
            if version > SCHEMA_VERSION:
                raise StoreVersionError(
                    f"SQLite schema version {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < SCHEMA_VERSION:
                raise StoreVersionError(
                    f"SQLite schema version {version} needs an explicit migration to {SCHEMA_VERSION}"
                )
            row = conn.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()
            if row is None:
                raise StoreSchemaError("SQLite metadata table is missing its revision record")
        except StoreError:
            raise
        except sqlite3.DatabaseError as exc:
            SQLiteStore._raise_database_error(exc)

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(write=False)
        try:
            yield conn
        except StoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise StoreIntegrityError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:
            self._raise_database_error(exc)
        finally:
            conn.close()

    @contextmanager
    def _writer(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(write=True)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except StoreError:
            conn.rollback()
            raise
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if "UNIQUE constraint failed: operations.lab_id" in str(exc):
                raise ActiveOperationError("A mutating operation is already active for this lab") from exc
            raise StoreIntegrityError(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            conn.rollback()
            self._raise_database_error(exc)
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            self._raise_database_error(exc)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _write_scope(self, transaction: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        """Reuse a caller's writer transaction or open a normal one."""
        if transaction is None:
            with self._writer() as conn:
                yield conn
        else:
            yield transaction

    @staticmethod
    def _bump_revision(conn: sqlite3.Connection, lab_id: str | None = None) -> int:
        conn.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
        if lab_id is None:
            row = conn.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()
            return int(row[0])
        now = _now()
        conn.execute("UPDATE labs SET revision=revision+1, updated_at=? WHERE lab_id=?", (now, lab_id))
        row = conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_id,)).fetchone()
        if row is None:
            raise EntityNotFoundError(f"Lab not found: {lab_id}")
        return int(row[0])

    def _ensure_schema(self) -> None:
        """Create/migrate schema only from the explicit post-confirmation init path."""
        self.home.mkdir(parents=True, exist_ok=True)
        with _global_init_lock(self.home / ".autoresearch.init.lock"):
            # sqlite3.connect creates the file only here, while the global lock is held.
            try:
                conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=15000")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA synchronous=FULL")
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                journal = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                objects = conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type IN ('table','index','trigger','view') "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise StoreVersionError(
                        f"SQLite schema version {version} is newer than supported {SCHEMA_VERSION}"
                    )
                if version == SCHEMA_VERSION:
                    if journal not in ("delete", "wal"):
                        raise StoreSchemaError("Database must use journal_mode=DELETE or WAL")
                    self._check_schema(conn)
                    return
                if version == 1:
                    if journal not in ("delete", "wal"):
                        raise StoreSchemaError("Database must use journal_mode=DELETE or WAL")
                    # Keep the existing database and every lab row; lifecycle is additive.
                    conn.executescript(
                        "BEGIN IMMEDIATE;\n"
                        "ALTER TABLE labs ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'active' "
                        "CHECK(lifecycle IN ('active','archived'));\n"
                        "UPDATE metadata SET value='2' WHERE key='schema_version';\n"
                        f"PRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                    )
                    self._check_schema(conn)
                    return
                if version != 0 or objects:
                    raise StoreCorruptError("SQLite file has schema objects without a recognized user_version")
                use_wal = os.environ.get("AUTORESEARCH_SQLITE_WAL") == "1"
                target_journal = "WAL" if use_wal else "DELETE"
                if journal != target_journal.lower():
                    conn.execute(f"PRAGMA journal_mode={target_journal}")
                # executescript starts by committing an open transaction, so put
                # BEGIN/COMMIT in the script to keep every DDL statement atomic.
                conn.executescript(
                    "BEGIN IMMEDIATE;\n" + _SCHEMA_SQL +
                    "\nINSERT INTO metadata(key,value) VALUES('revision','0');"
                    f"\nINSERT INTO metadata(key,value) VALUES('schema_version','{SCHEMA_VERSION}');"
                    f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                )
            except StoreError:
                raise
            except sqlite3.OperationalError as exc:
                self._raise_database_error(exc)
            except sqlite3.DatabaseError as exc:
                self._raise_database_error(exc)
            finally:
                if "conn" in locals():
                    conn.close()

    @staticmethod
    def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return default

    @classmethod
    def _validate_create_input(
        cls,
        config: dict[str, Any],
        initial_state: dict[str, Any],
        candidate_path: Path,
    ) -> dict[str, Any]:
        if not isinstance(config, dict) or not isinstance(initial_state, dict):
            raise StoreGateError("Confirmed config and initial state must be dictionaries")
        passport = config.get("_passport") if isinstance(config.get("_passport"), dict) else None
        if passport is None:
            passport = config.get("passport") if isinstance(config.get("passport"), dict) else {}
        confirmed = config.get("confirmed") is True or passport.get("confirmed") is True
        if not confirmed:
            raise StoreGateError("Lab creation requires an explicitly confirmed passport")
        goal = cls._first(config, "goal", "brief_goal", default=passport.get("goal"))
        if not isinstance(goal, str) or not goal.strip():
            raise StoreGateError("Lab passport requires a non-empty goal")
        source_root = cls._first(config, "source_root", "project", default=passport.get("project"))
        if not isinstance(source_root, (str, Path)) or not str(source_root).strip():
            raise StoreGateError("Lab passport requires a project path")
        metrics = config.get("metrics") if isinstance(config.get("metrics"), dict) else {}
        if not metrics and isinstance(passport.get("metrics"), dict):
            metrics = passport["metrics"]
        primary = cls._first(metrics, "primary", default=cls._first(config, "primary_metric", "primary"))
        direction = cls._first(metrics, "direction", default=config.get("direction", "max"))
        if not isinstance(primary, str) or not primary.strip():
            raise StoreGateError("Lab passport requires a primary metric")
        if direction not in {"min", "max"}:
            raise StoreGateError("Lab primary metric direction must be 'min' or 'max'")
        evaluator = config.get("evaluator") or passport.get("evaluator")
        if not isinstance(evaluator, (list, tuple)) or not evaluator or any(
            not isinstance(part, (str, dict)) or not part for part in evaluator
        ):
            raise StoreGateError("Lab passport requires a non-empty evaluator")
        if not any(key in config or key in passport for key in ("constraints", "limits", "budget")):
            raise StoreGateError("Lab passport must explicitly record constraints or limits")
        pins = config.get("pins") or passport.get("pins") or initial_state.get("pins")
        if not isinstance(pins, (dict, list, tuple)) or not pins:
            raise StoreGateError("Lab passport requires non-empty pin evidence")
        state_dir = cls._first(config, "state_dir", default=initial_state.get("state_dir"))
        config_path = cls._first(config, "config_path", "path", default=initial_state.get("config_path"))
        if not isinstance(state_dir, (str, Path)) or not str(state_dir).strip():
            raise StoreGateError("Lab config requires a state_dir path")
        if not isinstance(config_path, (str, Path)) or not str(config_path).strip():
            config_path = Path(state_dir) / "lab.json"
        candidate_path = Path(candidate_path).expanduser().resolve()
        if not candidate_path.is_dir() or not (candidate_path / ".git").exists():
            raise StoreGateError("Candidate snapshot is not a prepared Git repository")
        candidate_commit = cls._first(initial_state, "best_commit", "candidate_commit")
        if not isinstance(candidate_commit, str) or len(candidate_commit) not in {40, 64} or any(
            char not in "0123456789abcdef" for char in candidate_commit.lower()
        ):
            raise StoreGateError("Candidate snapshot must include its committed tree hash")
        state = dict(initial_state)
        state.setdefault("current_epoch", 1)
        epoch_id = state.get("current_epoch")
        if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 1:
            raise StoreGateError("Initial state requires a positive current_epoch")
        pin_rows = []
        if isinstance(pins, dict):
            pin_iter = pins.items()
        else:
            pin_iter = ((str(item.get("name") or item.get("id") or index), item) for index, item in enumerate(pins))
        for name, pin in pin_iter:
            if not isinstance(name, str) or not name.strip():
                continue
            payload = pin if isinstance(pin, dict) else {"value": pin}
            encoded = _json(payload).encode("utf-8")
            pin_rows.append((name, payload, _sha256(encoded), payload.get("path")))
        if not pin_rows:
            raise StoreGateError("Lab passport pin evidence has no usable records")
        return {
            "goal": goal.strip(),
            "source_root": _path_alias(str(source_root)),
            "state_dir": _path_alias(state_dir),
            "config_path": _path_alias(config_path),
            "candidate_path": str(candidate_path),
            "candidate_commit": candidate_commit.lower(),
            "primary_metric": primary.strip(),
            "direction": direction,
            "epoch_id": epoch_id,
            "metrics": metrics,
            "pins": pin_rows,
            "root_ref": cls._first(config, "root_ref", default=passport.get("root_ref")),
            "legacy_aliases": [
                str(value) for value in (
                    config.get("lab_id"), config.get("id"), config.get("contract_id"),
                    config.get("alias"), passport.get("contract_id"),
                ) if isinstance(value, str) and value.strip()
            ],
            "config": config,
            "state": state,
        }

    def create_lab(
        self,
        config: dict[str, Any],
        initial_state: dict[str, Any],
        candidate_path: str | Path,
        *,
        source_config_bytes: bytes | None = None,
        config_bytes: bytes | None = None,
        raw_config_sha256: str | None = None,
        _transaction: sqlite3.Connection | None = None,
    ) -> uuid.UUID:
        """Create schema and lab atomically, after validating the confirmation gate."""
        ready = self._validate_create_input(config, initial_state, Path(candidate_path))
        if source_config_bytes is not None and config_bytes is not None and source_config_bytes != config_bytes:
            raise ValueError("source_config_bytes and config_bytes disagree")
        source_config_bytes = source_config_bytes if source_config_bytes is not None else config_bytes
        if source_config_bytes is not None and not isinstance(source_config_bytes, bytes):
            raise TypeError("source_config_bytes must be bytes")
        clean_config = dict(ready["config"])
        clean_config.pop("_passport", None)
        for private_key in ("raw_config_bytes", "source_config_bytes", "_raw_config_bytes", "config_bytes"):
            raw = clean_config.pop(private_key, None)
            if source_config_bytes is None and isinstance(raw, bytes):
                source_config_bytes = raw
        source_bytes = source_config_bytes if source_config_bytes is not None else _json(clean_config).encode("utf-8")
        config_hash = _sha256(source_bytes)
        if raw_config_sha256 is not None and raw_config_sha256.lower() != config_hash:
            raise StoreGateError("Raw config bytes do not match the supplied source hash")
        if isinstance(clean_config.get("config_sha256"), str) and clean_config["config_sha256"] not in {config_hash, ""}:
            # A canonical input hash may refer to the user's source file; source bytes are the authority here.
            clean_config.setdefault("source_config_sha256", clean_config["config_sha256"])
        config_json = _json(clean_config)
        state_json = _json(ready["state"])
        now = _now()
        lab_id = uuid.uuid4()
        snapshot = {
            "candidate_path": ready["candidate_path"],
            "commit": ready["candidate_commit"],
            "source": ready["state"].get("source", []),
            "inputs": ready["state"].get("inputs", []),
        }

        candidate_projection = _json(snapshot)
        if _transaction is None:
            self._ensure_schema()
        prepared_objects: dict[str, str] = {}
        with self._write_scope(_transaction) as conn:
            for name, content in (
                ("config", config_json.encode("utf-8")),
                ("state", state_json.encode("utf-8")),
                ("candidate", candidate_projection.encode("utf-8")),
                ("source-config", source_bytes),
            ):
                if len(content) > INLINE_ARTIFACT_LIMIT:
                    object_path, digest = self._write_object(content, conn=conn)
                    if digest != _sha256(content):
                        raise ArtifactIntegrityError(f"Prepared {name} object hash changed")
                    prepared_objects[name] = object_path
            conn.execute(
                "INSERT INTO labs(lab_id,state_dir,config_path,source_root,candidate_path,candidate_commit,"
                "goal,primary_metric,direction,status,created_at,updated_at,revision,snapshot_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,'ready',?,?,0,?)",
                (str(lab_id), ready["state_dir"], ready["config_path"], ready["source_root"],
                 ready["candidate_path"], ready["candidate_commit"], ready["goal"],
                 ready["primary_metric"], ready["direction"], now, now, _json(snapshot)),
            )
            conn.execute(
                "INSERT INTO aliases(kind,scope_id,alias,lab_id) VALUES('lab',?,?,?)",
                (_NIL_UUID, str(lab_id), str(lab_id)),
            )
            aliases = {( _NIL_UUID, ready["state_dir"]), (_NIL_UUID, ready["config_path"])}
            if ready["root_ref"]:
                aliases.update((str(ready["root_ref"]), value) for value in ready["legacy_aliases"])
            aliases.update((_NIL_UUID, value) for value in ready["legacy_aliases"])
            aliases.add((_NIL_UUID, Path(ready["state_dir"]).name))
            for scope, alias in aliases:
                conn.execute(
                    "INSERT OR IGNORE INTO aliases(kind,scope_id,alias,lab_id) VALUES('lab',?,?,?)",
                    (scope, alias, str(lab_id)),
                )
            config_payload, config_artifact_id = self._projection_storage_prepared(
                conn, str(lab_id), "config", config_json, prepared_objects.get("config")
            )
            state_payload, state_artifact_id = self._projection_storage_prepared(
                conn, str(lab_id), "state", state_json, prepared_objects.get("state")
            )
            candidate_payload, candidate_artifact_id = self._projection_storage_prepared(
                conn, str(lab_id), "candidate", candidate_projection, prepared_objects.get("candidate")
            )
            source_artifact_id = None
            inline_source = source_bytes
            if len(source_bytes) > INLINE_ARTIFACT_LIMIT:
                source_artifact_id = self._insert_large_payload_artifact(
                    conn, str(lab_id), "source-config.json", source_bytes,
                    prepared_object_path=prepared_objects.get("source-config"),
                )
                inline_source = None
            conn.execute(
                "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,source_bytes,source_sha256,"
                "source_artifact_id,revision,updated_at) VALUES(?,?,?,?,?,?,?,0,?)",
                (str(lab_id), "config", config_payload, config_artifact_id,
                 sqlite3.Binary(inline_source) if inline_source is not None else None, config_hash,
                 source_artifact_id, now),
            )
            conn.execute(
                "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,revision,updated_at) "
                "VALUES(?,?,?,?,0,?)",
                (str(lab_id), "state", state_payload, state_artifact_id, now),
            )
            conn.execute(
                "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,revision,updated_at) "
                "VALUES(?,?,?,?,0,?)",
                (str(lab_id), "candidate", candidate_payload, candidate_artifact_id, now),
            )
            epoch_rows = ready["state"].get("epochs")
            if not isinstance(epoch_rows, list) or not epoch_rows:
                epoch_rows = [{"id": ready["epoch_id"], "reason": "initial confirmed contract"}]
            found_active = False
            for epoch in epoch_rows:
                if not isinstance(epoch, dict):
                    continue
                epoch_id = int(epoch.get("id") or ready["epoch_id"])
                status = "superseded" if epoch.get("superseded_at") else "active"
                found_active = found_active or (epoch_id == ready["epoch_id"] and status == "active")
                conn.execute(
                    "INSERT INTO epochs(lab_id,epoch_id,status,created_at,superseded_at,reason,contract_hash,payload_json) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (str(lab_id), epoch_id, status, str(epoch.get("created_at") or now),
                     epoch.get("superseded_at"), epoch.get("reason"), epoch.get("contract_hash"), _json(epoch)),
                )
            if not found_active:
                conn.execute(
                    "INSERT INTO epochs(lab_id,epoch_id,status,created_at,reason,payload_json) VALUES(?,?,'active',?,?,?)",
                    (str(lab_id), ready["epoch_id"], now, "initial confirmed contract", _json({"id": ready["epoch_id"]})),
                )
            self._insert_metric_definitions(conn, str(lab_id), ready["epoch_id"], ready["metrics"])
            for name, payload, digest, source_path in ready["pins"]:
                conn.execute(
                    "INSERT INTO pins(lab_id,name,sha256,source_path,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                    (str(lab_id), name, digest, source_path, _json(payload), now),
                )
            self._bump_revision(conn, str(lab_id))
        return lab_id

    @staticmethod
    def _insert_metric_definitions(
        conn: sqlite3.Connection,
        lab_id: str,
        epoch_id: int,
        metrics: Mapping[str, Any],
    ) -> None:
        names: dict[str, tuple[str, str | None, str | None, dict[str, Any]]] = {}
        primary = metrics.get("primary")
        if isinstance(primary, str) and primary:
            names[primary] = ("primary", metrics.get("direction", "max"), metrics.get("unit"), {})
        guards = metrics.get("guards", [])
        if isinstance(guards, list):
            for guard in guards:
                if isinstance(guard, str):
                    name, direction, unit, detail = guard, None, None, {}
                elif isinstance(guard, dict):
                    name = guard.get("name") or guard.get("metric")
                    direction = guard.get("direction")
                    unit = guard.get("unit")
                    detail = guard
                else:
                    continue
                if isinstance(name, str) and name:
                    names[name] = ("guard", direction if direction in {"min", "max"} else None, unit, detail)
        for name, (role, direction, unit, detail) in names.items():
            conn.execute(
                "INSERT OR IGNORE INTO metric_definitions(lab_id,epoch_id,metric_name,role,direction,unit,payload_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (lab_id, epoch_id, name, role, direction, unit, _json(detail)),
            )

    def _sync_state_epochs_in_conn(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        config: Mapping[str, Any],
        state: Mapping[str, Any],
        source_bytes: bytes,
        previous_config_sha256: str | None,
        now: str,
    ) -> bool:
        """Append one verified epoch; existing relational contract rows are immutable."""
        epochs = state.get("epochs")
        current_id = state.get("current_epoch")
        if (not isinstance(epochs, list) or not epochs or isinstance(current_id, bool)
                or not isinstance(current_id, int) or current_id < 1):
            raise StoreIntegrityError("Epoch save requires a non-empty epoch history and positive current_epoch")
        digest = _sha256(source_bytes)
        try:
            source_config = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreIntegrityError("Epoch config source bytes are not valid UTF-8 JSON") from exc
        if source_config != config:
            raise StoreIntegrityError("Epoch config projection does not match its exact source bytes")
        if state.get("config_sha256") != digest:
            raise StoreIntegrityError("Epoch state config_sha256 does not match exact source bytes")

        by_id: dict[int, dict[str, Any]] = {}
        for epoch in epochs:
            if not isinstance(epoch, dict):
                raise StoreIntegrityError("Epoch history contains a non-object record")
            epoch_id = epoch.get("id")
            if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 1:
                raise StoreIntegrityError("Epoch history contains an invalid epoch id")
            if epoch_id in by_id:
                raise StoreIntegrityError(f"Epoch history repeats epoch id {epoch_id}")
            by_id[epoch_id] = epoch
        if set(by_id) != set(range(1, current_id + 1)):
            raise StoreIntegrityError("Epoch history must be contiguous from 1 through current_epoch")
        current = by_id[current_id]
        if current.get("config_sha256") != digest:
            raise StoreIntegrityError("Current epoch config hash does not match exact source bytes")
        if current.get("superseded_at"):
            raise StoreIntegrityError("State current_epoch is marked superseded")

        existing = [int(row[0]) for row in conn.execute(
            "SELECT epoch_id FROM epochs WHERE lab_id=? ORDER BY epoch_id", (lab_id,),
        )]
        if not existing or existing != list(range(1, existing[-1] + 1)):
            raise StoreIntegrityError("Stored epoch history is missing or non-contiguous")
        if existing[-1] >= current_id:
            if existing[-1] != current_id:
                raise StoreIntegrityError("State current_epoch is behind stored immutable epochs")
            # Re-saving the current contract/state is allowed, but cannot change its source config.
            if previous_config_sha256 != digest:
                raise StoreIntegrityError("Changing config requires opening a new immutable epoch")
            return False
        if current_id != existing[-1] + 1:
            raise StoreIntegrityError("Only the next sequential epoch may be appended")
        if current.get("previous_epoch_id") != existing[-1]:
            raise StoreIntegrityError("New epoch must name the latest stored epoch as its predecessor")
        if not by_id[existing[-1]].get("superseded_at"):
            raise StoreIntegrityError("Previous epoch must be superseded before appending a new epoch")
        metrics = config.get("metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("primary"), str) or not metrics["primary"]:
            raise StoreIntegrityError("New epoch config requires a primary metric definition")

        # Store the validated contract details with this immutable epoch. Existing
        # payloads and pins remain untouched; the exact raw config stays in projections.
        snapshot = dict(current)
        snapshot["metrics"] = metrics
        snapshot["goal"] = config.get("goal") or (
            config.get("_passport", {}).get("goal") if isinstance(config.get("_passport"), dict) else None
        )
        snapshot["evaluator"] = config.get("evaluator")
        snapshot["repeats"] = config.get("repeats")
        snapshot["config_sha256"] = digest
        lab_goal = conn.execute("SELECT goal FROM labs WHERE lab_id=?", (lab_id,)).fetchone()
        pin_rows = conn.execute(
            "SELECT name,sha256,source_path,payload_json,created_at FROM pins WHERE lab_id=? ORDER BY name", (lab_id,),
        ).fetchall()
        snapshot["pins"] = [
            {"name": row["name"], "sha256": row["sha256"], "source_path": row["source_path"],
             "payload": _loads(row["payload_json"], {}), "created_at": row["created_at"]}
            for row in pin_rows
        ]
        snapshot["goal"] = snapshot["goal"] or (lab_goal[0] if lab_goal else None)
        superseded_at = current.get("superseded_at")
        conn.execute(
            "INSERT INTO epochs(lab_id,epoch_id,status,created_at,superseded_at,reason,contract_hash,payload_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (lab_id, current_id, "superseded" if superseded_at else "active",
             str(current.get("started_at") or current.get("created_at") or now), superseded_at,
             current.get("supersede_reason") or current.get("reason"), digest, _json(snapshot)),
        )
        self._insert_metric_definitions(conn, lab_id, current_id, metrics)
        definitions = conn.execute(
            "SELECT metric_name,role FROM metric_definitions WHERE lab_id=? AND epoch_id=?",
            (lab_id, current_id),
        ).fetchall()
        if not any(row["role"] == "primary" for row in definitions):
            raise StoreIntegrityError("New epoch did not produce a primary metric definition")
        return True

    @staticmethod
    def _metric_definition(conn: sqlite3.Connection, lab_id: str, epoch_id: int, name: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO metric_definitions(lab_id,epoch_id,metric_name,role,direction,unit,payload_json) "
            "VALUES(?,?,?,'additional',NULL,NULL,?)",
            (lab_id, epoch_id, name, _json({"inferred": True})),
        )

    def _lab_text(self, lab_id: uuid.UUID | str) -> str:
        return _uuid_text(lab_id, label="lab_id")

    def resolve_lab_id(self, ref: uuid.UUID | str | Path) -> uuid.UUID:
        if isinstance(ref, uuid.UUID):
            lab_text = str(ref)
            with self._reader() as conn:
                row = conn.execute("SELECT lab_id FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            return uuid.UUID(lab_text)
        value = str(ref)
        try:
            canonical = _uuid_text(value, label="lab reference")
        except ValueError:
            canonical = ""
        if canonical:
            with self._reader() as conn:
                row = conn.execute("SELECT lab_id FROM labs WHERE lab_id=?", (canonical,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Lab not found: {canonical}")
            return uuid.UUID(canonical)
        is_path = isinstance(ref, Path) or Path(value).is_absolute() or value.endswith("lab.json")
        alias = _path_alias(value) if is_path else value
        with self._reader() as conn:
            if is_path:
                rows = conn.execute(
                    "SELECT DISTINCT lab_id FROM aliases WHERE kind='lab' AND scope_id=? AND alias=?",
                    (_NIL_UUID, alias),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT lab_id FROM aliases WHERE kind='lab' AND alias=?",
                    (alias,),
                ).fetchall()
        if not rows:
            raise EntityNotFoundError(f"Lab not found for reference: {value}")
        if len(rows) != 1:
            raise AmbiguousAliasError(f"Lab alias is ambiguous: {value}")
        return uuid.UUID(str(rows[0][0]))

    def resolve_id(
        self,
        kind: str,
        ref: uuid.UUID | str | Path,
        root_ref: uuid.UUID | str | Path | None = None,
    ) -> uuid.UUID:
        """Resolve one canonical identifier; legacy aliases must be unique in scope."""
        if kind == "lab":
            if root_ref is None:
                return self.resolve_lab_id(ref)
            root_text = str(root_ref)
            alias = _path_alias(ref) if isinstance(ref, Path) or Path(str(ref)).is_absolute() else str(ref)
            try:
                canonical = _uuid_text(ref, label="lab reference")
            except ValueError:
                canonical = ""
            with self._reader() as conn:
                if canonical:
                    row = conn.execute("SELECT lab_id FROM labs WHERE lab_id=?", (canonical,)).fetchone()
                    if row:
                        return uuid.UUID(canonical)
                rows = conn.execute(
                    "SELECT DISTINCT lab_id FROM aliases WHERE kind='lab' AND alias=? AND scope_id=?",
                    (alias, root_text),
                ).fetchall()
                if not rows:
                    rows = conn.execute(
                        "SELECT DISTINCT lab_id FROM aliases WHERE kind='lab' AND alias=? AND scope_id=?",
                        (alias, _NIL_UUID),
                    ).fetchall()
            if not rows:
                raise EntityNotFoundError(f"Lab not found for reference: {ref}")
            if len(rows) > 1:
                raise AmbiguousAliasError(f"Lab alias is ambiguous in root {root_text}: {ref}")
            return uuid.UUID(rows[0][0])
        if kind not in {"run", "operation"}:
            raise ValueError("kind must be 'lab', 'run', or 'operation'")
        if root_ref is None:
            try:
                canonical = _uuid_text(ref, label=f"{kind} reference")
            except ValueError:
                raise ValueError(f"root_ref is required to resolve a {kind} alias")
            with self._reader() as conn:
                table, column = ("runs", "run_id") if kind == "run" else ("operations", "operation_id")
                row = conn.execute(f"SELECT {column} FROM {table} WHERE {column}=?", (canonical,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"{kind.title()} not found: {canonical}")
            return uuid.UUID(canonical)
        lab_id = self.resolve_lab_id(root_ref)
        if kind == "run":
            return self.resolve_run_id(lab_id, ref)
        try:
            operation_text = _uuid_text(ref, label="operation reference")
        except ValueError:
            operation_text = ""
        with self._reader() as conn:
            if operation_text:
                row = conn.execute("SELECT operation_id FROM operations WHERE lab_id=? AND operation_id=?",
                                   (str(lab_id), operation_text)).fetchone()
                if row:
                    return uuid.UUID(operation_text)
            rows = conn.execute("SELECT operation_id FROM aliases WHERE kind='operation' AND scope_id=? AND alias=?",
                                (str(lab_id), str(ref))).fetchall()
        if not rows:
            raise EntityNotFoundError(f"Operation not found in lab {lab_id}: {ref}")
        if len(rows) > 1:
            raise AmbiguousAliasError(f"Operation alias is ambiguous in lab {lab_id}: {ref}")
        return uuid.UUID(rows[0][0])

    def get_lab(self, lab_id: uuid.UUID | str | Path) -> dict[str, Any]:
        resolved = self.resolve_lab_id(lab_id)
        with self._reader() as conn:
            row = conn.execute("SELECT * FROM labs WHERE lab_id=?", (str(resolved),)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Lab not found: {resolved}")
            summary = {
                "lab_id": resolved,
                "state_dir": row["state_dir"],
                "config_path": row["config_path"],
                "source_root": row["source_root"],
                "candidate_path": row["candidate_path"],
                "candidate_commit": row["candidate_commit"],
                "goal": row["goal"],
                "primary_metric": row["primary_metric"],
                "direction": row["direction"],
                "status": row["status"],
                "lifecycle": row["lifecycle"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "revision": row["revision"],
                "snapshot": _loads(row["snapshot_json"], {}),
            }
        summary["config"] = self.load_projection(resolved, "config")
        summary["state"] = self.load_projection(resolved, "state")
        return summary

    def list_labs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        filters: Mapping[str, Any] | None = None,
        lifecycle: str | None = None,
        include_dashboard_summary: bool = False,
    ) -> dict[str, Any]:
        limit, offset = _safe_page(limit, offset)
        filter_lifecycle = (filters or {}).get("lifecycle")
        if lifecycle is not None and filter_lifecycle is not None and lifecycle != filter_lifecycle:
            raise ValueError("Conflicting lifecycle filters")
        selected_lifecycle = lifecycle if lifecycle is not None else filter_lifecycle
        if selected_lifecycle is not None and selected_lifecycle not in {"active", "archived"}:
            raise ValueError("lifecycle must be 'active' or 'archived'")
        if not self.is_initialized():
            return {"items": [], "next_cursor": None, "total": 0, "limit": limit, "offset": offset}
        clauses: list[str] = []
        params: list[Any] = []
        allowed = {"status", "source_root", "primary_metric", "lifecycle"}
        for key, value in (filters or {}).items():
            if key not in allowed or value is None:
                continue
            clauses.append(f"{key}=?")
            params.append(_path_alias(value) if key == "source_root" else value)
        if lifecycle is not None and filter_lifecycle is None:
            clauses.append("lifecycle=?")
            params.append(lifecycle)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._reader() as conn:
            total = int(conn.execute("SELECT count(*) FROM labs" + where, params).fetchone()[0])
            rows = conn.execute(
                "SELECT lab_id,state_dir,config_path,source_root,goal,primary_metric,direction,status,lifecycle,"
                "created_at,updated_at,revision FROM labs" + where +
                " ORDER BY created_at DESC,lab_id DESC LIMIT ? OFFSET ?",
                (*params, limit + 1, offset),
            ).fetchall()
            items = [dict(row) for row in rows[:limit]]
            if include_dashboard_summary and items:
                summaries = self._dashboard_summaries(conn, items)
                for item in items:
                    item["dashboard_summary"] = summaries[str(item["lab_id"])]
        has_more = len(rows) > limit
        for item in items:
            item["lab_id"] = uuid.UUID(item["lab_id"])
        return {
            "items": items,
            "next_cursor": str(items[-1]["lab_id"]) if has_more and items else None,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _dashboard_summaries(conn: sqlite3.Connection, items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """Build small overview summaries with a fixed number of queries per page."""
        if not items:
            return {}
        lab_ids = [str(item["lab_id"]) for item in items]
        lab_marks = ",".join("?" for _ in lab_ids)
        current_rows = conn.execute(
            f"SELECT lab_id,MAX(epoch_id) AS epoch_id FROM epochs WHERE status='active' "
            f"AND lab_id IN ({lab_marks}) GROUP BY lab_id", lab_ids,
        ).fetchall()
        current_epochs = {str(row["lab_id"]): int(row["epoch_id"]) for row in current_rows}
        summaries = {
            str(item["lab_id"]): {
                "current_epoch": current_epochs.get(str(item["lab_id"])),
                "metric_definitions": {
                    "primary": {"name": item.get("primary_metric"), "direction": item.get("direction"), "unit": None},
                    "guards": [],
                },
                "baseline_id": None, "best_id": None, "latest_id": None,
                "baseline": None, "best": None, "latest": None,
                "trend": [], "active_run": None, "active_operation": None,
                "statistic": "worst",
                "goal_status": "unknown",
                "cost_usd": None,
                "confirmed_cost_usd": 0.0,
                "signals": [],
            }
            for item in items
        }

        config_rows = conn.execute(
            f"SELECT lab_id,payload_json FROM projections WHERE view='config' AND lab_id IN ({lab_marks})",
            lab_ids,
        ).fetchall()
        goal_criteria_by_lab: dict[str, dict[str, Any] | None] = {}
        for row in config_rows:
            lab_id_str = str(row["lab_id"])
            cfg = _loads(row["payload_json"], {})
            if isinstance(cfg, dict):
                acceptance = cfg.get("acceptance") if isinstance(cfg.get("acceptance"), dict) else {}
                stat = acceptance.get("statistic")
                summaries[lab_id_str]["statistic"] = stat if stat in {"worst", "mean", "median"} else "worst"
                goal_criteria_by_lab[lab_id_str] = cfg.get("goal_criteria") if isinstance(cfg.get("goal_criteria"), dict) else None

        if current_epochs:
            current_marks = ",".join("(?,?)" for _ in current_epochs)
            current_params = [part for pair in current_epochs.items() for part in pair]
            definitions = conn.execute(
                "WITH current(lab_id,epoch_id) AS (VALUES " + current_marks + "), "
                "ranked AS (SELECT d.lab_id,d.metric_name,d.role,d.direction,d.unit,"
                "ROW_NUMBER() OVER (PARTITION BY d.lab_id ORDER BY "
                "CASE d.role WHEN 'primary' THEN 0 ELSE 1 END,d.metric_name) AS position "
                "FROM metric_definitions d JOIN current c ON c.lab_id=d.lab_id AND c.epoch_id=d.epoch_id "
                "WHERE d.role IN ('primary','guard')) "
                "SELECT lab_id,metric_name,role,direction,unit FROM ranked WHERE position<=64 "
                "ORDER BY lab_id,position",
                current_params,
            ).fetchall()
            for row in definitions:
                target = summaries[str(row["lab_id"])]["metric_definitions"]
                definition = {"name": str(row["metric_name"]), "direction": row["direction"], "unit": row["unit"]}
                if row["role"] == "primary":
                    target["primary"] = definition
                else:
                    target["guards"].append(definition)

        state_rows = conn.execute(
            f"SELECT lab_id,payload_json FROM projections WHERE view='state' AND lab_id IN ({lab_marks})",
            lab_ids,
        ).fetchall()
        state_by_lab: dict[str, dict[str, Any]] = {}
        for row in state_rows:
            value = _loads(row["payload_json"], {})
            if isinstance(value, dict):
                state_by_lab[str(row["lab_id"])] = value
        for lab_id, state in state_by_lab.items():
            if summaries[lab_id]["current_epoch"] is None:
                epoch = state.get("current_epoch")
                if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0:
                    summaries[lab_id]["current_epoch"] = epoch

        current_cte = "WITH current(lab_id,epoch_id) AS (VALUES " + ",".join("(?,?)" for _ in current_epochs) + ") "
        current_params = [part for pair in current_epochs.items() for part in pair]
        if current_epochs:
            recent_rows = conn.execute(
                current_cte +
                ", ranked AS (SELECT r.lab_id,r.run_id,r.alias,r.kind,r.status,r.parent_run_id,"
                "parent.alias AS parent_alias,r.created_at,r.updated_at,"
                "ROW_NUMBER() OVER (PARTITION BY r.lab_id ORDER BY r.created_at DESC,r.run_id DESC) AS position "
                "FROM runs r JOIN current c ON c.lab_id=r.lab_id AND c.epoch_id=r.epoch_id "
                "LEFT JOIN runs parent ON parent.lab_id=r.lab_id AND parent.run_id=r.parent_run_id) "
                "SELECT lab_id,run_id,alias,kind,status,parent_run_id,parent_alias,created_at,updated_at "
                "FROM ranked WHERE position<=5 ORDER BY lab_id,created_at,run_id",
                current_params,
            ).fetchall()
            baseline_rows = conn.execute(
                current_cte +
                ", ranked AS (SELECT r.lab_id,r.run_id,r.alias,r.kind,r.status,r.parent_run_id,"
                "parent.alias AS parent_alias,r.created_at,r.updated_at,"
                "ROW_NUMBER() OVER (PARTITION BY r.lab_id ORDER BY r.created_at DESC,r.run_id DESC) AS position "
                "FROM runs r JOIN current c ON c.lab_id=r.lab_id AND c.epoch_id=r.epoch_id "
                "LEFT JOIN runs parent ON parent.lab_id=r.lab_id AND parent.run_id=r.parent_run_id "
                "WHERE r.kind='baseline') "
                "SELECT lab_id,run_id,alias,kind,status,parent_run_id,parent_alias,created_at,updated_at "
                "FROM ranked WHERE position=1",
                current_params,
            ).fetchall()
        else:
            recent_rows = []
            baseline_rows = []

        def run_view(row: sqlite3.Row) -> dict[str, Any]:
            return {
                "run_id": str(row["run_id"]), "alias": str(row["alias"]), "kind": str(row["kind"]),
                "status": str(row["status"]),
                "parent_run_id": str(row["parent_run_id"]) if row["parent_run_id"] else None,
                "parent_alias": row["parent_alias"], "created_at": row["created_at"],
                "time": row["created_at"], "updated_at": row["updated_at"],
                "metric_means": {}, "metrics": {},
            }

        runs_by_lab: dict[str, list[dict[str, Any]]] = {lab_id: [] for lab_id in lab_ids}
        all_runs: dict[tuple[str, str], dict[str, Any]] = {}
        for row in recent_rows:
            lab_id = str(row["lab_id"])
            view = run_view(row)
            runs_by_lab[lab_id].append(view)
            all_runs[(lab_id, view["run_id"])] = view
        for row in baseline_rows:
            lab_id = str(row["lab_id"])
            view = all_runs.setdefault((lab_id, str(row["run_id"])), run_view(row))
            summaries[lab_id]["baseline"] = view

        refs: list[tuple[str, int, str, str]] = []
        for lab_id, state in state_by_lab.items():
            epoch_id = current_epochs.get(lab_id)
            if epoch_id is None:
                continue
            for slot in ("baseline", "best"):
                ref = state.get(f"{slot}_id")
                if isinstance(ref, str) and ref.strip():
                    refs.append((lab_id, epoch_id, slot, ref.strip()))
        if refs:
            ref_values = ",".join("(?,?,?,?)" for _ in refs)
            ref_params = [part for entry in refs for part in entry]
            selected_rows = conn.execute(
                "WITH refs(lab_id,epoch_id,slot,ref) AS (VALUES " + ref_values + ") "
                "SELECT refs.lab_id,refs.slot,r.run_id,r.alias,r.kind,r.status,r.parent_run_id,"
                "parent.alias AS parent_alias,r.created_at,r.updated_at "
                "FROM refs JOIN runs r ON r.lab_id=refs.lab_id AND r.epoch_id=refs.epoch_id "
                "AND (r.run_id=refs.ref OR r.alias=refs.ref) "
                "LEFT JOIN runs parent ON parent.lab_id=r.lab_id AND parent.run_id=r.parent_run_id",
                ref_params,
            ).fetchall()
            for row in selected_rows:
                lab_id = str(row["lab_id"])
                view = all_runs.setdefault((lab_id, str(row["run_id"])), run_view(row))
                summaries[lab_id][str(row["slot"])] = view

        for lab_id, summary in summaries.items():
            trend = runs_by_lab[lab_id]
            summary["trend"] = trend
            summary["latest"] = trend[-1] if trend else None
            if summary["baseline"] is None and trend:
                summary["baseline"] = next((run for run in trend if run["kind"] == "baseline"), None)
            if summary["best"] is None:
                summary["best"] = summary["baseline"] or summary["latest"]
            for slot in ("baseline", "best", "latest"):
                record = summary[slot]
                summary[f"{slot}_id"] = record["run_id"] if record else None

        selected_run_ids = sorted({run_id for _lab_id, run_id in all_runs})
        if selected_run_ids:
            means_rows = conn.execute(
                "SELECT m.lab_id,m.run_id,m.metric_name,AVG(m.value) AS metric_mean,"
                "MIN(m.value) AS min_val,MAX(m.value) AS max_val,COUNT(*) AS completed_repeats "
                "FROM metric_values m JOIN repeats rep ON rep.lab_id=m.lab_id AND rep.run_id=m.run_id "
                "AND rep.repeat_index=m.repeat_index "
                f"WHERE m.run_id IN ({','.join('?' for _ in selected_run_ids)}) "
                "AND rep.status NOT IN ('running','interrupted','quarantined') "
                "GROUP BY m.lab_id,m.run_id,m.metric_name",
                selected_run_ids,
            ).fetchall()
            for row in means_rows:
                lab_id = str(row["lab_id"])
                view = all_runs.get((lab_id, str(row["run_id"])))
                if view is not None:
                    metric_name = str(row["metric_name"])
                    mean_val = float(row["metric_mean"])
                    view["metric_means"][metric_name] = mean_val
                    defs = summaries.get(lab_id, {}).get("metric_definitions", {})
                    primary = defs.get("primary") or {}
                    if primary.get("name") == metric_name:
                        direction = primary.get("direction") or "max"
                    else:
                        direction = "max"
                        for g in defs.get("guards", []):
                            if g.get("name") == metric_name:
                                direction = g.get("direction") or "max"
                                break
                    decision_worst = float(row["min_val"] if direction == "max" else row["max_val"])
                    view.setdefault("metrics", {})[metric_name] = {
                        "decision_worst": decision_worst,
                        "mean": mean_val,
                        "completed_repeats": int(row["completed_repeats"]),
                    }

        for lab_id, summary in summaries.items():
            goal_criteria = goal_criteria_by_lab.get(lab_id)
            thresholds = goal_criteria.get("thresholds") if isinstance(goal_criteria, dict) else None
            if isinstance(thresholds, dict) and thresholds:
                best = summary.get("best")
                best_metrics = best.get("metrics") if isinstance(best, dict) else None
                if not best or not best_metrics:
                    summary["goal_status"] = "unconfirmed"
                else:
                    conf_repeats = int(goal_criteria.get("confirmation_repeats") or 1)
                    all_passed = True
                    any_failed = False
                    all_repeats_met = True
                    stat = summary.get("statistic", "worst")
                    for metric_name, bounds in thresholds.items():
                        if not isinstance(bounds, dict):
                            continue
                        if metric_name not in best_metrics:
                            all_passed = False
                            continue
                        m_info = best_metrics[metric_name]
                        val = m_info.get("mean") if stat == "mean" else m_info.get("decision_worst")
                        if val is None:
                            val = m_info.get("mean") if m_info.get("mean") is not None else m_info.get("decision_worst")
                        if val is None:
                            all_passed = False
                            continue
                        failed_this = False
                        if "min" in bounds and bounds["min"] is not None and not isinstance(bounds["min"], bool):
                            if val < float(bounds["min"]):
                                failed_this = True
                        if "max" in bounds and bounds["max"] is not None and not isinstance(bounds["max"], bool):
                            if val > float(bounds["max"]):
                                failed_this = True
                        if failed_this:
                            any_failed = True
                            break
                        repeats = int(m_info.get("completed_repeats") or 0)
                        if repeats < conf_repeats:
                            all_repeats_met = False
                    if any_failed:
                        summary["goal_status"] = "not_met"
                    elif all_passed and all_repeats_met:
                        summary["goal_status"] = "met"
                    else:
                        summary["goal_status"] = "unconfirmed"

            trend = summary.get("trend") or []
            rejected_streak = 0
            for item in reversed(trend):
                if item.get("kind") == "baseline":
                    continue
                if item.get("status") == "rejected":
                    rejected_streak += 1
                else:
                    break
            if rejected_streak >= 2:
                action = (
                    "Consult the openresearch skill (`orx discover` / `orx paper`) for new algorithmic framing before the next trial."
                    if rejected_streak >= 3
                    else "Change the mechanism or narrow the hypothesis scope."
                )
                summary["signals"].append({
                    "kind": "repeated_rejection",
                    "message": f"{rejected_streak} consecutive rejected trials in current trend",
                    "action": action,
                })

        cost_rows = conn.execute(
            f"SELECT lab_id, SUM(cost_usd) AS total_cost FROM attempts "
            f"WHERE lab_id IN ({lab_marks}) AND cost_usd IS NOT NULL GROUP BY lab_id",
            lab_ids,
        ).fetchall()
        for row in cost_rows:
            lab_id = str(row["lab_id"])
            if lab_id in summaries:
                summaries[lab_id]["confirmed_cost_usd"] = float(row["total_cost"] or 0.0)

        active_runs = conn.execute(
            f"SELECT r.lab_id,r.run_id,r.alias,r.kind,r.status,r.parent_run_id,parent.alias AS parent_alias,"
            f"r.created_at,r.updated_at FROM runs r LEFT JOIN runs parent "
            f"ON parent.lab_id=r.lab_id AND parent.run_id=r.parent_run_id "
            f"WHERE r.lab_id IN ({lab_marks}) AND r.status IN ('queued','running','finalizing') "
            "ORDER BY r.lab_id,r.updated_at DESC,r.run_id DESC",
            lab_ids,
        ).fetchall()
        active_by_lab: dict[str, dict[str, Any]] = {}
        for row in active_runs:
            lab_id = str(row["lab_id"])
            active_by_lab.setdefault(lab_id, run_view(row))

        operation_rows = conn.execute(
            f"SELECT lab_id,status,progress_json FROM operations WHERE lab_id IN ({lab_marks}) "
            "AND status IN ('queued','running','finalizing','recovery_required')",
            lab_ids,
        ).fetchall()
        for row in operation_rows:
            lab_id = str(row["lab_id"])
            progress = _loads(row["progress_json"], {}) if row["progress_json"] else {}
            safe_progress: dict[str, Any] = {}
            if isinstance(progress, dict):
                phase = progress.get("phase")
                if (isinstance(phase, str) and 0 < len(phase) <= 64
                        and all(char.isalnum() or char in "_.-" for char in phase)):
                    safe_progress["phase"] = phase
                for key in ("completed", "total"):
                    value = progress.get(key)
                    if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(float(value)) and 0 <= value <= 1_000_000_000_000):
                        safe_progress[key] = value
                if isinstance(progress.get("cancel_requested"), bool):
                    safe_progress["cancel_requested"] = progress["cancel_requested"]
                pending_reason = progress.get("cancel_pending_reason")
                if pending_reason in {
                    "worker_identity_unavailable", "worker_identity_mismatch",
                    "worker_tree_unavailable", "worker_exit_unconfirmed",
                }:
                    safe_progress["cancel_pending_reason"] = pending_reason
            progress_view = safe_progress or None
            summaries[lab_id]["active_operation"] = {"status": str(row["status"]), "progress": progress_view}
            active = active_by_lab.get(lab_id)
            if active is not None:
                active["progress"] = progress_view
                active["operation_status"] = str(row["status"])

        for lab_id, active in active_by_lab.items():
            active.setdefault("progress", None)
            summaries[lab_id]["active_run"] = active
        return summaries

    def set_lab_lifecycle(
        self,
        lab_id: uuid.UUID | str,
        action: str,
        *,
        confirm_lab_id: str | None = None,
    ) -> dict[str, Any]:
        """Archive, restore, or remove one SQLite lab without touching project files."""
        lab_text = self._lab_text(lab_id)
        if action not in {"archive", "restore", "delete"}:
            raise ValueError("action must be 'archive', 'restore', or 'delete'")
        if action == "delete" and confirm_lab_id != lab_text:
            raise ValueError("confirm_lab_id must exactly match the lab UUID for delete")
        if confirm_lab_id is not None and confirm_lab_id != lab_text:
            raise ValueError("confirm_lab_id must match the lab UUID")

        object_paths: dict[str, str] = {}
        with self._writer() as conn:
            lab = conn.execute("SELECT lifecycle FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
            if lab is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            active = conn.execute(
                "SELECT 1 FROM operations WHERE lab_id=? AND status IN "
                "('queued','running','finalizing','recovery_required') LIMIT 1",
                (lab_text,),
            ).fetchone()
            if active is not None:
                raise StoreConflictError("Lab lifecycle cannot change while an operation is active or needs recovery")

            if action in {"archive", "restore"}:
                lifecycle = "archived" if action == "archive" else "active"
                changed = lab["lifecycle"] != lifecycle
                if changed:
                    conn.execute("UPDATE labs SET lifecycle=? WHERE lab_id=?", (lifecycle, lab_text))
                    self._bump_revision(conn, lab_text)
                revision = int(conn.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])
                return {"lab_id": lab_text, "lifecycle": lifecycle, "deleted": False,
                        "changed": changed, "revision": revision}

            artifacts = conn.execute(
                "SELECT object_path,sha256 FROM artifacts WHERE lab_id=? AND object_path IS NOT NULL",
                (lab_text,),
            ).fetchall()
            object_paths = {str(row["object_path"]): str(row["sha256"]) for row in artifacts}
            conn.execute("PRAGMA defer_foreign_keys=ON")
            self._bump_revision(conn, lab_text)
            table_names = [str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]
            for table_name in table_names:
                if table_name in {"labs", "catalog_metadata"}:
                    continue
                quoted = '"' + table_name.replace('"', '""') + '"'
                columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({quoted})").fetchall()}
                if "lab_id" in columns:
                    conn.execute(f"DELETE FROM {quoted} WHERE lab_id=?", (lab_text,))
            conn.execute("DELETE FROM labs WHERE lab_id=?", (lab_text,))
            revision = int(conn.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

        self._remove_unreferenced_objects(object_paths)
        return {"lab_id": lab_text, "lifecycle": "deleted", "deleted": True,
                "changed": True, "revision": revision}

    def _remove_unreferenced_objects(self, object_paths: Mapping[str, str]) -> None:
        """Collect only content-addressed files no remaining artifact row references."""
        candidates = list(object_paths.items())
        for start in range(0, len(candidates), 400):
            batch = candidates[start:start + 400]
            paths = [path for path, _digest in batch]
            marks = ",".join("?" for _ in paths)
            # Serialize the final reference check and unlink with every publisher.
            with self._writer() as conn:
                references = {str(row[0]) for row in conn.execute(
                    f"SELECT DISTINCT object_path FROM artifacts WHERE object_path IN ({marks})", paths,
                ).fetchall()}
                for object_path, digest in batch:
                    if object_path in references or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                        continue
                    relative = Path(object_path)
                    expected = Path("objects") / "sha256" / digest[:2] / digest
                    if relative.is_absolute() or ".." in relative.parts or relative != expected:
                        continue
                    unresolved = self.home / relative
                    probe = self.home
                    unsafe = False
                    for part in relative.parts:
                        probe = probe / part
                        if probe.exists() and _is_link_or_reparse(probe):
                            unsafe = True
                            break
                    if unsafe:
                        continue
                    try:
                        target = unresolved.resolve(strict=True)
                    except OSError:
                        continue
                    if not target.is_relative_to(self._object_root.resolve()) or not target.is_file():
                        continue
                    try:
                        target.unlink()
                    except OSError:
                        # The database deletion is already durable; a locked object can be collected later.
                        continue

    def load_config(self, lab_id: uuid.UUID | str) -> dict[str, Any]:
        return self.load_projection(lab_id, "config")

    def load_config_bytes(self, lab_id: uuid.UUID | str) -> bytes:
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT source_bytes,source_sha256,source_artifact_id FROM projections "
                "WHERE lab_id=? AND view='config'", (lab_text,)
            ).fetchone()
        if row is None or (row["source_bytes"] is None and row["source_artifact_id"] is None):
            raise EntityNotFoundError(f"Config bytes not found for lab: {lab_text}")
        raw = (self.load_artifact(row["source_artifact_id"]) if row["source_artifact_id"]
               else bytes(row["source_bytes"]))
        if _sha256(raw) != row["source_sha256"]:
            raise StoreCorruptError("Stored source config bytes do not match their SHA-256")
        return raw

    def save_config(self, lab_id: uuid.UUID | str, payload: dict[str, Any]) -> None:
        raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self.save_config_bytes(lab_id, payload, raw)

    def _save_config_in_conn(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        payload: dict[str, Any],
        source_bytes: bytes,
        now: str,
    ) -> bool:
        encoded = _json(payload)
        digest = _sha256(source_bytes)
        old = conn.execute(
            "SELECT payload_json,payload_artifact_id,source_bytes,source_sha256,source_artifact_id "
            "FROM projections WHERE lab_id=? AND view='config'", (lab_id,),
        ).fetchone()
        if old:
            old_projection = (self.load_artifact(old["payload_artifact_id"]).decode("utf-8")
                              if old["payload_artifact_id"] else old["payload_json"])
            old_bytes = (self.load_artifact(old["source_artifact_id"])
                         if old["source_artifact_id"] else bytes(old["source_bytes"] or b""))
            if old_projection == encoded and old_bytes == source_bytes:
                return False
        stored_payload, payload_artifact = self._projection_storage(conn, lab_id, "config", encoded)
        source_artifact = None
        inline_source = source_bytes
        if len(source_bytes) > INLINE_ARTIFACT_LIMIT:
            source_artifact = self._insert_large_payload_artifact(conn, lab_id, "source-config.json", source_bytes)
            inline_source = b""
        conn.execute(
            "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,source_bytes,source_sha256,"
            "source_artifact_id,revision,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(lab_id,view) DO UPDATE SET payload_json=excluded.payload_json,"
            "payload_artifact_id=excluded.payload_artifact_id,source_bytes=excluded.source_bytes,"
            "source_sha256=excluded.source_sha256,source_artifact_id=excluded.source_artifact_id,updated_at=excluded.updated_at",
            (lab_id, "config", stored_payload, payload_artifact,
             sqlite3.Binary(inline_source) if source_artifact is None else None, digest, source_artifact, 0, now),
        )
        if old:
            for artifact_id in (old["payload_artifact_id"], old["source_artifact_id"]):
                if artifact_id and artifact_id not in {payload_artifact, source_artifact}:
                    conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,))
        return True

    def _save_state_in_conn(
        self, conn: sqlite3.Connection, lab_id: str, payload: dict[str, Any], now: str,
    ) -> bool:
        encoded = _json(payload)
        old = conn.execute(
            "SELECT payload_json,payload_artifact_id FROM projections WHERE lab_id=? AND view='state'", (lab_id,)
        ).fetchone()
        if old:
            old_payload = (self.load_artifact(old["payload_artifact_id"]).decode("utf-8")
                           if old["payload_artifact_id"] else old["payload_json"])
            if old_payload == encoded:
                return False
        stored_payload, artifact_id = self._projection_storage(conn, lab_id, "state", encoded)
        conn.execute(
            "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,revision,updated_at) "
            "VALUES(?, 'state', ?, ?,0,?) ON CONFLICT(lab_id,view) DO UPDATE SET "
            "payload_json=excluded.payload_json,payload_artifact_id=excluded.payload_artifact_id,updated_at=excluded.updated_at",
            (lab_id, stored_payload, artifact_id, now),
        )
        if old and old["payload_artifact_id"] and old["payload_artifact_id"] != artifact_id:
            conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (old["payload_artifact_id"],))
        return True

    def save_config_bytes(self, lab_id: uuid.UUID | str, config: dict[str, Any], raw_config_bytes: bytes) -> int:
        """Atomically update the config projection and exact source bytes/hash."""
        lab_text = self._lab_text(lab_id)
        if not isinstance(config, dict) or not isinstance(raw_config_bytes, bytes):
            raise TypeError("config must be a dictionary and raw_config_bytes must be bytes")
        now = _now()
        with self._writer() as conn:
            if conn.execute("SELECT 1 FROM labs WHERE lab_id=?", (lab_text,)).fetchone() is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            changed = self._save_config_in_conn(conn, lab_text, config, raw_config_bytes, now)
            if changed:
                revision = self._bump_revision(conn, lab_text)
                conn.execute("UPDATE projections SET revision=? WHERE lab_id=? AND view='config'",
                             (revision, lab_text))
            else:
                revision = int(conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()[0])
            return revision

    def save_config_and_state(
        self,
        lab_id: uuid.UUID | str,
        config: dict[str, Any],
        state: dict[str, Any],
        *,
        raw_config_bytes: bytes | None = None,
    ) -> int:
        """Commit an epoch/config and its reset state in one SQLite transaction."""
        lab_text = self._lab_text(lab_id)
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise TypeError("config and state must be dictionaries")
        source_bytes = raw_config_bytes
        if source_bytes is None:
            source_bytes = (json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if not isinstance(source_bytes, bytes):
            raise TypeError("raw_config_bytes must be bytes")
        now = _now()
        with self._writer() as conn:
            if conn.execute("SELECT 1 FROM labs WHERE lab_id=?", (lab_text,)).fetchone() is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            prior_config = conn.execute(
                "SELECT source_sha256 FROM projections WHERE lab_id=? AND view='config'", (lab_text,),
            ).fetchone()
            previous_config_sha256 = prior_config[0] if prior_config else None
            changed = self._save_config_in_conn(conn, lab_text, config, source_bytes, now)
            changed = self._save_state_in_conn(conn, lab_text, state, now) or changed
            changed = self._sync_state_epochs_in_conn(
                conn, lab_text, config, state, source_bytes, previous_config_sha256, now,
            ) or changed
            if changed:
                revision = self._bump_revision(conn, lab_text)
                conn.execute("UPDATE projections SET revision=? WHERE lab_id=? AND view IN ('config','state')",
                             (revision, lab_text))
            else:
                revision = int(conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()[0])
            return revision

    def load_state(self, lab_id: uuid.UUID | str) -> dict[str, Any]:
        return self.load_projection(lab_id, "state")

    def save_state(self, lab_id: uuid.UUID | str, payload: dict[str, Any], expected_revision: int | None = None) -> int:
        return self.save_projection(lab_id, "state", payload, expected_revision=expected_revision)

    def load_projection(self, lab_id: uuid.UUID | str, view: str) -> dict[str, Any]:
        if view not in {"config", "state", "candidate"}:
            raise ValueError(f"Unsupported lab projection: {view}")
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT payload_json,payload_artifact_id FROM projections WHERE lab_id=? AND view=?",
                (lab_text, view)
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"Projection not found: {view} for lab {lab_text}")
        raw = self.load_artifact(row["payload_artifact_id"]) if row["payload_artifact_id"] else row["payload_json"]
        value = _loads(raw)
        if not isinstance(value, dict):
            raise StoreSchemaError(f"Projection {view} must be a JSON object")
        return value

    def save_projection(
        self,
        lab_id: uuid.UUID | str,
        view: str,
        payload: dict[str, Any],
        expected_revision: int | None = None,
    ) -> int:
        if view not in {"config", "state", "candidate"}:
            raise ValueError(f"Unsupported lab projection: {view}")
        if not isinstance(payload, dict):
            raise TypeError("projection payload must be a dictionary")
        lab_text = self._lab_text(lab_id)
        encoded = _json(payload)
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            if expected_revision is not None and int(row[0]) != expected_revision:
                raise RevisionConflictError(f"Expected revision {expected_revision}, found {row[0]}")
            old = conn.execute(
                "SELECT payload_json,payload_artifact_id FROM projections WHERE lab_id=? AND view=?",
                (lab_text, view)
            ).fetchone()
            if old:
                old_payload = (self.load_artifact(old["payload_artifact_id"]).decode("utf-8")
                               if old["payload_artifact_id"] else old["payload_json"])
                if old_payload == encoded:
                    return int(row[0])
            stored_payload, artifact_id = self._projection_storage(conn, lab_text, view, encoded)
            conn.execute(
                "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,revision,updated_at) "
                "VALUES(?,?,?,?,0,?) ON CONFLICT(lab_id,view) DO UPDATE SET "
                "payload_json=excluded.payload_json,payload_artifact_id=excluded.payload_artifact_id,updated_at=excluded.updated_at",
                (lab_text, view, stored_payload, artifact_id, now),
            )
            if old and old["payload_artifact_id"]:
                conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (old["payload_artifact_id"],))
            revision = self._bump_revision(conn, lab_text)
            conn.execute("UPDATE projections SET revision=? WHERE lab_id=? AND view=?", (revision, lab_text, view))
            return revision

    def load_history(self, lab_id: uuid.UUID | str) -> list[dict[str, Any]]:
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM history WHERE lab_id=? ORDER BY sequence", (lab_text,)
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def append_history(self, lab_id: uuid.UUID | str, record: dict[str, Any]) -> None:
        lab_text = self._lab_text(lab_id)
        if not isinstance(record, dict):
            raise TypeError("history record must be a dictionary")
        payload = _json(record)
        run_ref = record.get("run_id") or record.get("id")
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT coalesce(max(sequence),0)+1 FROM history WHERE lab_id=?", (lab_text,)).fetchone()
            run_id = None
            if isinstance(run_ref, str):
                found = conn.execute("SELECT run_id FROM runs WHERE lab_id=? AND (run_id=? OR alias=?)", (lab_text, run_ref, run_ref)).fetchone()
                run_id = found[0] if found else None
            conn.execute(
                "INSERT INTO history(lab_id,sequence,run_id,kind,created_at,payload_json) VALUES(?,?,?,?,?,?)",
                (lab_text, int(row[0]), run_id, str(record.get("kind") or "record"), now, payload),
            )
            self._bump_revision(conn, lab_text)

    def resolve_run_id(self, lab_id: uuid.UUID | str, run_ref: uuid.UUID | str) -> uuid.UUID:
        lab_text = self._lab_text(lab_id)
        try:
            run_text = _uuid_text(run_ref, label="run reference")
        except ValueError:
            run_text = ""
        if run_text:
            with self._reader() as conn:
                row = conn.execute("SELECT run_id FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Run not found in lab {lab_text}: {run_text}")
            return uuid.UUID(run_text)
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT run_id FROM aliases WHERE kind='run' AND scope_id=? AND alias=?",
                (lab_text, str(run_ref)),
            ).fetchall()
        if not rows:
            raise EntityNotFoundError(f"Run not found in lab {lab_text}: {run_ref}")
        if len(rows) > 1:
            raise AmbiguousAliasError(f"Run alias is ambiguous in lab {lab_text}: {run_ref}")
        return uuid.UUID(str(rows[0][0]))

    def create_run(
        self,
        lab_id: uuid.UUID | str,
        kind: str,
        alias: str,
        epoch_id: int,
        payload: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        lab_text = self._lab_text(lab_id)
        if not kind or not alias:
            raise ValueError("run kind and alias are required")
        if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 1:
            raise ValueError("epoch_id must be a positive integer")
        payload = payload or {}
        if not isinstance(payload, dict):
            raise TypeError("run payload must be a dictionary")
        run_id = uuid.uuid4()
        now = _now()
        with self._writer() as conn:
            lab = conn.execute("SELECT candidate_commit FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
            if lab is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            old = conn.execute("SELECT run_id,payload_json,epoch_id,kind FROM runs WHERE lab_id=? AND alias=?",
                               (lab_text, alias)).fetchone()
            if old:
                if old["epoch_id"] == epoch_id and old["kind"] == kind and old["payload_json"] == _json(payload):
                    return uuid.UUID(old["run_id"])
                raise IdempotencyConflictError(f"Run alias already exists with different data: {alias}")
            conn.execute(
                "INSERT INTO runs(lab_id,run_id,epoch_id,kind,alias,status,parent_run_id,candidate_commit,"
                "created_at,updated_at,payload_json) VALUES(?,?,?,?,?,'created',?,?,?,?,?)",
                (lab_text, str(run_id), epoch_id, kind, alias, payload.get("parent_run_id"),
                 payload.get("candidate_commit") or lab[0], now, now, _json(payload)),
            )
            conn.execute(
                "INSERT INTO aliases(kind,scope_id,alias,run_id) VALUES('run',?,?,?)",
                (lab_text, alias, str(run_id)),
            )
            self._bump_revision(conn, lab_text)
        return run_id

    def load_manifest(self, lab_id: uuid.UUID | str, run_id: uuid.UUID | str) -> dict[str, Any] | None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT payload_json FROM run_documents WHERE lab_id=? AND run_id=? AND document_kind='manifest'",
                (lab_text, run_text),
            ).fetchone()
            run = conn.execute("SELECT alias,epoch_id,status,candidate_commit FROM runs WHERE lab_id=? AND run_id=?",
                                (lab_text, run_text)).fetchone()
        if not run:
            return None
        manifest = _loads(row[0]) if row else {
            "id": run["alias"], "epoch": run["epoch_id"], "status": run["status"],
            "candidate_commit": run["candidate_commit"],
        }
        raw_runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
        results = self.load_results(lab_text, run_text)
        if results:
            if len(raw_runs) > len(results):
                results = [*results, *raw_runs[len(results):]]
            manifest["runs"] = results
        return manifest

    def list_recent_manifests(
        self,
        lab_id: uuid.UUID | str,
        epoch_id: int,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent manifests, synthesizing only documents missing from older stores."""
        lab_text = self._lab_text(lab_id)
        cap = max(1, min(limit, 100))
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT r.run_id, d.payload_json FROM runs r "
                "LEFT JOIN run_documents d ON d.lab_id=r.lab_id AND d.run_id=r.run_id "
                "AND d.document_kind='manifest' "
                "WHERE r.lab_id=? AND r.epoch_id=? "
                "ORDER BY r.created_at DESC,r.run_id DESC LIMIT ?",
                (lab_text, epoch_id, cap),
            ).fetchall()

        manifests = []
        for row in rows:
            if row["payload_json"] is not None:
                manifests.append(json.loads(row["payload_json"]))
                continue
            manifest = self.load_manifest(lab_text, row["run_id"])
            if manifest is not None:
                manifests.append(manifest)
        return manifests

    def save_manifest(self, lab_id: uuid.UUID | str, run_id: uuid.UUID | str, payload: dict[str, Any]) -> None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if not isinstance(payload, dict):
            raise TypeError("manifest payload must be a dictionary")
        encoded = _json(payload)
        now = _now()
        with self._writer() as conn:
            run = conn.execute("SELECT status FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if run is None:
                raise EntityNotFoundError(f"Run not found: {run_text}")
            if self._save_manifest_in_conn(conn, lab_text, run_text, payload, now):
                self._bump_revision(conn, lab_text)

    def _save_manifest_in_conn(
        self,
        conn: sqlite3.Connection,
        lab_id: str,
        run_id: str,
        payload: dict[str, Any],
        now: str,
    ) -> bool:
        encoded = _json(payload)
        existing = conn.execute(
            "SELECT payload_json,revision FROM run_documents WHERE lab_id=? AND run_id=? AND document_kind='manifest'",
            (lab_id, run_id),
        ).fetchone()
        run = conn.execute("SELECT status,parent_run_id FROM runs WHERE lab_id=? AND run_id=?", (lab_id, run_id)).fetchone()
        if run is None:
            raise EntityNotFoundError(f"Run not found: {run_id}")
        parent_id = run["parent_run_id"]
        if isinstance(payload.get("parent"), str):
            parent = conn.execute("SELECT run_id FROM runs WHERE lab_id=? AND (run_id=? OR alias=?)",
                                  (lab_id, payload["parent"], payload["parent"])).fetchone()
            if parent:
                parent_id = parent[0]
        new_status = str(payload.get("status") or run["status"])
        if (existing and existing["payload_json"] == encoded and run["status"] == new_status
                and run["parent_run_id"] == parent_id):
            return False
        revision = int(existing["revision"] + 1) if existing else 0
        conn.execute(
            "INSERT INTO run_documents(lab_id,run_id,document_kind,payload_json,revision,updated_at) "
            "VALUES(?,?,'manifest',?,?,?) ON CONFLICT(lab_id,run_id,document_kind) DO UPDATE SET "
            "payload_json=excluded.payload_json,revision=excluded.revision,updated_at=excluded.updated_at",
            (lab_id, run_id, encoded, revision, now),
        )
        conn.execute("UPDATE runs SET status=?,parent_run_id=?,updated_at=? WHERE lab_id=? AND run_id=?",
                     (new_status, parent_id, now, lab_id, run_id))
        return True

    def finalize_run(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        state: dict[str, Any],
        manifest: dict[str, Any],
        results: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Commit terminal state, manifest and any final repeats as one transaction."""
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if not isinstance(state, dict) or not isinstance(manifest, dict):
            raise TypeError("state and manifest must be dictionaries")
        if results is not None and not isinstance(results, (list, tuple)):
            raise TypeError("results must be a sequence of dictionaries")
        state_json = _json(state)
        if results is not None:
            for item in results:
                if not isinstance(item, dict):
                    raise TypeError("each result must be a dictionary")
                self._result_fields(item)
        now = _now()
        changed = False
        with self._writer() as conn:
            run = conn.execute("SELECT 1 FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if run is None:
                raise EntityNotFoundError(f"Run not found: {run_text}")
            for index, item in enumerate(results or (), 1):
                repeat_index = item.get("repeat_index", index)
                if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
                    raise ValueError("repeat_index must be a positive integer")
                changed = self._append_result_in_conn(conn, lab_text, run_text, repeat_index, item) or changed
            prior_state = conn.execute(
                "SELECT payload_json,payload_artifact_id FROM projections WHERE lab_id=? AND view='state'",
                (lab_text,),
            ).fetchone()
            prior_state_json = None
            if prior_state is not None:
                prior_state_json = (self.load_artifact(prior_state["payload_artifact_id"]).decode("utf-8")
                                    if prior_state["payload_artifact_id"] else prior_state["payload_json"])
            if prior_state_json != state_json:
                stored_state, state_artifact_id = self._projection_storage(conn, lab_text, "state", state_json)
                conn.execute(
                    "INSERT INTO projections(lab_id,view,payload_json,payload_artifact_id,revision,updated_at) "
                    "VALUES(?,'state',?,?,0,?) ON CONFLICT(lab_id,view) DO UPDATE SET "
                    "payload_json=excluded.payload_json,payload_artifact_id=excluded.payload_artifact_id,updated_at=excluded.updated_at",
                    (lab_text, stored_state, state_artifact_id, now),
                )
                if prior_state and prior_state["payload_artifact_id"]:
                    conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (prior_state["payload_artifact_id"],))
                changed = True
            changed = self._save_manifest_in_conn(conn, lab_text, run_text, manifest, now) or changed
            if changed:
                revision = self._bump_revision(conn, lab_text)
                conn.execute("UPDATE projections SET revision=? WHERE lab_id=? AND view='state'", (revision, lab_text))
                conn.execute("UPDATE run_documents SET revision=revision+1 WHERE lab_id=? AND run_id=? AND document_kind='manifest'",
                             (lab_text, run_text))
            else:
                revision = int(conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()[0])
        return revision

    def _run_text(self, lab_id: str, run_id: uuid.UUID | str) -> str:
        try:
            run_text = _uuid_text(run_id, label="run_id")
        except ValueError:
            run_text = str(self.resolve_run_id(lab_id, str(run_id)))
        return run_text

    def load_results(self, lab_id: uuid.UUID | str, run_id: uuid.UUID | str) -> list[dict[str, Any]]:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT repeat_index,status,cost_usd,requests,started_at,finished_at,result_json,result_artifact_id "
                "FROM repeats WHERE lab_id=? AND run_id=? AND status NOT IN ('running','interrupted') "
                "ORDER BY repeat_index",
                (lab_text, run_text),
            ).fetchall()
            metrics = conn.execute(
                "SELECT m.repeat_index,m.metric_name,m.value FROM metric_values m "
                "JOIN repeats r ON r.lab_id=m.lab_id AND r.run_id=m.run_id "
                "AND r.repeat_index=m.repeat_index AND r.status<>'quarantined' "
                "WHERE m.lab_id=? AND m.run_id=?",
                (lab_text, run_text),
            ).fetchall()
        by_repeat: dict[int, dict[str, float]] = {}
        for metric in metrics:
            by_repeat.setdefault(int(metric[0]), {})[str(metric[1])] = float(metric[2])
        results = []
        for row in rows:
            payload = {} if row["result_artifact_id"] else _loads(row["result_json"], {})
            payload["metrics"] = by_repeat.get(int(row["repeat_index"]), {})
            result = {
                **payload, "repeat_index": row["repeat_index"], "status": row["status"],
                "cost_usd": row["cost_usd"], "requests": row["requests"],
                "started_at": row["started_at"], "finished_at": row["finished_at"],
            }
            if row["result_artifact_id"]:
                result["details_available"] = True
            results.append(result)
        return results

    def load_result(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int,
    ) -> dict[str, Any] | None:
        """Return the exact submitted result document for one repeat."""
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT result_json,status,result_artifact_id,result_sha256 FROM repeats "
                "WHERE lab_id=? AND run_id=? AND repeat_index=?",
                (lab_text, run_text, repeat_index),
            ).fetchone()
        if row is None or row["status"] in {"running", "interrupted", "quarantined"}:
            return None
        raw = self.load_artifact(row["result_artifact_id"]) if row["result_artifact_id"] else row["result_json"].encode("utf-8")
        if _sha256(raw) != row["result_sha256"]:
            raise StoreCorruptError("Stored result payload does not match its SHA-256")
        value = _loads(raw)
        return value if isinstance(value, dict) else None

    @staticmethod
    def _result_fields(payload: dict[str, Any]) -> tuple[str, list[tuple[str, float]], float | None, int | None, str, str | None]:
        encoded = _json(payload)
        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            raise ValueError("result metrics must be an object")
        checked: list[tuple[str, float]] = []
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Metric {name!r} must be a finite number or null")
            checked.append((str(name), float(value)))
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        raw_cost = usage.get("cost_usd", payload.get("cost_usd"))
        cost: float | None = None
        if raw_cost is not None:
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not math.isfinite(float(raw_cost)) or raw_cost < 0:
                raise ValueError("cost_usd must be a non-negative finite number or null")
            cost = float(raw_cost)
        raw_requests = usage.get("requests", payload.get("requests"))
        requests: int | None = None
        if raw_requests is not None:
            if isinstance(raw_requests, bool) or not isinstance(raw_requests, int) or raw_requests < 0:
                raise ValueError("requests must be a non-negative integer or null")
            requests = raw_requests
        # A successfully validated result with no explicit status is a completed case.
        return encoded, checked, cost, requests, str(payload.get("status") or "ok"), payload.get("started_at")

    @staticmethod
    def _ensure_repeat_in_conn(
        conn: sqlite3.Connection,
        lab_id: str,
        run_id: str,
        repeat_index: int,
        started_at: str | None = None,
    ) -> bool:
        run = conn.execute("SELECT epoch_id FROM runs WHERE lab_id=? AND run_id=?", (lab_id, run_id)).fetchone()
        if run is None:
            raise EntityNotFoundError(f"Run not found: {run_id}")
        exists = conn.execute("SELECT 1 FROM repeats WHERE lab_id=? AND run_id=? AND repeat_index=?",
                              (lab_id, run_id, repeat_index)).fetchone()
        if exists:
            return False
        conn.execute(
            "INSERT INTO repeats(lab_id,run_id,epoch_id,repeat_index,status,started_at,finished_at,result_sha256,result_json) "
            "VALUES(?,?,?,?,'running',?,NULL,?,'{}')",
            (lab_id, run_id, int(run[0]), repeat_index, started_at or _now(), _sha256(b"{}")),
        )
        return True

    def ensure_repeat_started(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int,
        started_at: str | None = None,
    ) -> None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")
        with self._writer() as conn:
            if self._ensure_repeat_in_conn(conn, lab_text, run_text, repeat_index, started_at):
                self._bump_revision(conn, lab_text)

    def interrupt_running_repeats(self, lab_id: uuid.UUID | str, run_id: uuid.UUID | str) -> int:
        """Close open repeat placeholders without inventing result or usage values."""
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        now = _now()
        with self._writer() as conn:
            changed = conn.execute(
                "UPDATE repeats SET status='interrupted',finished_at=? "
                "WHERE lab_id=? AND run_id=? AND status='running'",
                (now, lab_text, run_text),
            ).rowcount
            if changed:
                self._bump_revision(conn, lab_text)
        return int(changed)

    def _append_result_in_conn(
        self,
        conn: sqlite3.Connection,
        lab_text: str,
        run_text: str,
        repeat_index: int,
        payload: dict[str, Any],
        prepared_object_path: str | None = None,
    ) -> bool:
        encoded, checked_metrics, cost, requests, status, started = self._result_fields(payload)
        result_bytes = encoded.encode("utf-8")
        result_digest = _sha256(result_bytes)
        run = conn.execute("SELECT epoch_id FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
        if run is None:
            raise EntityNotFoundError(f"Run not found: {run_text}")
        prior = conn.execute(
            "SELECT result_sha256,status,started_at,result_artifact_id FROM repeats "
            "WHERE lab_id=? AND run_id=? AND repeat_index=?",
            (lab_text, run_text, repeat_index),
        ).fetchone()
        if prior and prior["status"] != "running":
            if prior["result_sha256"] == result_digest:
                return False
            raise IdempotencyConflictError("Repeat already has a different result")
        epoch_id = int(run[0])
        finished = str(payload.get("finished_at") or _now())
        result_artifact_id = None
        stored_result_json = encoded
        if len(result_bytes) > INLINE_ARTIFACT_LIMIT:
            result_artifact_id = self._insert_large_payload_artifact(
                conn, lab_text, f"result-{repeat_index}.json", result_bytes, run_id=run_text,
                prepared_object_path=prepared_object_path,
            )
            stored_result_json = "{}"
        if prior:
            conn.execute(
                "UPDATE repeats SET status=?,cost_usd=?,requests=?,started_at=coalesce(started_at,?),finished_at=?,"
                "result_sha256=?,result_json=?,result_artifact_id=? "
                "WHERE lab_id=? AND run_id=? AND repeat_index=?",
                (status, cost, requests, started, finished, result_digest, stored_result_json,
                 result_artifact_id, lab_text, run_text, repeat_index),
            )
            if prior["result_artifact_id"]:
                conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (prior["result_artifact_id"],))
            conn.execute("DELETE FROM metric_values WHERE lab_id=? AND run_id=? AND repeat_index=?",
                         (lab_text, run_text, repeat_index))
        else:
            conn.execute(
                "INSERT INTO repeats(lab_id,run_id,epoch_id,repeat_index,status,cost_usd,requests,started_at,finished_at,"
                "result_sha256,result_json,result_artifact_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (lab_text, run_text, epoch_id, repeat_index, status, cost, requests, started, finished,
                 result_digest, stored_result_json, result_artifact_id),
            )
        for name, value in checked_metrics:
            self._metric_definition(conn, lab_text, epoch_id, name)
            conn.execute(
                "INSERT INTO metric_values(lab_id,run_id,epoch_id,repeat_index,metric_name,value) VALUES(?,?,?,?,?,?)",
                (lab_text, run_text, epoch_id, repeat_index, name, value),
            )
        conn.execute("UPDATE runs SET status=?,updated_at=? WHERE lab_id=? AND run_id=?",
                     (status, finished, lab_text, run_text))
        return True

    def append_result(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int,
        payload: dict[str, Any],
    ) -> None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
            raise ValueError("repeat_index must be a positive integer")
        if not isinstance(payload, dict):
            raise TypeError("result payload must be a dictionary")
        self._result_fields(payload)
        with self._writer() as conn:
            if self._append_result_in_conn(conn, lab_text, run_text, repeat_index, payload):
                self._bump_revision(conn, lab_text)

    def load_checkpoints(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int | None = None,
    ) -> list[dict[str, Any]]:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        query = "SELECT repeat_index,case_id,payload_json FROM cases WHERE lab_id=? AND run_id=?"
        params: list[Any] = [lab_text, run_text]
        if repeat_index is not None:
            if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
                raise ValueError("repeat_index must be a positive integer")
            query += " AND repeat_index=?"
            params.append(repeat_index)
        query += " ORDER BY repeat_index,updated_at,case_id"
        with self._reader() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{**_loads(row[2], {}), "repeat_index": row[0], "id": row[1]} for row in rows]

    def append_checkpoint(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        case_id: str,
        payload: dict[str, Any],
    ) -> None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id is required")
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload must be a dictionary")
        encoded = _json(payload)
        repeat_index = payload.get("repeat_index", payload.get("repeat", 1))
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
            raise ValueError("checkpoint repeat_index must be a positive integer")
        status = str(payload.get("status") or "pending")
        now = _now()
        with self._writer() as conn:
            run = conn.execute("SELECT epoch_id FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if run is None:
                raise EntityNotFoundError(f"Run not found: {run_text}")
            prior = conn.execute(
                "SELECT payload_json FROM cases WHERE lab_id=? AND run_id=? AND repeat_index=? AND case_id=?",
                (lab_text, run_text, repeat_index, case_id),
            ).fetchone()
            if prior:
                if prior[0] == encoded:
                    return
                raise IdempotencyConflictError(f"Checkpoint already exists with different payload: {case_id}")
            self._ensure_repeat_in_conn(conn, lab_text, run_text, repeat_index)
            conn.execute(
                "INSERT INTO cases(lab_id,run_id,epoch_id,repeat_index,case_id,status,updated_at,payload_json) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (lab_text, run_text, int(run[0]), repeat_index, case_id, status, now, encoded),
            )
            self._bump_revision(conn, lab_text)

    def append_attempt(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        attempt_id: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if not attempt_id or event not in {"started", "finished", "usage", "attempt", "wire"}:
            raise ValueError("attempt_id and a supported event are required")
        if not isinstance(payload, dict):
            raise TypeError("attempt payload must be a dictionary")
        encoded = _json(payload)
        repeat_index = payload.get("repeat_index", payload.get("repeat", 1))
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
            raise ValueError("attempt repeat_index must be a positive integer")
        raw_cost = payload.get("cost_usd")
        cost: float | None = None
        if raw_cost is not None:
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not math.isfinite(float(raw_cost)) or raw_cost < 0:
                raise ValueError("attempt cost_usd must be a non-negative finite number or null")
            cost = float(raw_cost)
        raw_requests = payload.get("requests")
        if raw_requests is None and isinstance(payload.get("usage"), dict):
            raw_requests = payload["usage"].get("requests")
        requests: int | None = None
        if raw_requests is not None:
            if isinstance(raw_requests, bool) or not isinstance(raw_requests, int) or raw_requests < 0:
                raise ValueError("attempt requests must be a non-negative integer or null")
            requests = raw_requests
        case_id = payload.get("case_id")
        if case_id is not None and (not isinstance(case_id, str) or not case_id):
            raise ValueError("attempt case_id must be a non-empty string or null")
        operation_id = payload.get("operation_id")
        operation_text = _uuid_text(operation_id, label="operation_id") if operation_id else None
        now = str(payload.get("created_at") or _now())
        with self._writer() as conn:
            run = conn.execute("SELECT epoch_id FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if run is None:
                raise EntityNotFoundError(f"Run not found: {run_text}")
            self._ensure_repeat_in_conn(conn, lab_text, run_text, repeat_index)
            prior = conn.execute(
                "SELECT payload_json FROM attempts WHERE lab_id=? AND run_id=? AND repeat_index=? AND attempt_id=? AND event=?",
                (lab_text, run_text, repeat_index, attempt_id, event),
            ).fetchone()
            if prior:
                if prior[0] == encoded:
                    return
                raise IdempotencyConflictError("Attempt event already exists with different payload")
            conn.execute(
                "INSERT INTO attempts(lab_id,run_id,epoch_id,repeat_index,operation_id,attempt_id,case_id,event,created_at,cost_usd,requests,payload_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (lab_text, run_text, int(run[0]), repeat_index, operation_text, attempt_id, case_id,
                 event, now, cost, requests, encoded),
            )
            self._bump_revision(conn, lab_text)

    def load_attempts(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer")
            limit = min(limit, MAX_PAGE_SIZE)
        query = "SELECT attempt_id,event,repeat_index,created_at,cost_usd,requests,case_id,payload_json FROM attempts WHERE lab_id=? AND run_id=?"
        params: tuple[Any, ...] = (lab_text, run_text)
        if repeat_index is not None:
            if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
                raise ValueError("repeat_index must be a positive integer")
            query += " AND repeat_index=?"
            params += (repeat_index,)
        query += " ORDER BY created_at DESC,attempt_id DESC,event DESC"
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        with self._reader() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{"attempt_id": row[0], "event": row[1], "repeat_index": row[2], "created_at": row[3],
                 "cost_usd": row[4], "requests": row[5], "case_id": row[6], **_loads(row[7], {})} for row in rows]

    def usage_totals(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int | None = None,
    ) -> dict[str, int | float]:
        """Aggregate lifecycle attempts or fallback usage events without double counting."""
        lab_text = self._lab_text(lab_id)
        run_text = self._run_text(lab_text, run_id)
        where = "lab_id=? AND run_id=?"
        params: tuple[Any, ...] = (lab_text, run_text)
        if repeat_index is not None:
            if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 1:
                raise ValueError("repeat_index must be a positive integer")
            where += " AND repeat_index=?"
            params += (repeat_index,)
        with self._reader() as conn:
            lifecycle = conn.execute(
                "WITH by_attempt AS ("
                " SELECT repeat_index,attempt_id,"
                " max(CASE WHEN event='started' THEN 1 ELSE 0 END) AS started,"
                " max(CASE WHEN event='finished' THEN 1 ELSE 0 END) AS finished,"
                " max(CASE WHEN event='finished' THEN cost_usd END) AS cost_usd"
                " FROM attempts WHERE " + where + " AND event IN ('started','finished')"
                " GROUP BY repeat_index,attempt_id)"
                " SELECT repeat_index,count(*),"
                " sum(CASE WHEN started=1 AND finished=0 THEN 1 ELSE 0 END),"
                " coalesce(sum(CASE WHEN finished=1 THEN cost_usd ELSE 0 END),0.0),"
                " sum(CASE WHEN finished=0 OR cost_usd IS NULL THEN 1 ELSE 0 END)"
                " FROM by_attempt GROUP BY repeat_index",
                params,
            ).fetchall()
            lifecycle_repeat_ids = {int(row[0]) for row in lifecycle}
            legacy = conn.execute(
                "SELECT repeat_index,count(*),coalesce(sum(cost_usd),0.0),"
                "sum(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) FROM attempts "
                "WHERE " + where + " AND event='attempt' AND repeat_index NOT IN ("
                "SELECT DISTINCT repeat_index FROM attempts WHERE " + where + " AND event IN ('started','finished')) "
                "GROUP BY repeat_index",
                params + params,
            ).fetchall()
            legacy_repeat_ids = {int(row[0]) for row in legacy}
            usage_events = conn.execute(
                "SELECT repeat_index,requests,cost_usd,payload_json FROM attempts "
                "WHERE " + where + " AND event='usage' AND repeat_index NOT IN ("
                "SELECT DISTINCT repeat_index FROM attempts WHERE " + where + " AND event IN ('started','finished')) "
                "AND repeat_index NOT IN (SELECT DISTINCT repeat_index FROM attempts WHERE " + where + " AND event='attempt')",
                params + params + params,
            ).fetchall()

        requests_total = unknown_requests = unknown_costs = 0
        cost_total = 0.0
        for row in lifecycle:
            requests_total += int(row[1])
            unknown_requests += int(row[2] or 0)
            cost_total += float(row[3] or 0.0)
            unknown_costs += int(row[4] or 0)
        for row in legacy:
            requests_total += int(row[1])
            cost_total += float(row[2] or 0.0)
            unknown_costs += int(row[3] or 0)
        for row in usage_events:
            if int(row[0]) in lifecycle_repeat_ids or int(row[0]) in legacy_repeat_ids:
                continue
            payload = _loads(row[3], {})
            payload = payload if isinstance(payload, dict) else {}
            event_requests = payload.get("requests", row[1])
            event_unknown_requests = payload.get("unknown_requests", 0)
            event_cost = payload.get("cost_usd", row[2])
            if isinstance(event_requests, int) and not isinstance(event_requests, bool) and event_requests >= 0:
                requests_total += event_requests
            if isinstance(event_unknown_requests, int) and not isinstance(event_unknown_requests, bool):
                unknown_requests += max(0, event_unknown_requests)
            if isinstance(event_cost, (int, float)) and not isinstance(event_cost, bool):
                cost_total += float(event_cost)
            else:
                unknown_costs += max(1, int(event_requests or 0))
        return {
            "requests": requests_total,
            "unknown_requests": unknown_requests,
            "cost_usd": cost_total,
            "unknown_costs": unknown_costs,
        }

    def lab_usage_totals(self, lab_id: uuid.UUID | str) -> dict[str, int | float]:
        """Aggregate usage for a lab in indexed SQL without loading run history."""
        lab_text = self._lab_text(lab_id)
        lifecycle_sql = (
            "WITH by_attempt AS ("
            " SELECT run_id,repeat_index,attempt_id,"
            " max(CASE WHEN event='started' THEN 1 ELSE 0 END) AS started,"
            " max(CASE WHEN event='finished' THEN 1 ELSE 0 END) AS finished,"
            " max(CASE WHEN event='finished' THEN cost_usd END) AS cost_usd"
            " FROM attempts WHERE lab_id=? AND event IN ('started','finished')"
            " GROUP BY run_id,repeat_index,attempt_id)"
            " SELECT count(*),"
            " coalesce(sum(CASE WHEN started=1 AND finished=0 THEN 1 ELSE 0 END),0),"
            " coalesce(sum(CASE WHEN finished=1 THEN cost_usd ELSE 0 END),0.0),"
            " coalesce(sum(CASE WHEN finished=0 OR cost_usd IS NULL THEN 1 ELSE 0 END),0)"
            " FROM by_attempt"
        )
        legacy_sql = (
            "SELECT count(*),coalesce(sum(a.cost_usd),0.0),"
            " coalesce(sum(CASE WHEN a.cost_usd IS NULL THEN 1 ELSE 0 END),0)"
            " FROM attempts a WHERE a.lab_id=? AND a.event='attempt' AND NOT EXISTS ("
            " SELECT 1 FROM attempts x WHERE x.lab_id=a.lab_id AND x.run_id=a.run_id"
            " AND x.repeat_index=a.repeat_index AND x.event IN ('started','finished'))"
        )
        usage_sql = (
            "SELECT coalesce(sum(coalesce(a.requests,CAST(json_extract(a.payload_json,'$.requests') AS INTEGER))),0),"
            " coalesce(sum(CAST(coalesce(json_extract(a.payload_json,'$.unknown_requests'),0) AS INTEGER)),0),"
            " coalesce(sum(coalesce(a.cost_usd,CAST(json_extract(a.payload_json,'$.cost_usd') AS REAL))),0.0),"
            " coalesce(sum(CASE WHEN coalesce(a.cost_usd,CAST(json_extract(a.payload_json,'$.cost_usd') AS REAL)) IS NULL"
            " AND coalesce(a.requests,CAST(json_extract(a.payload_json,'$.requests') AS INTEGER),0)>0"
            " THEN 1 ELSE 0 END),0)"
            " FROM attempts a WHERE a.lab_id=? AND a.event='usage' AND NOT EXISTS ("
            " SELECT 1 FROM attempts x WHERE x.lab_id=a.lab_id AND x.run_id=a.run_id"
            " AND x.repeat_index=a.repeat_index AND x.event IN ('started','finished','attempt'))"
        )
        result_sql = (
            "SELECT coalesce(sum(coalesce(r.requests,CAST(json_extract(r.result_json,'$.usage.requests') AS INTEGER),"
            "CAST(json_extract(r.result_json,'$.requests') AS INTEGER))),0),"
            " coalesce(sum(CAST(coalesce(json_extract(r.result_json,'$.usage.unknown_requests'),0) AS INTEGER)),0),"
            " coalesce(sum(coalesce(r.cost_usd,CAST(json_extract(r.result_json,'$.usage.cost_usd') AS REAL),"
            "CAST(json_extract(r.result_json,'$.cost_usd') AS REAL))),0.0),"
            " coalesce(sum(CASE WHEN coalesce(r.cost_usd,CAST(json_extract(r.result_json,'$.usage.cost_usd') AS REAL),"
            "CAST(json_extract(r.result_json,'$.cost_usd') AS REAL)) IS NULL"
            " AND coalesce(r.requests,CAST(json_extract(r.result_json,'$.usage.requests') AS INTEGER),"
            "CAST(json_extract(r.result_json,'$.requests') AS INTEGER),0)>0 THEN 1 ELSE 0 END),0),"
            " coalesce(sum(CAST(coalesce(json_extract(r.result_json,'$.duration_seconds'),0) AS REAL)),0.0)"
            " FROM repeats r WHERE r.lab_id=? AND r.status<>'quarantined' AND NOT EXISTS ("
            " SELECT 1 FROM attempts a WHERE a.lab_id=r.lab_id AND a.run_id=r.run_id"
            " AND a.repeat_index=r.repeat_index AND a.event IN ('started','finished','usage','attempt'))"
        )
        seconds_sql = (
            "SELECT coalesce(sum(CAST(json_extract(result_json,'$.duration_seconds') AS REAL)),0.0) "
            "FROM repeats WHERE lab_id=? AND status NOT IN ('running','quarantined')"
        )
        with self._reader() as conn:
            lifecycle = conn.execute(lifecycle_sql, (lab_text,)).fetchone()
            legacy = conn.execute(legacy_sql, (lab_text,)).fetchone()
            usage = conn.execute(usage_sql, (lab_text,)).fetchone()
            results = conn.execute(result_sql, (lab_text,)).fetchone()
            seconds = conn.execute(seconds_sql, (lab_text,)).fetchone()[0]
        requests = int(lifecycle[0] or 0) + int(legacy[0] or 0) + int(usage[0] or 0) + int(results[0] or 0)
        unknown_requests = int(lifecycle[1] or 0) + int(usage[1] or 0) + int(results[1] or 0)
        cost = float(lifecycle[2] or 0.0) + float(legacy[1] or 0.0) + float(usage[2] or 0.0) + float(results[2] or 0.0)
        unknown_costs = int(lifecycle[3] or 0) + int(legacy[2] or 0) + int(usage[3] or 0) + int(results[3] or 0)
        return {"requests": requests, "unknown_requests": unknown_requests, "cost_usd": cost,
                "unknown_costs": unknown_costs, "unknown_cost_usd": float(unknown_costs),
                "seconds": float(seconds or 0.0)}

    def append_usage(self, lab_id: uuid.UUID | str, run_id: uuid.UUID | str,
                     repeat_index: int, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("usage payload must be a dictionary")
        usage_payload = {**payload, "repeat_index": repeat_index}
        attempt_id = str(uuid.uuid4())
        self.append_attempt(lab_id, run_id, attempt_id, "usage", usage_payload)

    def append_wire_event(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        self.append_attempt(lab_id, run_id, event_id or str(uuid.uuid4()), "wire",
                            {**payload, "repeat_index": repeat_index})

    def append_wire_events_batch(
        self,
        lab_id: str | Path | uuid.UUID,
        run_id: str | uuid.UUID,
        repeat_index: int | None,
        events: list[tuple[dict[str, Any], str | None]],
    ) -> list[str]:
        if not events:
            return []
        if isinstance(lab_id, Path) or not isinstance(lab_id, uuid.UUID):
            try:
                lab_text = self._lab_text(lab_id)
            except ValueError:
                lab_text = str(self.resolve_lab_id(lab_id))
        else:
            lab_text = str(lab_id)
        run_text = self._run_text(lab_text, run_id)
        event_ids: list[str] = []
        with self._writer() as conn:
            run = conn.execute("SELECT epoch_id FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
            if run is None:
                raise EntityNotFoundError(f"Run not found: {run_text}")
            epoch_id = int(run[0])
            for payload, event_id in events:
                if not isinstance(payload, dict):
                    raise TypeError("wire payload must be a dictionary")
                eid = event_id or str(uuid.uuid4())
                event_ids.append(eid)
                rep = repeat_index if repeat_index is not None else payload.get("repeat_index", payload.get("repeat", 1))
                if isinstance(rep, bool) or not isinstance(rep, int) or rep < 1:
                    raise ValueError("attempt repeat_index must be a positive integer")
                event_payload = {**payload, "repeat_index": rep}
                encoded = _json(event_payload)
                raw_cost = event_payload.get("cost_usd")
                cost: float | None = None
                if raw_cost is not None:
                    if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not math.isfinite(float(raw_cost)) or raw_cost < 0:
                        raise ValueError("attempt cost_usd must be a non-negative finite number or null")
                    cost = float(raw_cost)
                raw_requests = event_payload.get("requests")
                if raw_requests is None and isinstance(event_payload.get("usage"), dict):
                    raw_requests = event_payload["usage"].get("requests")
                requests: int | None = None
                if raw_requests is not None:
                    if isinstance(raw_requests, bool) or not isinstance(raw_requests, int) or raw_requests < 0:
                        raise ValueError("attempt requests must be a non-negative integer or null")
                    requests = raw_requests
                case_id = event_payload.get("case_id")
                if case_id is not None and (not isinstance(case_id, str) or not case_id):
                    raise ValueError("attempt case_id must be a non-empty string or null")
                operation_id = event_payload.get("operation_id")
                operation_text = _uuid_text(operation_id, label="operation_id") if operation_id else None
                now = str(event_payload.get("created_at") or _now())

                self._ensure_repeat_in_conn(conn, lab_text, run_text, rep)
                prior = conn.execute(
                    "SELECT payload_json FROM attempts WHERE lab_id=? AND run_id=? AND repeat_index=? AND attempt_id=? AND event='wire'",
                    (lab_text, run_text, rep, eid),
                ).fetchone()
                if prior:
                    if prior[0] == encoded:
                        continue
                    raise IdempotencyConflictError("Attempt event already exists with different payload")
                conn.execute(
                    "INSERT INTO attempts(lab_id,run_id,epoch_id,repeat_index,operation_id,attempt_id,case_id,event,created_at,cost_usd,requests,payload_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (lab_text, run_text, epoch_id, rep, operation_text, eid, case_id,
                     "wire", now, cost, requests, encoded),
                )
            self._bump_revision(conn, lab_text)
        return event_ids

    def load_wire_events(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str,
        repeat_index: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return [item for item in self.load_attempts(lab_id, run_id, repeat_index, limit)
                if item.get("event") == "wire"]

    def enqueue_operation(
        self,
        lab_id: uuid.UUID | str,
        kind: str,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        lab_text = self._lab_text(lab_id)
        if not kind or not idempotency_key:
            raise ValueError("kind and idempotency_key are required")
        if not isinstance(payload, dict):
            raise TypeError("operation payload must be a dictionary")
        encoded = _json(payload)
        digest = _sha256(encoded.encode("utf-8"))
        operation_id = uuid.uuid4()
        now = _now()
        with self._writer() as conn:
            if conn.execute("SELECT 1 FROM labs WHERE lab_id=?", (lab_text,)).fetchone() is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            prior = conn.execute(
                "SELECT * FROM operations WHERE lab_id=? AND idempotency_key=?", (lab_text, idempotency_key)
            ).fetchone()
            if prior:
                if prior["kind"] != kind or prior["payload_hash"] != digest or prior["payload_json"] != encoded:
                    raise IdempotencyConflictError("Idempotency key was already used for a different request")
                item = self._operation_dict(prior)
                item["replayed"] = True
                return item
            active = conn.execute(
                "SELECT operation_id FROM operations WHERE lab_id=? AND status IN ('queued','running','finalizing','recovery_required')",
                (lab_text,),
            ).fetchone()
            if active:
                raise ActiveOperationError("A mutating operation is already active for this lab")
            conn.execute(
                "INSERT INTO operations(lab_id,operation_id,kind,idempotency_key,payload_hash,payload_json,status,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                (lab_text, str(operation_id), kind, idempotency_key, digest, encoded, now, now),
            )
            conn.execute(
                "INSERT INTO aliases(kind,scope_id,alias,operation_id) VALUES('operation',?,?,?)",
                (lab_text, idempotency_key, str(operation_id)),
            )
            self._bump_revision(conn, lab_text)
            row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (str(operation_id),)).fetchone()
            item = self._operation_dict(row)
            item["replayed"] = False
            return item

    @staticmethod
    def _operation_dict(row: sqlite3.Row) -> dict[str, Any]:
        error = _loads(row["error_json"])
        return {
            "operation_id": uuid.UUID(row["operation_id"]),
            "lab_id": uuid.UUID(row["lab_id"]),
            "kind": row["kind"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "pid": row["pid"],
            "heartbeat_at": row["heartbeat_at"],
            "progress": _loads(row["progress_json"]),
            "result": _loads(row["result_json"]),
            "error": error,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "payload": _loads(row["payload_json"], {}),
        }

    def get_operation(self, operation_id: uuid.UUID | str) -> dict[str, Any]:
        operation_text = _uuid_text(operation_id, label="operation_id")
        with self._reader() as conn:
            row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (operation_text,)).fetchone()
        if row is None:
            raise EntityNotFoundError(f"Operation not found: {operation_text}")
        return self._operation_dict(row)

    def request_operation_cancel(self, operation_id: uuid.UUID | str) -> dict[str, Any]:
        """Persist a bounded cancel marker without copying request or config data."""
        op_text = _uuid_text(operation_id, label="operation_id")
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            error = _loads(row["error_json"], {})
            if row["status"] == "failed" and isinstance(error, dict) and error.get("code") == "operation_cancelled":
                return self._operation_dict(row)
            if row["status"] not in {"queued", "running"}:
                raise OperationStateError(f"Operation cannot be cancelled from status {row['status']}")
            progress = _loads(row["progress_json"], {})
            progress = progress if isinstance(progress, dict) else {}
            if progress.get("cancel_requested") is not True:
                progress["cancel_requested"] = True
                progress["cancel_requested_at"] = now
                conn.execute(
                    "UPDATE operations SET progress_json=?,updated_at=? WHERE operation_id=? AND status=?",
                    (_json(progress), now, op_text, row["status"]),
                )
                self._bump_revision(conn, row["lab_id"])
                row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            return self._operation_dict(row)

    def set_operation_cancel_pending(self, operation_id: uuid.UUID | str, reason: str) -> dict[str, Any]:
        """Persist why a requested cancellation could not safely stop its worker."""
        allowed = {"worker_identity_unavailable", "worker_identity_mismatch", "worker_tree_unavailable",
                   "worker_exit_unconfirmed"}
        if reason not in allowed:
            raise ValueError("unsupported cancellation-pending reason")
        op_text = _uuid_text(operation_id, label="operation_id")
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            if row["status"] not in {"queued", "running"}:
                raise OperationStateError(f"Cancellation is no longer pending from status {row['status']}")
            progress = _loads(row["progress_json"], {})
            progress = progress if isinstance(progress, dict) else {}
            if progress.get("cancel_requested") is not True:
                raise OperationStateError("Cancellation must be requested before recording pending state")
            progress["cancel_pending_reason"] = reason
            conn.execute(
                "UPDATE operations SET progress_json=?,updated_at=? WHERE operation_id=? AND status=?",
                (_json(progress), now, op_text, row["status"]),
            )
            self._bump_revision(conn, row["lab_id"])
            updated = conn.execute("SELECT * FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            return self._operation_dict(updated)

    def claim_operation(self, operation_id: uuid.UUID | str, pid: int) -> bool:
        op_text = _uuid_text(operation_id, label="operation_id")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT lab_id,status FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            if row["status"] != "queued":
                return False
            changed = conn.execute(
                "UPDATE operations SET status='running',pid=?,heartbeat_at=?,updated_at=? WHERE operation_id=? AND status='queued'",
                (pid, now, now, op_text),
            ).rowcount
            if changed:
                self._bump_revision(conn, row["lab_id"])
            return bool(changed)

    def record_operation_spawn(
        self,
        operation_id: uuid.UUID | str,
        pid: int,
        *,
        worker_identity: dict[str, Any] | None = None,
        worker_job_handle: int | None = None,
    ) -> None:
        """Record the launched child while it remains queued for its own claim."""
        op_text = _uuid_text(operation_id, label="operation_id")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if worker_identity is not None:
            kind = worker_identity.get("kind") if isinstance(worker_identity, dict) else None
            start_token = worker_identity.get("start_token") if isinstance(worker_identity, dict) else None
            if (kind not in {"linux_proc", "windows_job"} or not isinstance(start_token, str)
                    or not start_token or len(start_token) > 128):
                raise ValueError("worker_identity must contain a supported kind and start token")
            if kind == "windows_job":
                job_name = worker_identity.get("job_name")
                if job_name != f"Local\\AutoResearch-{op_text}" or len(job_name) > 128:
                    raise ValueError("windows worker identity requires an operation-scoped job name")
                identity = {"kind": kind, "start_token": start_token, "job_name": job_name}
            else:
                process_group = worker_identity.get("process_group")
                session_id = worker_identity.get("session_id")
                if (isinstance(process_group, bool) or not isinstance(process_group, int) or process_group != pid
                        or isinstance(session_id, bool) or not isinstance(session_id, int) or session_id != pid):
                    raise ValueError("linux worker identity requires its isolated process group")
                identity = {
                    "kind": kind, "start_token": start_token,
                    "process_group": process_group, "session_id": session_id,
                }
        else:
            identity = None
        if worker_job_handle is not None and (
            isinstance(worker_job_handle, bool) or not isinstance(worker_job_handle, int)
            or worker_job_handle <= 0 or worker_job_handle > 0xFFFFFFFFFFFFFFFF
            or not isinstance(identity, dict) or identity.get("kind") != "windows_job"
        ):
            raise ValueError("worker_job_handle requires a positive Windows worker handle")
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT lab_id,status,progress_json FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            if row["status"] != "queued":
                raise OperationStateError("A child PID can be recorded only while the operation is queued")
            progress = _loads(row["progress_json"], {})
            progress = progress if isinstance(progress, dict) else {}
            progress["worker_identity"] = identity
            if worker_job_handle is not None:
                progress["worker_job_handle"] = worker_job_handle
            conn.execute(
                "UPDATE operations SET pid=?,progress_json=?,heartbeat_at=?,updated_at=? "
                "WHERE operation_id=? AND status='queued'",
                (pid, _json(progress), now, now, op_text),
            )
            self._bump_revision(conn, row["lab_id"])

    def heartbeat_operation(
        self,
        operation_id: uuid.UUID | str,
        pid: int,
        progress: dict[str, Any] | None = None,
    ) -> None:
        op_text = _uuid_text(operation_id, label="operation_id")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        progress_json = _json(progress) if progress is not None else None
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT lab_id,status,pid FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            if row["status"] not in {"running", "finalizing"} or row["pid"] != pid:
                raise OperationStateError("Only the active operation worker may send a heartbeat")
            conn.execute(
                "UPDATE operations SET heartbeat_at=?,progress_json=coalesce(?,progress_json),updated_at=? WHERE operation_id=?",
                (now, progress_json, now, op_text),
            )
            self._bump_revision(conn, row["lab_id"])

    def transition_operation(
        self,
        operation_id: uuid.UUID | str,
        expected_status: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        op_text = _uuid_text(operation_id, label="operation_id")
        allowed = {
            "queued": {"running", "failed", "recovery_required"},
            "running": {"finalizing", "failed", "recovery_required"},
            "finalizing": {"succeeded", "failed", "recovery_required"},
            "recovery_required": {"running", "failed"},
        }
        if expected_status not in allowed or status not in allowed[expected_status]:
            raise ValueError(f"Invalid operation transition: {expected_status} -> {status}")
        result_json = _json(result) if result is not None else None
        error_json = _json(error) if error is not None else None
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT lab_id,status FROM operations WHERE operation_id=?", (op_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Operation not found: {op_text}")
            if row["status"] != expected_status:
                raise OperationStateError(f"Expected status {expected_status}, found {row['status']}")
            conn.execute(
                "UPDATE operations SET status=?,result_json=coalesce(?,result_json),error_json=coalesce(?,error_json),updated_at=? "
                "WHERE operation_id=? AND status=?",
                (status, result_json, error_json, now, op_text, expected_status),
            )
            self._bump_revision(conn, row["lab_id"])

    def load_worker(self, lab_id: uuid.UUID | str) -> dict[str, Any] | None:
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT operation_id,kind,status,pid,heartbeat_at,progress_json,result_json,error_json,created_at,updated_at "
                "FROM operations WHERE lab_id=? ORDER BY updated_at DESC,created_at DESC LIMIT 1",
                (lab_text,),
            ).fetchone()
        if row is None:
            return None
        error = _loads(row["error_json"])
        return {"operation_id": uuid.UUID(row["operation_id"]), "kind": row["kind"], "status": row["status"],
                "pid": row["pid"], "heartbeat": row["heartbeat_at"], "progress": _loads(row["progress_json"]),
                "result": _loads(row["result_json"]), "error": error,
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def get_latest_operation(self, lab_id: uuid.UUID | str) -> dict[str, Any] | None:
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            row = conn.execute("SELECT * FROM operations WHERE lab_id=? ORDER BY updated_at DESC,created_at DESC LIMIT 1",
                               (lab_text,)).fetchone()
        return self._operation_dict(row) if row else None

    def get_active_operation(self, lab_id: uuid.UUID | str) -> dict[str, Any] | None:
        """Return the lab's unique nonterminal operation, if one exists."""
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT * FROM operations WHERE lab_id=? AND status IN "
                "('queued','running','finalizing','recovery_required') "
                "ORDER BY created_at DESC,operation_id DESC LIMIT 2",
                (lab_text,),
            ).fetchall()
        if len(rows) > 1:
            raise StoreConflictError("Lab has multiple nonterminal operations")
        return self._operation_dict(rows[0]) if rows else None

    def list_active_operations(self) -> list[dict[str, Any]]:
        """Return active operations with the lab paths needed by doctor preflight."""
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT o.operation_id, o.lab_id, o.kind, o.status, o.pid, o.heartbeat_at, "
                "l.state_dir, l.source_root "
                "FROM operations o LEFT JOIN labs l ON o.lab_id=l.lab_id "
                "WHERE o.status IN ('queued','running','finalizing')"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, lab_id: uuid.UUID | str, run_ref: uuid.UUID | str) -> dict[str, Any] | None:
        lab_text = self._lab_text(lab_id)
        try:
            run_text = self._run_text(lab_text, run_ref)
        except EntityNotFoundError:
            return None
        with self._reader() as conn:
            row = conn.execute("SELECT * FROM runs WHERE lab_id=? AND run_id=?", (lab_text, run_text)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["lab_id"] = uuid.UUID(result["lab_id"])
        result["run_id"] = uuid.UUID(result["run_id"])
        result["payload"] = _loads(result.pop("payload_json"), {})
        result["manifest"] = self.load_manifest(lab_text, run_text)
        result["repeats"] = self.load_results(lab_text, run_text)
        return result

    def get_active_run(self, lab_id: uuid.UUID | str) -> dict[str, Any] | None:
        """Return the latest active run projection without loading its repeat history."""
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            row = conn.execute(
                "SELECT r.run_id,r.alias,r.epoch_id,r.kind,r.status,r.candidate_commit,d.payload_json "
                "FROM runs r LEFT JOIN run_documents d ON d.lab_id=r.lab_id AND d.run_id=r.run_id "
                "AND d.document_kind='manifest' WHERE r.lab_id=? AND r.status IN ('queued','running','finalizing') "
                "ORDER BY r.updated_at DESC,r.run_id DESC LIMIT 1",
                (lab_text,),
            ).fetchone()
        if row is None:
            return None
        manifest = _loads(row[6], {}) if row[6] else {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.pop("runs", None)
        manifest.update({"id": row[1], "status": row[4], "epoch": row[2],
                         "candidate_commit": row[5], "run_id": uuid.UUID(row[0]),
                         "kind": row[3]})
        return manifest

    def list_runs(
        self,
        lab_id: uuid.UUID | str,
        *,
        limit: int = 100,
        offset: int = 0,
        cursor: uuid.UUID | str | None = None,
        selected: Sequence[str | uuid.UUID] | None = None,
        epoch_id: int | None = None,
        include_metrics: bool = False,
    ) -> dict[str, Any]:
        limit, offset = _safe_page(limit, offset)
        lab_text = self._lab_text(lab_id)
        clauses = ["lab_id=?"]
        params: list[Any] = [lab_text]
        if epoch_id is not None:
            if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 1:
                raise ValueError("epoch_id must be a positive integer")
            clauses.append("epoch_id=?")
            params.append(epoch_id)
        if selected is not None:
            selected_ids = [str(self.resolve_run_id(lab_text, ref)) for ref in selected]
            if not selected_ids:
                return {"items": [], "next_cursor": None, "total": 0, "limit": limit, "offset": offset}
            clauses.append("run_id IN (" + ",".join("?" for _ in selected_ids) + ")")
            params.extend(selected_ids)
        count_clauses = list(clauses)
        count_params = list(params)
        if cursor is not None:
            cursor_id = str(self.resolve_run_id(lab_text, cursor))
            with self._reader() as conn:
                cursor_row = conn.execute("SELECT created_at FROM runs WHERE lab_id=? AND run_id=?", (lab_text, cursor_id)).fetchone()
            if cursor_row is None:
                raise EntityNotFoundError(f"Run cursor not found: {cursor_id}")
            clauses.append("(created_at,run_id)<(?,?)")
            params.extend((cursor_row[0], cursor_id))
        where = " WHERE " + " AND ".join(clauses)
        count_where = " WHERE " + " AND ".join(count_clauses)
        with self._reader() as conn:
            total = int(conn.execute("SELECT count(*) FROM runs" + count_where, count_params).fetchone()[0])
            rows = conn.execute(
                "SELECT lab_id,run_id,epoch_id,kind,alias,status,parent_run_id,candidate_commit,created_at,updated_at "
                "FROM runs" + where + " ORDER BY created_at DESC,run_id DESC LIMIT ? OFFSET ?",
                (*params, limit + 1, offset),
            ).fetchall()
            items = [dict(row) for row in rows[:limit]]
            if include_metrics and items:
                run_ids = [str(item["run_id"]) for item in items]
                run_marks = ",".join("?" for _ in run_ids)
                manifest_rows = conn.execute(
                    f"SELECT run_id, payload_json FROM run_documents "
                    f"WHERE lab_id=? AND document_kind='manifest' AND run_id IN ({run_marks})",
                    (lab_text, *run_ids),
                ).fetchall()
                manifests_by_run: dict[str, dict[str, Any]] = {}
                for m_row in manifest_rows:
                    m_val = _loads(m_row["payload_json"], {})
                    if isinstance(m_val, dict):
                        manifests_by_run[str(m_row["run_id"])] = m_val

                for item in items:
                    run_id_str = str(item["run_id"])
                    manifest = manifests_by_run.get(run_id_str, {})
                    hyp = manifest.get("hypothesis") if isinstance(manifest.get("hypothesis"), dict) else None
                    hyp_title = hyp.get("title") if isinstance(hyp, dict) else None
                    item["title"] = hyp_title or manifest.get("title") or ""
                    item["reason"] = manifest.get("reason") or ""
                    item["hypothesis"] = hyp
                    regressed = manifest.get("regressed_case_ids")
                    item["regressed_case_ids"] = regressed if isinstance(regressed, list) else []

                raw_metric_rows = conn.execute(
                    "SELECT m.run_id,m.metric_name,m.value "
                    "FROM metric_values m JOIN repeats r ON r.lab_id=m.lab_id AND r.run_id=m.run_id "
                    "AND r.epoch_id=m.epoch_id AND r.repeat_index=m.repeat_index AND r.status='ok' "
                    f"WHERE m.lab_id=? AND m.run_id IN ({run_marks})",
                    (lab_text, *run_ids),
                ).fetchall()
                vals_by_run_metric: dict[tuple[str, str], list[float]] = {}
                for r_row in raw_metric_rows:
                    val = r_row["value"]
                    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                        vals_by_run_metric.setdefault((str(r_row["run_id"]), str(r_row["metric_name"])), []).append(float(val))

                metric_rows = conn.execute(
                    "SELECT m.run_id,m.metric_name,d.direction,MIN(m.value) AS min_value,"
                    "MAX(m.value) AS max_value,AVG(m.value) AS mean_value,COUNT(*) AS completed_repeats "
                    "FROM metric_values m JOIN repeats r ON r.lab_id=m.lab_id AND r.run_id=m.run_id "
                    "AND r.epoch_id=m.epoch_id AND r.repeat_index=m.repeat_index AND r.status='ok' "
                    "LEFT JOIN metric_definitions d ON d.lab_id=m.lab_id AND d.epoch_id=m.epoch_id "
                    "AND d.metric_name=m.metric_name "
                    f"WHERE m.lab_id=? AND m.run_id IN ({run_marks}) "
                    "GROUP BY m.run_id,m.metric_name,d.direction",
                    (lab_text, *run_ids),
                ).fetchall()
                metrics_by_run: dict[str, dict[str, dict[str, Any]]] = {run_id: {} for run_id in run_ids}
                for row in metric_rows:
                    direction = row["direction"]
                    rid_str = str(row["run_id"])
                    mname_str = str(row["metric_name"])
                    vals = vals_by_run_metric.get((rid_str, mname_str), [])
                    median_val = float(statistics.median(vals)) if vals else float(row["mean_value"])
                    metrics_by_run[rid_str][mname_str] = {
                        "decision_worst": float(row["min_value"] if direction == "max" else row["max_value"]),
                        "mean": float(row["mean_value"]),
                        "median": median_val,
                        "completed_repeats": int(row["completed_repeats"]),
                    }
                for item in items:
                    item["metrics"] = metrics_by_run[str(item["run_id"])]
        has_more = len(rows) > limit
        for item in items:
            item["lab_id"] = uuid.UUID(item["lab_id"])
            item["run_id"] = uuid.UUID(item["run_id"])
        return {"items": items, "next_cursor": str(items[-1]["run_id"]) if has_more and items else None,
                "total": total, "limit": limit, "offset": offset}

    def reconcile_terminal_run_statuses(
        self, lab_id: uuid.UUID | str, *, apply: bool = False,
    ) -> list[dict[str, Any]]:
        """Repair statuses contradicted by a saved terminal manifest, without inferring results."""
        lab_text = self._lab_text(lab_id)
        terminal = {"accepted", "rejected", "failed", "interrupted", "budget", "crash",
                    "timeout", "incomparable", "invalid", "succeeded"}
        repairable_from = {"created", "queued", "running", "finalizing", "ok", "crash",
                           "timeout", "invalid", "budget", "interrupted"}
        findings: list[dict[str, Any]] = []
        changed = False
        connection = self._writer() if apply else self._reader()
        with connection as conn:
            rows = conn.execute(
                "SELECT r.run_id,r.alias,r.status,d.payload_json "
                "FROM runs r LEFT JOIN run_documents d ON d.lab_id=r.lab_id AND d.run_id=r.run_id "
                "AND d.document_kind='manifest' WHERE r.lab_id=? ORDER BY r.created_at,r.run_id",
                (lab_text,),
            ).fetchall()
            for row in rows:
                if row["payload_json"] is None:
                    continue
                try:
                    manifest = _loads(row["payload_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                document_status = manifest.get("status") if isinstance(manifest, dict) else None
                if document_status not in terminal:
                    continue
                if row["status"] in repairable_from and row["status"] != document_status:
                    findings.append({
                        "kind": "run_status", "run_id": row["run_id"], "alias": row["alias"],
                        "stored_status": row["status"], "manifest_status": document_status,
                        "status": "reconciled" if apply else "repairable",
                    })
                    if apply:
                        changed = bool(conn.execute(
                            "UPDATE runs SET status=?,updated_at=? WHERE lab_id=? AND run_id=? AND status=?",
                            (document_status, _now(), lab_text, row["run_id"], row["status"]),
                        ).rowcount) or changed
                # A terminal manifest proves an open repeat was abandoned. It says
                # nothing about that repeat's result, metrics, attempts, or usage.
                repeats = conn.execute(
                    "SELECT repeat_index FROM repeats WHERE lab_id=? AND run_id=? AND status='running' "
                    "ORDER BY repeat_index", (lab_text, row["run_id"]),
                ).fetchall()
                for repeat in repeats:
                    findings.append({
                        "kind": "repeat_status", "run_id": row["run_id"], "alias": row["alias"],
                        "repeat_index": int(repeat["repeat_index"]), "stored_status": "running",
                        "manifest_status": document_status, "target_status": "interrupted",
                        "status": "reconciled" if apply else "repairable",
                    })
                    if apply:
                        changed = bool(conn.execute(
                            "UPDATE repeats SET status='interrupted',finished_at=? "
                            "WHERE lab_id=? AND run_id=? AND repeat_index=? AND status='running'",
                            (_now(), lab_text, row["run_id"], int(repeat["repeat_index"])),
                        ).rowcount) or changed
            if changed:
                self._bump_revision(conn, lab_text)
        return findings

    def get_document(self, lab_id: uuid.UUID | str, key: str | Path, default: Any = None) -> Any:
        lab_text = self._lab_text(lab_id)
        name = self._document_key(lab_text, key)
        if name == "catalog":
            return self.load_catalog()
        if name == "worker":
            return self.load_worker(lab_text)
        if name.startswith("run:"):
            _, run_ref, doc = name.split(":", 2)
            run_id = self.resolve_run_id(lab_text, run_ref)
            if doc == "manifest":
                return self.load_manifest(lab_text, run_id)
            if doc.startswith("result:"):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                return next((item for item in self.load_results(lab_text, run_id)
                             if item.get("repeat_index") == repeat), default)
        if name.startswith("run:"):
            _, run_ref, doc = name.split(":", 2)
            run_id = self.resolve_run_id(lab_text, run_ref)
            if doc.startswith("cases:"):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                return self.load_checkpoints(lab_text, run_id, repeat)
            if doc.startswith("attempts:"):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                return self.load_attempts(lab_text, run_id, repeat)
        if name in {"config", "state", "candidate"}:
            try:
                return self.load_projection(lab_text, name)
            except EntityNotFoundError:
                return default
        if name == "history":
            return self.load_history(lab_text)
        with self._reader() as conn:
            row = conn.execute("SELECT payload_json FROM lab_documents WHERE lab_id=? AND document_key=?",
                               (lab_text, name)).fetchone()
        return _loads(row[0], default) if row else default

    def set_document(
        self,
        lab_id: uuid.UUID | str,
        key: str | Path,
        payload: Any,
        expected_revision: int | None = None,
    ) -> int:
        lab_text = self._lab_text(lab_id)
        name = self._document_key(lab_text, key)
        if name == "catalog":
            if not isinstance(payload, dict):
                raise TypeError("catalog document must be a dictionary")
            return self.save_catalog(payload)
        if name == "worker":
            raise StoreConflictError("Worker state is owned by the operation repository methods")
        if name.startswith("run:"):
            _, run_ref, doc = name.split(":", 2)
            run_id = self.resolve_run_id(lab_text, run_ref)
            if doc == "manifest":
                self.save_manifest(lab_text, run_id, payload)
                return int(self.get_lab(lab_text)["revision"])
            if doc.startswith("result:"):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                self.append_result(lab_text, run_id, repeat, payload)
                return int(self.get_lab(lab_text)["revision"])
            if doc.startswith("cases:") and isinstance(payload, list):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        self.append_checkpoint(lab_text, run_id, item["id"], {**item, "repeat_index": repeat})
                return int(self.get_lab(lab_text)["revision"])
            if doc.startswith("attempts:") and isinstance(payload, list):
                repeat = int(doc.split(":", 1)[1].removeprefix("repeat_"))
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("attempt_id"), str) and item.get("event") in {
                        "started", "finished", "usage", "attempt", "wire"
                    }:
                        self.append_attempt(lab_text, run_id, item["attempt_id"], item["event"], {**item, "repeat_index": repeat})
                return int(self.get_lab(lab_text)["revision"])
        if name in {"config", "state", "candidate"}:
            if not isinstance(payload, dict):
                raise TypeError("lab projections must be dictionaries")
            return self.save_projection(lab_text, name, payload, expected_revision)
        if name == "history":
            if not isinstance(payload, list):
                raise TypeError("history projection must be a list")
            current = self.load_history(lab_text)
            if payload[:len(current)] != current:
                raise StoreConflictError("History is append-only; use append_history for new records")
            additions = payload[len(current):]
            if not additions:
                return int(self.get_lab(lab_text)["revision"])
            encoded_items = [_json(item) for item in additions]
            now = _now()
            with self._writer() as conn:
                row = conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
                if row is None:
                    raise EntityNotFoundError(f"Lab not found: {lab_text}")
                if expected_revision is not None and int(row[0]) != expected_revision:
                    raise RevisionConflictError(f"Expected revision {expected_revision}, found {row[0]}")
                sequence = int(conn.execute("SELECT coalesce(max(sequence),0)+1 FROM history WHERE lab_id=?",
                                            (lab_text,)).fetchone()[0])
                for offset, encoded in enumerate(encoded_items):
                    item = additions[offset]
                    conn.execute("INSERT INTO history(lab_id,sequence,kind,created_at,payload_json) VALUES(?,?,?,?,?)",
                                 (lab_text, sequence + offset,
                                  str(item.get("kind") or "record") if isinstance(item, dict) else "record", now, encoded))
                return self._bump_revision(conn, lab_text)
        encoded = _json(payload)
        now = _now()
        with self._writer() as conn:
            row = conn.execute("SELECT revision FROM labs WHERE lab_id=?", (lab_text,)).fetchone()
            if row is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            if expected_revision is not None and int(row[0]) != expected_revision:
                raise RevisionConflictError(f"Expected revision {expected_revision}, found {row[0]}")
            old = conn.execute("SELECT payload_json FROM lab_documents WHERE lab_id=? AND document_key=?",
                               (lab_text, name)).fetchone()
            if old and old[0] == encoded:
                return int(row[0])
            conn.execute(
                "INSERT INTO lab_documents(lab_id,document_key,payload_json,revision,updated_at) VALUES(?,?,?,0,?) "
                "ON CONFLICT(lab_id,document_key) DO UPDATE SET payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (lab_text, name, encoded, now),
            )
            revision = self._bump_revision(conn, lab_text)
            conn.execute("UPDATE lab_documents SET revision=? WHERE lab_id=? AND document_key=?",
                         (revision, lab_text, name))
            return revision

    def _document_key(self, lab_id: str, key: str | Path) -> str:
        value = str(key).replace("\\", "/")
        base = Path(value).name.lower()
        known = {"lab.json": "config", "config": "config", "state.json": "state", "state": "state",
                 "catalog.json": "catalog", "catalog": "catalog", "history.json": "history", "history": "history",
                 "candidate.json": "candidate", "candidate": "candidate", "worker.json": "worker", "worker": "worker"}
        if base in known:
            if Path(value).is_absolute():
                expected = self.get_lab(lab_id)
                allowed = (Path(expected["state_dir"]).resolve(), Path(expected["config_path"]).resolve())
                target = Path(value).resolve()
                if target not in allowed and not target.is_relative_to(allowed[0]):
                    raise ValueError("Document path is outside this lab")
            return known[base]
        if not value or ".." in Path(value).parts:
            raise ValueError("Invalid document key")
        if Path(value).is_absolute():
            expected = self.get_lab(lab_id)
            target = Path(value).resolve()
            state_dir = Path(expected["state_dir"]).resolve()
            try:
                value = target.relative_to(state_dir).as_posix()
            except ValueError as exc:
                raise ValueError("Document path is outside this lab") from exc
        if value.startswith("runs/"):
            parts = value.split("/")
            if len(parts) >= 3 and parts[-1] == "manifest.json":
                return "run:" + parts[1] + ":manifest"
            if len(parts) >= 4 and parts[-1] == "result.json" and parts[-2].startswith("repeat_"):
                return "run:" + parts[1] + ":result:" + parts[-2]
            if len(parts) >= 4 and parts[-1] == "cases.jsonl" and parts[-2].startswith("repeat_"):
                return "run:" + parts[1] + ":cases:" + parts[-2]
            if len(parts) >= 4 and parts[-1] == "attempts.jsonl" and parts[-2].startswith("repeat_"):
                return "run:" + parts[1] + ":attempts:" + parts[-2]
            if len(parts) >= 4 and parts[-1] == "usage.json" and parts[-2].startswith("repeat_"):
                return "run:" + parts[1] + ":usage:" + parts[-2]
        return value

    def load_catalog(self, root_ref: str | Path | None = None) -> dict[str, Any]:
        root_text = str(root_ref) if root_ref is not None else "global"
        if not self.is_initialized():
            return {"root_ref": root_text, "projects": []}
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT c.project_ref,c.lab_id,c.updated_at,l.state_dir,l.source_root,l.goal,l.primary_metric,l.status,l.revision,"
                "CASE WHEN a.lab_id=c.lab_id THEN 1 ELSE 0 END AS active "
                "FROM catalog_metadata c JOIN labs l ON l.lab_id=c.lab_id "
                "LEFT JOIN active_labs a ON a.root_ref=c.root_ref AND a.project_ref=c.project_ref "
                "WHERE c.root_ref=? ORDER BY c.project_ref",
                (root_text,),
            ).fetchall()
            if not rows and root_text == "global":
                rows = conn.execute(
                    "SELECT source_root,lab_id,updated_at,state_dir,source_root,goal,primary_metric,status,revision,0 "
                    "FROM labs ORDER BY created_at DESC,lab_id DESC LIMIT ?",
                    (MAX_PAGE_SIZE,),
                ).fetchall()
        return {"root_ref": root_text, "projects": [
            {"project_ref": row[0], "lab_id": uuid.UUID(row[1]), "active": bool(row[9]),
             "state_dir": row[3], "source_root": row[4], "goal": row[5], "primary_metric": row[6],
             "status": row[7], "revision": row[8], "updated_at": row[2]}
            for row in rows
        ]}

    def save_catalog(self, payload: dict[str, Any], root_ref: str | Path | None = None) -> int:
        if not isinstance(payload, dict):
            raise TypeError("catalog payload must be a dictionary")
        root_text = str(root_ref) if root_ref is not None else str(payload.get("root_ref") or "global")
        projects = payload.get("projects", [])
        if not isinstance(projects, list):
            raise ValueError("catalog projects must be a list")
        now = _now()
        with self._writer() as conn:
            conn.execute("DELETE FROM catalog_metadata WHERE root_ref=?", (root_text,))
            for item in projects:
                if not isinstance(item, dict):
                    continue
                project_ref = str(item.get("project_ref") or item.get("id") or item.get("path") or "")
                if not project_ref:
                    continue
                lab_id = None
                if item.get("lab_id"):
                    lab_id = self._resolve_lab_in_conn(conn, item["lab_id"])
                elif item.get("source_root"):
                    found = conn.execute("SELECT lab_id FROM labs WHERE source_root=? ORDER BY created_at DESC LIMIT 1",
                                         (_path_alias(item["source_root"]),)).fetchone()
                    lab_id = found[0] if found else None
                if lab_id is None:
                    continue
                conn.execute(
                    "INSERT INTO catalog_metadata(root_ref,project_ref,lab_id,updated_at) VALUES(?,?,?,?)",
                    (root_text, project_ref, lab_id, now),
                )
            if isinstance(root_ref, Path):
                self._register_root_in_conn(conn, root_text, str(root_ref), {})
            return self._bump_revision(conn)

    def set_active_lab(self, root_ref: str | Path, project_ref: str, lab_id: uuid.UUID | str) -> None:
        root_text = str(root_ref)
        lab_text = self._lab_text(lab_id)
        now = _now()
        with self._writer() as conn:
            if conn.execute("SELECT 1 FROM labs WHERE lab_id=?", (lab_text,)).fetchone() is None:
                raise EntityNotFoundError(f"Lab not found: {lab_text}")
            conn.execute(
                "INSERT INTO catalog_metadata(root_ref,project_ref,lab_id,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(root_ref,project_ref) DO UPDATE SET lab_id=excluded.lab_id,updated_at=excluded.updated_at",
                (root_text, project_ref, lab_text, now),
            )
            conn.execute(
                "INSERT INTO active_labs(root_ref,project_ref,lab_id,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(root_ref,project_ref) DO UPDATE SET lab_id=excluded.lab_id,updated_at=excluded.updated_at",
                (root_text, project_ref, lab_text, now),
            )
            self._bump_revision(conn, lab_text)

    def get_active_lab(self, root_ref: str | Path, project_ref: str) -> uuid.UUID | None:
        with self._reader() as conn:
            row = conn.execute(
                "SELECT lab_id FROM active_labs WHERE root_ref=? AND project_ref=?",
                (str(root_ref), project_ref),
            ).fetchone()
        return uuid.UUID(row[0]) if row and row[0] else None

    @staticmethod
    def _register_root_in_conn(conn: sqlite3.Connection, root_ref: str, path: str, payload: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO roots(root_ref,path,created_at,payload_json) VALUES(?,?,?,?) "
            "ON CONFLICT(root_ref) DO UPDATE SET path=excluded.path,payload_json=excluded.payload_json",
            (root_ref, _path_alias(path), _now(), _json(payload)),
        )

    def register_root(self, root_ref: str, path: str | Path, payload: dict[str, Any] | None = None) -> None:
        if not root_ref:
            raise ValueError("root_ref is required")
        with self._writer() as conn:
            self._register_root_in_conn(conn, root_ref, str(path), payload or {})
            self._bump_revision(conn)

    def load_roots(self) -> list[dict[str, Any]]:
        with self._reader() as conn:
            rows = conn.execute("SELECT root_ref,path,created_at,payload_json FROM roots ORDER BY root_ref").fetchall()
        return [{"id": row[0], "path": row[1], "created_at": row[2], "metadata": _loads(row[3], {})} for row in rows]

    @staticmethod
    def _resolve_lab_in_conn(conn: sqlite3.Connection, value: uuid.UUID | str | Path) -> str:
        text = str(value)
        try:
            canonical = _uuid_text(text, label="lab_id")
            found = conn.execute("SELECT lab_id FROM labs WHERE lab_id=?", (canonical,)).fetchone()
            if found:
                return canonical
        except ValueError:
            pass
        alias = _path_alias(text) if Path(text).is_absolute() or text.endswith("lab.json") else text
        rows = conn.execute("SELECT lab_id FROM aliases WHERE kind='lab' AND scope_id=? AND alias=?",
                            (_NIL_UUID, alias)).fetchall()
        if not rows:
            raise EntityNotFoundError(f"Lab not found: {text}")
        if len({row[0] for row in rows}) > 1:
            raise AmbiguousAliasError(f"Lab alias is ambiguous: {text}")
        return str(rows[0][0])

    def register_pin(
        self,
        lab_id: uuid.UUID | str,
        name: str,
        payload: dict[str, Any],
        source_path: str | Path | None = None,
        sha256: str | None = None,
    ) -> None:
        lab_text = self._lab_text(lab_id)
        if not name or not isinstance(payload, dict):
            raise ValueError("pin requires a name and object payload")
        encoded = _json(payload)
        digest = sha256 or _sha256(encoded.encode("utf-8"))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("pin sha256 must be a 64-character hexadecimal digest")
        now = _now()
        with self._writer() as conn:
            previous = conn.execute("SELECT sha256,payload_json FROM pins WHERE lab_id=? AND name=?", (lab_text, name)).fetchone()
            if previous and previous[0] == digest and previous[1] == encoded:
                return
            conn.execute(
                "INSERT INTO pins(lab_id,name,sha256,source_path,payload_json,created_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(lab_id,name) DO UPDATE SET sha256=excluded.sha256,source_path=excluded.source_path,"
                "payload_json=excluded.payload_json,created_at=excluded.created_at",
                (lab_text, name, digest.lower(), str(source_path) if source_path is not None else None, encoded, now),
            )
            self._bump_revision(conn, lab_text)

    def load_pins(self, lab_id: uuid.UUID | str) -> list[dict[str, Any]]:
        lab_text = self._lab_text(lab_id)
        with self._reader() as conn:
            rows = conn.execute("SELECT name,sha256,source_path,payload_json,created_at FROM pins WHERE lab_id=? ORDER BY name",
                                (lab_text,)).fetchall()
        return [{"name": row[0], "sha256": row[1], "source_path": row[2],
                 "payload": _loads(row[3], {}), "created_at": row[4]} for row in rows]

    def register_artifact(
        self,
        lab_id: uuid.UUID | str,
        name: str,
        content: bytes,
        *,
        run_id: uuid.UUID | str | None = None,
        evidence_key: str | None = None,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lab_text = self._lab_text(lab_id)
        if not name or not isinstance(content, bytes):
            raise TypeError("artifact requires a name and bytes content")
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            raise TypeError("artifact metadata must be an object")
        encoded_metadata = _json(metadata)
        digest = _sha256(content)
        artifact_id = uuid.uuid4()
        run_text = self._run_text(lab_text, run_id) if run_id is not None else None
        object_rel = None
        inline = sqlite3.Binary(content) if len(content) <= INLINE_ARTIFACT_LIMIT else None
        now = _now()
        with self._writer() as conn:
            if inline is None:
                object_rel, written_digest = self._write_object(content, conn=conn)
                if written_digest != digest:
                    raise ArtifactIntegrityError("Materialized artifact hash changed")
            conn.execute(
                "INSERT INTO artifacts(artifact_id,lab_id,run_id,name,evidence_key,media_type,sha256,size_bytes,"
                "inline_data,object_path,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(artifact_id), lab_text, run_text, name, evidence_key, media_type, digest, len(content),
                 inline, object_rel, encoded_metadata, now),
            )
            self._bump_revision(conn, lab_text)
        return {"artifact_id": artifact_id, "lab_id": uuid.UUID(lab_text),
                "run_id": uuid.UUID(run_text) if run_text else None, "name": name,
                "evidence_key": evidence_key, "media_type": media_type, "sha256": digest,
                "size_bytes": len(content), "object_path": object_rel, "created_at": now,
                "metadata": metadata}

    def load_artifact(self, artifact_id: uuid.UUID | str) -> bytes:
        artifact_text = _uuid_text(artifact_id, label="artifact_id")
        with self._reader() as conn:
            row = conn.execute("SELECT sha256,size_bytes,inline_data,object_path FROM artifacts WHERE artifact_id=?",
                               (artifact_text,)).fetchone()
        if row is None:
            raise EntityNotFoundError(f"Artifact not found: {artifact_text}")
        if row["inline_data"] is not None:
            data = bytes(row["inline_data"])
        else:
            relative = Path(row["object_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ArtifactIntegrityError("Artifact object path is unsafe")
            unresolved = self.home / relative
            probe = self.home
            for part in relative.parts:
                probe = probe / part
                if _is_link_or_reparse(probe):
                    raise ArtifactIntegrityError("Artifact path contains a symlink or reparse point")
            try:
                path = unresolved.resolve(strict=True)
            except OSError as exc:
                raise ArtifactIntegrityError(f"Artifact object is unavailable: {exc}") from exc
            if not path.is_relative_to(self._object_root.resolve()) or not path.is_file():
                raise ArtifactIntegrityError("Artifact object path is outside the content store")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise ArtifactIntegrityError(f"Artifact object is unavailable: {exc}") from exc
        if len(data) != row["size_bytes"] or _sha256(data) != row["sha256"]:
            raise ArtifactIntegrityError("Artifact bytes do not match their recorded size and SHA-256")
        return data

    def get_artifact(self, lab_id: uuid.UUID | str, artifact_id: uuid.UUID | str) -> dict[str, Any] | None:
        lab_text = self._lab_text(lab_id)
        artifact_text = _uuid_text(artifact_id, label="artifact_id")
        with self._reader() as conn:
            row = conn.execute(
                "SELECT artifact_id,lab_id,run_id,name,evidence_key,media_type,sha256,size_bytes,object_path,metadata_json,created_at "
                "FROM artifacts WHERE lab_id=? AND artifact_id=?",
                (lab_text, artifact_text),
            ).fetchone()
        if row is None:
            return None
        return {"artifact_id": uuid.UUID(row[0]), "lab_id": uuid.UUID(row[1]),
                "run_id": uuid.UUID(row[2]) if row[2] else None, "name": row[3], "evidence_key": row[4],
                "media_type": row[5], "sha256": row[6], "size_bytes": row[7], "object_path": row[8],
                "metadata": _loads(row[9], {}), "created_at": row[10]}

    def list_artifacts(
        self,
        lab_id: uuid.UUID | str,
        run_id: uuid.UUID | str | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List bounded artifact metadata; callers fetch bytes through load_artifact."""
        limit, offset = _safe_page(limit, offset)
        lab_text = self._lab_text(lab_id)
        clauses = ["lab_id=?"]
        params: list[Any] = [lab_text]
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(self._run_text(lab_text, run_id))
        params.extend((limit, offset))
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT artifact_id,lab_id,run_id,name,evidence_key,media_type,sha256,size_bytes,object_path,metadata_json,created_at "
                "FROM artifacts WHERE " + " AND ".join(clauses) +
                " ORDER BY created_at DESC,artifact_id DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [
            {"artifact_id": uuid.UUID(row[0]), "lab_id": uuid.UUID(row[1]),
             "run_id": uuid.UUID(row[2]) if row[2] else None, "name": row[3], "evidence_key": row[4],
             "media_type": row[5], "sha256": row[6], "size_bytes": row[7], "object_path": row[8],
             "metadata": _loads(row[9], {}), "created_at": row[10]}
            for row in rows
        ]

__all__ = [
    "SQLiteStore", "StoreError", "StoreNotInitializedError", "StoreCorruptError", "StoreVersionError",
    "StoreSchemaError", "StoreBusyError", "StoreIntegrityError", "StoreConflictError",
    "RevisionConflictError", "IdempotencyConflictError", "ActiveOperationError", "OperationStateError",
    "AmbiguousAliasError", "EntityNotFoundError", "StoreGateError", "ArtifactIntegrityError",
]
