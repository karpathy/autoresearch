"""Aging instrumentation probe for credit-card-fastapi (finding bax-fastapi-credit-card-F1).

Measurement code only: it must not change application behavior. Everything is gated on
AGING_INSTRUMENT=1; when unset, every hook returns immediately.

Output: JSONL at $AGING_INSTRUMENT_LOG
  {"ts", "kind": "start", "probes": [...]}
  {"ts", "kind": "snapshot", "metrics": {...}}   every $AGING_SNAPSHOT_INTERVAL s (default 10),
                                                 plus one at start and one at shutdown
  {"ts", "kind": "event", "probe": ..., "fields": {...}}   rare paths, <= 1/s per probe

State is bounded: fixed scalar counters and windowed maxima reset at every snapshot. No
containers keyed by request data. All hooks swallow their own exceptions.

DB gauges: the app has no lock of its own (one sqlite3 connection per request, WAL mode), so
the snapshot thread opens its own short-lived READ-ONLY connection (URI mode=ro), runs a few
index-only aggregate queries inside one read transaction, and closes it. mode=ro matters:
a read-only connection never checkpoints or deletes the -wal on close, so the probe does not
alter the app's WAL lifecycle. While the read transaction is open (snapshot.query_ms) a
concurrent checkpoint cannot reset the WAL past the probe's read mark; that window is short
and measured.
"""
import json
import os
import sqlite3
import threading
import time

ENABLED = os.environ.get("AGING_INSTRUMENT") == "1"

try:
    _INTERVAL = float(os.environ.get("AGING_SNAPSHOT_INTERVAL", "10"))
    if _INTERVAL <= 0:
        _INTERVAL = 10.0
except ValueError:
    _INTERVAL = 10.0

_BUSY_TIMEOUT_MS = 5000

COUNTERS = (
    "calls.associate_card",
    "associate.inserted",            # INSERT OR IGNORE with rowcount == 1 (new distinct pair)
    "associate.ignored",             # rowcount == 0 (duplicate pair, nothing accumulates)
    "calls.retrieve_cards",
    "retrieve.rows_returned.total",  # cumulative len(rows) of /retrieve_cards
    "retrieve.not_found",            # len(rows) == 0 -> 404
    "errors.validation_400",
)
WINDOW_MAX = (
    "associate.db_ms.max",           # connect + INSERT OR IGNORE + commit + close
    "retrieve.rows.max",             # max len(rows) returned by one /retrieve_cards query
    "retrieve.phones.max",           # max number of (deduplicated) phones in one request
    "retrieve.db_ms.max",            # connect + execute + fetchall + commit + close
    "retrieve.handler_ms.max",       # handler body incl. response model build (excl. JSON encoding)
)
WINDOW_COUNT = (
    "window.calls.associate_card",
    "window.associate.inserted",
    "window.calls.retrieve_cards",
)
GAUGES = (
    "rows.associations",
    "rows.sqlite_sequence",
    "rows.total",
    "seq.associations",              # AUTOINCREMENT high-water mark (never decreases)
    "cards.distinct",
    "phones.distinct",
    "rows.max_per_phone",            # max cards associated with one phone (hot-phone fan-out)
    "db.bytes",                      # db.sqlite3 + -journal + -wal + -shm
    "db.main_bytes",
    "db.wal_bytes",
    "db.shm_bytes",
    "db.page_count",
    "db.freelist_count",
    "db.page_size",
    "snapshot.query_ms",
    "snapshot.db_error",
    "snapshot.seq",
    "snapshot.interval_s",
)
EVENTS = ("validation_error", "probe_error")

_lock = threading.Lock()           # protects the scalars below (held for microseconds)
_counters = dict.fromkeys(COUNTERS, 0)
_wmax = dict.fromkeys(WINDOW_MAX, 0.0)
_wcount = dict.fromkeys(WINDOW_COUNT, 0)
_last_event = dict.fromkeys(EVENTS, 0.0)

_out_lock = threading.Lock()
_out = None
_db_path = None
_started = False
_stop = threading.Event()
_thread = None
_seq = 0


# --------------------------------------------------------------------------- output
def _write(obj) -> None:
    if _out is None:
        return
    line = json.dumps(obj, separators=(",", ":"))
    with _out_lock:
        try:
            _out.write(line + "\n")
            _out.flush()
        except Exception:
            pass


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _max(name: str, value: float) -> None:
    if value > _wmax[name]:
        _wmax[name] = value


# --------------------------------------------------------------------------- hooks
def t0() -> float:
    """Timestamp (ms) taken right before a suspect path starts; 0.0 when disabled."""
    if not ENABLED:
        return 0.0
    return _now_ms()


def associate_done(rowcount: int, t_start: float) -> None:
    """After the `with db_conn()` block of associate_card (insert committed, conn closed)."""
    if not ENABLED:
        return
    try:
        dt = _now_ms() - t_start
        with _lock:
            _counters["calls.associate_card"] += 1
            _wcount["window.calls.associate_card"] += 1
            if rowcount and rowcount > 0:
                _counters["associate.inserted"] += 1
                _wcount["window.associate.inserted"] += 1
            else:
                _counters["associate.ignored"] += 1
            _max("associate.db_ms.max", dt)
    except Exception:
        pass


def retrieve_db_done(n_phones: int, n_rows: int, t_start: float) -> None:
    """After the `with db_conn()` block of retrieve_cards (rows fetched, conn closed)."""
    if not ENABLED:
        return
    try:
        dt = _now_ms() - t_start
        with _lock:
            _counters["calls.retrieve_cards"] += 1
            _wcount["window.calls.retrieve_cards"] += 1
            _counters["retrieve.rows_returned.total"] += n_rows
            if n_rows == 0:
                _counters["retrieve.not_found"] += 1
            _max("retrieve.rows.max", float(n_rows))
            _max("retrieve.phones.max", float(n_phones))
            _max("retrieve.db_ms.max", dt)
    except Exception:
        pass


def retrieve_handler_done(t_start: float) -> None:
    if not ENABLED:
        return
    try:
        dt = _now_ms() - t_start
        with _lock:
            _max("retrieve.handler_ms.max", dt)
    except Exception:
        pass


def event(probe: str, **fields) -> None:
    """Rare-path event, rate-limited to 1/s per probe name (fixed set of names)."""
    if not ENABLED:
        return
    try:
        if probe == "validation_error":
            with _lock:
                _counters["errors.validation_400"] += 1
        if probe not in _last_event:
            return
        now = time.time()
        with _lock:
            if now - _last_event[probe] < 1.0:
                return
            _last_event[probe] = now
        _write({"ts": now, "kind": "event", "probe": probe,
                "fields": {k: (v if isinstance(v, (int, float)) else str(v)[:200])
                           for k, v in fields.items()}})
    except Exception:
        pass


# --------------------------------------------------------------------------- snapshot
def _file_gauges(m: dict) -> None:
    total = 0
    for suffix, key in (("", "db.main_bytes"), ("-journal", None), ("-wal", "db.wal_bytes"),
                        ("-shm", "db.shm_bytes")):
        try:
            size = os.path.getsize(_db_path + suffix)
        except OSError:
            size = 0
        total += size
        if key:
            m[key] = size
    m["db.bytes"] = total


def _db_gauges(m: dict) -> None:
    if not os.path.exists(_db_path):
        return
    t_start = _now_ms()
    conn = sqlite3.connect("file:" + _db_path + "?mode=ro", uri=True,
                           timeout=_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None,
                           check_same_thread=False)
    try:
        conn.execute("BEGIN")  # one consistent read snapshot for all gauges
        try:
            # Each query is answered from an existing index (see EXPLAIN QUERY PLAN in the
            # report): COUNT(*) -> smallest covering index; GROUP BY phone -> idx_assoc_phone;
            # COUNT(DISTINCT credit_card) -> idx_assoc_card.
            n_assoc = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(c), 0) FROM "
                "(SELECT COUNT(*) AS c FROM associations GROUP BY phone)"
            ).fetchone()
            m["phones.distinct"] = row[0]
            m["rows.max_per_phone"] = row[1]
            m["cards.distinct"] = conn.execute(
                "SELECT COUNT(DISTINCT credit_card) FROM associations").fetchone()[0]
            n_seq_rows = conn.execute("SELECT COUNT(*) FROM sqlite_sequence").fetchone()[0]
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM sqlite_sequence WHERE name='associations'"
            ).fetchone()[0]
            m["db.page_count"] = conn.execute("PRAGMA page_count").fetchone()[0]
            m["db.freelist_count"] = conn.execute("PRAGMA freelist_count").fetchone()[0]
            m["db.page_size"] = conn.execute("PRAGMA page_size").fetchone()[0]
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()
    m["rows.associations"] = n_assoc
    m["rows.sqlite_sequence"] = n_seq_rows
    m["rows.total"] = n_assoc + n_seq_rows
    m["seq.associations"] = seq
    m["snapshot.query_ms"] = _now_ms() - t_start


def snapshot() -> None:
    global _seq
    if not _started:
        return
    try:
        with _lock:
            m = dict(_counters)
            m.update(_wmax)
            m.update(_wcount)
            for k in _wmax:
                _wmax[k] = 0.0
            for k in _wcount:
                _wcount[k] = 0
            _seq += 1
            m["snapshot.seq"] = _seq
        m["snapshot.interval_s"] = _INTERVAL
        m["snapshot.db_error"] = 0
        _file_gauges(m)
        try:
            _db_gauges(m)
        except Exception as e:  # e.g. table not yet created, busy
            m["snapshot.db_error"] = 1
            event("probe_error", where="db_gauges", error=repr(e))
        _write({"ts": time.time(), "kind": "snapshot", "metrics": m})
    except Exception:
        pass


def _loop() -> None:
    while not _stop.wait(_INTERVAL):
        snapshot()


def start(db_path: str) -> None:
    """Called from the app's startup path (after init_db). Opens the log, writes the start
    line, takes the baseline snapshot and starts the daemon snapshot thread."""
    global _out, _db_path, _started, _thread
    if not ENABLED or _started:
        return
    try:
        log_path = os.environ.get("AGING_INSTRUMENT_LOG") or "aging_instrument.jsonl"
        _out = open(log_path, "a", buffering=1, encoding="utf-8")
        _db_path = os.path.abspath(db_path)
        _started = True
        _write({"ts": time.time(), "kind": "start",
                "probes": list(COUNTERS + WINDOW_MAX + WINDOW_COUNT + GAUGES)
                + ["event." + e for e in EVENTS],
                "pid": os.getpid(), "interval_s": _INTERVAL, "db_path": _db_path})
        snapshot()
        _thread = threading.Thread(target=_loop, name="aging-probe-snapshot", daemon=True)
        _thread.start()
    except Exception:
        pass


def stop() -> None:
    """Called from the app's shutdown path: final snapshot, then stop the thread."""
    if not ENABLED or not _started:
        return
    try:
        _stop.set()
        snapshot()
    except Exception:
        pass
