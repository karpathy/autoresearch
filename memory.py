"""
SQLite-backed experiment memory.

Functions are organized in flow order:
1. Connection + table creation
2. Storage (save / retrieve)
3. Normalizing hyperparameters
4. Cosine similarity retrieval
5. Confidence calculation
6. LLM conflict resolution
7. Accept / Reject verdict flow
"""

import json
import math
import os
import requests
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from schema import ExperimentRecord, Verdict, VerdictSnapshot

# ---------------------------------------------------------------------------
# 1. Connection + table creation  (already reviewed)
# ---------------------------------------------------------------------------

def connect(db_path: str = "memory.db") -> sqlite3.Connection:
    """Open (or create) the experiment memory database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,   -- JSON blob
            confidence      REAL,
            llm_used        INTEGER NOT NULL DEFAULT 0,
            last_verdict    TEXT CHECK(last_verdict IN ('ACCEPT', 'REJECT'))
        );

        CREATE TABLE IF NOT EXISTS verdict_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id   TEXT    NOT NULL
                            REFERENCES experiments(id) ON DELETE CASCADE,
            recorded_at     TEXT    NOT NULL,
            verdict         TEXT    NOT NULL CHECK(verdict IN ('ACCEPT', 'REJECT')),
            val_bpb         REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_experiment
            ON verdict_snapshots(experiment_id);
    """)
    conn.commit()


def update_experiment(conn: sqlite3.Connection, record: ExperimentRecord) -> None:
    conn.execute(
        """
        UPDATE experiments
        SET confidence = ?,
            last_verdict = ?,
            llm_used = ?
        WHERE id = ?
        """,
        (
            record.confidence,
            record.last_verdict.value if record.last_verdict else None,
            1 if record.llm_used else 0,
            record.id,
        )
    )
    conn.commit()


def save_experiment(conn: sqlite3.Connection, record: ExperimentRecord) -> None:
    conn.execute(
        """
        INSERT INTO experiments (
            id, created_at, hyperparameters, confidence, llm_used, last_verdict
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.created_at.isoformat(),
            json.dumps(record.hyperparameters),
            record.confidence,
            1 if record.llm_used else 0,
            record.last_verdict.value if record.last_verdict else None
        )
    )
    conn.commit()


def save_snapshot(conn: sqlite3.Connection, snapshot: VerdictSnapshot) -> None:
    conn.execute(
        """
        INSERT INTO verdict_snapshots (
            experiment_id, recorded_at, verdict, val_bpb
        ) VALUES (?, ?, ?, ?)
        """,
        (
            snapshot.experiment_id,
            snapshot.recorded_at.isoformat(),
            snapshot.verdict.value,
            snapshot.val_bpb
        )
    )
    conn.commit()


def _row_to_record(row) -> ExperimentRecord:
    return ExperimentRecord(
        id=row[0],
        created_at=datetime.fromisoformat(row[1]),
        hyperparameters=json.loads(row[2]),
        confidence=row[3],
        llm_used=bool(row[4]),
        last_verdict=Verdict(row[5]) if row[5] else None
    )


def get_experiment(conn: sqlite3.Connection, experiment_id: str) -> Optional[ExperimentRecord]:
    cursor = conn.execute(
        """
        SELECT id, created_at, hyperparameters, confidence, llm_used, last_verdict
        FROM experiments
        WHERE id = ?
        """,
        (experiment_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def get_all_experiments(conn: sqlite3.Connection) -> list[ExperimentRecord]:
    cursor = conn.execute(
        """
        SELECT id, created_at, hyperparameters, confidence, llm_used, last_verdict
        FROM experiments
        ORDER BY created_at ASC
        """
    )
    return [_row_to_record(row) for row in cursor.fetchall()]


def compute_stats(experiments: list[ExperimentRecord]) -> dict[str, tuple[float, float]]:
    values_by_key = {}
    for exp in experiments:
        for k, v in exp.hyperparameters.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k not in values_by_key:
                    values_by_key[k] = []
                values_by_key[k].append(float(v))

    stats = {}
    for k, vals in values_by_key.items():
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
            
        if std == 0.0:
            std = 1.0
            
        stats[k] = (mean, std)
        
    return stats


def normalize(hyperparameters: dict, stats: dict[str, tuple[float, float]]) -> dict:
    normalized = {}
    for k, v in hyperparameters.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k in stats:
                mean, std = stats[k]
                normalized[k] = (float(v) - mean) / std
            else:
                normalized[k] = float(v)
        else:
            normalized[k] = v
    return normalized


def cosine_similarity(dict1: dict, dict2: dict) -> float:
    """Compute cosine similarity between two normalized hyperparameter dicts.
    Returns a float between -1.0 and 1.0.
    """
    keys = set(dict1.keys()).union(dict2.keys())
    
    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for k in keys:
        v1 = dict1.get(k)
        v2 = dict2.get(k)
        
        val1 = float(v1) if isinstance(v1, (int, float)) and not isinstance(v1, bool) else 0.0
        val2 = float(v2) if isinstance(v2, (int, float)) and not isinstance(v2, bool) else 0.0
        
        dot_product += val1 * val2
        norm1_sq += val1 ** 2
        norm2_sq += val2 ** 2
        
    if norm1_sq == 0.0 or norm2_sq == 0.0:
        return 0.0
        
    return dot_product / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))


def retrieve_similar(
    conn: sqlite3.Connection,
    query_hyperparameters: dict,
    k: int = 5
) -> list[tuple[ExperimentRecord, float]]:
    """
    Retrieve top-k most similar past experiments.
    Returns a list of tuples (ExperimentRecord, similarity_score) sorted descending by similarity.
    """
    all_experiments = get_all_experiments(conn)
    if not all_experiments:
        return []

    stats = compute_stats(all_experiments)
    query_norm = normalize(query_hyperparameters, stats)

    results = []
    for exp in all_experiments:
        exp_norm = normalize(exp.hyperparameters, stats)
        sim = cosine_similarity(query_norm, exp_norm)
        results.append((exp, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def resolve_with_llm(
    exp1: ExperimentRecord,
    snap1: VerdictSnapshot,
    exp2: ExperimentRecord,
    snap2: VerdictSnapshot,
) -> Verdict:
    """
    Call an LLM to resolve a contradiction between two highly similar experiments.
    """
    prompt = f"""We have two highly similar experiments with contradicting verdicts.

Experiment 1 (ID: {exp1.id}):
Hyperparameters: {json.dumps(exp1.hyperparameters)}
Validation BPB: {snap1.val_bpb:.4f}
Verdict: {exp1.last_verdict.value if exp1.last_verdict else 'None'}

Experiment 2 (ID: {exp2.id}):
Hyperparameters: {json.dumps(exp2.hyperparameters)}
Validation BPB: {snap2.val_bpb:.4f}
Verdict: {exp2.last_verdict.value if exp2.last_verdict else 'None'}

Reason about which verdict to trust based on these hyperparameters and their Validation BPB metrics (lower is better). Then, provide your final answer as solely 'ACCEPT' or 'REJECT'.
"""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Defaulting to REJECT.")
        return Verdict.REJECT

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an AI research assistant. Output 'ACCEPT' or 'REJECT' as your final word."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip().upper()
        
        if "ACCEPT" in content:
            return Verdict.ACCEPT
        return Verdict.REJECT
    except Exception as e:
        print(f"LLM resolution failed: {e}")
        return Verdict.REJECT


def _get_latest_snapshot(conn: sqlite3.Connection, experiment_id: str) -> Optional[VerdictSnapshot]:
    cursor = conn.execute(
        """
        SELECT experiment_id, recorded_at, verdict, val_bpb
        FROM verdict_snapshots
        WHERE experiment_id = ?
        ORDER BY recorded_at DESC
        LIMIT 1
        """,
        (experiment_id,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return VerdictSnapshot(
        experiment_id=row[0],
        recorded_at=datetime.fromisoformat(row[1]),
        verdict=Verdict(row[2]),
        val_bpb=row[3]
    )


def record_verdict(conn: sqlite3.Connection, experiment_id: str, verdict: Verdict, val_bpb: float) -> None:
    """Orchestrator function called externally to record an experiment's final verdict."""
    exp = get_experiment(conn, experiment_id)
    if not exp:
        return

    snap = VerdictSnapshot(
        experiment_id=experiment_id,
        verdict=verdict,
        val_bpb=val_bpb
    )

    similar_exps = retrieve_similar(conn, exp.hyperparameters, k=5)
    similar_exps = [(e, sim) for e, sim in similar_exps if e.id != experiment_id]

    final_verdict = verdict
    new_confidence = exp.confidence if exp.confidence is not None else 1.0
    llm_was_used = exp.llm_used

    if similar_exps:
        most_similar, sim_score = similar_exps[0]
        
        if sim_score >= 0.90 and most_similar.last_verdict:
            if most_similar.last_verdict == verdict:
                # High similarity + same verdict -> update confidence
                new_confidence = min(1.0, new_confidence + 0.5)
            else:
                # High similarity + opposite verdict -> call LLM to resolve
                past_snap = _get_latest_snapshot(conn, most_similar.id)
                if past_snap:
                    final_verdict = resolve_with_llm(exp, snap, most_similar, past_snap)
                    llm_was_used = True
                    new_confidence = 1.0  # Reset confidence due to contradiction

    exp.last_verdict = final_verdict
    exp.confidence = max(0.0, min(1.0, new_confidence))
    exp.llm_used = llm_was_used
    
    with conn:
        save_snapshot(conn, snap)
        update_experiment(conn, exp)
