"""Install the NanoChat-specific paper/method fixture into ResearchFS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import researchfs.db as db_module
from researchfs.bundle import import_corpus_memory_bundle
from researchfs.db import initialize_db, migrate_db


BENCHMARK_ROOT = Path(__file__).resolve().parent
SEED_BUNDLE_PATH = BENCHMARK_ROOT / "fixtures" / "seed.bundle.json"
EXPERIMENT_BRIEF_PATH = BENCHMARK_ROOT / "docs" / "experiment_brief.md"
EXPERIMENT_BRIEF_NAME = "nanochat_autoresearch"


def seed_initial_data(conn=None) -> None:
    """Populate an initialized database with the optional NanoChat fixture."""
    close = conn is None
    if conn is None:
        initialize_db()
        conn = db_module.get_connection()
    else:
        initialize_db(conn)
    try:
        migrate_db(conn)
        bundle = json.loads(SEED_BUNDLE_PATH.read_text(encoding="utf-8"))
        import_corpus_memory_bundle(bundle, conn=conn, commit=False)
        conn.execute(
            """
            INSERT INTO experiment_briefs (name, objective, constraints, hypotheses, content)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                objective = excluded.objective,
                constraints = excluded.constraints,
                hypotheses = excluded.hypotheses,
                content = excluded.content
            """,
            (
                EXPERIMENT_BRIEF_NAME,
                "Minimize validation bits per byte (val_bpb) within a fixed training time budget.",
                "Single device; no new dependencies; do not edit prepare.py; time budget enforced after step 10.",
                "SwiGLU, GQA, Muon learning-rate tuning, attention-window patterns, normalization, "
                "regularization, value embeddings, and token shifting are controlled hypotheses.",
                EXPERIMENT_BRIEF_PATH.read_text(encoding="utf-8"),
            ),
        )
        conn.commit()
    finally:
        if close:
            conn.close()


def ensure_seed(conn=None) -> None:
    """Install the fixture only when the target database has no methods."""
    close = conn is None
    if conn is None:
        initialize_db()
        conn = db_module.get_connection()
    try:
        if conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] == 0:
            seed_initial_data(conn)
    finally:
        if close:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=None, help="ResearchFS SQLite database path")
    args = parser.parse_args(argv)
    conn = db_module.get_connection(args.db_path)
    try:
        seed_initial_data(conn)
    finally:
        conn.close()
    print(f"Seeded NanoChat fixture into {args.db_path or 'researchfs.db'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
