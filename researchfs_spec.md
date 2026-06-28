# ResearchFS v0 Core Spec

We are implementing a lightweight database-backed reference and logging package called `researchfs` using SQLite. This package will store research papers, chunks, methods, and experiment history.

## 1. Directory Structure
```text
researchfs/
  __init__.py
  db.py          # Database connection, cursor helpers, and schema creation
  ingest.py      # Database seeder script
  retrieve.py    # Methods for retrieving context packs/briefs and logging metrics
```

---

## 2. SQLite Database Schema
The database should be saved at the repository root as `researchfs.db`.
Create the following tables (refer to `design_spec.md` for column guidelines):
*   `papers`: Ingested papers (title, authors, abstract, url, publish_date).
*   `chunks`: Paragraph-level text chunks linked to papers.
*   `methods`: Extracted research moves (name, description, implementation_hint, risks).
*   `method_sources`: M-to-M mapping of `methods` to `chunks` for citation tracking.
*   `experiment_briefs`: Experiment scope configurations (name, objective, constraints, hypotheses).
*   `experiments`: Ratchet loop outcomes (commit_hash, val_bpb, memory_gb, status, description, timestamp).
*   `retrieval_events`: Audit logs of retrieved content (experiment_id, event_type, query, retrieved_content, timestamp).
*   `token_usage`: Cost and token tracking per step (experiment_id, prompt_tokens, completion_tokens, cost, timestamp).

---

## 3. Module Requirements

### `researchfs/db.py`
*   Define a helper `get_connection()` that returns a `sqlite3.Connection` to `researchfs.db`.
*   Implement `initialize_db()` which creates all tables if they do not exist. Call this automatically on package import or setup.

### `researchfs/ingest.py`
*   Create a seeding function `seed_initial_data()` that populates:
    *   Papers: Abstracts/metadata for the GQA, SwiGLU, and Muon papers.
    *   Chunks: Actual text chunks from the papers detailing implementation.
    *   Methods: Entries for GQA, SwiGLU, and Muon, mapping them to the source chunks in `method_sources`.
    *   Experiment Briefs: The default objective and hypotheses (based on the current `experiment_brief.md`).
*   Include a `__main__` entry point to allow running: `python -m researchfs.ingest`.

### `researchfs/retrieve.py`
*   `get_method_pack()`: Queries all `methods` and their source `chunks` (for provenance) and formats them into a clean string to be injected into the agent prompt.
*   `get_experiment_brief(name)`: Queries the target brief and returns it as a string.
*   `log_experiment(commit_hash, val_bpb, memory_gb, status, description)`: Inserts a run log and returns the `experiment_id`.
*   `log_retrieval_event(experiment_id, event_type, query, retrieved_content)`: Logs a retrieval event.
*   `log_token_usage(experiment_id, prompt_tokens, completion_tokens, cost)`: Logs token usage.
