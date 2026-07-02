# ResearchFS v0 Core Spec

ResearchFS is a lightweight SQLite-backed paper/lab memory layer for labs and
auto-research agents. It stores research artifact text, compressed method
records, provenance chunks, retrieval events, experiments, and lessons. The
NanoChat autoresearch loop is an evaluation fixture for this memory layer, not
the canonical product surface.

## 1. Directory Structure
```text
researchfs/
  __init__.py
  api.py         # Stdlib local lab API dispatch and optional server entry point
  db.py          # Database connection, cursor helpers, and schema creation
  ingest.py      # Paper ingestion, seed data, chunking/extraction orchestration
  retrieve.py    # Context retrieval, experiment logging, and lesson history
  extractor.py   # Optional LLM extraction helpers
  evaluation.py  # Deterministic expected-method/source/evidence evaluator
  fetcher.py     # Stdlib paper/source fetch helpers
  lab_client.py  # SDK wrapper over the local lab API dispatch surface
  memory.py      # Local compressed-memory query and bundle CLI
  client.py      # DB-backed session client for autoresearch agents
```

---

## 2. SQLite Database Schema
The database should be saved at the repository root as `researchfs.db`.
Create the following tables (refer to `design_spec.md` for column guidelines):
*   `papers`: Ingested papers (title, authors, abstract, url, publish_date).
*   `paper_extraction_status`: Explicit extraction/compression status,
    source labels, character counts, nullable `extractor_model`, nullable
    `extractor_prompt_version`, nullable `confidence_score`, and nullable
    `confidence_reason` for each ingested paper. `source_type` belongs here, not
    on `papers`; local file/API ingestion may preserve labels such as `paper`,
    `lab_note`, `experiment_report`, or `repo_readme` for corpus inspection and
    retrieval provenance. `confidence_score` is a deterministic
    extraction-quality heuristic over observable ResearchFS memory support, not
    verified citation correctness and not an LLM self-confidence claim.
*   `chunks`: Paragraph-level text chunks linked to papers.
*   `methods`: Extracted research moves (name, description, implementation_hint, risks).
*   `method_sources`: M-to-M mapping of `methods` to `chunks` for citation tracking.
    Extraction may process each paper through bounded rolling windows whose
    prompts use local chunk numbers, but persisted links must remap every
    `source_chunk_indices` value to the original paper-wide chunk index before
    insertion. Duplicate extracted methods with the same name should be
    deterministically merged by unioning those remapped global chunk indices.
    Each link may include nullable `evidence_text`, `evidence_start`,
    `evidence_end`, and `evidence_strategy` fields. `evidence_text` is a
    deterministic excerpt from the linked global chunk; it is provenance
    evidence, not a verified PDF-coordinate citation span.
*   `experiment_briefs`: Experiment scope configurations (name, objective, constraints, hypotheses).
*   `agent_runs`, `experiments`, `retrieval_events`, `retrieved_methods`,
    `experiment_methods`, and `token_usage`: Agent run, retrieval, method-use,
    and cost audit logs.
*   `method_feedback`: Durable lab/agent ratings for methods. Each row has
    nullable `event_id`, required `method_id`, `rating` constrained to
    `[-1.0, 1.0]`, optional `label`, `note`, `source`, and timestamp.
*   `lessons`: Experiment lessons linked to methods and/or experiments. Compact
    method-linked lessons, including lessons imported from compressed-memory
    bundles, participate in existing method retrieval through deterministic
    local keyword matching; they are not a vector index or a semantic retrieval
    backend.
*   FTS5 tables for early keyword retrieval over methods, chunks, and code symbols.

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
*   Include a `__main__` entry point to allow running: `uv run python -m researchfs.ingest`.
*   Support local file ingestion with `--file PATH` for `.txt`, `.md`, `.markdown`, `.tex`, and `.pdf` notes. Support `--source-type LABEL` for `--file`, `--arxiv`, and `--repo` ingestion; the CLI must strip whitespace, reject empty labels through argparse, and apply the label to metadata before ingesting. Local file/API ingestion should default unlabeled local files to extraction-status `source_type` `file` and preserve optional metadata labels such as `paper`, `lab_note`, `experiment_report`, or `repo_readme` when supplied. PDF input should be extracted with `pypdf` text extraction only; do not claim OCR, scanned-image parsing, PDF-coordinate citation validation, vector retrieval, or hosted-service behavior.
*   Duplicate paper ingestion should support retrying extraction for the
    existing paper record instead of requiring title changes to create a second
    paper. A retry may fill extraction status, method links, evidence spans, and
    extractor metadata for a paper that was previously ingested without
    extraction, while preserving the same paper identity. Duplicate ingestion
    may also correct `paper_extraction_status.source_type` when the caller
    supplies an explicit source-type override; inferred defaults such as the
    local file `file` label must not relabel an existing paper.
*   Optional LLM extraction must support compressed-memory construction from
    research artifacts beyond deep-learning papers, including papers, lab notes,
    experiment reports, and repository READMEs. Extraction should identify
    specific, actionable methods, architectures, procedures, or operational
    tricks visible in the source chunks without assuming PyTorch-only content.
    Method categories must preserve the canonical backward-compatible tokens
    `attention`, `mlp`, `optimizer`, `normalization`, `precision`,
    `regularization`, and `embeddings`; broader artifact labels such as
    `paper`, `lab_note`, `experiment_report`, and `repo_readme` remain
    extraction-status `source_type` metadata, not method categories. Extraction
    must cover the full chunk sequence through bounded rolling windows, not by
    sending a full artifact in one prompt and not by silently ignoring chunks
    after the first window. Window prompts may number chunks from zero or one
    locally, but any extracted `source_chunk_indices` must be translated to the
    original artifact-wide chunk positions before creating `method_sources`
    rows. Duplicate extracted methods with the same name must deterministically
    union their remapped source chunks. This preserves provenance for methods
    found late in long artifacts while keeping prompts bounded. Extraction
    remains local compressed-memory construction, not vector search, OCR,
    PDF-coordinate citation validation, or a hosted service.

### `researchfs/memory.py`
*   Provide local operational CLI entry points for compressed-memory query,
    extraction-quality evaluation, and bundle movement:
    `uv run python -m researchfs.memory query ...`,
    `uv run python -m researchfs.memory evaluate ...`,
    `uv run python -m researchfs.memory export ...`, and
    `uv run python -m researchfs.memory import ...`.
*   `query` must call the existing `get_research_brief(...)` implementation over
    stored SQLite rows and write that same deterministic brief JSON shape to
    stdout. It accepts `--query TEXT`, optional `--max-chars N`, optional
    `--max-papers N`, repeated `--source-type LABEL`, `--include-scores`,
    `--log`, optional logging metadata `--event-type TYPE`, `--agent-run-id ID`,
    `--experiment-id ID`, and `--db-path PATH`. Invalid arguments or retrieval
    errors must return nonzero and write a machine-readable JSON error object to
    stderr. The command is not ingestion, bundle export/import, an HTTP server,
    hosted sync/auth, vector search, OCR, or an LLM briefing pass.
*   `query` must preserve `/brief`/`ResearchFSLabClient.research_brief(...)`
    parity. Its response includes the existing brief fields (`ok`, `query`,
    `context`, `paper_ids`, `method_ids`, `chunk_ids`, `lesson_ids`,
    `max_chars`, `max_papers`, `source_types`) plus `scores` when requested and
    `event_id` when logged. Returned context must be bounded compressed memory
    from `get_research_brief(...)`, not full raw paper `content` or full
    `chunks.content`.
*   `evaluate` must call the existing
    `researchfs.evaluation.evaluate_extraction(conn, paper_id,
    expected_methods, ...)` helper over stored SQLite rows and write
    `{ok: true, evaluation: ...}` to stdout. It accepts required `--paper-id N`,
    exactly one expected-method source (`--expected-methods JSON` or
    `--expected-methods-file PATH`), optional `--min-recall N`, optional
    `--min-evidence-coverage N`, optional `--notes TEXT`, and `--db-path PATH`.
    The expected-method JSON is the same non-empty list accepted by
    `/evaluate/extraction`: strings or objects with required `name` plus
    optional `source_chunk_ids`, `source_chunk_indices`, `source_contains`, and
    `evidence_contains`. Invalid arguments, invalid JSON, empty expected-method
    lists, and runtime failures must return nonzero and write machine-readable
    JSON errors to stderr. A missing paper must map to error code
    `paper_not_found`. The command is a deterministic local SQLite/stdlib audit
    of stored extraction/compression quality; it must not call an LLM, run OCR,
    perform vector search, repaste full papers, mutate extraction confidence, or
    duplicate the existing evaluator with a second quality implementation.
*   `export` and `import` must call the existing `researchfs.bundle`
    import/export helpers and `researchfs.db` connection/migration helpers. They
    must not duplicate bundle construction, validation, or raw SQLite schema
    setup logic.
*   `export` must require exactly one explicit selection mode: `--paper-id N`
    for one paper, `--paper-ids 1,2,3` for a selected corpus bundle, or `--all`
    for the entire local corpus. It may accept `--max-chars N` to apply the same
    nonnegative per-paper context bound as bundle export helpers.
*   `import` must accept either a paper bundle or corpus bundle and reconstruct
    local compressed memory through `import_memory_bundle(...)`.
*   `query`, `evaluate`, `export`, and `import` must accept `--db-path PATH` to
    choose the local SQLite database. `export` writes bundle JSON to
    `--output PATH` when
    supplied and to stdout when omitted. `import` reads bundle JSON from
    `--input PATH` when supplied and from stdin when omitted, then writes a
    concise machine-readable JSON import result to stdout. Bundle import/export
    is a local file/stdio workflow, not a hosted endpoint.
*   The CLI must preserve the compressed-only contract: reject or surface helper
    rejection for raw paper `content`, full original paper text, or full
    `chunks.content`; do not claim vector search, OCR, PDF-coordinate citation
    verification, authentication, or hosted sync behavior.

### `researchfs/api.py`
*   Provide local stdlib/in-process routes, not a hosted service requirement.
    The stdlib HTTP server is explicitly local-only, has no built-in
    authentication, and should be documented with the loopback command
    `uv run python -m researchfs.api --host 127.0.0.1 --port 8765 --db-path ...`.
*   The stdlib HTTP handler should add CORS headers and support `OPTIONS`
    preflight for local browser clients without changing
    `ResearchFSLabAPI.dispatch` response shapes.
*   The landing page may include a tiny browser smoke-test workflow for this
    local server. The console must default its configurable base URL to
    `http://127.0.0.1:8765`, persist the user-configured value in
    `localStorage`, and send all smoke-test requests to that configured value.
    The local browser workflow may call only existing stdlib API routes in this
    order: `POST /ingest` for direct paper/note metadata, content, and explicit
    method metadata; `POST /retrieve` with `log: true` and
    `include_scores: true`; `GET /method/<id>` for method provenance
    inspection; `POST /feedback` with the latest method ID plus logged retrieval
    event ID; and `POST /evaluate/extraction` using the latest `paper_id` from
    ingest plus `expected_methods` parsed from the console's small JSON textarea
    or, as a fallback, the current method name. It may also use existing
    read-only smoke routes such as `/health`, `/stats`, `POST /brief`, and
    `GET /paper/<id>/memory` for local status or compact paper reuse inspection.
    It must store returned `paper_id`, `method_ids`, and `event_id` in page
    state, display JSON responses or request errors in-page, and require the
    local loopback server command above to be running. It may describe local PDF
    text extraction via file/API/SDK ingestion, but must not imply a hosted
    service, production authentication, OCR or scanned-image PDF parsing,
    PDF-coordinate citation validation, vector search, or any unsupported
    backend route.
*   `GET /`: return a compact API index with `ok`, ResearchFS service/product
    name, local-only caveat, and the implemented local route list.
*   `GET /health`: return `ok: true` plus service and database status suitable
    for local smoke checks.
*   `POST /ingest`: accept in-process paper metadata (`title`, `authors`,
    `abstract`, `content`, `url`, `publish_date`, optional `source_type`, and
    optional extracted `methods`), ingest it into SQLite, store `source_type` on
    `paper_extraction_status` when provided, and return paper/chunk IDs plus
    ingestion metadata.
*   `POST /ingest/arxiv`: accept an arXiv `id`, fetch source metadata/text via
    the configured stdlib fetcher, ingest it, and return the same ingestion
    response shape.
*   `POST /ingest/repo`: accept a GitHub `repo` value (`owner/repo`,
    `owner/repo.git`, or repository-root URL), reject branch/tree/query/fragment
    or extra path components as `400 invalid_repo`,
    fetch README metadata/text via the configured stdlib fetcher, default the
    explicit extraction-status `source_type` to `repo_readme` unless overridden,
    return `400 invalid_repo` for malformed repository input, return
    `502 fetch_failed` for fetch/no-README failures, and return the same
    ingestion response shape.
*   `POST /ingest/file`: accept a local `.txt`, `.md`, `.markdown`, `.tex`, or
    `.pdf` `path` plus optional metadata overrides, including `source_type`
    labels such as `paper`, `lab_note`, `experiment_report`, or `repo_readme`,
    load it through the file metadata path, ingest it, and return the same
    ingestion response shape. PDF support uses `pypdf` text extraction only: no
    OCR, scanned-image parsing, PDF-coordinate citation spans, or hosted
    document service behavior is implied.
*   `GET /papers`: return paper metadata with each paper's extraction status,
    including nullable `confidence_score` and `confidence_reason`.
*   `GET /stats`: return local corpus observability via `counts`,
    `extraction_statuses`, `source_types`, `compression`, `confidence`, and
    `feedback`. `counts` includes `papers`, `chunks`, `methods`, `lessons`,
    `retrieval_events`, and `method_feedback`; `compression` includes
    `paper_count`, `source_char_count`, `compressed_char_count`,
    `average_compression_ratio`, and `overall_compression_ratio`; `confidence`
    includes `count`, `average_score`, `low_confidence_count`, and
    `high_confidence_count` over non-null scores (`< 0.5` low, `>= 0.8` high);
    `feedback` includes the corpus feedback count and average rating.
*   `GET /paper/<id>/memory`: return one deterministic compact paper memory card
    assembled only from stored SQLite rows: paper metadata, extraction status
    including nullable `extractor_model`/`extractor_prompt_version`, confidence
    telemetry, compression metadata when available, linked methods,
    method-source evidence snippets with nullable `evidence_start`,
    `evidence_end`, and `evidence_strategy`, feedback summaries/records, and
    related lessons/experiments. It accepts optional query parameter `max_chars`
    as a nonnegative integer and limits `context` with the existing
    bounded-context semantics. The response includes `ok`, `paper_id`,
    `max_chars`, `context`, `method_ids`, `chunk_ids`, `lesson_ids`, and
    `experiment_ids`, plus compact metadata such as extraction
    status/confidence/compression when available. The `context` is
    human/agent-readable markdown-style text for paper reuse; it is not a fresh
    LLM summary, not raw paper/chunk transport, not a new extraction pass, and
    not a citation verifier.
*   `GET /paper/<id>/memory/export`: return a portable compressed-memory bundle
    for one paper. The bundle is JSON with `kind`, `version`, exported paper
    metadata, extraction status metadata including nullable `extractor_model`
    and `extractor_prompt_version`, bounded `GET /paper/<id>/memory` `context`,
    method metadata, source IDs, method-source provenance evidence snippets and
    nullable `evidence_start`/`evidence_end`/`evidence_strategy` metadata,
    compact method-linked lessons, extraction/confidence/compression metadata,
    and optional feedback summaries. Each exported lesson is a compact object
    with `source_lesson_id`, `source_method_id`, optional
    `source_experiment_id`, `label`, `content`, `outcome`, and `created_at`;
    those source IDs are provenance/audit metadata only, not IDs that importers
    must preserve locally. It must be compressed-only: do not include full
    `chunks.content` values, raw paper `content`, or original paper full text.
    The export route may accept the same nonnegative integer `max_chars` bound
    used by `GET /paper/<id>/memory` for the included context. Original paper,
    method, chunk, lesson, and experiment IDs are audit metadata only.
*   `GET /memory/export`: return a local corpus/set compressed-memory bundle
    layered on the single-paper bundle export. The response uses
    `kind: "researchfs.corpus_memory_bundle"`, `version: 1`, and a `papers` list
    containing existing paper-memory bundle objects. Optional `paper_ids` selects
    which local papers to include; optional `max_chars` applies the per-paper
    context bound to each nested paper bundle. The route must preserve the same
    compressed-only guarantee and metadata contract as single-paper export: no
    raw paper `content`, no full original paper text, and no full
    `chunks.content`.
*   `POST /memory/import`: accept either one compressed-memory bundle for a
    paper or one corpus bundle whose `papers` list contains paper-memory bundle
    objects, then reconstruct usable local memory through the same paper import
    flow without requiring original paper text. Imported papers use
    `source_type` `memory_bundle`; imported chunks are compact evidence-snippet
    chunks created from bundle evidence, not original paper chunks. The importer
    remaps paper, method, and chunk IDs into local SQLite IDs while preserving
    original IDs only as audit metadata; method-source links are rebuilt through
    the remapped local method/chunk IDs. Import compact lessons only when
    `source_method_id` can be remapped to a local method. Set imported lesson
    `experiment_id` to `NULL` unless the same import flow also implements an
    explicit local experiment mapping. Because `lessons` has no uniqueness
    constraint, skip and reuse any existing local lesson with the same
    `(method_id, label, content, outcome)`. A single-paper import returns the
    local paper ID, enough imported method/chunk IDs for callers to retrieve the
    imported method memory, imported or reused `lesson_ids`, and a
    `source_id_map["lesson_ids"]` mapping source lesson IDs to local lesson IDs
    analogous to chunk mapping. A corpus import iterates that existing paper
    importer atomically, explicitly commits the successful import, returns
    per-paper import results, and also returns aggregate local `paper_ids`,
    `method_ids`, `chunk_ids`, and `lesson_ids` for convenience. If any nested
    paper bundle is invalid, no earlier nested paper from that corpus import is
    left in the target database. Reject bundles that are missing required
    `kind`/`version` metadata, that contain raw full-text `chunks.content`/paper
    `content`, or that claim unavailable vector/OCR/PDF-coordinate citation
    verification behavior.
*   `GET /paper/<id>`: return paper metadata, extraction status, linked methods,
    chunks, and provenance links for lab inspection. Extraction status includes
    nullable `extractor_model`, `extractor_prompt_version`, `confidence_score`,
    and `confidence_reason`. Provenance links include nullable `evidence_text`,
    `evidence_start`, `evidence_end`, and `evidence_strategy`.
*   `GET /method/<id>`: return method metadata, `feedback_summary`, and
    `provenance_chunks`; `feedback_summary` includes `method_id`,
    `feedback_count`, `feedback_avg_rating`, `positive_feedback_count`,
    `negative_feedback_count`, `neutral_feedback_count`, and
    `last_feedback_at`; each provenance chunk includes the same nullable
    evidence fields.
*   `POST /feedback`: accept required `method_id` and `rating` in `[-1.0, 1.0]`,
    plus optional `event_id`, `label`, `note`, and `source`; persist the
    feedback record and return `ok` with the durable `feedback_id`.
*   `POST /retrieve`: accept JSON `{query: str, max_chars?: int,
    mode?: "method" | "graph" | "memory", max_papers?: int,
    source_types?: str | list[str], source_type?: str, include_scores?: bool,
    log?: bool, event_type?: str, agent_run_id?: int, experiment_id?: int}`.
    Default `mode: "method"` returns bounded context and method IDs through the
    existing method retrieval path. `mode: "graph"` must keep the existing
    graph-context behavior. `source_types` and the single-label `source_type`
    shorthand apply only to `mode: "memory"`; they must not change method or
    graph retrieval behavior. Both method and graph modes search compressed
    method fields, method/chunk provenance via linked chunks, and compact
    method-linked lesson text (including lessons imported from
    compressed-memory bundles). Lesson matches must feed the same
    `search_methods_and_ids`, `search_methods_with_scores`, API, SDK, logged
    retrieval, and `mode: "graph"` call paths rather than adding another
    method-only retrieval route. Matching is deterministic local SQLite/stdlib
    keyword matching, not vector semantic retrieval and not a hosted search
    service. The returned method pack should expose relevant bounded lesson
    evidence with a `Matched lessons:` section when lesson text contributed to
    retrieval so agents can see why the method was retrieved, while unrelated
    queries should keep existing no-match behavior.

    Explicit `mode: "memory"` routes through the existing `get_research_brief`
    implementation rather than duplicating chunk/card assembly, giving the
    primary agent-facing `/retrieve` helper parity with the one-shot `/brief`
    route. The response keeps the `/retrieve` shape (`ok`, `query`, `mode`,
    `context`, `method_ids`, `max_chars`) and adds `paper_ids`, `chunk_ids`,
    `lesson_ids`, `max_papers`, and `source_types`. `source_types` is the
    normalized non-empty label list when filtered and JSON `null` when
    unfiltered. It returns `scores` only when requested and `event_id` only when
    logged. `max_papers` bounds selected paper memory cards and must be optional
    for non-memory modes. Memory context is deterministic compressed markdown
    assembled from stored SQLite/stdlib memory only and preserves available
    extraction status plus method-source evidence span/strategy metadata: no raw
    paper `content`, no raw `chunks.content`, no fresh LLM call, no full-paper
    repaste, no vector search, no OCR, and no hosted search or briefing service.
    Direct `get_research_brief(...)` callers accept the same `source_types`
    filter as a string or list/tuple of strings.

    When `log: true`, accept optional `event_type`, `agent_run_id`, and
    `experiment_id`, log the returned context and method IDs through the
    existing retrieval log path, and return `event_id`. By default the response
    stays compact (`ok`, `query`, `mode`, `context`, `method_ids`, `max_chars`,
    and optional memory fields/event ID). When callers pass opt-in
    `include_scores: true`, return a `scores` array in returned evidence order,
    including logged responses. Score rows include `method_id`, `name`,
    `category`, `final_score`, `base_score`, `composite_score`, `rank_score`,
    `delta_score`, `success_rate_contribution`, `fatigue_multiplier`,
    `feedback_multiplier`, `method_rank`, `normalized_rank`,
    `lesson_match_count`, `lesson_match_score`, `feedback_count`,
    `feedback_avg_rating`, and empirical `stats`. These scores are
    diagnostic/debug signals for lab review, not stable public model weights.
*   `POST /brief`: accept JSON `{query: str, max_chars?: int, max_papers?: int,
    source_types?: str | list[str], source_type?: str, include_scores?: bool,
    log?: bool, event_type?: str, agent_run_id?: int, experiment_id?: int}` and
    return one bounded compressed research brief for the question. `source_types`
    accepts one label string or multiple labels; `source_type` is a single-label
    shorthand. Labels are extraction-status source classes such as `paper`,
    `lab_note`, `experiment_report`, `repo_readme`, `memory_bundle`, and `file`.
    Filtering must be deterministic SQLite/stdlib selection, not vector or
    hosted search, and it must be applied while selecting method-linked papers
    and chunk-FTS fallback papers so excluded source classes do not consume
    `max_papers`. The response includes `ok`, `query`, `context`, `paper_ids`,
    `method_ids`, `chunk_ids`, `lesson_ids`, `max_chars`, `max_papers`, and
    `source_types`, plus `scores` when requested and `event_id` when logged.
    `context` is deterministic markdown assembled from current `search_methods*`
    retrieval results, compact method-linked lessons, bounded `get_paper_memory`
    cards for method-linked papers, and a chunk-FTS fallback when no
    method-linked paper is selected but stored paper text matches the query. It
    must honor the requested budget, cap paper memory cards by `max_papers`,
    preserve available extraction status plus evidence span/strategy metadata,
    and preserve the compressed-only guarantee: no raw paper `content`, no raw
    `chunks.content`, no fresh LLM call, no full-paper repaste, no vector search,
    no OCR, no PDF-coordinate citation verification, and no hosted briefing
    service.
*   `POST /evaluate/extraction`: compare a paper's stored extraction results
    with a non-empty `expected_methods` list. The body requires `paper_id` and
    `expected_methods`; each expected method may be a string name or an object
    with required `name` plus optional `source_chunk_ids`,
    `source_chunk_indices`, `source_contains`, and `evidence_contains`.
    Optional request fields are `min_precision`, `min_recall`, `min_f1`,
    `min_source_coverage`, `min_evidence_coverage`, `require_source_coverage`,
    and `notes`. Return `ok` and `evaluation`, whose required fields include
    `paper_id`, `expected_count`, `extracted_count`, `matched_count`,
    `missing_methods`, `extra_methods`, `precision`, `recall`, `f1`,
    `source_link_coverage`, `evidence_coverage`, `confidence_score`,
    `confidence_reason`, and `passed`. It also includes diagnostic/API fields:
    `true_positive`, `false_negative`, `false_positive`,
    `expected_source_coverage`, `explicit_source_expectations`,
    `explicit_evidence_expectations`, `matched_methods`, `unexpected_methods`,
    `thresholds`, and `notes`. This route is deterministic, not an LLM judge,
    not verified citation correctness, not PDF-coordinate validation, and not a
    confidence-score calibration path. This slice has no
    extraction-evaluation persistence, no `persist` request behavior, and no
    `persisted`/`eval_id` result fields.
*   `POST /experiment`: record an agent experiment/result against the local
    SQLite memory store and return the existing experiment response shape.
*   `POST /lesson`: record a reusable lesson against the local SQLite memory
    store and return the existing lesson response shape.

### `researchfs/lab_client.py`
*   Define `ResearchFSLabClient` as the recommended SDK wrapper for labs and
    auto-research agents that want convenient local route access without a web
    framework. It wraps `ResearchFSLabAPI.dispatch`, returns response bodies for
    2xx API responses, and raises a small client exception for non-2xx
    responses.
*   Keep `ResearchFSLabClient` separate from `ResearchFSClient` and
    `ResearchFSSession` in `researchfs/client.py`; those remain DB-backed
    agent-session helpers for query reuse, prompt assembly, and experiment
    logging rather than route wrappers.
*   Expose convenience methods for the primary ingestion, retrieval, brief,
    inspection, paper-memory, bundle, feedback, evaluation, and stats routes:
    `ingest`, `ingest_arxiv`, `ingest_file`, `retrieve(query, max_chars=None,
    max_papers=None, mode="method", log=False, include_scores=False,
    event_type=None, agent_run_id=None, experiment_id=None, source_types=None,
    source_type=None)`, `research_brief(query, max_chars=None, max_papers=None,
    include_scores=False, log=False, event_type=None, agent_run_id=None,
    experiment_id=None, source_types=None, source_type=None)`, `list_papers`,
    `get_paper`,
    `get_paper_memory(paper_id, max_chars=None)`,
    `export_paper_memory_bundle(paper_id, max_chars=None)`,
    `import_paper_memory_bundle(bundle)`, `get_method`, `feedback`,
    `evaluate_extraction`, and `stats`; keep a low-level `dispatch` passthrough
    available for other implemented local routes.

### Extraction confidence semantics
*   `confidence_score` is nullable and ranges from `0.0` to `1.0` when present.
    It is a deterministic heuristic for the stored ResearchFS extraction record:
    status must be parsed/valid, normalized methods should be nonempty,
    `source_chunk_indices` should point within the paper's chunk bounds,
    linked provenance/evidence should cover the extracted methods, and
    compression should be sane relative to the source text.
*   `confidence_reason` should be a concise diagnostic explanation of the
    observable checks behind the score. Failed and skipped records should be
    low-confidence with explicit reasons.
*   The heuristic is not a verified citation-correctness score, not PDF
    coordinate validation, and not an LLM self-confidence claim. Extractor model
    and prompt metadata are audit metadata, not score inputs.

### Extraction evaluation semantics
*   `researchfs.evaluation.evaluate_extraction(conn, paper_id,
    expected_methods, ...)` is the pure stdlib helper behind
    `POST /evaluate/extraction`; it operates over SQLite rows already present in
    the supplied connection and does not require hosted services or an LLM.
*   Precision, recall, and F1 compare expected method names to stored method
    names for the paper. `missing_methods` are expected names not found;
    `extra_methods` are extracted method names for that paper that were not in
    the expected set; `unexpected_methods` carries the same extra-method
    information with method IDs for API diagnostics.
*   Source/evidence coverage is diagnostic. `source_link_coverage` /
    `source_coverage` measures whether matched methods have stored source links;
    `evidence_coverage` measures whether matched methods have nonempty
    `evidence_text`; `expected_source_coverage` measures explicit
    `source_chunk_ids`, `source_contains`, and `evidence_contains` expectations
    provided in the request.
*   Thresholds (`min_precision`, `min_recall`, `min_f1`,
    `min_source_coverage`, `min_evidence_coverage`, and
    `require_source_coverage`) only affect `thresholds` and `passed`; they do
    not tune retrieval, extraction, or `confidence_score`.
*   `confidence_score` remains heuristic telemetry from
    `paper_extraction_status`, while extraction evaluation is a separate
    expected-method/source/evidence check. It must not claim LLM-judge status,
    verified citation correctness, or PDF-coordinate correctness.

### `researchfs/retrieve.py`
*   `get_method_pack()`: Queries all `methods` and their source `chunks` (for provenance) and formats them into a clean string to be injected into the agent prompt.
*   `get_experiment_brief(name)`: Queries the target brief and returns it as a string.
*   `log_experiment(commit_hash, val_bpb, memory_gb, status, description)`: Inserts a run log and returns the `experiment_id`.
*   `log_retrieval_event(event_type, query, retrieved_content, agent_run_id=None, experiment_id=None, ...)`: Logs a retrieval event and returns `event_id`.
*   `log_method_feedback(method_id, rating, event_id=None, label=None, note=None, source=None, conn=None)`: Records durable lab or agent feedback and returns the inserted feedback id.
*   `search_methods_with_scores(...)`: opt-in retrieval helper returning the formatted context, ranked `method_ids`, and score rows in the same order; existing `search_methods()` and `search_methods_and_ids()` keep their current compact behavior.
*   `log_token_usage(experiment_id, prompt_tokens, completion_tokens, cost)`: Logs token usage.
