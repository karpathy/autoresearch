# ResearchFS

ResearchFS is a compressed paper/lab memory layer for labs and autonomous
research agents. It ingests research artifact text into durable SQLite storage,
compresses it into method and provenance memory, retrieves bounded context for a
question, and records experiments and lessons so later agents can reuse what
the lab has already learned.

The NanoChat/autoresearch training loop in this repository is now an
evaluation fixture: it gives ResearchFS a small, real research environment for
measuring whether paper-grounded memory helps an agent choose better
experiments. NanoChat remains useful, but it is no longer the product surface.

## Product path

The first shippable path is intentionally local and small:

1. **Ingest a research artifact** from a local `.txt`, `.md`, `.markdown`, `.tex`, or
   `.pdf` file, arXiv source text, a repository README, or an in-process
   metadata dictionary through the CLI, SDK, or local API. File/API metadata can
   carry a `source_type` label such as `paper`,
   `lab_note`, `experiment_report`, or `repo_readme`; ResearchFS stores it on
   the extraction status row for corpus inspection and retrieval provenance, not
   as a new `papers` schema field.
2. **Compress research artifacts into memory**: chunks preserve source
   provenance, and optional LLM extraction can turn papers, lab notes,
   experiment reports, or repository READMEs into method records with
   implementation hints and risks. The extractor is not limited to
   deep-learning/PyTorch papers, but it keeps the canonical category tokens
   `attention`, `mlp`, `optimizer`, `normalization`, `precision`,
   `regularization`, and `embeddings` for backward compatibility with existing
   retrieval behavior. Broader lab artifact labels, such as `paper`, `lab_note`,
   `experiment_report`, and `repo_readme`, belong in extraction-status
   `source_type` metadata rather than method categories. Extraction is bounded
   and rolling: long artifacts are processed across chunk windows instead of
   repasting a full source into one prompt or silently ignoring chunks after the
   first window. A window prompt may number chunks locally, but stored
   method-source links are remapped back to the artifact's global chunk
   positions before they are persisted; when duplicate extracted methods share a
   name, ResearchFS deterministically unions their remapped source chunks. Each
   method-source link stores a deterministic `evidence_text` excerpt from the
   linked global chunk, plus nullable `evidence_start`, `evidence_end`, and
   `evidence_strategy` metadata for provenance inspection; these snippets are
   provenance evidence, not verified PDF-coordinate spans. Each extraction
   status also carries nullable `extractor_model` and `extractor_prompt_version`
   values when extraction ran, plus nullable `confidence_score` and
   `confidence_reason` fields: a deterministic heuristic for whether the
   compressed artifact memory is high-confidence, skipped, or weakly supported.
   This is compressed local paper/lab memory, not vector search, OCR,
   PDF-coordinate citation validation, or a hosted service.
3. **Inspect the local corpus** through paper list/detail routes and `GET /stats`
   for extraction status, heuristic extraction confidence, compression
   summaries, counts, chunks, linked methods, feedback summaries, and provenance.
4. **Retrieve bounded context** through `POST /retrieve`/SDK when an agent needs
   a method pack (`mode="method"`), graph context (`mode="graph"`), or bounded
   paper memory cards (`mode="memory"`). `POST /brief`/SDK remains the one-shot
   route for a deterministic compressed research brief. Both paths use local
   keyword matching and character budgets; memory/brief context adds relevant
   paper memory cards and lesson IDs without repasting full papers or calling an
   LLM.
5. **Rate retrieved, briefed, or inspected methods** with lab/agent feedback
   after an audited retrieval, brief, or method review. Ratings are durable, can
   point back to a retrieval `event_id`, and lightly influence later ranking.
6. **Log the experiment and lesson** after the agent tries a method.
7. **Retrieve the lesson later** through the same method retrieval/briefing path
   and method history/recommendation APIs.

## Quick start: local paper/lab memory

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/). ResearchFS
uses SQLite from the Python standard library. The NanoChat fixture additionally
needs a single NVIDIA GPU for training experiments.

```bash
# 1. Install dependencies
uv sync

# 2. Seed the local SQLite database with the bundled method corpus
uv run python -m researchfs.ingest --seed

# 3. Ingest a local markdown/text paper note without method extraction
mkdir -p notes
cat > notes/clip-matrix.md <<'EOF'
# Clip Matrix Update Norms

Method: clip matrix optimizer updates before applying each optimizer step.
Risk: clipping can add overhead or suppress useful large updates.
EOF
uv run python -m researchfs.ingest --file notes/clip-matrix.md

# Optional: ingest arXiv source text instead
uv run python -m researchfs.ingest --arxiv 2305.13245

# Optional: request LLM method extraction for the artifact if OPENAI_API_KEY is configured
uv run python -m researchfs.ingest --file notes/clip-matrix.md --extract
```

Local file ingestion defaults its extraction-status `source_type` to `file`.
Use `--source-type LABEL` with `--file`, `--arxiv`, or `--repo` to set a more
specific extraction-status label such as `paper`, `lab_note`,
`experiment_report`, or `repo_readme` when labs need corpus inspection and
retrieval provenance to distinguish artifact types without repasting full
papers. Re-ingesting a duplicate paper with an explicit `--source-type` corrects
that extraction-status label without reclassifying duplicates from the default
local-file `file` label. The label is stored on extraction status metadata, not
as a new `papers` schema field.

If a paper was first ingested without extraction, retry ingestion with
`--extract` for the same paper instead of changing the title to create a second
record. Duplicate-paper retry is the path for filling in extraction status,
method links, evidence spans, and extractor metadata once extraction support is
available for that paper.

By default the database is `researchfs.db` at the repository root. Set
`RESEARCHFS_DB_PATH=/path/to/researchfs.db` to use another SQLite file.

## Quick start: local ResearchFS API

ResearchFS exposes a local stdlib API that agents, tests, browser tools, and
lab scripts can call without a web framework. It is a local-only development
surface: there is no hosted ResearchFS service here, no built-in authentication,
and no claim that binding it beyond `127.0.0.1` is safe. Use
`ResearchFSLabClient` for in-process SDK access to the same route surface: it
wraps `ResearchFSLabAPI.dispatch`, returns response bodies for successful API
responses, and raises a small client exception for non-2xx responses. Keep it
distinct from `ResearchFSClient` and `ResearchFSSession`, which are DB-backed
helpers for autoresearch agent sessions, query reuse, prompt assembly, and
experiment logging.

```python
from researchfs import ResearchFSLabClient, ResearchFSLabClientError

client = ResearchFSLabClient()

try:
    ingested = client.ingest({
        "title": "Example Optimizer Note",
        "authors": "Lab Team",
        "abstract": "A short note about clipping matrix updates before training.",
        "content": "Method: clip matrix update norms before applying the optimizer step.",
        "url": "file://example",
        "publish_date": "2026-06-30",
        "source_type": "lab_note",
        "methods": [{
            "name": "Clip matrix update norms",
            "description": "Clip matrix optimizer updates before applying the step.",
            "implementation_hint": "Bound update norms in the optimizer step.",
            "risks": "May add overhead or suppress useful large updates.",
            "category": "optimizer",
            "source_chunk_indices": [0],
        }],
    })

    audited = client.retrieve(
        query="clip matrix",
        max_chars=2000,
        log=True,
        event_type="agent_search",
    )
    context = audited["context"]
    method_ids = audited["method_ids"]
    event_id = audited["event_id"]

    brief = client.research_brief(
        query="clip matrix optimizer risk",
        max_chars=3000,
        max_papers=2,
        source_types=["paper", "lab_note"],
        include_scores=True,
        log=True,
        event_type="agent_brief",
    )

    memory = client.retrieve(
        query="clip matrix optimizer risk",
        max_chars=3000,
        mode="memory",
        max_papers=2,
        source_types=["paper", "lab_note"],
        include_scores=True,
        log=True,
        event_type="agent_memory",
    )
    memory_context = memory["context"]
    memory_paper_ids = memory["paper_ids"]
    brief_context = brief["context"]
    brief_paper_ids = brief["paper_ids"]

    if method_ids:
        feedback = client.feedback(
            method_id=method_ids[0],
            event_id=event_id,
            rating=0.75,          # -1.0 = harmful, 0.0 = neutral, 1.0 = useful
            label="useful",
            source="lab-review",
            note="The retrieved method was relevant to this inspection.",
        )
        method_detail = client.get_method(method_ids[0])
    else:
        feedback = None
        method_detail = {"provenance_chunks": [], "feedback_summary": None}

    extraction_eval = client.evaluate_extraction(
        paper_id=ingested["paper_id"],
        expected_methods=[{
            "name": "Clip matrix update norms",
            "source_chunk_ids": ingested["chunk_ids"][:1],
            "source_contains": ["clip matrix optimizer"],
            "evidence_contains": ["clip matrix"],
        }],
        min_recall=1.0,
        min_f1=0.8,
        min_evidence_coverage=0.5,
        notes="local gold-method smoke check",
    )["evaluation"]

    paper_id = ingested["paper_id"]
    paper_memory = client.get_paper_memory(paper_id, max_chars=2000)

    papers = client.list_papers()
    stats = client.stats()
    paper_detail = client.get_paper(paper_id)
except ResearchFSLabClientError as exc:
    raise RuntimeError(f"ResearchFS local API failed: {exc.status} {exc.code}") from exc
```

Use `context` from the default `mode="method"` as the bounded
method/provenance/lesson pack for an agent prompt, and keep `event_id` with the
agent run or experiment record when `log: True` is set. `POST /retrieve` also
accepts optional `agent_run_id` and `experiment_id` fields; logged retrievals
return `event_id`, while unlogged retrievals keep the same response shape
without it.

The same primary route and `ResearchFSLabClient.retrieve(...)` SDK helper now
also accept explicit `mode="memory"` plus optional `max_papers` and
`source_types`. Memory mode routes through the existing research brief
implementation so agents can reuse bounded paper memory cards without switching
to `/brief`: the response keeps the standard `/retrieve` fields (`ok`, `query`,
`mode`, `context`, `method_ids`, `max_chars`) and adds `paper_ids`,
`chunk_ids`, `lesson_ids`, `max_papers`, and `source_types`. `source_types`
filters paper selection itself by extraction-status labels such as `paper`,
`lab_note`, `experiment_report`, `repo_readme`, `memory_bundle`, and `file`; pass
a string for one label or a list/tuple of strings for multiple labels, for
example `["paper", "lab_note"]` to exclude `repo_readme` cards. The filter is
only for `mode="memory"` and `/brief`, not default method retrieval or graph
mode. Set `include_scores: True` only when a lab or agent needs to inspect
ranking behavior; the optional `scores` array is aligned with returned
method/brief evidence and contains explanatory rows. With `log: True`, the
returned memory context and method IDs are written through the existing
retrieval log path and the response includes `event_id`.

`POST /brief`, `ResearchFSLabClient.research_brief(...)`, and the
`researchfs.memory query` CLI remain the one-shot brief surfaces for callers
that want the brief-shaped JSON directly. All three are deterministic local
SQLite/stdlib keyword assembly over compressed markdown memory only: no fresh
LLM call, no raw full-paper context, no raw `chunks.content`, no vector search,
no OCR, and no hosted search or briefing service. Their optional `source_types`
filter narrows method-source traversal and the chunk-FTS fallback while papers
are selected, so
`max_papers` is not consumed by excluded source classes. When available, the
compressed memory cards inside these surfaces expose extraction status metadata
(`source_type`, `extractor_model`, `extractor_prompt_version`, confidence
fields) and method-source evidence metadata (`evidence_text`, `evidence_start`,
`evidence_end`, `evidence_strategy`) so agents can audit where compressed
knowledge came from without receiving the full paper. Treat score fields as
bounded diagnostic/debug signals for ResearchFS review, not as stable public
model weights.

Use `POST /brief`, `ResearchFSLabClient.research_brief(...)`, or
`researchfs.memory query` when a lab or auto-research agent wants one
deterministic research brief for a question rather than a method pack alone. The
API JSON body is `{query: str, max_chars?: int, max_papers?: int,
source_types?: str | list[str], include_scores?: bool, log?: bool,
event_type?: str, agent_run_id?: int, experiment_id?: int}`; the local API also
accepts `source_type?: str` as a single-label shorthand. The CLI exposes the
same fields as flags. The response returns `ok`, `query`, `context`,
`paper_ids`, `method_ids`,
`chunk_ids`, `lesson_ids`, `max_chars`, `max_papers`, and `source_types`, plus
`scores` when requested and `event_id` when logged. `source_types` is the
normalized selected-label list when filtered and `null` when unfiltered. The
SDK helpers, the direct `get_research_brief(...)` helper, and the
`researchfs.memory query` CLI accept the same filter as a string or list/tuple
of strings; JSON requests use a string or array. The
brief context is compressed-only markdown assembled from current method search
results, compact method-linked lessons, bounded `GET /paper/<id>/memory` cards
for method-linked papers, and a chunk-FTS paper fallback when a query matches
stored paper text but no method-linked paper was selected. It is deterministic
SQLite/stdlib retrieval over stored ResearchFS memory, not a fresh LLM summary,
not vector search, not OCR, not a hosted briefing service, not PDF-coordinate
citation verification, and not raw full-paper or raw chunk context.

### Local compressed-memory query CLI

Use `researchfs.memory query` when a lab script or auto-research agent wants the
same deterministic research brief JSON as `POST /brief` and
`ResearchFSLabClient.research_brief(...)` without starting the local HTTP API:

```bash
uv run python -m researchfs.memory query \
  --query "clip matrix optimizer risk" \
  --source-type paper \
  --source-type lab_note \
  --max-chars 3000 \
  --max-papers 2 \
  --include-scores \
  --log \
  --event-type agent_brief \
  --agent-run-id 17 \
  --experiment-id 3 \
  --db-path ./researchfs.db
```

The command writes the brief JSON to stdout. It accepts `--query TEXT`,
`--max-chars N`, `--max-papers N`, repeated `--source-type LABEL` filters,
`--include-scores`, `--log`, optional logging metadata (`--event-type TYPE`,
`--agent-run-id ID`, `--experiment-id ID`), and `--db-path PATH`. Invalid
arguments or retrieval failures return nonzero and write a JSON error to stderr
so shell scripts can branch on exit status while still parsing errors. The
response shape matches the one-shot brief route: `ok`, `query`, `context`,
`paper_ids`, `method_ids`, `chunk_ids`, `lesson_ids`, `max_chars`,
`max_papers`, `source_types`, plus `scores` when requested and `event_id` when
logged.

This query subcommand is a local compressed-memory retrieval surface only. It
uses stored SQLite rows and the existing `get_research_brief(...)` assembly path;
it does not ingest new files, export/import bundles, start a server, host sync or
auth, call an LLM, run vector search, perform OCR, or repaste full papers. The
returned `context` stays bounded to the same evidence snippets and compact
memory cards already produced by `get_research_brief(...)`.

### Local extraction evaluation CLI

Use `researchfs.memory evaluate` when a lab script or auto-research agent wants
to audit stored extraction/compression quality against a caller-supplied
expected-method checklist without starting the local HTTP API or repasting a
full paper. The command uses the same deterministic
`researchfs.evaluation.evaluate_extraction(...)` helper behind
`POST /evaluate/extraction`; it reads the selected local SQLite database and
does not call an LLM, run OCR, perform vector search, mutate extraction
confidence, persist evaluation rows, or host a service.

```bash
# Inline expected methods JSON.
uv run python -m researchfs.memory evaluate \
  --paper-id 42 \
  --expected-methods '[{"name":"Clip matrix update norms","source_contains":["clip matrix optimizer"],"evidence_contains":["clip matrix"]}]' \
  --min-recall 1.0 \
  --min-evidence-coverage 0.5 \
  --notes "local gold-method smoke check" \
  --db-path ./researchfs.db

# File-based expected methods JSON.
uv run python -m researchfs.memory evaluate \
  --paper-id 42 \
  --expected-methods-file expected-methods.json \
  --min-recall 1.0 \
  --db-path ./researchfs.db
```

`--expected-methods` and `--expected-methods-file` are alternatives; both supply
the same non-empty JSON list accepted by `/evaluate/extraction`, where each
expected method may be a string name or an object with `name` plus optional
`source_chunk_ids`, `source_chunk_indices`, `source_contains`, and
`evidence_contains`. The CLI also accepts `--min-recall`,
`--min-evidence-coverage`, optional `--notes`, and `--db-path`. Successful runs
write machine-readable JSON to stdout with `ok: true` and `evaluation`; invalid
input or runtime errors return nonzero and write JSON errors to stderr. A missing
paper is reported with error code `paper_not_found`.

After a human or agent audits a retrieved method, `POST /feedback` records a
required `method_id` and `rating` from `-1.0` to `1.0`, plus optional
`event_id`, `label`, `source`, and `note`; the response contains `ok` and the
durable `feedback_id`. `POST /evaluate/extraction` compares stored extraction
results with a deterministic expected-method checklist. Use `GET /papers` for
the corpus inventory, `GET /stats` for local corpus observability via `counts`,
`extraction_statuses`, `source_types`, `compression`, `confidence`, and
`feedback`, `GET /paper/<id>/memory` as the compact paper reuse surface, and
`GET /paper/<id>` for paper status/chunks/method links/provenance with evidence
fields and extraction status metadata (`source_type`, `extractor_model`,
`extractor_prompt_version`, `confidence_score`, and `confidence_reason`).
`source_type` is an extraction-status label for the ingested artifact source
(`file` by default for local file ingestion, or labels such as `paper`,
`lab_note`, `experiment_report`, and `repo_readme` when provided), so labs can
inspect provenance without adding a paper-table column.
`GET /paper/<id>/memory` composes a deterministic markdown-style memory card
from stored metadata, extraction status, confidence/compression telemetry when
available, linked methods, method-source evidence snippets and span/strategy
metadata, feedback, experiments, and lessons. It returns `ok`, `paper_id`,
`max_chars`, `context`, `method_ids`, `chunk_ids`, `lesson_ids`, and
`experiment_ids`; the optional nonnegative integer `max_chars` query parameter
limits `context` with the same bounded-context semantics used elsewhere. It is
a compact reuse card for agents, not a fresh LLM summary, not a new extraction
pass, not raw paper/chunk transport, and not a citation verifier. Use
`GET /method/<id>` for method metadata, `feedback_summary` (`feedback_count`,
`feedback_avg_rating`, positive/negative/neutral counts, and `last_feedback_at`),
and provenance chunks with evidence fields.

### Portable compressed-memory bundles

ResearchFS paper memory can be shared between local labs or agents without
repasting full papers through compressed-memory bundles. Bundle export/import is
the local CLI file/stdio path, distinct from the `query` subcommand above, and it
emits machine-readable JSON for lab scripts:

```bash
# Export one paper bundle from the default researchfs.db.
uv run python -m researchfs.memory export --paper-id 42 --output paper-42.bundle.json

# Export a selected corpus bundle.
uv run python -m researchfs.memory export --paper-ids 12,18,21 --output optimizer-corpus.bundle.json

# Export every paper in a local database, bounding each nested paper memory card.
uv run python -m researchfs.memory export --all --max-chars 4000 --db-path /tmp/source-researchfs.db --output full-corpus.bundle.json

# Import into another local SQLite database.
uv run python -m researchfs.memory import --db-path /tmp/target-researchfs.db --input optimizer-corpus.bundle.json

# Stdio mode: omit --output to write the bundle JSON to stdout; omit --input to read a bundle from stdin.
uv run python -m researchfs.memory export --paper-id 42 > paper-42.bundle.json
uv run python -m researchfs.memory import --db-path /tmp/target-researchfs.db < paper-42.bundle.json
```

Use `--db-path` to override the default `researchfs.db` (equivalent to choosing
which local SQLite file to open). Export selection is explicit: use
`--paper-id` for one paper, `--paper-ids` for a comma-separated selected corpus,
or `--all` for the whole corpus. `--output` writes the bundle JSON to a file;
without it, export writes the bundle JSON to stdout. `--input` reads a bundle
JSON file; without it, import reads JSON from stdin. Imports return a compact
JSON result on stdout with imported/reused local IDs. The CLI calls the
same local bundle and database helpers as the in-process/API surfaces; it is not
a hosted sync service.

The same bundle contract is also available through the local loopback API routes
`GET /paper/<id>/memory/export` for one paper, `GET /memory/export` for a
corpus/set bundle, `POST /memory/import` for import, and SDK helpers
`export_paper_memory_bundle(paper_id, max_chars=None)`,
`export_corpus_memory_bundle(paper_ids=None, max_chars=None)`,
`import_paper_memory_bundle(bundle)`, `import_corpus_memory_bundle(bundle)`, and
`import_memory_bundle(bundle)`.

Bundle exports are compressed-only. A single-paper bundle contains a JSON
`kind`/`version`, paper metadata, extraction status metadata (including
`extractor_model`, `extractor_prompt_version`, and confidence fields when
available), the bounded `GET /paper/<id>/memory` context, method metadata,
source IDs, method-source provenance evidence snippets plus their nullable
`evidence_start`, `evidence_end`, and `evidence_strategy` metadata, compact
method-linked lessons, extraction/confidence/compression metadata, and optional
feedback summaries. A corpus export uses
`kind: "researchfs.corpus_memory_bundle"`, `version: 1`, and a `papers` list of
the same paper-memory bundle objects; optional `paper_ids` selects which papers
to include and optional `max_chars` bounds each nested paper bundle context.
Neither shape may contain full `chunks.content` values or raw paper `content`;
imports reconstruct usable compressed memory for retrieval from the metadata,
method records, source IDs, evidence snippets and span/strategy metadata, lesson
summaries, feedback summaries, and bounded context rather than requiring the
original full-text paper.
Because the existing retrieve path also matches method-linked lesson text, imported
compressed bundles can surface a method from compact lesson content alone; the
returned method pack includes a bounded `Matched lessons:` section so agents
can see why a lesson-backed method matched. Lesson entries carry only reusable
lab knowledge and audit metadata:
`source_lesson_id`, `source_method_id`, optional `source_experiment_id`,
`label`, `content`, `outcome`, and `created_at`. Imported papers use
`source_type` `memory_bundle`; imported chunks are compact evidence-snippet
chunks, not original paper chunks. Original paper, method, chunk, lesson, and
experiment IDs are kept only as audit metadata, while local database IDs are
remapped on import; method-source links are rebuilt through those local IDs.
Lesson import is method-linked: the importer only carries over lessons whose
source method maps to a local method, sets imported lesson `experiment_id` to
`NULL` unless a local experiment mapping is implemented, skips an existing local
lesson with the same `(method_id, label, content, outcome)`, and reports the
imported or reused `lesson_ids` plus `source_id_map["lesson_ids"]`.
Corpus imports are atomic and explicitly committed on success: if any nested
paper bundle is invalid, no earlier nested paper from that corpus import is left
in the target database, and a successful file-backed import is visible after the
API/client is reopened.

Bundle import/export is still a local-only ResearchFS surface. It is not hosted
sync, not an authentication boundary, not vector/OCR/citation infrastructure,
not a guarantee that evidence snippets are citation-verified, and not a
replacement for keeping the source paper under the lab's own controls when
full-text audit is required. `POST /memory/import` accepts either one paper
bundle or a corpus bundle and returns the same per-paper import result shape for
each paper, with aggregate local `paper_ids`, `method_ids`, `chunk_ids`, and
`lesson_ids` for convenience.

Implemented local API routes:

- `GET /` — compact API index with `ok`, service/product name, local-only caveat,
  and implemented routes.
- `GET /health` — local smoke-check status with `ok: true` and service/database
  status.
- `GET /papers`
- `GET /paper/<id>/memory`
- `GET /paper/<id>/memory/export`
- `GET /memory/export`
- `GET /paper/<id>`
- `GET /method/<id>`
- `GET /stats`
- `POST /ingest` — direct paper/note metadata and content. Payloads may include
  optional `source_type` labels such as `paper`, `lab_note`,
  `experiment_report`, or `repo_readme`; ResearchFS stores the label in
  `paper_extraction_status` and includes it in extraction status responses.
- `POST /ingest/arxiv`
- `POST /ingest/repo` — GitHub `owner/repo`, `owner/repo.git`, or repository
  URL README ingestion. URLs must point at the repository root; branch/tree,
  query, fragment, or extra path components return `invalid_repo`.
  Unlabeled repository README payloads default to explicit
  `source_type: "repo_readme"` so GitHub URLs do not collapse to inferred
  `github`; provided labels are stripped and preserved in extraction status.
- `POST /ingest/file` — local `.txt`, `.md`, `.markdown`, `.tex`, or `.pdf`
  paths plus optional metadata overrides. Unlabeled local files default to
  `source_type: "file"`; provided labels are preserved in extraction status for
  corpus inspection and provenance.
- `POST /evaluate/extraction`
- `POST /retrieve`
- `POST /brief`
- `POST /memory/import`
- `POST /feedback`
- `POST /experiment`
- `POST /lesson`

The stdlib HTTP handler adds CORS headers and supports `OPTIONS` preflight so
local browser clients can call the loopback server during development. Serve it
locally with:

```bash
uv run python -m researchfs.api --host 127.0.0.1 --port 8765 --db-path /path/to/researchfs.db
```

Then inspect the route index at `http://127.0.0.1:8765/` or smoke-check the
database connection at `http://127.0.0.1:8765/health`.

### Local browser workflow

The duplicated landing pages (`index.html` and `public/index.html`) include a
tiny browser workflow for this loopback API. Open either page from the local
checkout, set **Base URL** to the loopback server above (the default is
`http://127.0.0.1:8765`, and the page persists your choice in `localStorage`),
then run the flow in order:

1. **Ingest** a text paper or lab note. The page sends `POST /ingest` with
   browser-provided metadata/content (`title`, `abstract`, and `content`) plus
   explicit method metadata (`methods` with `name`, `description`,
   `implementation_hint`, and `source_chunk_indices`). The returned `paper_id`
   and `method_ids` stay in browser state for the next steps.
2. **Retrieve** the stored method memory. The page sends `POST /retrieve` to
   the configured base URL with `log: true` and `include_scores: true`, displays
   the bounded context plus score diagnostics, and keeps the returned `event_id`
   and latest `method_ids`.
3. **Inspect provenance** for a selected method with `GET /method/<id>`. This
   shows method metadata, feedback summary, and deterministic provenance chunks
   with evidence fields.
4. **Submit feedback** after the inspection. The page sends `POST /feedback`
   using the latest `method_id` and logged retrieval `event_id`, a rating in
   `[-1.0, 1.0]`, and a label/source for the browser review.
5. **Evaluate extraction quality** with the latest ingested paper. The page sends
   `POST /evaluate/extraction` using the `paper_id` saved from step 1 and
   `expected_methods` parsed from the console's small JSON textarea, falling
   back to the current method name when no checklist is supplied. This is a
   deterministic expected-method/source/evidence check, not confidence scoring,
   citation verification, or PDF validation.

All browser requests go to the configured loopback base URL and use only the
existing local stdlib routes above. The page displays JSON responses or request
errors in-place. It is not a hosted console, does not add authentication, does
not upload or OCR PDFs itself, and does not provide vector retrieval; PDF file
ingestion stays on the local file/API/SDK path.

## Extraction evaluation

Extraction evaluation is separate from extraction confidence. Use it when a lab
has a gold or expected-method checklist for a compressed paper memory and wants
to verify that ResearchFS stored the expected method names and source/evidence
support. It is deterministic SQLite/stdlib computation, not an LLM judge, not a
citation-correctness verifier, and not a PDF-coordinate validator.

The pure helper is `researchfs.evaluation.evaluate_extraction(conn, paper_id,
expected_methods, ...)`; the local API exposes the same result at
`POST /evaluate/extraction`, and `researchfs.memory evaluate` exposes it for
local scripts that do not want to start the API. The API request body requires
`paper_id` and a non-empty `expected_methods` list; the CLI supplies the same
values with `--paper-id` plus `--expected-methods` or
`--expected-methods-file`. Each expected method may be a string method
name or an object with:

- `name` (required): expected extracted method name; matching is deterministic,
  using normalized exact slug matches or high-overlap token matches.
- `source_chunk_ids` (optional): positive stored chunk IDs that should be linked
  to the matched method.
- `source_chunk_indices` (optional): zero-based chunk positions within the
  paper that should be linked to the matched method.
- `source_contains` (optional): source chunk substrings expected in linked
  chunks.
- `evidence_contains` (optional): substrings expected in linked
  `method_sources.evidence_text`.

Optional request fields are `min_precision`, `min_recall`, `min_f1`,
`min_source_coverage`, `min_evidence_coverage`, `require_source_coverage`,
and `notes`. Threshold fields are `0.0` to `1.0` checks reported under
`evaluation["thresholds"]`; they do not tune retrieval or confidence. This
slice has no extraction-evaluation persistence: the helper/API response is a
pure computation over the provided connection and no `extraction_evals` table,
`persist` request behavior, `persisted`, `eval_id`, or `result_json` surface is
exposed.

The evaluation result includes `paper_id`, `expected_count`, `extracted_count`,
`matched_count`, `missing_methods`, `extra_methods`, `precision`, `recall`,
`f1`, `source_link_coverage`, `evidence_coverage`, `confidence_score`,
`confidence_reason`, and `passed`. It also includes diagnostic/API fields:
`true_positive`, `false_negative`, `false_positive`,
`expected_source_coverage`, `explicit_source_expectations`,
`explicit_evidence_expectations`, `matched_methods`, `unexpected_methods`,
`thresholds`, and `notes`.
Coverage metrics are diagnostic: `source_link_coverage` measures whether matched
methods have stored source links and requested source text, `evidence_coverage`
measures whether matched methods have nonempty/requested evidence excerpts, and
`expected_source_coverage` aliases source-link coverage for compatibility.

`confidence_score` and `confidence_reason` remain heuristic telemetry from
`paper_extraction_status`; extraction evaluation reads stored rows but does not
recompute, mutate, or calibrate those confidence fields. The local CLI prints
the deterministic evaluation JSON to stdout and writes parse/runtime errors to
stderr; it is an audit surface, not a second confidence or quality path.

## Extraction confidence

`paper_extraction_status.confidence_score` is a nullable `0.0` to `1.0`
ResearchFS heuristic for the extraction/compression record attached to a paper.
It is deterministic and based on observable extraction quality: parsed status,
nonempty normalized methods, valid `source_chunk_indices` within the paper's
chunk bounds, provenance/evidence coverage from linked chunks, and compression
sanity. Skipped or failed extraction records are low-confidence and should carry
an explicit `confidence_reason`; extracted records explain the method,
source-link/provenance, and compression checks that support the score.

This field is deliberately not a verified citation-correctness score, not a
PDF-coordinate validation signal, and not an LLM self-confidence claim. Extractor
metadata such as model or prompt version may be recorded for auditability, but
the heuristic score should reflect only the observable quality of the stored
ResearchFS memory record.

## Current limitations

- Retrieval is early FTS5 keyword/hybrid ranking, not production vector search.
- Human/agent feedback is a modest ranking signal layered onto keyword rank,
  empirical experiment outcomes, BPB deltas, and fatigue. A positive average
  rating can nudge a method upward and a negative average can nudge it
  downward, but feedback alone should not override measured experiment results.
- LLM method extraction is optional. Without `--extract` or an injected
  `llm_fn`, ingest stores the paper and chunks but does not synthesize methods;
  that skipped extraction is reflected as low heuristic extraction confidence,
  not as a judgement about the source paper itself.
- The API is a local stdlib/Python module surface, not a hosted service.
- The browser workflow is local-only and unauthenticated; keep it on loopback
  and treat it as a lab smoke-test console for the stdlib API, not a production
  product portal.
- Local file ingestion accepts `.txt`, `.md`, `.markdown`, `.tex`, and `.pdf`
  papers through the local file/API/SDK path. PDF support uses `pypdf` text
  extraction only: it does not perform OCR, parse scanned-image PDFs, validate
  PDF-coordinate citations, or turn ResearchFS into a hosted document service.
- Provenance evidence is deterministic chunk/excerpt evidence (`evidence_text`
  with nullable `evidence_start`, `evidence_end`, and `evidence_strategy`), not
  a verified exact citation span or PDF-coordinate citation.
- Extraction confidence is a deterministic ResearchFS memory-quality heuristic,
  not verified citation correctness and not an LLM self-confidence claim.

## Development checks

Focused regression coverage lives under `tests/`; keep new targeted tests there
instead of growing the legacy monolithic `test_runner.py` unless an existing
fixture helper must be reused. Run the current checks with:

```bash
uv run pytest tests/test_regressions.py -q
uv run pytest test_runner.py tests/test_regressions.py -q
```

`test_runner.py` remains the broad compatibility suite for existing ResearchFS,
API, bundle, and NanoChat fixture behavior.

The Codex experiment guard permits generated agent edits only in `train.py`.
Outside `train.py`, it compares tracked/staged diffs, untracked file hashes, and
a scoped ignored-artifact whitelist: `results.tsv`, `researchfs.db`, `.asep/**`,
`results/**`, `queue/**`, `CLAUDE.md`, `AGENTS.md`, `run_*.log`, and
`run_baseline.log`. It deliberately does not scan every gitignored path such as
`.venv/`, `data/`, `worktrees/`, or Python caches.

## Project structure

```
researchfs/                — ResearchFS ingestion, retrieval, evaluation, ranking, SDK, and DB helpers
researchfs_spec.md         — compact v0 product/schema notes
program.md                 — baseline prompt for the NanoChat evaluation fixture
prepare.py                 — NanoChat fixture data prep, tokenizer, dataloader, fixed BPB eval
train.py                   — NanoChat fixture model, optimizer, and training loop
run_agent.py               — CLI loop for fixture experiments
agents/harness_agent.py    — in-process agent loop used by tests/ASEP wrappers
evals/nanochat_bpb/        — fixture run metrics, benchmark comparison, safety screens
compare_runs.py            — compare fixture runs against each other and published baselines
tests/test_regressions.py  — focused regression tests for recent safety fixes
pyproject.toml             — dependencies
```

## NanoChat evaluation fixture

The original autoresearch workflow gives ResearchFS a concrete benchmark: an
agent edits `train.py`, trains for a fixed 5-minute budget, checks validation
bits per byte (`val_bpb`; lower is better), and keeps or discards the change.
ResearchFS supplies retrieved method context before the experiment and stores
the result afterward.

To run the fixture directly:

```bash
# One-time data/tokenizer preparation for the NanoChat fixture
uv run prepare.py

# Manual 5-minute fixture training run
uv run train.py
```

List and compare recorded fixture runs:

```bash
uv run python compare_runs.py --list
uv run python compare_runs.py <baseline_run_id> <loopgraph_run_id>
uv run python compare_runs.py --benchmarks <baseline_run_id> <loopgraph_run_id>
uv run python compare_runs.py --benchmark-only <run_id>
uv run python compare_runs.py --aggregate-seeds <run_id_1> <run_id_2> [run_id_3 ...]
```

The benchmark reports include published NanoChat Autoresearch reference points
from Recursive/autoresearch@home and flag suspicious single-seed claims that
need multi-seed validation.

## Fixture platform support

ResearchFS itself is local SQLite/Python and does not require a GPU. The
NanoChat evaluation fixture currently targets a single NVIDIA GPU. CPU, MPS,
AMD, and Windows variants are out of scope for the canonical ResearchFS path;
the forks below may be useful if you want to run the fixture on other hardware.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
