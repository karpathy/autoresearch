# AUTORESEARCH_HOME layout

State lives outside the measured project. Override the default `~/.autoresearch` with the `AUTORESEARCH_HOME` environment variable.

| Path | Purpose |
| --- | --- |
| `briefs/<project-id>/brief.md` | Passport saved by intake via `prepare --brief-text` (or `-` for stdin). This is the normal brief. `--program PATH` is optional |
| `projects/` | Candidate and snapshot files for labs in this storage root |
| `autoresearch.sqlite3` | Authoritative metadata, state, manifests, results, evidence metadata, and durable operations for confirmed SQLite-backed labs |
| `<lab-state>/.lock.guard` | Persistent OS advisory lock file for serializing lab mutations; the file is not replaced or deleted |
| `<lab-state>/.lock` | Temporary owner/heartbeat diagnostics while a process holds the advisory lock |
| `.dashboard-server.json` | Local dashboard discovery data, including its stop token; keep this home directory private |
| `.dashboard-start.lock` | Cross-process lock so concurrent CLI commands share one dashboard service |
| `.dashboard-server.log` | Local dashboard service diagnostics; startup errors returned to the CLI are intentionally generic |

Prepare accepts `--brief-text TEXT` (the usual intake path) or an optional `--program PATH` when the user already wrote a brief. Do not create `program.md` in the measured project. Resume follows the project's active lab recorded in SQLite.

The dashboard service starts as a background process on the first regular, non-headless CLI command and is shared by this home. Startup does not open or focus a browser. On Windows its child process runs without a console window. Open a browser only by an explicit `--open-browser` request, such as `lab.py dashboard ensure --open-browser`. Use global `--headless` or set `AUTORESEARCH_NO_DASHBOARD=1` to skip service startup. Before the first confirmed lab, the empty dashboard does not create the SQLite database. Labs use the authenticated v1 API.

Lab mutations use an OS advisory lock held by the process. The `.lock` heartbeat is diagnostic metadata; its age never authorizes taking a lock from a live process. A crashed process releases the OS lock automatically. After upgrading to a version using `.lock.guard`, stop old AutoResearch workers and commands before starting new ones; older processes do not participate in this lock protocol. Do not delete or replace `.lock.guard` while a process may be using the lab.
