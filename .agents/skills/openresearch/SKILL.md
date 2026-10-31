---
name: openresearch
description: Use for scholarly literature retrieval (`orx discover`, `orx paper`) as a read-only companion to AutoResearch labs, or for standalone OpenResearch projects, experiment trees, and compute.
---

# OpenResearch CLI (`orx`)

## AutoResearch Tandem Rule (Read-Only Literature Companion Mode)

> [!IMPORTANT]
> **When paired with `autoresearch` (default in this workspace), `autoresearch` (`lab.py`) is the sole owner of the experiment loop, git state (`candidate/`), and evaluation.**

- **Literature Companion Mode (Default in AutoResearch labs)**: In an AutoResearch lab, ONLY use `orx` for scholarly literature retrieval:
  - `orx discover keyword "<query>"` (alphaXiv full-text keyword retrieval)
  - `orx discover embedding "<query>"` (alphaXiv semantic retrieval)
  - `orx discover openalex "<query>"` (cross-disciplinary scholarly graph)
  - `orx discover biorxiv "<query>"` (bioRxiv preprints)
  - `orx discover pubmed "<query>"` (PubMed biomedical literature)
  - `orx paper <id|url> [--source ...] [--full]` (fetch paper content/abstract)
  - `orx skill orx-lit-review` (cross-corpus retrieval guidelines and policies)
- **Strictly Read-Only — No Orchestration Mutations**: NEVER run `orx up`, `orx create-experiment`, `orx project edit`, `orx exp run`, `orx compute`, or create/switch Git branches inside `candidate/` when `autoresearch` is active.
- **Hypothesis Testing**: Formulate ideas from the retrieved literature and implement and evaluate them strictly inside `candidate/` via `lab.py`.

### Literature & papers — alphaXiv / OpenAlex / bioRxiv / PubMed (no login required) — module `orx-lit-review`

Use before any web search for academic/research queries (paper, author, blog, model release).

| Command | What it does |
|---|---|
| `orx discover keyword "<query>"` | Call the alphaXiv full-text retrieval primitive with match snippets. |
| `orx discover embedding "<query>"` | Call the alphaXiv semantic retrieval primitive. The main agent ranks candidates and decides focused follow-ups; see `orx-lit-review`. |
| `orx discover openalex "<query>"` | Search the cross-disciplinary OpenAlex scholarly graph. |
| `orx discover biorxiv "<query>"` | Search bioRxiv preprints through OpenAlex's bioRxiv index. |
| `orx discover pubmed "<query>"` | Search PubMed biomedical literature through NCBI E-utilities. |
| `orx paper <id\|url> [--source ...] [--full]` | Fetch a paper: alphaXiv report with automatic full-text fallback (`--full` forces raw text), or OpenAlex/bioRxiv/PubMed metadata+abstract. Source auto-detected from the id. |

---

## Standalone OpenResearch Mode ONLY (Do NOT use inside an AutoResearch lab)

`orx up` owns projects, experiment branches, and execution in a local Git
repository and local database. Use the local session worktree to read, diff,
and edit code (see `orx-git`).

> [!WARNING]
> The rules and commands in this section apply **ONLY** when running a standalone OpenResearch project without `autoresearch`. Do NOT use these commands or create branches inside an `autoresearch` lab.

### Cardinal rules — read before doing anything else

These four govern everything below. Breaking any one silently invalidates your
results — they are not style preferences. The `orx-experiment-tree` module
expands on the why; these are the non-negotiables.

1. **Never edit a node once a run has answered it.** A node freezes the moment
   a run establishes its baseline or tests its hypothesis — that includes the
   root — and freezing is permanent: a disappointing result is still a result.
   Until then it is **provisional**: seeding it, fixing its deps, and making it
   run all happen on its own branch (`orx-experiment-tree`). To test a new
   hypothesis, branch a **child** and edit the child.
2. **The run command *and* the environment are a fixed contract — identical on
   every node.** A child inherits its parent's run command verbatim; leave it
   alone. Do **not** give nodes different start commands, and do **not** vary
   behavior through environment variables or env-prefixed commands
   (`LR=3e-4 python …`). The *only* thing that may differ between nodes is the
   **committed code/config** on the node's git branch. Set the local project's
   command once with `orx project edit <projectId> --run-command '<cmd>'`.
3. **Vary code, not knobs-in-the-command.** Encode hyperparameters in the
   code/config files and branch a child per variant — never sweep them by editing
   the run command or passing env vars. Every node runs the *same* command over
   *different code*, so their logged result summaries stay comparable.
4. **Grow the tree downward, not sideways.** Fan a little *within* a round (the
   options of one decision), then **descend onto that round's winner** for the
   next round. A root with a long row of direct children and no grandchildren is
   the failure mode. See "Shape the tree" in the `orx-experiment-tree` module.

If you're ever tempted to change the command, pass an env var, or pile another
node onto the root instead of branching a child, editing its branch, and
descending — stop. That's the anti-pattern, not a shortcut.

### Setup (Standalone)

#### Windows installation and preflight

Windows CLI support is currently beta and requires Git for Windows. Before
running `orx`, check that `git.exe` is the Git for Windows installation and
that its `bin\bash.exe` exists. OpenResearch discovers that shell from the Git
installation; `C:\Windows\System32\bash.exe` is the WSL launcher and is not a
replacement. In PowerShell, inspect both explicitly:

```powershell
Get-Command git.exe
Test-Path 'C:\Program Files\Git\bin\bash.exe'
Get-Command orx
orx --version
```

Install only from the official [OpenResearch releases](https://github.com/alphaXiv/OpenResearch/releases)
page. For the Windows x64 ZIP, download its adjacent `.sha256` file, compare it
with `Get-FileHash -Algorithm SHA256`, and extract only after they match. Put
`orx.exe` in `%USERPROFILE%\.cargo\bin`, add that directory to the user `PATH`,
and open a new terminal. Verify `orx --version`, then inspect needed instructions
with `orx skill orx-experiment-tree` or `orx skill orx-lit-review`. Modules are
provided by the installed CLI and may change with its version.

To test only the local interface without opening a browser or spawning an
agent, run `orx up --no-browser --no-agent`, confirm that it serves on
`127.0.0.1`, and stop it with Ctrl+C. `orx projects` is a read-only inventory;
do not create a project or launch compute as part of setup verification.

```sh
orx login          # opens a browser, stores a token at ~/.config/openresearch/credentials.json
orx logout         # remove the stored token
```

- The API base URL resolves from `--api-url` → `OPENRESEARCH_API_URL` → a built-in
  default. Set `OPENRESEARCH_API_URL` for non-local use.
- Local project and run commands do not require a token. OpenResearch compute,
  instance provisioning, and account settings require `orx login`.

### Standalone Command Quick-Reference

Project-scoped commands take a **project id**; experiment-scoped commands take an
**experiment id**; run-scoped commands take a **run id**. Don't mix them — get
ids from `orx projects`, `orx project view`, and `orx runs` respectively. Each
group below has a module (`orx skill <name>`) with the full flags and rules.

#### Auth
| Command | What it does |
|---|---|
| `orx login [--api-url <url>]` | Open a browser, do loopback OAuth, store a token. |
| `orx logout` | Remove the stored token. |

#### Discover (project- and experiment-scoped)
| Command | What it does |
|---|---|
| `orx projects [--json]` | List projects in the local `orx` store. |
| `orx orgs [--json]` | List organization ids available for OpenResearch compute (login required). |
| `orx project view <projectId>` | Show a local project's details and experiment tree. **Experiment ids come from here.** |
| `orx runs <projectId> [--experiment <id>]` | List runs as a table, newest first. **Run ids come from here.** |

#### Run evidence (run-scoped) — module `orx-evidence`
| Command | What it does |
|---|---|
| `orx logs <runId>` | Show the local log path, size, and short preview; inspect the file for full evidence. |

#### Create and run experiments (write) — modules `orx-create`, `orx-compute`, `orx-git`
| Command | What it does |
|---|---|
| `orx up` | Open the local dashboard to import or create a local project. |
| `orx project edit <localProjectId> [--name "<n>"] [--run-command "<cmd>"]` | Edit a local project's name or fixed run command. |
| `orx create-experiment <localProjectId> --title "<t>" [...]` | Add a local experiment node; prints its Git branch. |
| `orx compute status` / `show <backend>` / `test <backend>` | Inspect machine-wide compute configuration and readiness. |
| `orx compute configure <backend> --help` / `default set <backend>` / `connect <backend>` | Configure compute or authenticate; load `orx-compute`. |
| `orx compute instructions show --json` | Read the machine-wide custom recipe and its revision before configuring or launching. |
| `orx compute [--gpu <id>] [--count <n>] [--provider <name>]` / `orx compute --cpu` | List the GPU/CPU compute catalog. |
| `orx instance create <orgId> (--gpu <id> … \| --cpu <flavor> …)` | Spin up a standalone instance in an org; see `orx-instances`. |
| `orx exp status/run/cancel/wait/wake <localExpId>` | Inspect, run, cancel, wait on, or register a wake-up for a local experiment node. |
| `orx exp desc <expId> [--set "<text>" \| --stdin]` | Read or overwrite the experiment's description. |
| `orx agent spawn "<task>" [--title "<t>"] [--stdin] [--no-wake]` | Delegate an independent task to a helper session; see `orx-agent-delegation`. |

To **read or edit** a node's code—including diffing what a run changed—use plain
Git in the local session worktree. See the `orx-git` module.

#### Skills & templates — module `orx-customize`
| Command | What it does |
|---|---|
| `orx skills add <path>` | Save a reusable skill from a `SKILL.md` file or skill ZIP across projects. |
| `orx templates add <path>` | Save a reusable LaTeX template from a `.tex` file or template ZIP across projects. |

#### Meta
| Command | What it does |
|---|---|
| `orx skill [name[/resource]]` | Print this overview, one bundled module, or a lazily loaded module resource such as `compute/hf`. |
| `orx feedback --kind <bug\|feature_request\|frustration> --summary ... --details ...` | Report a meaningful OpenResearch bug, feature request, or user frustration to its maintainers. See the `orx-feedback` module. |

### Modules

The detail lives in focused modules — load one with `orx skill <name>` (the bundled
list, with one-line descriptions, is printed at the end of `orx skill` output):

- **orx-experiment-tree** — the experiment-tree model, the auto-research loop, and `orx exp desc`.
- **orx-create** — initialize a local project and add local experiment nodes.
- **orx-compute** — launch and monitor runs; after resolving the backend, read its bundled reference.
- **orx-instances** — create persistent standalone machines for manual work.
- **orx-git** — read, edit, and diff a node's code with plain git.
- **orx-agent-delegation** — delegate independent work to helper sessions safely.
- **orx-evidence** — capture and inspect experiment results through run logs.
- **orx-reports** — write durable research outputs into the project's artifacts directory.
- **orx-figures** — publication-quality figures in matplotlib or TikZ. Load it **before** writing any plotting code, then read the one reference for that figure type.
- **orx-customize** — add reusable skills and LaTeX templates across projects.
- **orx-paper** — draft a paper or preprint as LaTeX that renders and compiles to PDF.
- **orx-feedback** — report meaningful OpenResearch bugs, feature requests, and user frustration without leaking research details.
- **orx-lit-review** — main-agent cross-corpus retrieval, source-selective follow-up policy, and paper content; the preferred starting point for academic/research queries.

### Typical standalone workflow

Orienting in a project (read-only discovery):

```sh
orx projects                     # find the project id
orx project view <projectId>     # see the tree, pick an experiment id
orx skill experiment-tree        # the model + the auto-research loop
orx runs <projectId>             # find a run id
orx logs <runId>                 # locate its log, then inspect the file
```

To actually **drive** a project toward a goal — edit each node's code on its Git
branch and keep the round moving — follow the auto-research loop in
the `orx-experiment-tree` module. Every completed run is a decision point with
four moves: **repair** the same node when a run answered nothing, **refill**
the round with another sibling, **promote** the winner and descend, or **stop**.
