# Web Listener Evaluator Plan

## Motivation

The current evaluator (`evaluate.py`) polls for remote `proposals/*` branches. This works, but requires collaborators to have push access to the repo. The natural GitHub model for contributions is pull requests — anyone with read access can fork, change, and open a PR.

A lightweight web server that listens for GitHub PR webhooks solves this:

- **Automatic evaluation** — PR opens, server evaluates, posts score as comment, merges or closes. No human in the loop.
- **Rich feedback** — PR comments are visible, searchable, and threaded. Agents (and humans) see exactly what score they got and why.
- **Clean workflow** — accepted PRs get merged; rejected ones get closed with the score. The PR history IS the evaluation history.
- **Foundation for public challenges** — right now it's small-team (collaborators push branches or open PRs from the same repo). Opening to forks/external contributors is a small extension later (just allow PRs from forks, which GitHub webhooks already support).

The server runs on a machine with the scoring infrastructure (GPU, private oracle, etc.) behind a reverse proxy the user already has. It replaces the polling loop with a webhook-driven queue but reuses all the existing scoring, DB, and leaderboard logic.

## Goal

A single Python file (`evaluator/server.py`) that:

1. Receives GitHub webhook events for PRs targeting master
2. Validates the PR (modifies only allowed files, targets the right branch)
3. Queues it for serial evaluation
4. Checks out the PR code, runs `score.sh`, records the result
5. Comments on the PR with the score and comparison to incumbent
6. Merges the PR if it improved the score, closes it if not (or if it crashed)
7. Updates `leaderboard.md` and pushes

## Design

### Architecture

```
GitHub ──webhook POST──▶ reverse proxy ──▶ server.py :8000
                                              │
                                              ▼
                                         ┌─────────┐
                                         │  Queue   │  (in-memory deque,
                                         │  (FIFO)  │   serial processing)
                                         └────┬─────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Evaluation worker  │
                                    │  (background thread)│
                                    │                     │
                                    │  - git checkout PR  │
                                    │  - run score.sh     │
                                    │  - compare incumbent│
                                    │  - comment on PR    │
                                    │  - merge or close   │
                                    │  - update leaderboard│
                                    └─────────────────────┘
```

### Single file, minimal dependencies

`server.py` uses FastAPI + the existing evaluator logic:

- `fastapi` + `uvicorn` for the webhook endpoint (both already available in the runtime)
- `threading` for the background evaluation worker
- `subprocess` to call `gh` CLI for PR comments/merge/close (already authed in the env)
- Imports/reuses functions from `evaluate.py` for DB, scoring, leaderboard export

The `gh` CLI handles all GitHub API interaction — it's already authed, handles pagination, and is simpler than pulling in PyGithub.

### Why `gh` CLI over Python GitHub SDK

- Already authed in the environment (user stated this)
- Zero new dependencies
- Simple subprocess calls: `gh pr comment`, `gh pr merge`, `gh pr close`
- If we later want the Python SDK, it's a drop-in replacement for those calls

### Serial evaluation (unchanged)

Same invariant as the polling evaluator: one evaluation at a time. The webhook endpoint just appends to a queue; a single background thread drains it. Multiple PRs arriving while one is being scored just wait their turn.

### PR validation

Before scoring, the server should sanity-check the PR:

1. **Targets master** — reject PRs targeting other branches
2. **Modifies only allowed files** — parse `problem.yaml` `mutable` list, check the PR diff. If it touches anything else, comment and close without scoring.
3. **Is not already evaluated** — check commit SHA against the DB (same as current logic)
4. **Can be cleanly checked out** — if checkout fails (conflicts with master), comment and close

This is the "see if it has this problem implemented and do best effort" part — we try, and if anything is wrong, we close the PR with a clear explanation.

### Webhook verification

GitHub webhooks can include a secret-based HMAC signature (`X-Hub-Signature-256`). The server should verify this to prevent spoofed requests. The secret is configured once in GitHub repo settings and stored as an env var on the server.

For now (small team, behind reverse proxy), verification is optional but the code should support it from day one via an env var (`WEBHOOK_SECRET`). If not set, accept all requests (with a startup warning).

### PR comment format

On evaluation completion, comment on the PR:

```markdown
## AutoAnything Evaluation

| Metric | Value |
|--------|-------|
| **Score** | 7.342 |
| **Incumbent** | 6.891 |
| **Delta** | +0.451 |
| **Result** | ✅ Accepted — merging |
| **Duration** | 312s |

<details>
<summary>Additional metrics</summary>

| Metric | Value |
|--------|-------|
| peak_vram_mb | 4200 |
| training_seconds | 298 |
| ... | ... |

</details>
```

Or for a rejection:

```markdown
## AutoAnything Evaluation

| Metric | Value |
|--------|-------|
| **Score** | 6.500 |
| **Incumbent** | 6.891 |
| **Delta** | -0.391 |
| **Result** | ❌ Rejected — closing |
| **Duration** | 305s |
```

Or for a crash:

```markdown
## AutoAnything Evaluation

**Result:** 💥 Crash — closing

```
Error: Exit code 1
RuntimeError: CUDA out of memory...
```
```

### Merge strategy

When a PR improves the score:

1. Comment with the results
2. `gh pr merge <number> --merge` (merge commit, not squash — preserves the proposal branch history)
3. Update incumbent in DB
4. Export leaderboard, commit, push

When a PR doesn't improve (or crashes):

1. Comment with the results
2. `gh pr close <number>`

### Status updates

While a PR is being evaluated (scoring takes minutes), add a "pending" comment or label so agents waiting know their PR is in the queue:

- When queued: add label `evaluating` (or comment "Queued for evaluation, position N")
- When starting: comment "Evaluation started..."
- When done: post results, remove label

Keep this minimal. A simple "Evaluating..." comment when starting, results comment when done.

## Implementation Plan

### Step 1: Refactor evaluate.py for reuse

Extract the core logic from `evaluate.py` into importable functions. Currently `evaluate.py` is already fairly modular (init_db, run_score, record_evaluation, etc.), but a few things need adjustment:

- `evaluate_proposal` currently does git checkout + scoring + DB recording + leaderboard export all in one. This is fine — the web server calls it the same way, just with PR metadata alongside.
- Add a function to checkout a PR by number: `gh pr checkout <number>` (cleaner than manual fetch + checkout for PRs from forks)
- Add functions for PR commenting, merging, closing via `gh`

These can go directly in `server.py` since it imports from `evaluate.py`.

### Step 2: Build server.py

```
evaluator/server.py     ~200-300 lines
```

**Components:**

1. **Webhook handler** — `POST /webhook`
   - FastAPI route that receives the GitHub webhook payload
   - Verify `X-Hub-Signature-256` if `WEBHOOK_SECRET` is set
   - Filter: only `pull_request` events with action `opened` or `synchronize`
   - Filter: only PRs targeting master
   - Extract: PR number, branch name, head SHA, PR author
   - Append to evaluation queue
   - Return 200 immediately

2. **Health endpoint** — `GET /health`
   - Returns queue length, current evaluation status, incumbent score
   - Useful for monitoring

3. **Evaluation worker** — background thread (started via FastAPI lifespan)
   - Drains the queue serially
   - For each PR:
     a. Comment "Evaluating..."
     b. Validate: check diff for allowed files only
     c. Checkout the PR: `gh pr checkout <number>` (handles forks correctly)
     d. Run `score.sh`
     e. Record in DB
     f. Comment with results
     g. Merge or close
     h. Checkout master, export leaderboard, commit+push
   - On any unexpected error: comment on PR with the error, close, continue to next

4. **Main** — parse args, start uvicorn with the FastAPI app

**Args:**
- `--port` (default 8000)
- `--host` (default 0.0.0.0)
- `--push` (push leaderboard updates after each evaluation)

**Env vars:**
- `WEBHOOK_SECRET` — optional, for webhook signature verification
- `GITHUB_TOKEN` — used by `gh` CLI (already set if gh is authed)

### Step 3: PR checkout strategy

For PRs from collaborators (same repo): `gh pr checkout <number>` creates a local branch tracking the PR.

For PRs from forks (future): `gh pr checkout <number>` also works — it fetches the fork's branch automatically. This is why using `gh pr checkout` is better than manual `git fetch origin pull/<n>/head` — it handles forks transparently.

After scoring, return to master: `git checkout master`.

The scoring happens in the main repo working directory (same as the current evaluator). Since evaluation is serial, there's no contention.

### Step 4: Wire up to GitHub

On the repo's GitHub settings page:
1. Add a webhook: `https://<your-domain>/webhook`
2. Content type: `application/json`
3. Secret: (generate one, set as `WEBHOOK_SECRET` env var on server)
4. Events: select "Pull requests" only

### Step 5: Update documentation

**README.md** — add a "Web evaluator" section:
- What it is (webhook-driven alternative to the polling evaluator)
- How to set it up (run server.py, configure GitHub webhook)
- What it does (evaluates PRs, comments scores, merges/closes)

**CLAUDE.md** — add server.py command to the Commands section

**agent_instructions.md** — update to mention the PR workflow:
- "Open a PR targeting master" as an alternative to pushing a branch
- The evaluator will comment with your score

### Step 6: Handle edge cases

- **PR updated while being evaluated** — the `synchronize` event fires when new commits are pushed to a PR branch. If a PR is mid-evaluation and gets updated, the queue will have a new entry. When the current evaluation finishes, the next one picks up the new commit. The old evaluation's comment stays (useful history), and the new one adds another comment.
- **Multiple PRs queued** — FIFO. First-come, first-scored. If PR #2 is queued behind PR #1, and PR #1 gets merged (changing the incumbent), PR #2 is scored against the new incumbent. This is correct — same serial invariant.
- **Server restart** — the in-memory queue is lost, but the DB persists. Any PRs that were queued but not yet evaluated will be missed. A simple mitigation: on startup, scan for open PRs that haven't been evaluated (by checking commit SHAs against the DB) and enqueue them. This doubles as a polling fallback.
- **Score.sh fails to produce output** — handled by existing `run_score()` logic (returns error message). Server comments the error and closes the PR.
- **PR targets wrong branch** — webhook handler filters these out. Return 200 (acknowledge) but don't enqueue.
- **PR modifies disallowed files** — validate before scoring. Comment "This PR modifies files outside the allowed set" and close.

### File diff validation

Before scoring a PR, check what files it modifies:

```
gh pr diff <number> --name-only
```

Compare against `problem.yaml` `mutable` list. If any file outside the mutable set is modified, reject without scoring.

## What this does NOT do (yet)

- **Accept PRs from forks by untrusted users** — the scoring runs arbitrary code from the PR. For a public challenge, you'd need sandboxing (container, VM, or at minimum resource limits). For a small team, trust is assumed.
- **Rate limiting** — no throttling of PR submissions. For small team, unnecessary. For public, add later.
- **Web dashboard** — no UI beyond `leaderboard.md` and PR comments. A future enhancement.
- **Multiple problem support** — one server instance per challenge repo. Fine for now.

## Summary

One new file: `evaluator/server.py`. Reuses all existing scoring/DB/leaderboard logic from `evaluate.py`. Receives GitHub PR webhooks via FastAPI/uvicorn, validates, scores serially, comments results, merges or closes. Driven by `gh` CLI for all GitHub interaction. Documentation updated to explain both the polling evaluator and the web evaluator as two deployment options.
