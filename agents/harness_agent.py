"""Agent runner that uses an in-process LLM callable (e.g. the OMP harness completion() helper).

This module is designed to be driven from an eval cell or from a wrapper that supplies
an LLM function taking a prompt string and returning a response string.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from researchfs.db import initialize_db
from researchfs import retrieve as rfs
from researchfs.ingest import ensure_seed
from evals.nanochat_bpb.metrics import estimate_cost

from agents.editor import apply_hunks, check_syntax, HunkError
from run_agent import (
    parse_log,
    read_results_tsv,
)


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English/code text."""
    return max(1, len(text) // 4)


def _clean_query(query: str) -> str:
    """Normalize a query string for cache lookup."""
    return " ".join(query.lower().split())


def _get_reusable_query(agent_run_id: int, baseline_val_bpb: float, conn=None) -> Optional[str]:
    """Return the query from the most recent kept experiment that improved over baseline."""
    from researchfs.db import get_connection

    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT re.query
            FROM experiments e
            JOIN retrieval_events re ON e.experiment_id = re.experiment_id
            WHERE e.agent_run_id = ?
              AND e.status = 'keep'
              AND e.val_bpb < ?
              AND re.event_type IN ('agent_search', 'agent_search_reuse')
            ORDER BY e.experiment_id DESC
            LIMIT 1
            """,
            (agent_run_id, baseline_val_bpb),
        )
        row = cur.fetchone()
        return row["query"] if row else None
    finally:
        if close:
            conn.close()


def _git_env() -> dict:
    env = os.environ.copy()
    for key, value in {
        "GIT_AUTHOR_NAME": "Autoresearch Agent",
        "GIT_AUTHOR_EMAIL": "agent@autoresearch.local",
        "GIT_COMMITTER_NAME": "Autoresearch Agent",
        "GIT_COMMITTER_EMAIL": "agent@autoresearch.local",
    }.items():
        env.setdefault(key, value)
    return env


def _current_branch(env: dict) -> str:
    res = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return res.stdout.strip()


def _checkout(branch: str, env: dict) -> None:
    subprocess.run(["git", "checkout", branch], env=env, check=False)


def _run_train(time_budget: int, steps_budget: int, env: dict, log_path: str) -> tuple[float | None, float | None, str, str]:
    """Run train.py and parse the log. Returns (val_bpb, peak_vram_mb, status, error_msg)."""
    run_env = env.copy()
    run_env["TIME_BUDGET"] = str(time_budget)
    run_env["STEPS_BUDGET"] = str(steps_budget)
    with open(log_path, "w") as log_f:
        res = subprocess.run(
            ["uv", "run", "python", "train.py"],
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    with open(log_path, "r") as log_f:
        log_text = log_f.read()
    return parse_log(log_text, res.returncode)


def assemble_harness_prompt(
    mode: str,
    results_content: str,
    method_pack: str,
    experiment_brief: str,
    train_py_content: str,
    error_msg: Optional[str] = None,
) -> str:
    """Assemble a precise prompt asking for search-replace hunk edits."""
    prompt = (
        "You are an expert machine learning researcher agent.\n"
        "Your goal is to modify the training script `train.py` to minimize the validation bits per byte (val_bpb) on a pretraining task.\n\n"
    )
    if mode == "loopgraph":
        prompt += f"Method Pack:\n{method_pack}\n\nExperiment Brief:\n{experiment_brief}\n\n"

    prompt += f"Here is the history of previous experiments:\n{results_content}\n\n"
    prompt += f"Here is the current content of `train.py`:\n```python\n{train_py_content}\n```\n\n"

    if error_msg:
        prompt += f"CRITICAL: Your previous edit attempt failed with the following error:\n{error_msg}\n"
        prompt += "Analyze this error carefully and propose a different, corrected SEARCH/REPLACE block.\n\n"

    prompt += (
        "Propose the next experiment. Output your changes using one or more SEARCH/REPLACE blocks.\n"
        "This is the ONLY way you are allowed to modify the file. Do not output the entire file.\n\n"
        "Each block must follow this exact format:\n"
        "<<<<<<< SEARCH\n"
        "[exact lines in train.py to replace]\n"
        "=======\n"
        "[replacement lines]\n"
        ">>>>>>>\n\n"
        "Ensure the SEARCH block matches the file content exactly, including spacing and indents.\n"
        "Explain your changes briefly, and start your explanation with a line formatted exactly like:\n"
        "Description: <brief summary of your change>\n"
    )
    return prompt



def assemble_retrieval_prompt(results_content: str, experiment_brief: str) -> str:
    """Assemble prompt to ask the agent for a search query to query ResearchFS."""
    prompt = (
        "You are an expert machine learning researcher agent.\n"
        "Your goal is to optimize the training script `train.py` to minimize the validation bits per byte (val_bpb) on a pretraining task.\n\n"
        f"Experiment Brief:\n{experiment_brief}\n\n"
        f"Here is the history of previous experiments:\n{results_content}\n\n"
        "Based on the results, decide what optimization area you want to explore next (e.g. attention, MLP, optimizer learning rate, precision, regularization).\n"
        "Output a single search query (1-2 keywords, e.g. 'MLP SwiGLU', 'GQA attention', or 'Muon LR') to search our literature database for relevant methods and code implementations.\n"
        "Reply ONLY with the search query text, nothing else. Do not include markdown blocks or explanation."
    )
    return prompt

def run_agent_mode(
    mode: str,
    model: str,
    max_experiments: int,
    time_budget: int,
    llm_fn: Callable[[str], str],
    experiments: Optional[list[dict]] = None,
    steps_budget: int = 0,
    max_method_chars: int | None = None,
) -> int:
    """Run the propose-train-evaluate loop using the supplied LLM callable."""
    initialize_db()
    ensure_seed()

    env = _git_env()
    original_branch = _current_branch(env)
    print(f"Original branch: {original_branch}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"autoresearch/exp_{mode}_{timestamp}"
    subprocess.run(["git", "checkout", "-b", branch_name], check=True, env=env)

    results_path = Path("results.tsv")
    if not results_path.exists():
        results_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    method_pack = ""
    experiment_brief = ""
    agent_run_id = rfs.start_agent_run(mode, model)

    if mode == "loopgraph":
        method_pack = rfs.get_method_pack()
        experiment_brief = rfs.get_experiment_brief("nanochat_autoresearch")
        print(f"Loaded method pack ({len(method_pack)} chars) and experiment brief ({len(experiment_brief)} chars)")
        rfs.log_retrieval_event(
            event_type="method_pack",
            query="all",
            retrieved_content=method_pack[:2000],
            agent_run_id=agent_run_id,
        )
        rfs.log_retrieval_event(
            event_type="experiment_brief",
            query="nanochat_autoresearch",
            retrieved_content=experiment_brief[:2000],
            agent_run_id=agent_run_id,
        )

    # Baseline run
    print("\n=== Running baseline experiment ===")
    val_bpb, peak_vram_mb, status, error_msg = _run_train(time_budget, steps_budget, env, "run_baseline.log")
    if status == "crash" or val_bpb is None:
        raise RuntimeError(f"Baseline run failed: {error_msg}")

    baseline_val_bpb = val_bpb
    print(f"Baseline val_bpb: {val_bpb:.6f}")
    with open(results_path, "a") as f:
        f.write(f"baseline\t{val_bpb:.6f}\t{peak_vram_mb / 1024:.3f}\tkeep\tbaseline\n")

    rfs.log_experiment(
        commit_hash="baseline",
        val_bpb=baseline_val_bpb,
        memory_gb=peak_vram_mb / 1024 if peak_vram_mb is not None else None,
        status="keep",
        description="baseline",
        agent_run_id=agent_run_id,
    )

    best_val_bpb = baseline_val_bpb
    search_cache: dict[str, tuple[str, list[int]]] = {}

    # Iterative loop
    for exp_num in range(1, max_experiments + 1):
        print(f"\n=== Starting Experiment {exp_num}/{max_experiments} ===")
        results_content = read_results_tsv(str(results_path))

        # Store the original train.py content to allow stateless retries
        original_train_code = Path("train.py").read_text(encoding="utf-8")

        max_retries = 3
        last_error = None
        success = False
        description = f"experiment_{exp_num}"
        commit_hash = "pending"
        experiment_id = None
        prompt_tokens_accum = 0
        completion_tokens_accum = 0
        query = ""
        current_method_pack = ""
        method_ids = None
        query_reused = False
        if mode == "loopgraph":
            reused_query = _get_reusable_query(agent_run_id, baseline_val_bpb)
            if reused_query:
                query = reused_query
                query_reused = True
                print(f"Reusing prior query '{query}' (kept, improved BPB)")
            else:
                retrieval_prompt = assemble_retrieval_prompt(results_content, experiment_brief)
                prompt_tokens_accum += _estimate_tokens(retrieval_prompt)
                print("Calling harness LLM for retrieval query...")
                query_res = llm_fn(retrieval_prompt)
                completion_tokens_accum += _estimate_tokens(query_res)
                query = query_res.strip().strip('"').strip("'")
                print(f"Agent chose search query: '{query}'")

            cache_key = _clean_query(query)
            if cache_key in search_cache:
                current_method_pack, method_ids = search_cache[cache_key]
                print(f"Using cached method pack for query '{query}'")
            else:
                current_method_pack, method_ids = rfs.search_methods_and_ids(query, max_chars=max_method_chars)
                search_cache[cache_key] = (current_method_pack, method_ids)

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Attempt {attempt + 1}/{max_retries} (Self-Correction)...")
            else:
                print(f"Attempt 1/{max_retries}...")

            # Restore train.py to original state before applying edits
            Path("train.py").write_text(original_train_code, encoding="utf-8")

            # Assemble prompt with the error from previous attempt if any
            prompt = assemble_harness_prompt(
                mode, results_content, current_method_pack, experiment_brief, original_train_code, last_error
            )

            prompt_tokens = _estimate_tokens(prompt)
            prompt_tokens_accum += prompt_tokens

            print("Calling harness LLM...")
            response = llm_fn(prompt)

            completion_tokens = _estimate_tokens(response)
            completion_tokens_accum += completion_tokens

            # Try to extract a description
            for line in response.splitlines():
                if line.strip().lower().startswith("description:"):
                    description = line.strip().split(":", 1)[1].strip()
                    break

            # 1. Apply hunks
            try:
                modified_code = apply_hunks(original_train_code, response)
                Path("train.py").write_text(modified_code, encoding="utf-8")
            except HunkError as e:
                print(f"Hunk parsing failed: {e}")
                last_error = f"Hunk parsing/application error: {e}"
                continue

            # 2. Check diff
            res_diff = subprocess.run(["git", "diff", "--quiet", "train.py"], env=env)
            if res_diff.returncode == 0:
                print("No changes detected in train.py.")
                last_error = "No changes detected. Your SEARCH/REPLACE block did not modify train.py."
                continue

            # 3. Check syntax (lint gate)
            is_syntax_ok, syntax_err = check_syntax("train.py")
            if not is_syntax_ok:
                print(f"Syntax validation failed: {syntax_err}")
                last_error = f"Python syntax check error:\n{syntax_err}"
                continue

            # 4. Commit changes temporarily for the run
            commit_msg = f"experiment {exp_num} (attempt {attempt}): {description}"
            print(f"Committing: {commit_msg}")
            subprocess.run(["git", "commit", "-am", commit_msg], check=True, env=env)

            res_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, env=env)
            commit_hash = res_hash.stdout.strip()

            if experiment_id is None:
                experiment_id = rfs.log_experiment(
                    commit_hash=commit_hash,
                    val_bpb=None,
                    memory_gb=None,
                    status="pending",
                    description=description,
                    agent_run_id=agent_run_id,
                )
            if mode == "loopgraph":
                rfs.log_retrieval_event(
                    event_type="agent_search_reuse" if query_reused else "agent_search",
                    query=query,
                    retrieved_content=current_method_pack[:2000],
                    agent_run_id=agent_run_id,
                    experiment_id=experiment_id,
                    method_ids=method_ids,
                )

            # 5. Run training
            log_path = f"run_{exp_num}_attempt_{attempt}.log"
            val_bpb, peak_vram_mb, status, error_msg = _run_train(time_budget, steps_budget, env, log_path)

            if status == "crash":
                print(f"Training crashed: {error_msg}")
                # Revert git commit so we can start clean next attempt
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True, env=env)
                # Keep last few lines of log as traceback context
                lines = error_msg.splitlines()
                traceback_context = "\n".join(lines[-20:])
                last_error = f"Training crashed during execution:\n{traceback_context}"
                continue

            # Success! The run compiled and finished without crashing
            success = True
            break

        # Log total accumulated tokens for this experiment step
        if experiment_id is not None:
            rfs.log_token_usage(
                prompt_tokens=prompt_tokens_accum,
                completion_tokens=completion_tokens_accum,
                cost=estimate_cost(prompt_tokens_accum, completion_tokens_accum, model),
                experiment_id=experiment_id,
            )

        if not success:
            print(f"Experiment {exp_num} failed after {max_retries} attempts.")
            # Ensure file is clean and state is reverted
            Path("train.py").write_text(original_train_code, encoding="utf-8")
            if experiment_id is not None:
                rfs.update_experiment(
                    experiment_id=experiment_id,
                    val_bpb=None,
                    memory_gb=None,
                    status="crash",
                    description=f"{description} (Failed to compile/run)",
                )
            with open(results_path, "a") as f:
                f.write(f"failed\tNA\tNA\tcrash\t{description} (Failed after retries)\n")
            continue

        # Decide keep/discard
        keep_experiment = False
        if val_bpb is not None:
            if val_bpb < best_val_bpb:
                keep_experiment = True
                best_val_bpb = val_bpb
                print(f"New best val_bpb! Keeping changes.")
            else:
                print(f"val_bpb {val_bpb:.6f} not better than best {best_val_bpb:.6f}. Discarding.")
        else:
            print("Run completed but val_bpb was None. Discarding.")

        mem_gb = peak_vram_mb / 1024 if peak_vram_mb is not None else None
        status_str = "keep" if keep_experiment else "discard"

        if not keep_experiment:
            print("Reverting changes...")
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True, env=env)
            Path("train.py").write_text(original_train_code, encoding="utf-8")

        with open(results_path, "a") as f:
            val_str = f"{val_bpb:.6f}" if val_bpb is not None else "NA"
            mem_str = f"{mem_gb:.3f}" if mem_gb is not None else "NA"
            f.write(f"{commit_hash}\t{val_str}\t{mem_str}\t{status_str}\t{description}\n")

        rfs.update_experiment(
            experiment_id=experiment_id,
            val_bpb=val_bpb,
            memory_gb=mem_gb,
            status=status_str,
            description=description,
        )

        if experiments is not None:
            experiments.append({
                "exp_num": exp_num,
                "description": description,
                "val_bpb": val_bpb,
                "status": status_str,
            })

    rfs.finish_agent_run(agent_run_id)
    _checkout(original_branch, env)

    return agent_run_id
