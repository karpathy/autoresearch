"""Agent runner that uses an in-process LLM callable (e.g. the OMP harness completion() helper).

This module is designed to be driven from an eval cell or from a wrapper that supplies
an LLM function taking a prompt string and returning a response string.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from researchfs.client import ResearchFSSession

from researchfs.db import initialize_db
from researchfs import retrieve as rfs
from researchfs.client import ResearchFSClient, _estimate_tokens
from researchfs.ingest import ensure_seed
from evals.nanochat_bpb.metrics import estimate_cost

from agents.editor import apply_hunks, check_syntax, HunkError
from run_agent import (
    _commit_train_py,
    _drop_last_commit_and_restore_train,
    _ensure_clean_train_py,
    parse_log,
    read_results_tsv,
)



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
    branch = res.stdout.strip()
    if branch != "HEAD":
        return branch
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return res.stdout.strip()


def _checkout(branch: str, env: dict) -> None:
    res = subprocess.run(
        ["git", "checkout", branch],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        if detail:
            raise RuntimeError(f"Could not checkout {branch}: {detail}")
        raise RuntimeError(f"Could not checkout {branch} (git exited {res.returncode})")


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

def _refresh_loopgraph_code_map() -> None:
    """Refresh ResearchFS code symbols for the editable training targets."""
    import importlib

    try:
        codegraph = importlib.import_module("researchfs.codegraph")
    except ModuleNotFoundError as exc:
        if exc.name == "researchfs.codegraph":
            return
        raise

    refresh_code_map = getattr(codegraph, "refresh_code_map", None)
    link_methods_to_code = getattr(codegraph, "link_methods_to_code", None)
    if callable(refresh_code_map):
        refresh_code_map(paths=("train.py", "prepare.py"))
    if callable(link_methods_to_code):
        link_methods_to_code()


def _get_loopgraph_context(query: str, max_chars: int | None, agent_run_id: Optional[int]) -> str:
    """Return graph context when the ResearchFS graph API is available."""
    get_graph_context = getattr(rfs, "get_graph_context", None)
    if not callable(get_graph_context):
        return ""
    return get_graph_context(query, max_chars=max_chars, agent_run_id=agent_run_id)


def _link_loopgraph_methods(experiment_id: int, method_ids: Optional[list[int]], query: str) -> None:
    """Explicitly connect retrieved methods to an experiment when supported."""
    if not method_ids:
        return
    link_experiment_methods = getattr(rfs, "link_experiment_methods", None)
    if callable(link_experiment_methods):
        link_experiment_methods(experiment_id, method_ids, role="retrieved", note=query or None)


def _log_loopgraph_lesson(
    *,
    experiment_id: Optional[int],
    method_ids: Optional[list[int]],
    query: str,
    description: str,
    status: str,
    val_bpb: Optional[float],
    memory_gb: Optional[float],
) -> None:
    """Record a compact outcome lesson once the experiment status is final."""
    if experiment_id is None:
        return
    log_lesson = getattr(rfs, "log_lesson", None)
    if not callable(log_lesson):
        return
    unique_method_ids = list(dict.fromkeys(method_ids or []))
    method_id = unique_method_ids[0] if len(unique_method_ids) == 1 else None
    val_str = f"{val_bpb:.6f}" if val_bpb is not None else "NA"
    mem_str = f"{memory_gb:.3f} GB" if memory_gb is not None else "NA"
    method_query = query or "none"
    content = (
        f"method_query={method_query}; "
        f"description={description}; "
        f"status={status}; "
        f"val_bpb={val_str}; "
        f"memory={mem_str}"
    )
    log_lesson(
        label="experiment_outcome",
        content=content,
        outcome=status,
        method_id=method_id,
        experiment_id=experiment_id,
    )


def assemble_harness_prompt(
    mode: str,
    results_content: str,
    method_pack: str,
    experiment_brief: str,
    train_py_content: str,
    error_msg: Optional[str] = None,
    graph_context: str = "",
) -> str:
    """Assemble a precise prompt asking for search-replace hunk edits."""
    prompt = (
        "You are an expert machine learning researcher agent.\n"
        "Your goal is to modify the training script `train.py` to minimize the validation bits per byte (val_bpb) on a pretraining task.\n\n"
    )
    if mode == "loopgraph":
        prompt += (
            f"Method Pack:\n{method_pack}\n\n"
            f"Experiment Brief:\n{experiment_brief}\n\n"
            f"Research Graph Context:\n{graph_context}\n\n"
        )

    prompt += f"Here is the history of previous experiments:\n{results_content}\n\n"
    prompt += f"Here is the current content of `train.py`:\n```python\n{train_py_content}\n```\n\n"

    if error_msg:
        prompt += f"CRITICAL: Your previous edit attempt failed with the following error:\n{error_msg}\n"
        prompt += "Analyze this error carefully and propose a different, corrected SEARCH/REPLACE block.\n\n"

    prompt += (
        "Propose the next experiment. Output your changes using one or more SEARCH/REPLACE blocks.\n"
        "This is the ONLY way you are allowed to modify the file. Do not output the entire file.\n"
        "Do not modify or ask to modify any file other than `train.py`.\n\n"
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
    env = _git_env()
    original_branch = _current_branch(env)
    print(f"Original checkout target: {original_branch}")

    _ensure_clean_train_py(env)

    initialize_db()
    ensure_seed()
    if mode == "loopgraph":
        _refresh_loopgraph_code_map()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"autoresearch/exp_{mode}_{timestamp}"

    agent_run_id: Optional[int] = None
    research_client: Optional[ResearchFSClient] = None
    branch_created = False

    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, env=env)
        branch_created = True

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
        mem_gb_str = f"{peak_vram_mb / 1024:.3f}" if peak_vram_mb is not None else "NA"
        with open(results_path, "a") as f:
            f.write(f"baseline\t{val_bpb:.6f}\t{mem_gb_str}\tkeep\tbaseline\n")

        rfs.log_experiment(
            commit_hash="baseline",
            val_bpb=baseline_val_bpb,
            memory_gb=peak_vram_mb / 1024 if peak_vram_mb is not None else None,
            status="keep",
            description="baseline",
            agent_run_id=agent_run_id,
        )

        best_val_bpb = baseline_val_bpb

        # Use the ResearchFS SDK for RAG decision, caching, and history feedback.
        research_client = ResearchFSClient()
        session: Optional[ResearchFSSession] = None
        if mode == "loopgraph":
            session = research_client.start_session(
                mode=mode,
                model=model,
                run_id=agent_run_id,
                baseline_metric=baseline_val_bpb,
            )

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
            current_graph_context = ""
            method_ids = None
            query_reused = False
            if mode == "loopgraph" and session is not None:
                decision = session.suggest_query(results_content, experiment_brief, llm_fn)
                query = decision["query"]
                query_reused = bool(decision.get("reused", False))
                reason = str(decision.get("reason", ""))
                print(f"Agent structured decision: search={decision['search']}, query='{query}', reason='{reason}'")
                if decision["search"] and query:
                    current_method_pack, method_ids = session.retrieve_context(query, max_method_chars=max_method_chars)
                else:
                    print(f"Skipping search. search_decision={decision['search']}, query='{query}'")
                    current_method_pack = ""
                    method_ids = None
                current_graph_context = _get_loopgraph_context(
                    query,
                    max_chars=max_method_chars,
                    agent_run_id=agent_run_id,
                )
                prompt_tokens_accum += session.prompt_tokens_accum
                completion_tokens_accum += session.completion_tokens_accum
                # Reset per-step accumulators so each experiment is tracked separately.
                session.prompt_tokens_accum = 0
                session.completion_tokens_accum = 0

            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"Attempt {attempt + 1}/{max_retries} (Self-Correction)...")
                else:
                    print(f"Attempt 1/{max_retries}...")

                # Restore train.py to original state before applying edits
                Path("train.py").write_text(original_train_code, encoding="utf-8")

                # Assemble prompt with the error from previous attempt if any
                prompt = assemble_harness_prompt(
                    mode,
                    results_content,
                    current_method_pack,
                    experiment_brief,
                    original_train_code,
                    last_error,
                    current_graph_context,
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
                except HunkError as e:
                    print(f"Hunk parsing failed: {e}")
                    last_error = f"Hunk parsing/application error: {e}"
                    continue

                if modified_code == original_train_code:
                    print("No changes detected in train.py.")
                    last_error = "No changes detected. Your SEARCH/REPLACE block did not modify train.py."
                    continue

                Path("train.py").write_text(modified_code, encoding="utf-8")

                # 3. Check syntax (lint gate)
                is_syntax_ok, syntax_err = check_syntax("train.py")
                if not is_syntax_ok:
                    print(f"Syntax validation failed: {syntax_err}")
                    last_error = f"Python syntax check error:\n{syntax_err}"
                    continue

                # 4. Commit changes temporarily for the run
                commit_msg = f"experiment {exp_num} (attempt {attempt}): {description}"
                print(f"Committing: {commit_msg}")
                _commit_train_py(commit_msg, env)

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
                    _link_loopgraph_methods(experiment_id, method_ids, query)

                # 5. Run training
                log_path = f"run_{exp_num}_attempt_{attempt}.log"
                val_bpb, peak_vram_mb, status, error_msg = _run_train(time_budget, steps_budget, env, log_path)

                if status == "crash":
                    print(f"Training crashed: {error_msg}")
                    # Revert only the experiment commit so we can start clean next attempt
                    _drop_last_commit_and_restore_train(original_train_code, env)
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
                    if mode == "loopgraph":
                        _log_loopgraph_lesson(
                            experiment_id=experiment_id,
                            method_ids=method_ids,
                            query=query,
                            description=f"{description} (Failed to compile/run)",
                            status="crash",
                            val_bpb=None,
                            memory_gb=None,
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
                _drop_last_commit_and_restore_train(original_train_code, env)

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
            if mode == "loopgraph":
                _log_loopgraph_lesson(
                    experiment_id=experiment_id,
                    method_ids=method_ids,
                    query=query,
                    description=description,
                    status=status_str,
                    val_bpb=val_bpb,
                    memory_gb=mem_gb,
                )

            if experiments is not None:
                experiments.append({
                    "exp_num": exp_num,
                    "description": description,
                    "val_bpb": val_bpb,
                    "status": status_str,
                })

        return agent_run_id
    finally:
        active_exception = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []

        if agent_run_id is not None:
            try:
                rfs.finish_agent_run(agent_run_id)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if branch_created:
            try:
                _checkout(original_branch, env)
            except BaseException as exc:
                cleanup_errors.append(exc)

        if research_client is not None:
            try:
                research_client.close()
            except BaseException as exc:
                cleanup_errors.append(exc)

        if cleanup_errors and active_exception is None:
            raise cleanup_errors[0]
        if cleanup_errors and hasattr(active_exception, "add_note"):
            for cleanup_error in cleanup_errors:
                active_exception.add_note(f"Cleanup failed: {cleanup_error!r}")
