#!/usr/bin/env python3
import glob
import hashlib
import os
import sys
import argparse
import datetime
import re
import subprocess
from pathlib import Path
import requests

from researchfs.db import initialize_db
from researchfs import retrieve as rfs
from evals.nanochat_bpb.metrics import estimate_cost, load_run
from seed import ensure_seed


BENCHMARK_ROOT = Path(__file__).resolve().parent

def extract_code_block(response_text):
    """Extract code from the single Python markdown block in an LLM response."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("LLM response did not contain a Python markdown code block.")
    return match.group(1).strip()

def extract_description(llm_response, default_desc):
    """Extract a one-line description from the LLM response.
    
    Looks for lines starting with 'Description:'.
    """
    for line in llm_response.splitlines():
        if line.strip().lower().startswith("description:"):
            desc = re.sub(r"^description:\s*", "", line.strip(), flags=re.IGNORECASE)
            return desc.strip()
    return default_desc

def parse_log(log_text, exit_code=0):
    """Parse train.py output log to extract val_bpb, peak_vram_mb, status, and error message."""
    val_bpb = None
    peak_vram_mb = None
    
    for line in log_text.splitlines():
        if "val_bpb:" in line:
            parts = line.split("val_bpb:")
            if len(parts) > 1:
                try:
                    val_bpb = float(parts[1].strip())
                except ValueError:
                    pass
        elif "peak_vram_mb:" in line:
            parts = line.split("peak_vram_mb:")
            if len(parts) > 1:
                try:
                    peak_vram_mb = float(parts[1].strip())
                except ValueError:
                    pass
                    
    # Check for traceback or non-zero exit code
    if "Traceback (most recent call last):" in log_text or exit_code != 0:
        tb_index = log_text.find("Traceback (most recent call last):")
        if tb_index != -1:
            traceback_text = log_text[tb_index:]
        else:
            traceback_text = log_text[-1000:]
        return val_bpb, peak_vram_mb, "crash", traceback_text.strip()
        
    if val_bpb is not None:
        return val_bpb, peak_vram_mb, "success", ""
    else:
        return None, None, "crash", "No val_bpb found in log. Run may have failed/crashed silently."

def assemble_system_prompt(mode, method_pack="", experiment_brief=""):
    """Assemble the system prompt based on mode and reference docs."""
    system_prompt = (
        "You are an expert machine learning researcher agent.\n"
        "Your goal is to modify the training script `train.py` to minimize the validation bits per byte (val_bpb) on a pretraining task.\n\n"
    )
    if mode == "loopgraph":
        system_prompt += (
            "You must carefully study the method pack and the experiment brief provided below.\n"
            "Modify `train.py` strictly following the instructions, patterns, constraints, and ranges described in these files.\n\n"
            f"Method Pack:\n{method_pack}\n\n"
            f"Experiment Brief:\n{experiment_brief}\n\n"
        )
    system_prompt += (
        "You must reply with your proposed changes to `train.py`.\n"
        "Output the complete, updated code of `train.py` inside a single markdown code block starting with ```python.\n"
        "Do not include any other markdown code blocks.\n"
        "Explain your changes briefly before or after the code block, and start your explanation with a line "
        "format like 'Description: <brief summary of your proposed changes>' so the runner can log it.\n"
    )
    return system_prompt

def assemble_user_prompt(train_py_content, results_tsv_content):
    """Assemble the user prompt containing current code and results history."""
    user_prompt = (
        f"Here is the current content of `train.py`:\n"
        f"```python\n{train_py_content}\n```\n\n"
        f"Here is the experiment history so far:\n"
        f"{results_tsv_content}\n\n"
        "Propose the next experiment. Write the complete updated `train.py` with your improvements.\n"
    )
    return user_prompt

def assemble_codex_prompt(mode, results_tsv_content, method_pack="", experiment_brief="", search_enabled=False):
    """Assemble prompt for Codex CLI."""
    prompt = "You are an autonomous machine learning researcher. Your goal is to modify `train.py` to achieve a lower validation bits per byte (val_bpb) than the baseline.\n\n"
    if search_enabled:
        prompt += "Note: You have access to a web/arXiv search tool. If you need to verify literature, paper implementations, or hyperparameter setups (e.g. SwiGLU, GQA, Muon), feel free to search the web.\n\n"
    if mode == "loopgraph":
        prompt += f"Use the method pack and experiment brief to guide your changes. Here is the method pack:\n{method_pack}\n\nHere is the experiment brief:\n{experiment_brief}\n\n"
    prompt += f"Here is the history of previous experiments:\n{results_tsv_content}\n\n"
    prompt += "Please choose a new hypothesis/optimization to test, modify `train.py` accordingly, and ensure the code compiles and runs correctly.\n"
    prompt += "Do not modify any file other than `train.py`; leave existing uncommitted work in the repository untouched.\n"
    prompt += "CRITICAL: At the very end of your response, write exactly: DESCRIPTION: <a concise one-line description of your change>."
    return prompt
def extract_codex_description(stdout, default_desc):
    """Extract description from Codex CLI stdout."""
    match = re.search(r"DESCRIPTION:\s*(.*)", stdout, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default_desc

def call_llm(provider, model, messages):
    """Call the LLM API using requests."""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/google/loopgraph",
            "X-Title": "LoopGraph Agent"
        }
    elif provider == "venice":
        api_key = os.environ.get("VENICE_API_KEY")
        if not api_key:
            raise ValueError("VENICE_API_KEY environment variable is not set")
        url = "https://api.venice.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    else:
        raise ValueError(f"Unknown API provider: {provider}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    return {
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {}),
    }
def read_results_tsv(filepath):
    """Read the results.tsv file if it exists."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r") as f:
        return f.read()


def _run_git_checked(args, env, action):
    """Run a git command and raise an actionable error if it fails."""
    res = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        if detail:
            raise RuntimeError(f"Git {action} failed (exit {res.returncode}): {detail}")
        raise RuntimeError(f"Git {action} failed with exit code {res.returncode}")
    return res

# Codex is allowed to edit train.py only. For ignored files, protect a scoped
# whitelist of reproducibility artifacts that this runner creates or consumes;
# this is not a complete scanner for every gitignored path in the repo.
_CODEX_IGNORED_NON_TRAIN_PATTERNS = (
    "results.tsv",
    "researchfs.db",
    ".asep/**",
    "CLAUDE.md",
    "results/**",
    "queue/**",
    "AGENTS.md",
    "run_*.log",
    "run_baseline.log",
)


def _snapshot_file_entries(paths):
    """Return path, size, and content hash entries for existing files."""
    entries = []
    for path in sorted(set(paths)):
        if path == "train.py" or not os.path.isfile(path):
            continue
        try:
            stat = os.stat(path)
            digest = hashlib.sha256()
            with open(path, "rb") as file_obj:
                for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(f"{path}\t{stat.st_size}\t{digest.hexdigest()}")
        except OSError as exc:
            entries.append(f"{path}\tERROR\t{type(exc).__name__}:{exc}")
    return "\n".join(entries)


def _snapshot_untracked_non_train_files(env, pathspec):
    result = _run_git_checked(
        ["ls-files", "--others", "--exclude-standard", *pathspec],
        env,
        "snapshot untracked non-train.py changes",
    )
    return _snapshot_file_entries(line for line in result.stdout.splitlines() if line)


def _snapshot_ignored_codex_files():
    """Return a content-sensitive snapshot for ignored files Codex must not touch."""
    paths = []
    for pattern in _CODEX_IGNORED_NON_TRAIN_PATTERNS:
        paths.extend(glob.glob(pattern, recursive=True))
    return _snapshot_file_entries(paths)


def _top_level_train_path() -> str:
    """Return the train target relative to this checkout's git top level."""
    if Path.cwd().resolve() == BENCHMARK_ROOT:
        return "benchmarks/nanochat/train.py"
    return "train.py"


def _snapshot_tracked_non_train_diff(env):
    """Capture non-train changes plus scoped ignored reproducibility artifacts."""
    pathspec = ["--", ":(top)", f":(exclude,top){_top_level_train_path()}"]
    return {
        "unstaged": _run_git_checked(
            ["diff", "--binary", *pathspec],
            env,
            "snapshot unstaged non-train.py changes",
        ).stdout,
        "staged": _run_git_checked(
            ["diff", "--cached", "--binary", *pathspec],
            env,
            "snapshot staged non-train.py changes",
        ).stdout,
        "untracked": _snapshot_untracked_non_train_files(env, pathspec),
        "ignored": _snapshot_ignored_codex_files(),
    }


def _ensure_codex_only_changed_train_py(before_snapshot, env):
    """Fail if Codex changed files outside train.py since the snapshot."""
    after_snapshot = _snapshot_tracked_non_train_diff(env)
    changed = [
        label
        for label in sorted(set(before_snapshot) | set(after_snapshot))
        if after_snapshot.get(label) != before_snapshot.get(label)
    ]
    if changed:
        changed_kinds = ", ".join(changed)
        raise RuntimeError(
            "Codex modified files outside train.py "
            f"({changed_kinds} snapshot changed); refusing to continue. "
            "Inspect those changes and restore them to the pre-Codex state before rerunning."
        )
    return after_snapshot


def _ensure_clean_train_py(env):
    """Refuse to run if train.py already has user work that could be overwritten."""
    checks = [
        (["diff", "--quiet", "--", "train.py"], "unstaged"),
        (["diff", "--cached", "--quiet", "--", "train.py"], "staged"),
    ]
    for args, label in checks:
        res = subprocess.run(["git", *args], env=env)
        if res.returncode == 1:
            raise RuntimeError(
                f"Refusing to run with {label} changes in train.py; commit or stash them first."
            )
        if res.returncode != 0:
            raise RuntimeError(f"Could not check {label} train.py changes (git exited {res.returncode}).")


def _commit_train_py(commit_msg, env, amend=False):
    """Commit only train.py so unrelated user edits are never swept into agent commits."""
    args = ["commit"]
    if amend:
        args.append("--amend")
    args.extend(["--only", "-m", commit_msg, "--", "train.py"])
    _run_git_checked(args, env, "commit train.py")


def _drop_last_commit_and_restore_train(original_train_code, env):
    """Drop the last train.py experiment commit without resetting unrelated work."""
    _run_git_checked(["reset", "--soft", "HEAD~1"], env, "drop experiment commit")
    with open("train.py", "w") as f:
        f.write(original_train_code)
    _run_git_checked(["reset", "--", "train.py"], env, "unstage restored train.py")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Autoresearch Agent Loop")
    parser.add_argument("--mode", choices=["baseline", "loopgraph"], required=True, help="Agent search mode")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max-experiments", type=int, default=5, help="Number of loop iterations")
    parser.add_argument("--time-budget", type=int, default=300, help="Training budget per experiment in seconds")
    parser.add_argument("--steps-budget", type=int, default=0, help="Training budget per experiment in steps (overrides time-budget if > 0)")
    parser.add_argument("--api-provider", choices=["openai", "openrouter", "venice", "codex"], default="codex", help="API provider to use")
    parser.add_argument("--search", action="store_true", help="Enable live web search tool for the agent")
    
    args = parser.parse_args()

    # Initialize ResearchFS database.
    initialize_db()
    ensure_seed()
    # Configure git author env vars if not set
    env = os.environ.copy()
    if "GIT_AUTHOR_NAME" not in env:
        env["GIT_AUTHOR_NAME"] = "Autoresearch Agent"
    if "GIT_AUTHOR_EMAIL" not in env:
        env["GIT_AUTHOR_EMAIL"] = "agent@autoresearch.local"
    if "GIT_COMMITTER_NAME" not in env:
        env["GIT_COMMITTER_NAME"] = "Autoresearch Agent"
    if "GIT_COMMITTER_EMAIL" not in env:
        env["GIT_COMMITTER_EMAIL"] = "agent@autoresearch.local"
        
    _ensure_clean_train_py(env)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"autoresearch/exp_{args.mode}_{timestamp}"

    print(f"Creating and checking out branch: {branch_name}")
    subprocess.run(["git", "checkout", "-b", branch_name], check=True, env=env)
    agent_run_id = rfs.start_agent_run(args.mode, args.model)
    
    results_path = "results.tsv"
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
            
    # Load reference context from ResearchFS if mode is loopgraph.
    method_pack = ""
    experiment_brief = ""
    if args.mode == "loopgraph":
        method_pack = rfs.get_method_pack()
        experiment_brief = rfs.get_experiment_brief("nanochat_autoresearch")
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
    # Step 1: Run Baseline
    print("Running baseline experiment...")
    baseline_log = "run_baseline.log"
    run_env = env.copy()
    run_env["TIME_BUDGET"] = str(args.time_budget)
    run_env["STEPS_BUDGET"] = str(args.steps_budget)
    
    with open(baseline_log, "w") as log_f:
        res_run = subprocess.run([sys.executable, "train.py"], env=run_env, stdout=log_f, stderr=subprocess.STDOUT)
        
    with open(baseline_log, "r") as log_f:
        baseline_log_text = log_f.read()
        
    val_bpb, peak_vram_mb, status, error_msg = parse_log(baseline_log_text, res_run.returncode)
    
    if status == "crash" or val_bpb is None:
        raise RuntimeError(f"Baseline run failed to execute successfully: {error_msg}")
        
    baseline_val_bpb = val_bpb
    mem_gb_str = f"{peak_vram_mb / 1024:.3f}" if peak_vram_mb is not None else "NA"
    val_bpb_str = f"{val_bpb:.6f}"
    
    print(f"Baseline successful. val_bpb: {val_bpb_str}, memory_gb: {mem_gb_str}")
    
    with open(results_path, "a") as f:
        f.write(f"baseline\t{val_bpb_str}\t{mem_gb_str}\tkeep\tbaseline\n")

    rfs.log_experiment(
        commit_hash="baseline",
        val_bpb=baseline_val_bpb,
        memory_gb=peak_vram_mb / 1024 if peak_vram_mb is not None else None,
        status="keep",
        description="baseline",
        agent_run_id=agent_run_id,
    )
        
    best_val_bpb = baseline_val_bpb
    
    # Step 2: Iterative Propose-Train-Evaluate Loop
    for exp_num in range(1, args.max_experiments + 1):
        prompt_tokens = 0
        completion_tokens = 0
        print(f"\n=== Starting Experiment {exp_num}/{args.max_experiments} ===")
        
        results_content = read_results_tsv(results_path)
        with open("train.py", "r") as f:
            original_train_code = f.read()
        if args.api_provider == "codex":
            prompt = assemble_codex_prompt(args.mode, results_content, method_pack, experiment_brief, search_enabled=args.search)
            print("Calling Codex CLI...")
            cmd = ["codex"]
            if args.search:
                cmd.append("--search")
            cmd.extend(["exec", "--dangerously-bypass-approvals-and-sandbox", prompt])
            codex_non_train_snapshot = _snapshot_tracked_non_train_diff(env)
            res_codex = subprocess.run(
                cmd,
                capture_output=True, text=True, env=env
            )
            _ensure_codex_only_changed_train_py(codex_non_train_snapshot, env)
            if res_codex.returncode != 0:
                print(f"Codex run failed (code {res_codex.returncode}): {res_codex.stderr}", file=sys.stderr)
                print("Stopping experiment loop.")
                break
            description = extract_codex_description(res_codex.stdout, f"experiment_{exp_num}")
            print(f"Proposed change description: {description}")
        else:
            train_py_content = original_train_code

            sys_prompt = assemble_system_prompt(args.mode, method_pack, experiment_brief)
            usr_prompt = assemble_user_prompt(train_py_content, results_content)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ]
            
            print(f"Calling LLM ({args.api_provider}, model: {args.model})...")
            try:
                llm_response = call_llm(args.api_provider, args.model, messages)
            except Exception as e:
                print(f"LLM API call failed: {e}", file=sys.stderr)
                print("Stopping experiment loop due to API error.")
                break

            usage = llm_response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            try:
                proposed_code = extract_code_block(llm_response["content"])
            except ValueError as e:
                print(f"LLM response parsing failed: {e}", file=sys.stderr)
                print("Stopping experiment loop due to malformed LLM response.")
                break
            description = extract_description(llm_response["content"], f"experiment_{exp_num}")

            print(f"Proposed change description: {description}")

            with open("train.py", "w") as f:
                f.write(proposed_code)
        with open("train.py", "r") as f:
            current_train_code = f.read()
        if current_train_code == original_train_code:
            print("No changes detected in train.py. Skipping this iteration.")
            with open(results_path, "a") as f:
                f.write(f"no_change\tNA\tNA\tdiscard\tNo changes proposed by LLM\n")
            continue
            
        # Commit the proposed changes
        commit_msg = f"experiment {exp_num}: {description}"
        print(f"Committing changes: {commit_msg}")
        _commit_train_py(commit_msg, env)
        
        res_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, env=env)
        commit_hash = res_hash.stdout.strip()

        # Log this experiment in ResearchFS.
        experiment_id = rfs.log_experiment(
            commit_hash=commit_hash,
            val_bpb=None,
            memory_gb=None,
            status="pending",
            description=description,
            agent_run_id=agent_run_id,
        )
        if prompt_tokens or completion_tokens:
            rfs.log_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=estimate_cost(prompt_tokens, completion_tokens, args.model),
                experiment_id=experiment_id,
            )
        # Execute the training run with self-correction retry loop
        max_retries = 2
        for attempt in range(max_retries + 1):
            log_path = f"run_{exp_num}_attempt_{attempt}.log" if attempt > 0 else f"run_{exp_num}.log"
            print(f"Running training (attempt {attempt}/{max_retries}), logging to {log_path}...")
            
            run_env = env.copy()
            run_env["TIME_BUDGET"] = str(args.time_budget)
            run_env["STEPS_BUDGET"] = str(args.steps_budget)
            
            with open(log_path, "w") as log_f:
                res_run = subprocess.run([sys.executable, "train.py"], env=run_env, stdout=log_f, stderr=subprocess.STDOUT)
                
            # Parse the execution logs
            with open(log_path, "r") as log_f:
                log_text = log_f.read()
                
            val_bpb, peak_vram_mb, status, error_msg = parse_log(log_text, res_run.returncode)
            
            print(f"Run status (attempt {attempt}): {status}")
            if status != "crash":
                break  # Succeeded or silent fail without crash trace; exit retry loop
                
            if attempt == max_retries:
                print("Max retries reached. Experiment failed.")
                break
                
            print(f"Attempt {attempt} crashed. Attempting self-correction...")
            error_lines = log_text.splitlines()
            traceback_text = "\n".join(error_lines[-30:])
            
            if args.api_provider == "codex":
                fix_prompt = f"The previous edit you made to `train.py` caused a crash during execution. Here is the traceback:\n\n{traceback_text}\n\nPlease analyze the code and modify `train.py` to fix this issue. CRITICAL: Write DESCRIPTION: <one-line description> at the end."
                print("Calling Codex for self-correction...")
                cmd_fix = ["codex"]
                if args.search:
                    cmd_fix.append("--search")
                cmd_fix.extend(["exec", "--dangerously-bypass-approvals-and-sandbox", fix_prompt])
                codex_non_train_snapshot = _snapshot_tracked_non_train_diff(env)
                res_codex = subprocess.run(
                    cmd_fix,
                    capture_output=True, text=True, env=env
                )
                _ensure_codex_only_changed_train_py(codex_non_train_snapshot, env)
                if res_codex.returncode != 0:
                    print(f"Codex self-correction failed: {res_codex.stderr}", file=sys.stderr)
                    break
                description = extract_codex_description(res_codex.stdout, description)
            else:
                with open("train.py", "r") as f:
                    current_code = f.read()
                fix_usr_prompt = (
                    f"The code you proposed crashed during execution. Here is the traceback:\n\n{traceback_text}\n\n"
                    f"Here is the current content of `train.py`:\n```python\n{current_code}\n```\n\n"
                    f"Please modify `train.py` to fix the crash. Output the complete updated code inside a ```python code block. "
                    f"Include a line 'Description: <brief summary of your fix>'."
                )
                fix_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": fix_usr_prompt}
                ]
                try:
                    llm_response = call_llm(args.api_provider, args.model, fix_messages)
                except Exception as e:
                    print(f"LLM API self-correction failed: {e}", file=sys.stderr)
                    break
                try:
                    proposed_code = extract_code_block(llm_response["content"])
                except ValueError as e:
                    print(f"LLM self-correction parsing failed: {e}", file=sys.stderr)
                    break
                description = extract_description(llm_response["content"], description)
                with open("train.py", "w") as f:
                    f.write(proposed_code)
                    
            # Amend the commit with the fix
            commit_msg = f"experiment {exp_num} (fix): {description}"
            print(f"Amending commit: {commit_msg}")
            _commit_train_py(commit_msg, env, amend=True)
            
            res_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, env=env)
            commit_hash = res_hash.stdout.strip()
        keep_experiment = False
        if status == "success" and val_bpb is not None:
            if val_bpb < best_val_bpb:
                keep_experiment = True
                best_val_bpb = val_bpb
                print(f"New best val_bpb! Keeping this change.")
            else:
                print(f"val_bpb {val_bpb:.6f} is not better than best {best_val_bpb:.6f}. Discarding.")
        else:
            print(f"Experiment crashed or failed to log metrics: {error_msg}. Discarding.")
            
        mem_gb_str = f"{peak_vram_mb / 1024:.3f}" if peak_vram_mb is not None else "NA"
        val_bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "NA"
        
        if keep_experiment:
            status_str = "keep"
        else:
            status_str = "crash" if status == "crash" else "discard"
            print("Reverting changes...")
            _drop_last_commit_and_restore_train(original_train_code, env)
            
        # Append experiment result to results.tsv
        with open(results_path, "a") as f:
            f.write(f"{commit_hash}\t{val_bpb_str}\t{mem_gb_str}\t{status_str}\t{description}\n")

        # Update ResearchFS with the final outcome.
        rfs.update_experiment(
            experiment_id=experiment_id,
            val_bpb=val_bpb if val_bpb is not None else None,
            memory_gb=peak_vram_mb / 1024 if peak_vram_mb is not None else None,
            status=status_str,
            description=description,
        )

    rfs.finish_agent_run(agent_run_id)

    # Print a concise summary of this run.
    summary = load_run(agent_run_id)
    print("\n=== Agent Run Summary ===")
    print(f"  agent_run_id: {agent_run_id}")
    print(f"  mode: {summary['mode']}")
    print(f"  best_bpb: {summary['best_bpb']:.6f}" if summary['best_bpb'] is not None else "  best_bpb: N/A")
    print(f"  total_experiments: {summary['total_experiments']}")
    print(f"  kept/crashed/discarded: {summary['kept']}/{summary['crashed']}/{summary['discarded']}")
    if summary["total_tokens"] is None:
        print("  total_tokens: N/A")
    else:
        print(f"  total_tokens: {summary['total_tokens']:,}")
    if summary["cost"] is None:
        print("  estimated_cost: N/A")
    else:
        print(f"  estimated_cost: ${summary['cost']:.4f}")
    print(f"  retrieval_events: {summary['retrieval_count']}")
    print(f"\nCompare with: python compare_runs.py --list")
if __name__ == "__main__":
    main()
