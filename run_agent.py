#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import re
import subprocess
import requests

def extract_code_block(response_text):
    """Extract code from LLM response markdown block.
    
    Looks for ```python ... ```, ```py ... ```, or ``` ... ```.
    Falls back to the raw response if no block markers are found.
    """
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```py\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

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
    return data["choices"][0]["message"]["content"]

def read_results_tsv(filepath):
    """Read the results.tsv file if it exists."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="Autonomous Autoresearch Agent Loop")
    parser.add_argument("--mode", choices=["baseline", "loopgraph"], required=True, help="Agent search mode")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max-experiments", type=int, default=5, help="Number of loop iterations")
    parser.add_argument("--time-budget", type=int, default=300, help="Training budget per experiment in seconds")
    parser.add_argument("--api-provider", choices=["openai", "openrouter"], default="openai", help="API provider to use")
    
    args = parser.parse_args()
    
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
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"autoresearch/exp_{args.mode}_{timestamp}"
    
    print(f"Creating and checking out branch: {branch_name}")
    subprocess.run(["git", "checkout", "-b", branch_name], check=True, env=env)
    
    results_path = "results.tsv"
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
            
    # Load reference files if mode is loopgraph
    method_pack = ""
    experiment_brief = ""
    if args.mode == "loopgraph":
        try:
            with open("method_pack.md", "r") as f:
                method_pack = f.read()
        except FileNotFoundError:
            print("Warning: method_pack.md not found", file=sys.stderr)
        try:
            with open("experiment_brief.md", "r") as f:
                experiment_brief = f.read()
        except FileNotFoundError:
            print("Warning: experiment_brief.md not found", file=sys.stderr)
            
    # Step 1: Run Baseline
    print("Running baseline experiment...")
    baseline_log = "run_baseline.log"
    run_env = env.copy()
    run_env["TIME_BUDGET"] = str(args.time_budget)
    
    with open(baseline_log, "w") as log_f:
        res_run = subprocess.run(["python", "train.py"], env=run_env, stdout=log_f, stderr=subprocess.STDOUT)
        
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
        
    best_val_bpb = baseline_val_bpb
    
    # Step 2: Iterative Propose-Train-Evaluate Loop
    for exp_num in range(1, args.max_experiments + 1):
        print(f"\n=== Starting Experiment {exp_num}/{args.max_experiments} ===")
        
        with open("train.py", "r") as f:
            train_py_content = f.read()
            
        results_content = read_results_tsv(results_path)
        
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
            
        proposed_code = extract_code_block(llm_response)
        description = extract_description(llm_response, f"experiment_{exp_num}")
        
        print(f"Proposed change description: {description}")
        
        # Apply changes temporarily to check for diff
        with open("train.py", "w") as f:
            f.write(proposed_code)
            
        res_diff = subprocess.run(["git", "diff", "--quiet", "train.py"], env=env)
        if res_diff.returncode == 0:
            print("No changes detected in train.py. Skipping this iteration.")
            with open(results_path, "a") as f:
                f.write(f"no_change\tNA\tNA\tdiscard\tNo changes proposed by LLM\n")
            continue
            
        # Commit the proposed changes
        commit_msg = f"experiment {exp_num}: {description}"
        print(f"Committing changes: {commit_msg}")
        subprocess.run(["git", "commit", "-am", commit_msg], check=True, env=env)
        
        res_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, env=env)
        commit_hash = res_hash.stdout.strip()
        
        # Execute the training run
        log_path = f"run_{exp_num}.log"
        print(f"Running training, logging to {log_path}...")
        
        run_env = env.copy()
        run_env["TIME_BUDGET"] = str(args.time_budget)
        
        with open(log_path, "w") as log_f:
            res_run = subprocess.run(["python", "train.py"], env=run_env, stdout=log_f, stderr=subprocess.STDOUT)
            
        # Parse the execution logs
        with open(log_path, "r") as log_f:
            log_text = log_f.read()
            
        val_bpb, peak_vram_mb, status, error_msg = parse_log(log_text, res_run.returncode)
        
        print(f"Run status: {status}")
        if val_bpb is not None:
            print(f"val_bpb: {val_bpb:.6f}")
        if peak_vram_mb is not None:
            print(f"peak_vram_mb: {peak_vram_mb:.1f}")
            
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
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True, env=env)
            
        # Append experiment result to results.tsv
        with open(results_path, "a") as f:
            f.write(f"{commit_hash}\t{val_bpb_str}\t{mem_gb_str}\t{status_str}\t{description}\n")

if __name__ == "__main__":
    main()
