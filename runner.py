import subprocess
import requests
import re
import time
import shutil

MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TRAIN_FILE = "train.py"
RESULTS_FILE = "results.tsv"
LOG_FILE = "run.log"
MAX_LINE_CHANGE_RATIO = 0.3  # only allow small edits

# --- helper functions ---
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def run_training():
    """Run train.py and return val_bpb."""
    subprocess.run(f"python {TRAIN_FILE} > {LOG_FILE} 2>&1", shell=True)
    log = read_file(LOG_FILE)
    m = re.search(r"val_bpb:\s+([0-9.]+)", log)
    if m:
        return float(m.group(1))
    return None

def log_result(commit, val_bpb, status, description):
    line = f"{commit}\t{val_bpb:.6f}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)

def ask_llm(prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False}
    )

    data = resp.json()

    try:
        # Ollama usually returns: data["results"][0]["text"]
        text = data["results"][0]["text"]
    except (KeyError, IndexError):
        print("Unexpected response:", data)
        return ""

    # Sometimes the model returns code wrapped in ```python ... ```
    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(text.splitlines()) > len(read_file(TRAIN_FILE).splitlines()) * 1.5:
            print("Proposed change too large, skipping")
            return None
        # skip the first line ```python or ```
        if len(lines) > 2:
            # remove first and last line
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])  # fallback
    

    return text

def build_prompt():
    program = read_file("program.md")
    code = read_file(TRAIN_FILE)
    try:
        results = read_file(RESULTS_FILE)
    except FileNotFoundError:
        results = ""
    try:
        log = read_file(LOG_FILE)
    except FileNotFoundError:
        log = ""
    prompt = f"""
You are running an AutoResearch experiment.

Follow all instructions in program.md below.
Do not change files other than train.py.

--- program.md ---
{program}

--- train.py ---
{code}

--- previous results.tsv ---
{results}

--- last run.log ---
{log}

Propose the next modification to train.py.
Return ONLY the updated full train.py code, not explanations.
"""
    return prompt


best_score = None

while True:
    print("\n=== NEW EXPERIMENT ===")
    original_code = read_file(TRAIN_FILE)

    prompt = build_prompt()
    new_code = ask_llm(prompt)

    # --- safety check ---
    original_lines = original_code.splitlines()
    new_lines = new_code.splitlines()
    change_ratio = abs(len(new_lines) - len(original_lines)) / len(original_lines)
    if change_ratio > MAX_LINE_CHANGE_RATIO:
        print("Change too big, skipping...")
        time.sleep(2)
        continue

    # backup
    shutil.copyfile(TRAIN_FILE, TRAIN_FILE + ".bak")
    write_file(TRAIN_FILE, new_code)

    # commit
    subprocess.run("git commit -am 'experiment'", shell=True)

    # run training
    score = run_training()
    print("Score:", score)

    if score is None:
        print("Crash detected, reverting")
        subprocess.run("git reset --hard HEAD~1", shell=True)
        write_file(TRAIN_FILE, original_code)
        log_result("HEAD", 0.0, "crash", "Crash detected")
        continue

    # initialize baseline
    if best_score is None:
        best_score = score
        log_result("HEAD", score, "keep", "baseline")
        continue

    # compare
    if score < best_score:
        print("Improvement found!")
        best_score = score
        log_result("HEAD", score, "keep", "improved val_bpb")
    else:
        print("No improvement, reverting")
        subprocess.run("git reset --hard HEAD~1", shell=True)
        write_file(TRAIN_FILE, original_code)
        log_result("HEAD", score, "discard", "no improvement")

    time.sleep(2)