import os
import subprocess
import time

def run_command(command):
    """Executes a shell command and returns stdout, stderr, and exit code."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def get_git_hash():
    stdout, _, _ = run_command("git rev-parse --short HEAD")
    return stdout.strip()

def log_result(commit, bpb, vram, status, description):
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{bpb:.6f}\t{vram:.1f}\t{status}\t{description}\n")

def update_knowledge(note):
    with open("knowledge.md", "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')}: {note}\n")

def run_experiment(description):
    """Runs train.py, handles crashes with up to 3 retries, and records results."""
    max_retries = 3
    attempt = 0
    
    while attempt <= max_retries:
        print(f"\n>>> Attempt {attempt} for: {description}")
        stdout, stderr, code = run_command("uv run train.py")
        
        # Save full log for analysis
        with open("run.log", "w") as f:
            f.write(stdout)
            f.write(stderr)

        if code == 0:
            try:
                # Basic parsing of output metrics
                bpb = 0.0
                vram_gb = 0.0
                for line in stdout.split('\n'):
                    if line.startswith('val_bpb:'):
                        bpb = float(line.split(':')[1].strip())
                    if line.startswith('peak_vram_mb:'):
                        vram_gb = float(line.split(':')[1].strip()) / 1024
                return bpb, vram_gb, "success"
            except Exception as e:
                print(f"Error parsing metrics: {e}")
                return 0.0, 0.0, "crash"
        else:
            print(f"!!! Crash detected on attempt {attempt}.")
            if attempt < max_retries:
                print("Checking logs (run.log)...")
                print("AGENT ACTION REQUIRED: Fix the error in train.py and save the file.")
                input("Press Enter to retry once you have fixed the code...")
                attempt += 1
            else:
                print("Max retries reached. Experiment failed.")
                return 0.0, 0.0, "crash"

if __name__ == "__main__":
    if not os.path.exists("results.tsv"):
        with open("results.tsv", "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    if not os.path.exists("knowledge.md"):
        with open("knowledge.md", "w") as f:
            f.write("# Research Knowledge Base\n\n")

    desc = input("Experiment description: ")
    bpb, vram, status = run_experiment(desc)
    
    commit = get_git_hash()
    log_result(commit, bpb, vram, status, desc)
    
    if status == "success":
        print(f"Run completed. val_bpb: {bpb:.6f}")
        note = input("Add insight to knowledge.md (optional, press Enter to skip): ")
        if note:
            update_knowledge(note)
    else:
        update_knowledge(f"FAILED experiment: {desc}. Check logs for details.")
    
    print(f"Experiment logged. Status: {status}")
