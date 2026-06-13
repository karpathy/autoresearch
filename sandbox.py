#!/usr/bin/env python3
import subprocess
import sys
import os

def get_best_bpb():
    best_bpb = float('inf')
    if os.path.exists('results.tsv'):
        with open('results.tsv', 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]: # skip header
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].upper() == 'KEEP':
                    try:
                        bpb = float(parts[1])
                        if bpb < best_bpb:
                            best_bpb = bpb
                    except ValueError:
                        pass
    return best_bpb

def run_experiment(description):
    print(f"🔬 Sandbox Critic: Starting experiment '{description}'...")
    
    # 1. Check syntax first (Critic: Linter)
    print("🧹 Critic Phase 1: Linting and compiling train.py...")
    comp = subprocess.run([sys.executable, "-m", "py_compile", "train.py"], capture_output=True, text=True)
    if comp.returncode != 0:
        print("❌ CRASH: train.py has Python syntax errors!")
        print(comp.stderr)
        rollback()
        log_result("N/A", "N/A", "CRASH", description)
        return

    # 2. Run the training script
    print("⏳ Critic Phase 2: Running 5-minute training budget... (tail run.log for live output)")
    with open("run.log", "w", encoding="utf-8") as f:
        train = subprocess.run(["uv", "run", "train.py"], stdout=f, stderr=subprocess.STDOUT)
    
    # 3. Analyze the outcomes
    print("📊 Critic Phase 3: Analyzing results...")

    # Extract metrics
    val_bpb = None
    peak_vram = None
    if os.path.exists("run.log"):
        with open("run.log", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("val_bpb:"):
                    try:
                        val_bpb = float(line.split(":")[1].strip())
                    except ValueError: pass
                elif line.startswith("peak_vram_mb:"):
                    try:
                        peak_vram = float(line.split(":")[1].strip())
                    except ValueError: pass

    if train.returncode != 0 or val_bpb is None:
        print("❌ CRASH: Runtime error, OOM, or missing metrics detected in run.log.")
        rollback()
        log_result(val_bpb if val_bpb else "N/A", peak_vram if peak_vram else "N/A", "CRASH", description)
        return

    print(f"📈 Result: val_bpb = {val_bpb:.6f}, RAM = {peak_vram}MB")
    
    best_bpb = get_best_bpb()
    if best_bpb == float('inf'):
        print("ℹ️ No previous KEEP records found in results.tsv. Setting baseline.")
        best_bpb = 5.0 # Loose initial fallback

    if val_bpb < best_bpb:
        print(f"✅ SUCCESS: BPB ({val_bpb:.4f}) improved over best ({best_bpb:.4f})! Keeping changes.")
        subprocess.run(["git", "commit", "-am", f"KEEP: {description} (BPB: {val_bpb:.4f})"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_result(val_bpb, peak_vram, "KEEP", description)
    else:
        print(f"📉 FAIL: BPB {val_bpb:.6f} is not better than best {best_bpb:.6f}. Discarding.")
        rollback()
        log_result(val_bpb, peak_vram, "DISCARD", description)

def log_result(bpb, vram, status, desc):
    if not os.path.exists("results.tsv"):
        with open("results.tsv", "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
            
    commit_hash = "N/A"
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf-8').strip()
    except subprocess.CalledProcessError: pass
    
    mem_gb = f"{float(vram) / 1024:.2f}" if vram != "N/A" else "N/A"
    bpb_str = f"{float(bpb):.6f}" if bpb != "N/A" else "N/A"
    
    with open("results.tsv", "a", encoding="utf-8") as f:
        f.write(f"{commit_hash}\t{bpb_str}\t{mem_gb}\t{status}\t{desc}\n")

def rollback():
    print("⏪ Rolling back to previous stable git state...")
    subprocess.run(["git", "reset", "--hard", "HEAD"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sandbox.py \"Description of your experiment\"")
        sys.exit(1)
    
    run_experiment(" ".join(sys.argv[1:]))
