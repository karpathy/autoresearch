#!/usr/bin/env python3
import os
import sys
import argparse
import json

def analyze_results(tsv_path):
    if not os.path.exists(tsv_path):
        return None
        
    results = []
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return None
            headers = lines[0].strip().split("\t")
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= len(headers):
                    row = dict(zip(headers, parts))
                    # Parse numerical fields
                    try:
                        row["val_bpb"] = float(row["val_bpb"]) if row["val_bpb"] != "N/A" else None
                    except ValueError:
                        row["val_bpb"] = None
                    try:
                        row["memory_gb"] = float(row["memory_gb"]) if row["memory_gb"] != "N/A" else None
                    except ValueError:
                        row["memory_gb"] = None
                    results.append(row)
    except Exception as e:
        print(f"Error reading TSV: {e}", file=sys.stderr)
        return None
        
    if not results:
        return None
        
    total_experiments = len(results)
    kept = sum(1 for r in results if r.get("status", "").upper() == "KEEP")
    discarded = sum(1 for r in results if r.get("status", "").upper() == "DISCARD")
    crashed = sum(1 for r in results if r.get("status", "").upper() == "CRASH")
    
    valid_bpbs = [r["val_bpb"] for r in results if r["val_bpb"] is not None]
    
    baseline_bpb = valid_bpbs[0] if valid_bpbs else None
    best_bpb = min(valid_bpbs) if valid_bpbs else None
    
    improvement = (baseline_bpb - best_bpb) if (baseline_bpb is not None and best_bpb is not None) else 0.0
    improvement_pct = (improvement / baseline_bpb * 100) if (baseline_bpb and improvement) else 0.0
    
    best_experiment = None
    if best_bpb is not None:
        for r in results:
            if r["val_bpb"] == best_bpb:
                best_experiment = r.get("description", "")
                break
                
    # Detect trajectory (improving, plateauing, stuck)
    trajectory = "stuck"
    if len(valid_bpbs) >= 2:
        recent = valid_bpbs[-5:]
        best_recent = min(recent)
        if best_recent < baseline_bpb:
            last_10 = valid_bpbs[-10:]
            if len(last_10) >= 5 and min(last_10) == min(valid_bpbs):
                trajectory = "improving"
            else:
                trajectory = "plateauing"
                
    summary = {
        "total_experiments": total_experiments,
        "kept": kept,
        "discarded": discarded,
        "crashed": crashed,
        "keep_rate": float(f"{kept / total_experiments:.4f}") if total_experiments else 0.0,
        "baseline_bpb": baseline_bpb,
        "best_bpb": best_bpb,
        "improvement": float(f"{improvement:.6f}"),
        "improvement_pct": float(f"{improvement_pct:.2f}"),
        "best_experiment": best_experiment,
        "trajectory": trajectory
    }
    return summary, results

def plot_results(results, output_path):
    try:
        import matplotlib.pyplot as plt
        valid_runs = [(i + 1, r["val_bpb"]) for i, r in enumerate(results) if r["val_bpb"] is not None]
        if not valid_runs:
            print("No valid BPB data to plot.", file=sys.stderr)
            return False
            
        x, y = zip(*valid_runs)
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        plt.plot(x, y, marker='o', color='#007bff', linewidth=2, label='Validation BPB')
        plt.title('Autoresearch Progress Trajectory')
        plt.xlabel('Experiment Index')
        plt.ylabel('Validation BPB (lower is better)')
        plt.grid(True, color='#444444', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"📈 Progress chart saved to {output_path}")
        return True
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Autoresearch experiment analysis CLI tool")
    parser.add_argument("--tsv", default="results.tsv", help="Path to results TSV file")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON format")
    parser.add_argument("--plot", help="Path to save progress plot image (e.g. progress.png)")
    
    args = parser.parse_args()
    
    analysis = analyze_results(args.tsv)
    if not analysis:
        if args.json:
            print(json.dumps({"error": "No results found or file does not exist"}))
        else:
            print(f"Error: No results found in {args.tsv} or file does not exist", file=sys.stderr)
        sys.exit(1)
        
    summary, results = analysis
    
    if args.plot:
        plot_results(results, args.plot)
        
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 50)
        print("📊 AUTORESEARCH EXPERIMENT ANALYSIS")
        print("=" * 50)
        print(f"Total Runs:        {summary['total_experiments']}")
        print(f"🟢 Successful Keeps: {summary['kept']} (Rate: {summary['keep_rate'] * 100:.1f}%)")
        print(f"🔴 Discarded Ideas:  {summary['discarded']}")
        print(f"⚠️ Crashes/OOMs:     {summary['crashed']}")
        print("-" * 50)
        print(f"Baseline BPB:      {summary['baseline_bpb']:.6f}" if summary['baseline_bpb'] else "Baseline BPB:      N/A")
        print(f"Best BPB:          {summary['best_bpb']:.6f}" if summary['best_bpb'] else "Best BPB:          N/A")
        print(f"Overall Gain:      {summary['improvement']:.6f} ({summary['improvement_pct']:.2f}%)")
        print(f"Best Experiment:   {summary['best_experiment']}")
        print(f"Trajectory:        {summary['trajectory'].upper()}")
        print("=" * 50)

if __name__ == "__main__":
    main()
