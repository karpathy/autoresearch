#!/usr/bin/env python3
"""
Experiment Novelty Score Checker for autoresearch.
Compares a proposed experiment description against previous runs in results.tsv
to prevent the agent from repeating the same hyperparameter/architectural changes.

Usage:
    python novelty.py "description of the proposed experiment"
"""

import sys
import os
import difflib

def get_past_experiments(tsv_path="results.tsv"):
    """Reads results.tsv and returns a list of past experiment dicts."""
    if not os.path.exists(tsv_path):
        return []
    
    experiments = []
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return []
            headers = lines[0].strip().split("\t")
            desc_idx = -1
            status_idx = -1
            bpb_idx = -1
            if "description" in headers:
                desc_idx = headers.index("description")
            if "status" in headers:
                status_idx = headers.index("status")
            if "val_bpb" in headers:
                bpb_idx = headers.index("val_bpb")
            
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) > max(desc_idx, status_idx, bpb_idx):
                    desc = parts[desc_idx] if desc_idx != -1 else ""
                    status = parts[status_idx] if status_idx != -1 else "UNKNOWN"
                    bpb = parts[bpb_idx] if bpb_idx != -1 else "N/A"
                    experiments.append({
                        "description": desc,
                        "status": status,
                        "val_bpb": bpb
                    })
    except Exception as e:
        print(f"Warning: error reading {tsv_path}: {e}", file=sys.stderr)
    return experiments

def calculate_similarity(desc1, desc2):
    """Calculates a normalized similarity score in [0, 1] between two descriptions."""
    # 1. Normalize strings: lowercase and strip extra whitespace
    d1 = " ".join(desc1.lower().split())
    d2 = " ".join(desc2.lower().split())
    
    if not d1 or not d2:
        return 0.0
        
    # 2. Word-level Jaccard similarity (vocabulary overlap)
    words1 = set(d1.split())
    words2 = set(d2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # 3. Character-level sequence match (handles substrings, prefixes, typos)
    seq_match = difflib.SequenceMatcher(None, d1, d2).ratio()
    
    # Weighted average: 60% Jaccard (concept match), 40% sequence match (textual structure)
    return 0.6 * jaccard + 0.4 * seq_match

def evaluate_novelty(proposed_desc, tsv_path="results.tsv"):
    """Evaluates the novelty of a proposed experiment description."""
    past_exps = get_past_experiments(tsv_path)
    
    if not past_exps:
        return {
            "novelty_score": 1.0,
            "status": "HIGH NOVELTY",
            "matches": [],
            "message": "No previous experiments found. This is the baseline run."
        }
        
    matches = []
    for exp in past_exps:
        sim = calculate_similarity(proposed_desc, exp["description"])
        matches.append({
            "description": exp["description"],
            "status": exp["status"],
            "val_bpb": exp["val_bpb"],
            "similarity": sim,
            "novelty": 1.0 - sim
        })
        
    # Sort matches by similarity descending (highest similarity first)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    most_similar = matches[0]
    max_similarity = most_similar["similarity"]
    novelty_score = 1.0 - max_similarity
    
    if max_similarity >= 0.8:
        status = "REPEATED PATTERN"
        message = "WARNING: This experiment is highly similar to a past run. Consider exploring a different idea."
    elif max_similarity >= 0.5:
        status = "LOW NOVELTY"
        message = "Note: This experiment shares significant similarities with previous runs."
    elif max_similarity >= 0.3:
        status = "MODERATE NOVELTY"
        message = "This experiment shows moderate novelty compared to past runs."
    else:
        status = "HIGH NOVELTY"
        message = "This experiment looks highly novel and distinct from previous runs."
        
    return {
        "novelty_score": novelty_score,
        "status": status,
        "most_similar": most_similar,
        "message": message,
        "matches": matches[:5]  # Top 5 closest matches
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python novelty.py \"proposed experiment description\"")
        sys.exit(1)
        
    proposed_desc = " ".join(sys.argv[1:])
    result = evaluate_novelty(proposed_desc)
    
    print("=" * 60)
    print("🔬 EXPERIMENT NOVELTY REPORT")
    print("=" * 60)
    print(f"Proposed:     {proposed_desc}")
    print(f"Novelty:      {result['novelty_score'] * 100:.1f}% ({result['status']})")
    print(f"Feedback:     {result['message']}")
    
    if 'most_similar' in result:
        sim = result['most_similar']
        print("-" * 60)
        print("Closest Historical Match:")
        print(f"  Description: {sim['description']}")
        print(f"  Status:      {sim['status']}")
        print(f"  Val BPB:     {sim['val_bpb']}")
        print(f"  Similarity:  {sim['similarity'] * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
