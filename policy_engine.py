from typing import NamedTuple

class Decision(NamedTuple):
    action: str
    reason: str

class Result(NamedTuple):
    val_bpb: float
    complexity: int
    status: str

def decide(candidate: Result, baseline: Result, improvement_threshold: float = 0.001) -> Decision:
    """
    Determines whether to keep or discard a candidate result based on a baseline.

    The policy aims to find results with lower val_bpb (bits per byte),
    while heavily penalizing increases in complexity. Simpler is better.

    Args:
        candidate: The new result to evaluate.
        baseline: The current best result to compare against.
        improvement_threshold: The minimum val_bpb improvement required to
                               outweigh a complexity increase.

    Returns:
        A Decision object with the action ('KEEP' or 'DISCARD') and a reason.
    """
    if candidate.status in ('crash', 'timeout'):
        return Decision('DISCARD', f"Candidate status is '{candidate.status}'.")

    val_bpb_change = baseline.val_bpb - candidate.val_bpb
    complexity_change = baseline.complexity - candidate.complexity

    # Rule 1: Lower val_bpb is a strong signal to keep.
    if val_bpb_change > improvement_threshold:
        return Decision('KEEP', f"Significant val_bpb improvement ({val_bpb_change:.4f}) outweighs complexity considerations.")

    # Rule 2: Worse or equal val_bpb with extra complexity is a clear discard.
    # This also handles the case where val_bpb is slightly better (within the threshold)
    # but complexity has increased.
    if val_bpb_change <= improvement_threshold and complexity_change < 0:
        return Decision('DISCARD', f"val_bpb not significantly better (change: {val_bpb_change:.4f}) and complexity increased by {-complexity_change}.")

    # Rule 3: For marginal or equal val_bpb changes, simplicity is the decider.
    if abs(val_bpb_change) <= improvement_threshold:
        if complexity_change > 0:
            return Decision('KEEP', f"val_bpb is comparable (change: {val_bpb_change:.4f}), but complexity is lower by {complexity_change}.")
        # If complexity is not lower, and val_bpb is not significantly better,
        # there's no reason to accept the candidate unless it's identical or nearly identical.
        if complexity_change == 0 and abs(val_bpb_change) == 0:
             return Decision('KEEP', "Candidate is identical to baseline.")
        else:
             return Decision('DISCARD', f"val_bpb is comparable (change: {val_bpb_change:.4f}), and complexity is not better (change: {complexity_change}).")

    # Rule 4: If val_bpb is worse (and not caught by rule 2, e.g. complexity is same/lower), discard.
    if val_bpb_change < 0:
        return Decision('DISCARD', f"val_bpb is worse (change: {val_bpb_change:.4f}).")

    # Default to keep if no other rule applies (e.g., identical results).
    return Decision('KEEP', "Candidate is identical or acceptably similar to baseline.")
