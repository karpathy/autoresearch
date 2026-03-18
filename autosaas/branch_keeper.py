def decide_keep_or_revert(required_gate_results: dict[str, bool]) -> str:
    if not required_gate_results:
        return "revert"
    return "keep" if all(required_gate_results.values()) else "revert"
