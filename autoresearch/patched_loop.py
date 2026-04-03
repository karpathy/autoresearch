"""
GNN-driven Research Automation Loop

This is the main integration point. It wraps your existing train.py logic
and augments it with:
- Graph-based experiment tracking
- GNN policy for deterministic decision-making
- Zero-hallucination action selection
"""

import torch
from autoresearch.core.state import ExperimentState, compute_reward
from autoresearch.core.graph import build_graph
from autoresearch.core.actions import apply_action, ACTIONS
from autoresearch.core.memory import Memory
from autoresearch.agents.gnn_policy import GNNPolicy
from autoresearch.agents.agent import ResearchAgent
from autoresearch.interface.emitter import SafeEmitter


def run_patched_loop(run_experiment_fn, initial_config, num_steps=50, 
                     device="cpu", verbose=True):
    """
    Main research loop with GNN-driven optimization.
    
    Args:
        run_experiment_fn: Function that takes config and returns result dict
                          with at least {"val_bpb": float}
        initial_config: Starting configuration (dict)
        num_steps: Number of optimization steps
        device: torch device (cpu or cuda)
        verbose: Print progress
        
    Returns:
        state: Final ExperimentState with all history
    """
    
    # Initialize state, memory, and graph
    state = ExperimentState()
    memory = Memory()
    
    # Initialize policy and agent
    policy = GNNPolicy(in_dim=1, hidden_dim=32, action_dim=len(ACTIONS), 
                       num_heads=2)
    agent = ResearchAgent(policy, agent_id=0, device=device)
    
    config = initial_config.copy()
    
    if verbose:
        print(f"🚀 Starting GNN-driven research loop for {num_steps} steps")
        print(f"Initial config: {config}")
        print()
    
    for step in range(num_steps):
        # Run experiment with current config
        result = run_experiment_fn(config)
        state.add_experiment(config, result)
        memory.add(config, result)
        
        reward = compute_reward(result)
        
        # Build graph from all past experiments
        graph = build_graph(state)
        
        if graph is None:
            # Not enough experiments yet, random action
            action_idx = torch.randint(0, len(ACTIONS), (1,)).item()
            action = ACTIONS[action_idx]
            if verbose:
                print(f"[STEP {step:3d}] Insufficient history, random action: {action}")
        else:
            # Use GNN to select action
            action, logits = agent.act(graph, use_sampling=False)
            
            if verbose:
                best_logit = torch.max(logits).item()
                print(f"[STEP {step:3d}] Action: {action:16s} | "
                      f"Val BPB: {result['val_bpb']:.6f} | "
                      f"Logit: {best_logit:.4f}")
        
        # Apply action to get next config
        config = apply_action(config, action)
    
    if verbose:
        print()
        print("=" * 70)
        print(SafeEmitter.summarize(state))
        print("=" * 70)
    
    return state


def main():
    """Default entry point for testing."""
    
    # Placeholder initial config
    initial_config = {
        "lr": 0.001,
        "layers": 2,
        "batch_size": 32,
        "vocab_size": 50257,
        "seq_len": 256,
        "hidden_dim": 768
    }
    
    # Placeholder experiment runner (you'll replace with actual train.py logic)
    def mock_run_experiment(config):
        # In practice, this calls your train.py
        import random
        val_bpb = 5.0 + random.random()  # Mock: random performance
        return {"val_bpb": val_bpb}
    
    # Run the patched loop
    state = run_patched_loop(
        run_experiment_fn=mock_run_experiment,
        initial_config=initial_config,
        num_steps=20,
        verbose=True
    )
    
    return state


if __name__ == "__main__":
    main()