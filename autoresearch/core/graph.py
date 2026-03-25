"""
Graph Construction Engine

Builds PyTorch Geometric graphs from experiment history.
Each experiment is a node; edges represent similarity.
"""

import torch
from torch_geometric.data import Data


def build_graph(state):
    """
    Build a PyG graph from the experiment state.
    
    Args:
        state: ExperimentState object
        
    Returns:
        torch_geometric.data.Data or None if insufficient experiments
    """
    num_nodes = len(state.experiments)
    
    if num_nodes == 0:
        return None
    
    # Node features: [val_bpb]
    x = []
    for exp in state.experiments:
        val_bpb = exp["result"].get("val_bpb", 0.0)
        x.append([val_bpb])
    
    x = torch.tensor(x, dtype=torch.float32)
    
    # If only one experiment, no edges yet
    if num_nodes == 1:
        return Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long))
    
    # Build edges: connect each experiment to all previous ones
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(i):
            sim = _compute_similarity(state.experiments[i], 
                                      state.experiments[j])
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected
            edge_attr.append([sim])
            edge_attr.append([sim])
    
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _compute_similarity(exp1, exp2):
    """
    Compute similarity between two experiments.
    Based on BPB distance (normalized).
    """
    bpb1 = exp1["result"].get("val_bpb", 0.0)
    bpb2 = exp2["result"].get("val_bpb", 0.0)
    
    # Similarity: inverse of normalized difference
    diff = abs(bpb1 - bpb2)
    sim = max(0.0, 1.0 - diff / (max(bpb1, bpb2, 1.0) + 1e-6))
    
    return sim
