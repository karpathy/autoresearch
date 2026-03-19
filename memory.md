# Persistent Memory Layer for Autoresearch

Built a persistent memory layer for @karpathy's `autoresearch` — an autonomous ML research agent that runs hundreds of experiments overnight.

## The Problem
The agent rediscovers the same dead ends every run. It has no memory of what it already tried.

## The Fix
- **Persistent Storage**: Every experiment (hyperparameters + `val_bpb`) gets stored in SQLite with a confidence score.
- **Memory Gate**: Before proposing the next experiment, the agent queries memory first — that's the gate.
- **Normalization**: Hyperparameters are std-normalized before comparison, so `DEPTH=8` and `LR=0.04` don't distort each other's scale.
- **Retrieval**: Cosine similarity finds the closest past experiments in normalized hyperparameter space.
- **Confidence Updates**: High similarity + same verdict → confidence goes up, no LLM needed.
- **Conflict Resolution**: High similarity + opposite verdict → single LLM-as-judge call to resolve the contradiction.
- **Ground Truth**: Resolved verdict gets written back as ground truth.

One expensive operation (LLM call) only when the math can't decide. Everything else is pure geometry.

## Inspiration
The architecture is directly inspired by CoALA (Sumers et al. 2023) — episodic memory for autonomous research agents.
