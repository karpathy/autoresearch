---
name: autoresearch
version: "1.0.0"
description: "Autonomous Machine Learning Research Agent. Iteratively modifies train.py, runs experiments, and optimizes for the lowest val_bpb within strict VRAM constraints."
argument-hint: 'Start autoresearch loop'
allowed-tools: Bash, Read, Write
author: karpathy-adapted
license: MIT
user-invocable: true
metadata:
  openclaw:
    emoji: "🔬"
    requires:
      bins:
        - uv
        - python3
        - git
    tags:
      - machine-learning
      - pytorch
      - research
      - optimization
      - autonomous
---

@E:/SKILLS/autoresearch/program.md