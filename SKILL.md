---
name: autoresearch
version: "1.0.0"
description: "AI agents running research on single-GPU nanochat training automatically"
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

@./program.md
@./README.md
