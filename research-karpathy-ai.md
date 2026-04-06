# Andrej Karpathy: Building with AI (2024-2026)

## Overview

Andrej Karpathy is one of the most influential figures in modern AI, known for making deep learning accessible through minimal, educational implementations. After co-founding OpenAI, leading Tesla's Autopilot AI, and creating Stanford's CS231n course, he shifted focus in 2024-2026 toward prolific open-source building and AI-native education.

His recent work follows a clear philosophy: strip away abstraction, build from scratch, and make AI understandable to anyone willing to read the code.

## Background

- **OpenAI** - Co-founding member and VP of Research
- **Tesla** - Director of AI, led Autopilot and Full Self-Driving vision systems
- **Stanford** - PhD under Fei-Fei Li; created and led CS231n (Convolutional Neural Networks for Visual Recognition), one of the first deep learning courses at Stanford
- **YouTube** - Ongoing educational content with millions of views across his "Neural Networks: Zero to Hero" and "Deep Dive into LLMs" series

## Major Projects

### nanoGPT (2023)

The project that established the "nano" philosophy. A minimal repository for training and finetuning medium-sized GPTs, nanoGPT prioritized readability and simplicity over feature completeness. It became the foundation and template for everything that followed.

- Repository: github.com/karpathy/nanoGPT

### llm.c (2024-2026)

LLM training in pure C and CUDA with no heavy dependencies. Approximately 3,000 lines of C/CUDA for the optimized version.

Key characteristics:
- Multi-GPU training with bfloat16 precision and flash attention
- ~7% faster than PyTorch nightly, ~46% faster than PyTorch stable 2.3.0
- Trains GPT-2 (124M parameters) at speeds matching or exceeding framework-based implementations
- Educational focus: demonstrates the progression from CPU baseline to GPU-accelerated CUDA kernels

The project proved that LLM training does not require Python or deep learning frameworks.

- Repository: github.com/karpathy/llm.c

### Eureka Labs (July 2024)

An "AI Native School" startup founded to transform education with AI teaching assistants.

- **Mission:** Scale expert teaching by pairing teacher-designed course materials with AI assistants
- **First product:** LLM101n - an undergraduate-level course titled "Let's build a Storyteller," a full-stack guide to training your own AI from scratch
- **Educational tracks:** "Neural Networks: Zero to Hero," "Deep Dive into LLMs like ChatGPT," "How I use LLMs"
- **Vision:** Address the scarcity of passionate, expert teachers by leveraging generative AI to deliver high-quality instruction at scale

### nanochat (October 2025)

A complete ChatGPT-style pipeline buildable for approximately $100.

- ~8,000 lines of PyTorch covering the full stack: tokenizer training through web interface
- Training cost: ~$100 for ~4 hours on an 8xH100 GPU node
- At $100: conversational, can write stories and poems, answers simple questions
- At ~$1,000: more coherent, handles simple math, code problems, and multiple choice tests
- Serves as the capstone project for the LLM101n course
- Maintains a community leaderboard for the "GPT-2 speedrun" (wall-clock time to reach GPT-2 capability)

- Repository: github.com/karpathy/nanochat

### microgpt (February 2026)

The extreme distillation of the nano philosophy: a complete GPT in a single file, 200 lines of pure Python, with zero dependencies.

Includes:
- Dataset handling and tokenization
- A from-scratch autograd engine
- GPT-2-like neural network architecture
- Adam optimizer
- Complete training and inference loops

All in 200 lines with no imports beyond the Python standard library.

- Blog post: karpathy.github.io/2026/02/12/microgpt/

### AutoResearch (March 2026)

An autonomous AI experimentation framework. This is the repository containing this document.

- ~630 lines of modifiable training code (train.py), MIT licensed
- AI agents read a markdown "program" (program.md), form hypotheses, modify code, run experiments, and evaluate results autonomously
- Works on git feature branches: the agent commits improvements and reverts failures
- In the initial experiment: 700+ experiments over 2 days, discovering 20 optimizations
- Achieved 21,000+ GitHub stars within days of release; 8.6+ million views on the announcement
- Future direction: massively asynchronous, collaborative AI agents (SETI@home style)

The key insight: the human writes the research strategy (program.md), and the AI agent executes the experiment loop indefinitely.

- Repository: github.com/karpathy/autoresearch

### LLM Wiki (2026)

A living archive for AI ideas using LLMs to generate, curate, and refine wiki-style articles. Treats AI-generated drafts as starting points for iterative refinement. Serves as an evolving "idea file" for the AI community.

## Ideas and Influence

### Vibe Coding

Karpathy coined the term "vibe coding" to describe a new mode of programming where developers describe their intent in natural language and AI generates the code. He predicted this would reshape software development, enabling hobbyists to build apps and websites through conversational prompts rather than traditional coding.

### 2025 LLM Year in Review

In December 2025, Karpathy published a comprehensive analysis identifying six paradigm shifts in the LLM landscape:

1. **Reinforcement Learning from Verifiable Rewards (RLVR)** - Emerged as the dominant training methodology, fundamentally altering the LLM production pipeline
2. **"Animals vs. Ghosts"** - LLMs as entities optimized under entirely different constraints than biological intelligence
3. **Crisis of Benchmarks** - Recognition that benchmarks are immediately susceptible to overfitting and synthetic data manipulation
4. **LLM Application Layer** - Rise of specialized orchestrators bundling multiple LLM calls (exemplified by Cursor in 2025)
5. **Vibe Coding** - AI enabling non-programmers to build software through natural language
6. **Claude Code and LLM Agents** - Recognition of effective agent patterns for extended problem solving

His key observation: humans have exploited less than 10% of this computing paradigm's potential.

### Personal Trajectory

By early 2026, Karpathy noted feeling "dramatically behind" as a programmer, observing that the profession itself is being refactored as human contributions become "sparse and between." He expressed belief that he could be "10X more powerful" by properly orchestrating AI tools - a perspective that directly informed the creation of AutoResearch.

## Themes and Patterns

Across all of Karpathy's 2024-2026 work, several consistent themes emerge:

- **Radical simplicity:** Each project strips away layers of abstraction. From frameworks to raw C, from thousands of lines to hundreds, from dependencies to zero dependencies.
- **Education-first:** Every project doubles as a teaching tool. The code is the curriculum.
- **Progressive minimalism:** The trajectory from nanoGPT to nanochat to microgpt represents a deliberate compression of ideas to their essence.
- **From human to agent:** The progression from manually-run training scripts to fully autonomous research agents (AutoResearch) reflects his vision of AI-augmented scientific discovery.
- **Democratization:** Proving that meaningful AI work does not require massive scale, corporate resources, or complex infrastructure. A $100 budget and readable code are enough.

## References

- Karpathy's personal site: karpathy.ai
- AutoResearch: github.com/karpathy/autoresearch
- nanochat: github.com/karpathy/nanochat
- llm.c: github.com/karpathy/llm.c
- nanoGPT: github.com/karpathy/nanoGPT
- microgpt blog post: karpathy.github.io/2026/02/12/microgpt/
- Eureka Labs announcement: TechCrunch, July 16, 2024
- 2025 LLM Year in Review: karpathy.bearblog.dev/year-in-review-2025/
- Neural Networks: Zero to Hero: karpathy.ai/zero-to-hero.html
