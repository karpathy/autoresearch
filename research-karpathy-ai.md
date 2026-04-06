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

### Dobby: Home Automation Agent (Early-April 2026)

Karpathy used AI agents to take over his home's IoT ecosystem, replacing six separate vendor apps with a single natural language interface.

**Lutron Discovery (early 2026):**
Using Claude Code, an AI agent autonomously discovered Lutron home automation controllers on Karpathy's local WiFi network. It performed IP scans and port checks, retrieved hardware metadata and firmware versions, searched the internet for Lutron system documentation, configured pairing and certificates, and successfully controlled kitchen lights — all without manual setup.

**Dobby the House Elf (April 2026):**
A persistent AI agent named "Dobby the House Elf Claw" that consolidates smart home control across multiple manufacturers and protocols:

- **Devices controlled:** Sonos sound system, lighting, security cameras, window shades, HVAC/climate, pool and spa heating, package detection
- **How it works:** Scanned the local network, discovered devices (including unprotected Sonos endpoints), reverse-engineered undocumented device APIs, and searched the web for documentation — all autonomously
- **Interface:** Natural language commands via WhatsApp (e.g., "bedtime" triggers a coordinated sequence of lights, blinds, and pool heating)
- **Vision:** Integrated models like Qwen to monitor security cameras and detect events such as FedEx trucks arriving

**Key insight:** Karpathy argued that software should expose clean API endpoints rather than complex GUIs, because the primary consumer of a system is increasingly an intelligent agent, not a human. The agent acts as the "glue" unifying fragmented ecosystems. As he put it: "I used to use six different apps, and I don't have to use these apps anymore."

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

### RSS Revival (February 2026)

Karpathy publicly advocated for a return to RSS/Atom feeds as an antidote to algorithmic social media, noting he was "finding myself going back to RSS/Atom feeds a lot more recently" because there's "a lot more higher quality longform and a lot less slop intended to provoke."

- Curated and shared a list of 92 RSS feeds from blogs popular on Hacker News in 2025, distributed as an importable .opml file
- Recommended NetNewsWire as an RSS reader
- Framed RSS as "open, pervasive, hackable" — a decentralized, user-controlled alternative to opaque algorithmic recommendation systems
- Started his own Bear blog (karpathy.bearblog.dev) with native RSS support
- His advocacy inspired community projects like "OneFeed," which combines RSS feeds with LLM agents for intelligent filtering and prioritization of content

The underlying argument: as AI-generated content ("slop") floods algorithmic feeds, curation shifts back to the user. RSS gives that control natively, and LLM agents can layer intelligent filtering on top of open standards rather than proprietary algorithms.

## Themes and Patterns

Across all of Karpathy's 2024-2026 work, several consistent themes emerge:

- **Radical simplicity:** Each project strips away layers of abstraction. From frameworks to raw C, from thousands of lines to hundreds, from dependencies to zero dependencies.
- **Education-first:** Every project doubles as a teaching tool. The code is the curriculum.
- **Progressive minimalism:** The trajectory from nanoGPT to nanochat to microgpt represents a deliberate compression of ideas to their essence.
- **From human to agent:** The progression from manually-run training scripts to fully autonomous research agents (AutoResearch) reflects his vision of AI-augmented scientific discovery.
- **Agent as glue:** The Dobby project demonstrates agents unifying fragmented app ecosystems. Instead of six vendor apps, one conversational agent connects everything — a pattern Karpathy sees as the future of software interaction.
- **Democratization:** Proving that meaningful AI work does not require massive scale, corporate resources, or complex infrastructure. A $100 budget and readable code are enough.

## Lessons: How to Replicate These Approaches

Each of Karpathy's projects encodes a replicable pattern. Below are concrete ways to apply them to your own work.

### 1. Build a "Nano" Version First

**What he did:** Rewrote GPT training in progressively fewer lines — nanoGPT, then microgpt (200 lines, zero deps).

**The principle:** You don't understand a system until you can rebuild its core in minimal code. Stripping dependencies forces you to confront what actually matters.

**How to apply:**
- Pick your app's most complex subsystem (auth, data pipeline, inference engine)
- Rewrite the core logic in a single file, under 500 lines, with minimal or zero dependencies
- Use the result as an onboarding tool, a test harness, or a reference implementation
- If a dependency does something you can write in 20 lines, write it yourself

### 2. Use AI Agents as IoT/API Glue (Dobby Pattern)

**What he did:** Pointed an LLM agent at his home network. It discovered devices, reverse-engineered APIs, and built a unified natural language interface replacing six vendor apps.

**The principle:** Agents can autonomously discover, document, and integrate APIs that were never designed to work together. The agent is the universal adapter.

**How to apply:**
- Inventory your devices or services (smart home, office tools, SaaS APIs)
- Give an LLM agent network access and ask it to discover available endpoints
- Have it read manufacturer docs, reverse-engineer undocumented APIs, and write integration code
- Build a single conversational interface (WhatsApp, Slack, CLI, or web) that controls everything
- Add vision capabilities for camera feeds or visual monitoring if needed

### 3. Autonomous Experimentation (AutoResearch Pattern)

**What he did:** Wrote a research strategy in markdown (program.md). An AI agent ran hundreds of experiments autonomously, modifying code, evaluating results, and keeping improvements.

**The principle:** Separate the research strategy (human) from the experiment execution (agent). The human defines what to optimize and the constraints; the agent runs the loop.

**How to apply:**
- Define your optimization target (latency, accuracy, cost, conversion rate)
- Write a `program.md` describing: the goal metric, what the agent can modify, constraints, and evaluation criteria
- Point the agent at your training script, config file, or deployment parameters
- Let it run overnight — review the results.tsv in the morning
- Start small: hyperparameter tuning before architecture search

### 4. RSS + LLM for Information Curation

**What he did:** Replaced algorithmic social media feeds with 92 curated RSS feeds + an RSS reader, advocating for open standards over opaque algorithms.

**The principle:** As AI slop floods algorithmic feeds, reclaim control over your information diet. RSS gives you the raw stream; an LLM agent can filter, score, and summarize it.

**How to apply:**
- Export your current news sources as an .opml file (most readers support this)
- Import into an RSS reader (NetNewsWire, Miniflux, Feedbin)
- Build an LLM filter that scores articles by relevance to your interests and summarizes the top items daily
- Combine RSS feeds from blogs, GitHub releases, arxiv, and niche forums into one prioritized stream
- Share your curated .opml with your team to bootstrap their information diet

### 5. Vibe-Code a Prototype, Then Harden

**What he did:** Transitioned from 80% manual coding to 80% AI-generated code with 20% human review and refinement.

**The principle:** Use natural language to scaffold quickly, then apply human judgment for correctness, security, and architecture. Speed on the first draft; rigor on the second pass.

**How to apply:**
- Describe your app or feature to an LLM agent in plain language
- Let it generate a working prototype end-to-end
- Review for: security vulnerabilities, edge cases, error handling, and architectural fit
- Iterate conversationally — refine by describing what's wrong, not by rewriting from scratch
- Reserve your manual coding effort for the parts that require deep domain knowledge

### 6. The $100 AI Pipeline

**What he did:** Built nanochat — a complete ChatGPT-style pipeline (tokenizer, training, inference, web UI) trainable for ~$100 on rented GPUs.

**The principle:** A useful AI product does not require millions of dollars. Small models fine-tuned on domain-specific data can outperform large general models for narrow tasks.

**How to apply:**
- Clone nanochat as a starting template
- Prepare your domain-specific dataset (support tickets, internal docs, product descriptions)
- Rent an 8xH100 node for ~4 hours (~$100) and train your own small chat model
- Deploy the resulting model behind a simple API for your specific use case
- For even cheaper experiments, start with microgpt to understand the full pipeline before scaling up

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
- Dobby home automation agent: storyboard18.com (April 2026)
- Lutron integration demo: x.com/karpathy/status/2005067301511630926
- RSS advocacy: x.com/karpathy/status/2018043254986703167
- Karpathy's Bear blog: karpathy.bearblog.dev
