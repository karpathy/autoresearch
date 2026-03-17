# Core Concepts: Autonomous Research System

Clear explanations of the key concepts behind this system.

---

## What is autoresearch?

**Autoresearch** is an autonomous research platform where AI agents run structured experiments on a fixed 5-minute time budget.

### The Loop

```
Modify Code → Run Experiment → Evaluate → Keep or Discard → Repeat
```

1. **Agent modifies** `train.py` (hyperparameters, architecture, etc.)
2. **Agent runs** training for exactly 5 minutes (fixed time budget)
3. **Agent evaluates** result: `val_bpb` (validation bits per byte) — lower is better
4. **Agent keeps** improvement or discards (reverts commit)
5. **Repeat** with new ideas

### Why 5 minutes?

- **Consistency**: All experiments are directly comparable (same time budget)
- **Velocity**: ~12 experiments per hour, ~100 per night of sleep
- **Platform-neutral**: Time budget works on any hardware (though results adjust)
- **Rapid iteration**: Fast feedback loop for agent exploration

### What can you research?

Not just neural networks:
- **Hyperparameters**: Learning rate, batch size, weight decay
- **Architectures**: Depth, width, attention patterns
- **Optimizers**: Adam, SGD, Muon, custom variants
- **Data**: Different datasets, preprocessing, augmentation
- **Training dynamics**: Gradient clipping, warm-up schedules, curricula
- **Regularization**: Dropout, layer norm, techniques
- **Loss functions**: Cross-entropy variants, custom objectives

The key: Anything you can implement in `train.py` in 5 minutes of computation.

---

## What is semantic linking?

**Semantic linking** means understanding meaning, not just keywords.

### Traditional Wiki (Keyword-Based)

```
Article: "Learning Rate"
Manual backlinks:
  - Hyperparameter Optimization
  - Stochastic Gradient Descent
  - Training Dynamics
```

Problem: Human must manually create every link.

### Murmur Wiki (Semantic)

```
Article: "Hyperparameter Optimization Results"
Content: "We found LR=0.04 optimal, BS=32 best..."

knowledge-tensor AI reads this and automatically:
  - Finds: "learning rate", "batch size", "optimization"
  - Links to: related articles on these topics
  - Creates backlinks: "optimization" articles mention this finding
  - Builds graph: relationships between concepts
```

Benefits:
- **Self-populating**: No manual link creation
- **Bidirectional**: Finding "B" mentions concept from article "A" → backlink appears
- **Semantic clusters**: Groups related articles (all about "efficiency trade-offs")
- **Growing**: Richer connections as more research accumulates

---

## What is constraint-theory validation?

**Constraint-theory** validates claims deterministically using geometry, not probability.

### Traditional Validation (Probabilistic)

```
Claim: "LR=0.04 is optimal"
Validation: Bayesian probability
  - Prior: Is LR optimization plausible? (90% yes)
  - Likelihood: How many experiments support it? (95% consistency)
  - Posterior: 90% × 95% = 85.5% confidence

Problem: Depends on choice of prior, can't explain reasoning
```

### Constraint-Theory Validation (Deterministic)

```
Claim: "LR=0.04 is optimal"
Validation: Geometric constraint solving

Step 1: Origin-centric geometry (Ω)
  Define ground state:
    - Baseline val_bpb = 1.041
    - Model depth = 8
    - All learning rates in valid range [0.001, 0.1]

Step 2: Φ-folding operator
  Map continuous space to discrete valid states:
    LR=0.001 → val_bpb=1.042 ✓
    LR=0.01  → val_bpb=1.035 ✓
    LR=0.04  → val_bpb=0.998 ← claim is here
    LR=0.1   → val_bpb=1.150 ✓

Step 3: Rigidity-curvature duality
  Check: Claim satisfies all constraints?
    - All LRs within valid range? ✓
    - All val_bpb values realistic? ✓
    - Curvature (improvement slope) consistent? ✓

Result: Confidence = 0.987 (deterministic)
Explanation: "Claim maps exactly to observed discrete state"
```

### Why Deterministic is Better

| Aspect | Probabilistic | Deterministic |
|--------|---------------|---------------|
| **Confidence** | 85.5% (depends on prior) | 0.987 (geometric) |
| **Explanation** | "Bayesian posterior" (opaque) | "Constraint satisfied" (clear) |
| **Reproducibility** | Depends on Bayesian choice | Same everywhere |
| **Speed** | Slow (MCMC sampling) | Fast (KD-tree lookup) |

---

## What is cold storage?

**Cold storage** is archival for old data you rarely access but want to keep.

### Three Tiers

```
Tier 1: HOT (Working)
├─ Latest experiments
├─ Current research
└─ Fast access (< 1ms)

Tier 2: WARM (Recent Archive, <30 days)
├─ Last month's research
├─ Compressed
└─ Slower access (~1s)

Tier 3: COLD (Long-term Archive, >30 days)
├─ Old experiments
├─ Highly compressed
└─ Very slow access (hours)
```

### Retention Policies

**Global Policy**: Applies to all research
```json
{
  "default_retention_days": 90,
  "archive_raw_data": true,
  "compress_after_days": 30
}
```

**Per-File Override**: For special cases
```json
{
  "file_path": "breakthrough_experiments/*.json",
  "retention_days": 365,
  "reason": "Keep longer for future reference"
}
```

### Why?

- **Space**: Raw experiments are huge (raw data, logs, metrics)
- **Cost**: Cloud storage costs scale with data size
- **Access**: Most old data never accessed again
- **Compliance**: Some data must be deleted after X days

### Summarization Before Archive

```
Raw: 500MB experiment data
  ├─ GPU traces (detailed, huge)
  ├─ All training checkpoints
  ├─ Full loss curves
  └─ Hyperparameter sweep logs

Summarized: 5MB archive
  ├─ Final metrics (val_bpb, memory, time)
  ├─ Best checkpoint
  ├─ Summary of findings
  └─ Link to published murmur article

Compression: 99% size reduction
```

---

## What is flowstate (sandbox mode)?

**Flowstate** lets agents explore radical, unvalidated ideas without contaminating the main knowledge graph.

### Normal Mode vs Flowstate

```
NORMAL MODE (Curated Research)
├─ Experiments must be sound
├─ Findings fact-checked before publication
├─ Directly updates murmur (public wiki)
└─ High-quality but slow

FLOWSTATE MODE (Exploratory)
├─ Try anything, no validation required
├─ Ideas stay in sandbox (temporary)
├─ Later manual review: promote to wiki?
└─ Fast exploration, some failures expected
```

### Flowstate Workflow

```
1. Agent declares: "Entering flowstate for radical hypothesis X"
   └─ Creates temporary sandbox knowledge graph

2. Agent explores freely
   ├─ Run wild experiments (may fail)
   ├─ Test contradictory hypotheses
   ├─ Record ALL thinking (not just successes)
   └─ No fact-checking, no validation

3. Record everything
   ├─ Timestamped logs of reasoning
   ├─ Failed hypotheses (valuable!)
   ├─ Reasoning chains
   └─ Insights even if hypothesis failed

4. Exit flowstate
   └─ Sandbox data collected for review

5. Manual curation
   ├─ Human reviews: which findings promote to main wiki?
   ├─ Which hypotheses worth investigating more?
   ├─ What did we learn even from failures?
   └─ Decision: keep, merge, or discard?

6. Constraint validation (for promoted findings)
   ├─ Run promoted claims through fact-checking
   └─ Only high-confidence findings update murmur

7. Archive
   └─ All flowstate data moves to cold storage
       (valuable for pattern mining later)
```

### Benefits

- **Radical exploration**: Not limited to safe incremental improvements
- **Learning from failure**: Failed hypotheses archived for future analysis
- **Rapid iteration**: No fact-checking delays during exploration
- **Discovery**: Novel connections found by free association
- **History**: Complete reasoning trace (not just results)

### Example

```
FLOWSTATE: "Testing if biological constraint systems
can improve neural network generalization"

Hypothesis 1: Protein folding patterns → weight initialization
  - Implementation: tried SE(3) rotational invariance
  - Result: FAILED (training unstable)
  - Learning: SE(3) constraints too restrictive
  - Archive: Failure analysis for future reference

Hypothesis 2: Evolutionary game theory → loss function
  - Implementation: Tried Nash equilibrium optimization
  - Result: SLOW (expensive computation)
  - Learning: Game theory concepts valuable, need simplification
  - Archive: Complexity analysis saved

Hypothesis 3: Cellular consensus → attention pattern
  - Implementation: Protein lattice coordination
  - Result: MODERATE SUCCESS (5% val_bpb improvement, 2x slower)
  - Learning: Good idea but needs optimization
  - Archive: Promote to main research, continue optimization

EXIT FLOWSTATE
→ Promote Hypothesis 3 to normal mode
→ Archive all analysis (even failures)
→ Continue with next flowstate or new direction
```

---

## What is spreader-tool orchestration?

**Spreader-tool** coordinates multiple specialist agents investigating in parallel.

### Sequential vs Parallel

```
SEQUENTIAL (1 agent)
Time: ████████████ (12 experiments/hour = 100/night)

PARALLEL (4 agents)
Agent A: ████████████
Agent B: ████████████  (parallel)
Agent C: ████████████  (12 exp/hr each)
Agent D: ████████████
Time: ██ (faster overall)
Total: ~48 experiments/hour = 400/night
```

### Specialization

```
Research question: "Optimize training efficiency"

Sequential agent:
├─ Try random ideas
├─ Discover LR=0.04 is good
├─ Discover BS=32 is good
└─ Takes 20 experiments to find both

Parallel with specialists:
├─ Agent A (hyperparameter expert):
│  └─ Tries LR variations, BS variations
│     (finds optimal in 6 experiments)
├─ Agent B (architecture expert):
│  └─ Tries depth/width/attention patterns
│     (finds architectural improvements in 8 experiments)
├─ Agent C (optimizer expert):
│  └─ Tries different optimizers
│     (finds Muon works 3% better in 5 experiments)
└─ Agent D (synthesis):
   └─ Combines findings: "LR+BS+optimizer stack"
      (validates combination in 3 experiments)

Total: 22 experiments (vs 20 sequential)
But: Deeper exploration of each dimension
And: Parallel execution (4x faster wall-clock)
```

### Agent Collaboration

```
Agent A (LR finder):
  "LR=0.04 optimal for BS≤32"
  → shares insight with Agent B

Agent B (Architecture):
  "With this LR, depth=6 works better than 8"
  → shares insight with Agent C

Agent C (Optimizer):
  "Muon + Depth6 + LR0.04 is sweet spot"
  → shares synthesis back to Agent A

Agent A (updated knowledge):
  "Confirmed: this combo is stable at BS=32"
  → final validation

RESULT: Collaborative finding beats individual discoveries
```

---

## What is a semantic knowledge graph?

**Knowledge graph** = interconnected web of research findings and concepts.

### Structure

```
Nodes (articles/facts):
├─ "Learning Rate Optimization Results"
├─ "Batch Size Trade-offs"
├─ "Hardware Efficiency Analysis"
├─ "Pareto Frontier Concept"
└─ "Transformer Architecture Basics"

Edges (relationships):
├─ LR article → "mentions" → Pareto Frontier
├─ BS article → "optimizes" → Hardware Efficiency
├─ Both articles → "related to" → Training Efficiency
└─ Training Efficiency → "applies to" → Transformer Basics
```

### Queries

```
"What affects training speed?"
→ Query knowledge graph
→ Return: LR, BS, architecture depth, optimizer
  with evidence from articles

"What did we learn about efficiency?"
→ Query knowledge graph
→ Return: all articles tagged "efficiency"
  in dependency order

"Why did experiment X fail?"
→ Query knowledge graph
→ Return: related findings that explain the failure
  with references to flowstate archives
```

### Dynamic Growth

```
Week 1: 10 articles, 15 relationships
Week 2: 15 articles, 28 relationships (adding new research)
Week 3: 22 articles, 47 relationships (connections between areas)
Week 4: 30 articles, 73 relationships (emerging patterns visible)

As graph grows:
├─ Researchers see connections across topics
├─ Duplicate work avoided ("already tested that")
├─ Novel insights emerge (unexpected combinations)
└─ Predictions improve ("these always work together")
```

---

## What is a fact-checking pipeline?

**Fact-checking** validates research claims before they affect decisions.

### Four Stages

```
1. CONSTRAINT CONSISTENCY CHECK
   ├─ Does claim satisfy geometric constraints?
   ├─ Example: "LR=0.04" must be in valid range [0.001, 0.1]
   └─ Confidence impact: 0.0 if violates, 1.0 if consistent

2. EXPERIMENTAL EVIDENCE REVIEW
   ├─ How much data supports claim?
   ├─ Example: "LR=0.04 optimal" backed by 4 independent runs
   └─ Confidence: N independent confirmations → higher score

3. CROSS-REFERENCE CHECK
   ├─ Does it contradict other known facts?
   ├─ Example: Any conflicting findings in murmur graph?
   └─ Confidence: 0 conflicts → higher; 1+ conflict → investigate

4. CONSENSUS LAYER
   ├─ Do community members agree?
   ├─ Weighted by expertise (expert vote > novice vote)
   └─ Confidence: agreement → higher score
```

### Output

```
Claim: "Batch size 32 is optimal"

Status: VERIFIED
Confidence: 0.987 (high)
Evidence:
  ✓ Satisfies constraints (valid range)
  ✓ 4 independent experiments
  ✓ No contradictions found
  ✓ 12 community votes, all positive

Caveats:
  ⚠ Only for model_depth ≤ 8
  ⚠ Only for VRAM ≥ 44GB
  ⚠ May not generalize to vision tasks

Revision history:
  2026-03-17 18:30 - Initial validation
  2026-03-18 09:15 - Community review completed
  2026-03-18 12:00 - Added depth constraint
```

---

## Putting It All Together

### A Complete Research Cycle

```
RESEARCH QUERY
"Optimize training efficiency under 48GB memory"
         ↓
ORCHESTRATION (spreader-tool)
Multiple specialist agents (hyperparameter, architecture, optimizer)
         ↓
EXPERIMENTATION (autoresearch)
Each agent runs 12 experiments/hour, 100+ total
         ↓
AGGREGATION
Synthesize findings into coherent research summary
         ↓
VALIDATION (constraint-theory)
Fact-check all claims with geometric validation
         ↓
KNOWLEDGE GRAPH (murmur)
Publish findings, auto-link to related articles
         ↓
MONITORING (spreadsheet-moment)
Display results on dashboard, show Pareto frontier
         ↓
FLOWSTATE REVIEW
Explore radical ideas in sandbox (separate from wiki)
         ↓
ARCHIVAL
Move old experiments to cold storage, summarize
         ↓
OUTPUT GENERATION
Podcast script, wiki updates, citations ready
         ↓
COMMUNITY REVIEW (murmur bulletin)
Researchers vote, challenge, extend findings
         ↓
NEXT ITERATION
Results inform next week's research direction
```

---

## Key Insights

1. **Fixed time budget** (5 min) makes experiments comparable regardless of what's tested
2. **Semantic linking** auto-organizes findings without manual curation
3. **Deterministic validation** provides repeatable, explainable confidence scores
4. **Cold storage** balances accessibility with cost
5. **Flowstate** enables radical exploration without contaminating main research
6. **Parallel agents** explore faster than sequential research
7. **Knowledge graph** reveals connections invisible in individual papers
8. **Community layer** adds distributed consensus to validation

---

## Questions?

See:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete system design
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — How to connect components
- **[superinstance-papers](https://github.com/SuperInstance/superinstance-papers)** — Theoretical foundations
