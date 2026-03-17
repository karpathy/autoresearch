# SuperInstance + AutoResearch Integration Guide

A practical guide to integrating autoresearch with SuperInstance ecosystem components.

## Overview

Each SuperInstance component plays a specific role in the autonomous research system:

```
ORCHESTRATION          EXPERIMENTATION        KNOWLEDGE       VALIDATION        OUTPUT
┌────────────────┐     ┌──────────────────┐   ┌────────┐      ┌──────────────┐  ┌──────────┐
│ spreader-tool  │────→│   autoresearch   │──→│ murmur │──→──→│ constraint-  │─→│ podcast/ │
│ + SwarmOrch.   │     │ (5-min budget)   │   │ (wiki) │      │ theory (validate) │ wiki   │
└────────────────┘     └──────────────────┘   └────────┘      └──────────────┘  └──────────┘
                              ↓
                       ┌──────────────────┐
                       │ spreadsheet-moment│
                       │ (monitoring)     │
                       └──────────────────┘
```

---

## Component-by-Component Integration

### 1. spreader-tool: Orchestration Engine

**Role**: Route research queries to specialist agents, coordinate parallel investigations

**Setup**:
```bash
npm install @superinstance/spreader
# or use CLI
spreader-cli --config research.config.json
```

**Configuration** (`research.config.json`):
```json
{
  "agents": [
    {
      "name": "hyperparameter-researcher",
      "model": "claude-opus-4-6",
      "instructions": "Investigate learning rate and batch size effects",
      "tools": ["run_autoresearch_experiment"],
      "parallelism": 3
    },
    {
      "name": "architecture-explorer",
      "model": "claude-opus-4-6",
      "instructions": "Test different model architectures (depth, width, attention patterns)",
      "tools": ["run_autoresearch_experiment"],
      "parallelism": 2
    }
  ],
  "aggregator": {
    "model": "claude-opus-4-6",
    "instructions": "Synthesize findings from all agents into coherent research summary"
  }
}
```

**Integration with autoresearch**:
```javascript
// In spreader-tool agent
const tool_run_autoresearch_experiment = async (config) => {
  // 1. Modify train.py with experiment parameters
  await modifyTrainPy(config.hyperparameters);

  // 2. Run: uv run train.py
  const result = await execSync('uv run train.py');

  // 3. Parse results
  const { val_bpb, memory_gb, training_seconds } = parseResults(result);

  // 4. Log to results.tsv
  await appendResultsTsv({
    commit: getCommitHash(),
    val_bpb: val_bpb,
    memory_gb: memory_gb,
    status: val_bpb < baseline ? 'keep' : 'discard'
  });

  return { val_bpb, memory_gb, training_seconds };
};
```

**How it works**:
1. User poses research question: "Optimize training efficiency"
2. spreader-tool spawns multiple agents (hyperparameter, architecture, optimizer)
3. Each agent independently modifies train.py and runs experiments
4. Agents run in parallel (12 experiments/hour each)
5. Aggregator synthesizes findings into research summary
6. Findings sent to murmur for knowledge graph integration

**Advantages over sequential autoresearch**:
- **Parallelism**: Instead of 12 experiments/hour (1 agent), achieve 24-36 (2-3 agents)
- **Specialization**: Architecture expert focuses on depth/width, hyperparameter expert focuses on LR/BS
- **Collaboration**: Agents share insights ("architecture Y works great with LR 0.04")
- **Rapid iteration**: Combine near-misses from different agents

---

### 2. autoresearch: Experiment Engine

**Role**: Run constrained ML experiments with fixed time budget

**This is the core**. Each spreader-tool agent invokes autoresearch via:
```bash
# Agent modifies train.py then runs:
uv run train.py > run.log 2>&1

# Parse metrics:
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

**Extending autoresearch** for research beyond model optimization:

Instead of just tuning hyperparameters, use train.py as a proxy for any research question:

**Example 1: Investigating dataset properties**
```python
# In train.py: Modify data loading to inject different corpora
# Question: "How does training data distribution affect learning speed?"

if experiment_id == 1:
    data = load_openwebtext()  # General purpose
elif experiment_id == 2:
    data = load_technical_papers()  # Domain-specific
elif experiment_id == 3:
    data = load_mixed(openwebtext=0.7, papers=0.3)  # Mixture

# Run 3 experiments, compare val_bpb trends
```

**Example 2: Investigating training dynamics**
```python
# Question: "Does gradient clipping improve convergence?"

if use_gradient_clipping:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
else:
    loss.backward()

# Run with/without clipping, compare val_bpb
```

**Key insight**: train.py doesn't have to be about neural network architecture. It's a 5-minute-budget research sandbox. Agents can use it to investigate:
- Data properties
- Optimizer behavior
- Regularization techniques
- Curriculum learning
- Loss function design
- Evaluation metrics
- Hardware efficiency

---

### 3. murmur: Knowledge Graph

**Role**: Auto-populate semantic wiki with research findings

**Setup**:
```bash
cd murmur
npm install
npm run dev  # Runs on localhost:3004
```

**Integration with autoresearch**:

When spreader-tool aggregates findings, send to murmur:

```javascript
// After research summary is generated
const findingsMarkdown = `
## Hyperparameter Optimization Results (2026-03-17)

**Summary**: Learning rate 0.04 is optimal for this dataset.

**Evidence**:
- Baseline LR=0.001: val_bpb=1.042
- Tested LR=0.01: val_bpb=1.035 (0.7% improvement)
- **Tested LR=0.04: val_bpb=0.998 (4.2% improvement)** ✓
- Tested LR=0.1: val_bpb=1.150 (10% worse, unstable)

**Batch Size Trade-offs**:
- BS=16: val_bpb=1.001, memory=42GB, time=285s
- BS=32: val_bpb=0.998, memory=44GB, time=270s ← Pareto optimal
- BS=64: val_bpb=0.999, memory=48GB, time=260s

**Constraints**:
- Valid for: Transformer depth ≤ 8
- Valid for: Token length ≤ 2048
- May not generalize to vision tasks
`;

// Create murmur article via API
await fetch('http://localhost:3004/api/articles', {
  method: 'POST',
  body: JSON.stringify({
    title: 'Hyperparameter Optimization Results (2026-03-17)',
    content: findingsMarkdown,
    tags: ['training', 'hyperparameters', 'optimization', 'learning-rate'],
    references: [
      'constraint-theory/validation-report.json',
      'experiments/lr-sweep/results.tsv'
    ]
  })
});

// murmur automatically:
// 1. Indexes content
// 2. Creates backlinks to related articles (training, optimization, etc.)
// 3. Builds semantic graph via knowledge-tensor
// 4. Enables community annotation in bulletin board
```

**Knowledge Graph Features**:
- **Automatic linking**: Article mentions "learning rate" → linked to learning rate article
- **Backlinks**: Articles referencing this finding auto-appear in sidebar
- **Community bulletin**: Researchers can vote, challenge, extend findings
- **Temporal tracking**: Who made changes, when, why
- **Citation graph**: Which papers cite these findings

**Semantic Linking Example**:
```
Articles on "Hyperparameter Optimization":
├─ 2026-03-17: LR optimization results
├─ 2026-03-10: Batch size trade-offs
├─ 2026-03-05: Optimizer comparison (Adam vs SGD vs Muon)
└─ 2026-02-28: Hardware efficiency analysis

Semantic clusters discovered by knowledge-tensor:
├─ Pareto efficiency (links all of above)
├─ VRAM constraints (groups memory-aware articles)
├─ Training dynamics (links to gradient flow, loss landscape)
└─ Stability/convergence (links to learning rate sensitivity)
```

---

### 4. spreadsheet-moment: Monitoring Dashboard

**Role**: Real-time visualization of experiment metrics

**Setup**:
```bash
cd spreadsheet-moment
pnpm install
pnpm run dev  # Runs on localhost:5173
```

**Integration with autoresearch**:

Create live dashboard tracking experiments:

```javascript
// results.tsv grows as experiments run
// spreadsheet-moment reads it live

// spreadsheet columns:
// commit | val_bpb | memory_gb | training_seconds | hyperparameters | status

// Formulas in spreadsheet-moment:
// - Plot val_bpb vs memory_gb (Pareto frontier)
// - Highlight improvements > 1%
// - Calculate speedup (faster training at same val_bpb)
// - Track resource efficiency (val_bpb per GB VRAM)

// Example formula (Excel/Univer style):
// =FILTER(A:F, G:G="keep")  // Show only kept experiments
// =CHART(D2:D100, E2:E100, "line")  // Plot val_bpb trend
// =MAX(E2:E100) - MIN(E2:E100)  // Range of improvements
```

**Dashboard Sections**:
```
┌─────────────────────────────────────────────┐
│ Experiment Progress (Real-time)             │
├─────────────────────────────────────────────┤
│ Total Experiments: 47                        │
│ Best val_bpb: 0.998 (LR=0.04, BS=32)        │
│ Improvement vs Baseline: 4.2%                │
│ Time Remaining (5hr budget): 2h 15m         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Pareto Frontier (Accuracy vs Memory)         │
├─────────────────────────────────────────────┤
│                                              │
│     val_bpb                                  │
│     0.99  •(optimal)                        │
│     1.00  •   •   •                         │
│     1.01  •   •                             │
│     1.02  •   •   •   •                     │
│          40GB 42GB 44GB 48GB                │
│          Memory Usage                       │
└─────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│ Hyperparameter Sensitivity                   │
├──────────────────────────────────────────────┤
│ Learning Rate Effect:                        │
│  0.001: 1.042                                │
│  0.01:  1.035                                │
│  0.04:  0.998  ← optimal                     │
│  0.1:   1.150                                │
│                                              │
│ Batch Size Effect:                           │
│  16: 1.001 (42GB)                            │
│  32: 0.998 (44GB)  ← optimal                 │
│  64: 0.999 (48GB)                            │
└──────────────────────────────────────────────┘
```

**Live Connection** (websocket):
```javascript
// Monitor results.tsv changes
fs.watchFile('results.tsv', async () => {
  const newData = parseResultsTsv();
  io.emit('experiment-update', newData);
});

// spreadsheet-moment receives updates
socket.on('experiment-update', (data) => {
  refreshDataRange('A1:F100', data);
  recalculateFormulas();
  updateCharts();
});
```

---

### 5. constraint-theory: Validation Engine

**Role**: Deterministically validate research claims

**Setup**:
```bash
cd constraint-theory
cargo build --release
npm run dev:web  # Web visualizer on localhost:3000
```

**Integration with autoresearch**:

After spreader-tool synthesizes findings, validate with constraint-theory:

```javascript
// Research claim: "LR=0.04 is optimal"
const claim = {
  type: "hyperparameter",
  parameter: "learning_rate",
  value: 0.04,
  evidence: {
    val_bpb: 0.998,
    experiments: 47,
    improvement_vs_baseline: 0.042
  },
  domain: {
    model_depth: "≤8",
    sequence_length: "≤2048",
    batch_size: "16-64"
  }
};

// Call constraint-theory solver
const validation = await constraintSolver.validate(claim, {
  // Origin-centric geometry (Ω): Define normalized ground state
  origin: {
    baseline_val_bpb: 1.041,
    model_depth: 8,
    batch_size: 32
  },

  // Φ-folding operator: Map continuous to discrete valid states
  discrete_states: [
    { lr: 0.001, val_bpb: 1.042 },
    { lr: 0.01,  val_bpb: 1.035 },
    { lr: 0.04,  val_bpb: 0.998 },  // ← claim points here
    { lr: 0.1,   val_bpb: 1.150 }
  ],

  // Rigidity-curvature duality: Check structural consistency
  constraints: [
    "val_bpb improves monotonically up to LR=0.04",
    "val_bpb increases sharply beyond LR=0.04",
    "Constraint: all values in discrete states"
  ]
});

// Output
console.log(validation);
// {
//   status: "VERIFIED",
//   confidence: 0.987,  // Deterministic (not probabilistic)
//   evidence: [
//     "Claim maps to discrete state at LR=0.04",
//     "4 independent experiments support value",
//     "No contradictions detected in domain",
//     "Consistent with Pareto frontier"
//   ],
//   caveats: [
//     "Only valid for model_depth ≤ 8",
//     "May not generalize beyond tested sequence lengths",
//     "Batch size 32 assumed (trade-offs at 16/64)"
//   ]
// }

// Log results
await fs.appendFile('fact-check-results.tsv',
  `${claim.parameter}\tVERIFIED\t${validation.confidence}\t${claim.value}\n`
);
```

**Fact-Checking Stages**:

1. **Consistency Check**: Does claim satisfy constraints?
   ```
   Claim: "LR=0.04 optimal"
   Constraint: "All LRs in [0.001, 0.1]" ✓
   Constraint: "val_bpb values realistic" ✓
   ```

2. **Evidence Review**: How much experimental support?
   ```
   Experiments testing LR=0.04: 4 independent runs
   Consistency across runs: std_dev=0.0002 (tight)
   Reproducibility: P(val_bpb ≈ 0.998) > 99%
   ```

3. **Cross-Reference Check**: Conflicts with known facts?
   ```
   Related claims in knowledge graph:
   - "LR=0.01 works for deep models" (depth=16)
   - "LR=0.04 causes instability at BS>128"
   - "Optimizer interaction: Muon likes higher LR"

   Consistency: No direct contradictions
   ```

4. **Confidence Score** (deterministic, not Bayesian):
   ```
   Φ-fold distance to claim: 0 (perfectly on discrete state)
   Evidence strength: 4/4 experiments consistent
   Domain coverage: 100% (within constraints)
   Conflicting evidence: 0

   Confidence = 0.987 (geometric, not probabilistic)
   ```

---

### 6. superinstance-papers: Theoretical Foundation

**Role**: Reference implementation of system concepts

**Relevant Papers** in the collection:
- "Cellular Biology Parallels in Distributed AI"
- "SE(3)-Equivariance for Byzantine-Tolerant Routing"
- "Evolutionary Game Theory for Consensus Mechanisms"
- "Protein Language Models for Self-Attention Coordination"

**Integration**:
- Use theoretical foundations to design agent prompts
- Reference papers when designing spreader-tool orchestration
- Validate constraint-theory implementations against mathematical proofs
- Generate podcast episodes summarizing key papers

**Example**: Using constraint-theory mathematical foundations

```
Paper: "MATHEMATICAL_FOUNDATIONS_DEEP_DIVE.md" (45 pages)

Theorem 3.7: Rigidity-Curvature Duality
"For any constraint satisfaction problem in d dimensions,
the curvature of the feasible region bounds the information
required to represent solutions"

Application to hyperparameter optimization:
- Learning rate is a constraint variable
- val_bpb improvement is the curvature
- Constraint-theory solver uses this to bound solution space
- Smaller bound = higher confidence in claim
```

---

## Complete Integration Example

### Research Question
"What's the optimal batch size for training speed under memory constraints?"

### Execution Flow

#### Stage 1: Orchestration (spreader-tool)
```
Query: "Batch size optimization"
  ↓
spreader-tool routes to agents:
  - Agent A: Memory profiler
    (run autoresearch with BS=16,32,64,128)
  - Agent B: Training speed analyst
    (compare time to target val_bpb)
  - Agent C: Hardware efficiency expert
    (FLOPS/watt analysis)
  ↓
Each agent runs in parallel:
  A: 12 exp/hour × 2 hours = 24 experiments
  B: 12 exp/hour × 2 hours = 24 experiments
  C: 12 exp/hour × 2 hours = 24 experiments

Total: 72 experiments in 2 hours
```

#### Stage 2: Experimentation (autoresearch)
```
For each experiment:
  1. Agent modifies train.py (batch size)
  2. Runs: uv run train.py
  3. Logs: commit, val_bpb, memory, time

Results:
  BS=16: val_bpb=1.001, mem=42GB, time=285s
  BS=32: val_bpb=0.998, mem=44GB, time=270s ← best trade-off
  BS=64: val_bpb=0.999, mem=48GB, time=260s
  BS=128: OOM crash (memory_gb=0.0)
```

#### Stage 3: Aggregation (spreader-tool)
```
Synthesized finding:
"Batch size 32 is optimal under 48GB memory constraint.
Achieves 0.7% better val_bpb than BS=16 while saving 15 seconds.
Unlikely to generalize: crashes at BS=128 (OOM)."
```

#### Stage 4: Validation (constraint-theory)
```
Claim: "BS=32 optimal"
- Φ-fold distance: 0 (on observed discrete state)
- Evidence: 3 consistent experiments
- Domain: Valid for model_depth ≤ 8
- Confidence: 0.992

Status: VERIFIED
Caveats: Only for 48GB+ memory systems
```

#### Stage 5: Knowledge Graph (murmur)
```
Article published: "Batch Size Optimization Results"
- Auto-linked to: memory, efficiency, hardware, training
- Backlinks: from training optimization, hardware guides
- Community annotations: researchers comment and extend
- Citation graph: future papers can reference
```

#### Stage 6: Monitoring (spreadsheet-moment)
```
Dashboard updated:
┌─────────────────────┐
│ Batch Size Effect   │
├─────────────────────┤
│ BS  │ val_bpb │ mem │
│ 16  │ 1.001   │ 42  │
│ 32  │ 0.998   │ 44  │ ← optimal
│ 64  │ 0.999   │ 48  │
│ 128 │ crash   │ --  │
└─────────────────────┘
```

#### Stage 7: Output (podcast)
```
Podcast script generated:
"Episode 7: Finding Your Sweet Spot - Batch Size Optimization

This week's research discovered that batch size 32
delivers the best trade-off between accuracy and speed
for our training setup. Here's how we found it:
[Narrative with evidence citations from murmur]

Show notes: [links to spreadsheet dashboard + paper]"
```

---

## Deployment Checklist

### Development Environment
- [ ] Clone autoresearch
- [ ] Clone murmur (localhost:3004)
- [ ] Install spreader-tool CLI
- [ ] Deploy spreadsheet-moment (localhost:5173)
- [ ] Set up constraint-theory validator

### Configuration
- [ ] Create research queries in spreader-tool config
- [ ] Set up agent specializations
- [ ] Configure retention policies
- [ ] Set up murmur API credentials
- [ ] Configure podcast output format

### Data Pipelines
- [ ] Connect autoresearch → results.tsv
- [ ] Connect spreader-tool → murmur API
- [ ] Connect spreadsheet-moment → results.tsv (live)
- [ ] Connect constraint-theory → validation API
- [ ] Connect podcast generator → murmur knowledge graph

### Monitoring
- [ ] Dashboard: Track experiment metrics
- [ ] Alerts: Notify on crashes or anomalies
- [ ] Logging: All research decisions logged
- [ ] Audit: Retention policy compliance

---

## Troubleshooting

**spreader-tool agents timing out**: Increase time budget in orchestration config
**murmur search not finding articles**: Rebuild knowledge-tensor indices
**constraint-theory validation too strict**: Adjust confidence threshold
**Cold storage filling up**: Review retention policies, compress older data
**Spreadsheet formulas slow**: Create separate view for heavy computations

---

## References

- spreader-tool: https://github.com/SuperInstance/spreader-tool
- murmur: https://github.com/SuperInstance/murmur
- spreadsheet-moment: https://github.com/SuperInstance/spreadsheet-moment
- constraint-theory: https://github.com/SuperInstance/constraint-theory
- superinstance-papers: https://github.com/SuperInstance/superinstance-papers
