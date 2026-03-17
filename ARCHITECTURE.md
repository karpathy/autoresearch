# AutoResearch + SuperInstance: Autonomous Fact-Checking Wiki Architecture

## Executive Summary

This document describes a system that combines **autoresearch** (autonomous LLM-driven model experimentation) with the **SuperInstance ecosystem** (multi-agent orchestration, semantic knowledge management, constraint-based computation) to create a **continually-bettering research and fact-checking platform**.

The system automatically:
- Conducts exploratory research through autonomous agent swarms
- Populates and refines semantic knowledge graphs (wiki)
- Fact-checks findings using deterministic constraint validation
- Records sandbox exploration and analysis in cold storage
- Generates podcast-ready content and wiki articles
- Respects global and per-file data retention policies

This enables **continuous research streams** for wikis, creative pontification, podcast content, or any domain requiring iterative investigation and validation.

---

## System Architecture

### Layer 1: Autonomous Research Engine (autoresearch)

**Core**: `autoresearch/` (this repo)
- **Engine**: GPT-style model training in 5-minute increments
- **Mechanism**: AI agents modify `train.py`, run experiments, evaluate `val_bpb`, keep improvements
- **Metric**: Validation bits-per-byte (lower = better, vocab-size-independent)
- **Output**: Continuous stream of experimental results and insights

**Extension Point**: Instead of just optimizing model architecture, agents can:
- Investigate research questions by modifying training dynamics
- Encode domain-specific knowledge into training procedures
- Synthesize findings from parallel research streams

### Layer 2: Multi-Agent Orchestration (spreader-tool)

**Purpose**: Coordinate multiple specialized research agents in parallel

**Integration**:
```
research-query
    ↓
spreader-tool (orchestration layer)
    ├→ [Agent 1] Literature researcher
    ├→ [Agent 2] Experimental validator
    ├→ [Agent 3] Fact-checker
    └→ [Agent 4] Synthesis & summarization
    ↓
consolidated findings
```

**Key Capabilities**:
- Parallel multi-agent investigation (accelerated research velocity)
- Context distribution (specialized knowledge per agent)
- Agent handoff and collaboration protocols
- Result aggregation and conflict resolution
- Progress monitoring and adaptive routing

**SuperInstance Component**: `@superinstance/spreader` (npm package)

### Layer 3: Semantic Knowledge Management (murmur)

**Purpose**: Auto-organize research findings into semantically-linked knowledge graph

**Integration**:
```
research findings
    ↓
knowledge-tensor (semantic AI linking)
    ↓
murmur (wiki interface)
    ├→ Automatic backlinks
    ├→ Semantic clusters
    ├→ Community bulletin board
    └→ TensorDB knowledge graph
```

**Features**:
- **Self-populating wiki**: New findings automatically integrated
- **Semantic linking**: AI-powered bidirectional references
- **Knowledge graph**: Relationships preserved and queryable
- **Community layer**: Researchers can annotate and challenge findings
- **Real-time updates**: Immediately reflects new research

**SuperInstance Component**: `murmur` + `@superinstance/knowledge-tensor`

### Layer 4: Data Interface & Monitoring (spreadsheet-moment)

**Purpose**: Universal interface for data observation, monitoring, and formula-driven analysis

**Integration**:
```
experimental data
    ↓
spreadsheet-moment (Univer-based)
    ├→ Real-time metrics dashboard
    ├→ Formula-driven analysis
    ├→ Collaborative data exploration
    └→ Cloudflare Workers deployment
    ↓
patterns & insights
```

**Capabilities**:
- **Real-time monitoring**: Live experiment metrics
- **Formula system**: Pivot tables, conditional formatting, custom calculations
- **Universal format**: Works as spreadsheet, document, or presentation
- **Collaborative**: Multiple researchers analyze simultaneously
- **Serverless**: Deploy via Cloudflare Workers, zero-config auth

**Use Cases**:
- Track `val_bpb` trends across experiment series
- Monitor resource utilization (VRAM, training time)
- Spot patterns in hyperparameter sweeps
- Collaborative hypothesis formation

**SuperInstance Component**: `spreadsheet-moment` (Univer fork)

### Layer 5: Deterministic Validation (constraint-theory)

**Purpose**: Validate research claims using geometric constraint-solving instead of probabilistic approximation

**Integration**:
```
research claim
    ↓
constraint-theory solver
    ├→ Origin-centric geometry (Ω)
    ├→ Φ-folding operator (continuous → discrete)
    └→ Rigidity-curvature verification
    ↓
confidence score (deterministic guarantee)
```

**Key Guarantees**:
- **Origin-centric geometry (Ω)**: Normalized ground state definition
- **Φ-folding operator**: Maps continuous inputs to discrete valid states
- **Rigidity-curvature duality**: Structural properties ↔ information theory
- **Performance**: ~109× speedup vs NumPy (KD-tree spatial indexing + SIMD)

**Fact-Checking Applications**:
- Verify claim consistency against knowledge graph
- Detect contradictions in research findings
- Provide deterministic confidence scores (not probabilistic)
- Validate constraint satisfaction in complex theories

**SuperInstance Component**: `constraint-theory` (Rust + GPU simulation)

### Layer 6: Memory & Context Preservation (hierarchical-memory + SmartCRDT)

**Purpose**: Maintain research context across agent tiers and improve over time

**Integration**:
```
research session
    ↓
hierarchical-memory (6-tier system)
    ├→ Tier 1: Immediate working context
    ├→ Tier 2: Session summary
    ├→ Tier 3: Consolidated insights
    ├→ Tier 4: Archival knowledge
    ├→ Tier 5: Vector embeddings
    └→ Tier 6: Identity persistence
    ↓
SmartCRDT (self-improving replicated state)
    ├→ Conflict-free data types
    ├→ Self-improving verification
    └→ Distributed consensus
```

**Benefits**:
- Agents retain context across multiple research sessions
- Self-improving: Corrections propagate automatically via CRDT
- Distributed: Multiple research teams can merge findings
- Identity persistence: Maintain researcher personas across contexts

**SuperInstance Components**: `hierarchical-memory` + `SmartCRDT`

### Layer 7: Output Pipelines

#### 7a: Wiki Generation Pipeline

```
murmur (semantic knowledge graph)
    ↓
[extract relevant articles]
    ↓
[format for wiki syntax]
    ↓
[apply retention policies]
    ↓
[publish to wiki repo]
```

**Output Format**: Markdown or MediaWiki syntax
**Frequency**: Continuous or batch (configurable)
**Retention**: Global + per-file allowances respected

#### 7b: Podcast Content Pipeline

```
research findings + synthesis
    ↓
[select high-interest topics]
    ↓
[generate narrative script]
    ↓
[add evidence citations]
    ↓
[apply cold storage policy]
    ↓
[publish to podcast repo/RSS]
```

**Content Types**: Episode scripts, transcripts, show notes
**Cold Storage**: Raw research data archived separately from narrative

---

## Sandbox/Flowstate Mode: Exploratory Research

### Purpose
Allow agents to explore radical, unvalidated ideas without contaminating primary knowledge graph. Record all exploration for later analysis.

### Architecture

```
flowstate mode
    ↓
[temporary sandbox knowledge graph]
    ├→ Exploratory findings (unvalidated)
    ├→ Radical hypotheses
    ├→ Failed experiments (recorded)
    └→ Analysis logs (timestamped)
    ↓
[after flowstate ends]
    ├→ Manual review: which findings promote to primary graph?
    ├→ Constraint validation: which survive deterministic checks?
    └→ Archive: record all exploration in cold storage
    ↓
[cold storage]
    ├→ Raw simulation data
    ├→ Analysis logs
    ├→ Rejected hypotheses (for pattern mining)
    └→ Researcher notes
```

### Configurable Retention Policies

**Global Policy** (applies to all research):
```json
{
  "default_retention_days": 90,
  "archive_raw_data": true,
  "compress_after_days": 30,
  "delete_after_days": 180
}
```

**Per-File Policy** (can override global):
```json
{
  "file_path": "experiments/radical_hypothesis_X.json",
  "retention_days": 365,
  "archive_raw_data": true,
  "compress_after_days": 14,
  "note": "Keep longer for longitudinal analysis"
}
```

### Flowstate Workflow

1. **Enter flowstate**: Agent declares exploratory mode
2. **Temporary sandbox**: All findings go to sandbox graph (not primary)
3. **Explore freely**: No constraint validation, no community review
4. **Record everything**: Logs, hypotheses, failures all timestamped
5. **Exit flowstate**: Review period begins
6. **Manual curation**: Researcher selects findings to promote
7. **Constraint validation**: Promoted findings must pass validation
8. **Archive exploration**: All sandbox data moved to cold storage

---

## Data Flow: Complete Example

### Research Query: "What hyperparameter configuration optimizes training efficiency?"

```
[User] Query: optimize training efficiency
    ↓
[spreader-tool] Orchestrate research agents
    ├→ Agent A: Analyze learning rate effects
    ├→ Agent B: Investigate batch size trade-offs
    ├→ Agent C: Study optimizer algorithms
    └→ Agent D: Synthesize pareto frontier
    ↓
[autoresearch] Each agent runs experiments (5-min budget)
    ├→ A: train.py variations, 12+ experiments/hour
    ├→ B: train.py variations, 12+ experiments/hour
    ├→ C: train.py variations, 12+ experiments/hour
    └→ D: Compare findings
    ↓
[constraint-theory] Validate claims
    ├→ "LR=0.04 is optimal" → verify consistency
    ├→ "Batch size > 128 always faster" → check constraints
    └→ Provide deterministic confidence scores
    ↓
[spreadsheet-moment] Dashboard update
    ├→ Real-time metrics visualization
    ├→ Pareto frontier plot (formulas)
    └→ Collaborative annotation
    ↓
[murmur] Knowledge graph update
    ├→ New article: "Hyperparameter Optimization Results (2026-03-17)"
    ├→ Auto-links to: training, efficiency, batch-size, learning-rate
    ├→ Backlinks added: from existing papers on optimization
    └→ Community bulletin: researchers can challenge/extend
    ↓
[podcast pipeline] Content generation
    ├→ Script: "Episode: Finding the Efficiency Sweet Spot"
    ├→ Evidence citations: auto-linked to murmur
    ├→ Show notes: link to spreadsheet dashboard
    └→ Publish: RSS feed + wiki repo
    ↓
[cold storage] Archive exploration
    ├→ All 120+ experiments logged (12/hr × 10hr research)
    ├→ Agent reasoning traces
    ├→ Rejected hypotheses
    └→ Retention policy: global 90 days, compress day 30
```

---

## Integration Points with SuperInstance Ecosystem

| Component | Role | Integration |
|-----------|------|-------------|
| **autoresearch** | Autonomous experimentation engine | Core: runs research experiments |
| **spreader-tool** | Multi-agent orchestration | Routes queries to specialist agents |
| **murmur** | Semantic knowledge management | Publishes findings to wiki/graph |
| **spreadsheet-moment** | Data interface & monitoring | Dashboard for experiment metrics |
| **constraint-theory** | Deterministic validation | Fact-checks research claims |
| **SmartCRDT** | Self-improving distributed state | Replicates knowledge graph across teams |
| **hierarchical-memory** | Context preservation | Maintains agent research context |
| **CognitiveEngine** | Pattern recognition & synthesis | Identifies insights from findings |
| **OpenClaw** | Device integration & voice | Enables voice-driven research queries |
| **SwarmOrchestration** | Distributed coordination | Scales to 1000+ research agents |

---

## Fact-Checking Pipeline Details

### Multi-Stage Validation

```
Claim: "Model A outperforms Model B by 5% on metric X"
    ↓
[Stage 1] Constraint Consistency Check
    ├→ Does claim satisfy geometric constraints?
    ├→ Does it contradict known facts in knowledge graph?
    └→ Φ-folding: Map claim to discrete valid state
    ↓
[Stage 2] Experimental Evidence Review
    ├→ How many independent runs support claim?
    ├→ Statistical significance (if probabilistic)
    ├→ Reproducibility across compute platforms
    └→ VRAM/time trade-offs acceptable?
    ↓
[Stage 3] Cross-Reference Check
    ├→ Does it align with related findings in murmur?
    ├→ Any contradictions in literature?
    ├→ Similar claims in historical archive?
    └→ Author credibility score
    ↓
[Stage 4] Consensus Layer
    ├→ Community votes in murmur bulletin
    ├→ Expert agents review
    ├→ Temporal validation (does it hold over time?)
    └→ Conditional acceptance: "true under X assumptions"
    ↓
[Output] Fact-Check Result
    ├→ Status: VERIFIED | DISPUTED | UNRESOLVED | FALSE
    ├→ Confidence: 0.95 (deterministic from constraint-theory)
    ├→ Evidence summary: [list of supporting experiments]
    ├→ Caveats: [conditions, limitations, open questions]
    └→ Revision history: timestamped edits
```

---

## Cold Storage & Retention Management

### Storage Tiers

```
Tier 1: Hot Storage (Working Research)
├→ Location: In-memory + local SSD
├→ Data: Active experiments, current knowledge graph
├→ TTL: Session duration (usually < 24 hours)
└→ Access: Instant

Tier 2: Warm Storage (Recent Archive)
├→ Location: Cloud blob storage (R2, S3, etc.)
├→ Data: Last 30 days of research + findings
├→ TTL: 30 days (configurable)
├→ Access: <1 second
└→ Format: Compressed JSON + metadata

Tier 3: Cold Storage (Long-Term Archive)
├→ Location: Archival storage (Glacier, etc.)
├→ Data: Older than 30 days, rarely accessed
├→ TTL: 180+ days (configurable)
├→ Access: Hours to days (async retrieval)
└→ Format: Compressed, deduplicated snapshots
```

### Summarization Policy

Before moving from Warm → Cold Storage:

```
raw_experimental_data (large)
    ├→ Extract: Key metrics, hyperparameters, results
    ├→ Summarize: Agent reasoning, decision logs
    ├→ Redact: Sensitive data (if needed)
    └→ Preserve: Links to published findings in murmur
    ↓
summarized_archive (90% size reduction typical)
    └→ Move to cold storage
```

### Per-File Configuration Example

```json
{
  "retention_policies": [
    {
      "pattern": "experiments/*/metrics.json",
      "tier_1_ttl_days": 1,
      "tier_2_ttl_days": 30,
      "tier_3_ttl_days": 180,
      "summarize_before_archive": true,
      "redact_fields": ["api_keys", "email"]
    },
    {
      "pattern": "flowstate_sandbox/*/analysis.md",
      "tier_1_ttl_days": 7,
      "tier_2_ttl_days": 90,
      "tier_3_ttl_days": 365,
      "summarize_before_archive": false,
      "reason": "Exploratory analysis valuable for longitudinal study"
    },
    {
      "pattern": "published_findings/*.md",
      "tier_1_ttl_days": 7,
      "tier_2_ttl_days": 1095,
      "tier_3_ttl_days": null,
      "reason": "Keep published work indefinitely"
    }
  ]
}
```

---

## Continuous Improvement Loop

### The Weekly Research Cycle

```
Monday 00:00 - Friday 23:59
    ↓
[Daily automated research]
    ├→ spreader-tool orchestrates 100+ experiments/day
    ├→ autoresearch runs 5-min experiments continuously
    ├→ murmur knowledge graph grows daily
    └→ constraint-theory validates new findings
    ↓
[Friday 18:00]
    ├→ Extract: top 10 insights from week
    ├→ Synthesize: using CognitiveEngine pattern recognition
    ├→ Fact-check: run through constraint validation
    ├→ Archive: week's experimental data to cold storage
    └→ Generate: podcast script + wiki updates
    ↓
[Friday 20:00]
    ├→ Podcast episode publishes
    ├→ Wiki articles update
    ├→ Community review begins (murmur bulletin)
    └→ Researchers vote/annotate findings
    ↓
[Weekend]
    ├→ Manual review & curation (if needed)
    ├→ Dispute resolution
    ├→ Planning next week's research directions
    └→ Update program.md (agent instructions)
    ↓
[Monday 00:00] ← cycle repeats
```

---

## Extensibility & Customization

### How to Add a New Research Stream

1. **Create research query**: "Investigate property X in domain Y"
2. **Extend program.md**: Add instructions for new research direction
3. **Configure spreader-tool**: Route to relevant specialist agents
4. **Set retention policy**: How long to keep this research data?
5. **Deploy**: Agents begin research autonomously

### How to Add a New Validation Layer

1. **Implement constraint solver**: Extend constraint-theory or add new backend
2. **Register in fact-checking pipeline**: Stage 2 (Experimental Evidence) or new stage
3. **Configure confidence weighting**: How much should this layer influence overall score?
4. **Test on historical data**: Validate against known true/false claims

### How to Add a New Output Pipeline

1. **Consume from murmur**: Subscribe to knowledge graph updates
2. **Filter findings**: What subset is relevant to your output format?
3. **Transform content**: Markdown → format X (podcast script, legal brief, etc.)
4. **Apply retention policy**: Which data can be published? What must stay in cold storage?
5. **Publish**: RSS feed, wiki repo, email, Slack, etc.

---

## Comparison: Traditional Research vs. Autonomous Research

| Aspect | Traditional | Autonomous (This System) |
|--------|-----------|-------------------------|
| **Speed** | Months to years | Hours to days |
| **Parallelism** | Sequential (1-2 researchers) | Parallel (100+ agents) |
| **Validation** | Manual peer review | Automated + community review |
| **Knowledge Graph** | Manual literature review | Auto-populated semantic graph |
| **Fact-Checking** | Probabilistic peer consensus | Deterministic constraint validation |
| **Data Retention** | Ad-hoc archival | Policy-driven cold storage |
| **Reproducibility** | Detailed lab notes | Complete experimental logs |
| **Output Format** | Papers only | Papers + podcasts + wiki + code |
| **Iteration** | Slow (feedback loops) | Fast (10+ experiments/hour) |
| **Cost** | High (human time) | Low (compute-dominated) |

---

## Open Questions & Future Work

1. **Consensus Mechanisms**: How to resolve disagreements between agents and community?
2. **Drift Detection**: When does the system lose calibration? How to detect?
3. **Adversarial Robustness**: Can agents or community members game the fact-checking?
4. **Privacy**: How to handle proprietary/sensitive research findings?
5. **Interpretability**: Can we explain constraint-theory determinations to domain experts?
6. **Scalability**: What's the limit? 1000 agents? 1M research items?
7. **Cross-Domain Transfer**: Can findings in domain A inform research in domain B?

---

## Getting Started

### Phase 1: Setup (Week 1)
- [ ] Fork autoresearch repo
- [ ] Deploy murmur locally (port 3004)
- [ ] Set up spreader-tool CLI
- [ ] Configure spreadsheet-moment dashboard
- [ ] Write initial retention policies

### Phase 2: Baseline Research (Week 2-3)
- [ ] Run 100+ autoresearch experiments
- [ ] Populate initial murmur knowledge graph
- [ ] Implement fact-checking pipeline
- [ ] Generate first podcast episode

### Phase 3: Refinement (Week 4+)
- [ ] Optimize agent prompts in program.md
- [ ] Add specialized research domains
- [ ] Extend retention policies
- [ ] Scale to full swarm orchestration

---

## References

- **autoresearch**: https://github.com/karpathy/autoresearch
- **spreader-tool**: https://github.com/SuperInstance/spreader-tool
- **murmur**: https://github.com/SuperInstance/murmur
- **spreadsheet-moment**: https://github.com/SuperInstance/spreadsheet-moment
- **constraint-theory**: https://github.com/SuperInstance/constraint-theory
- **superinstance-papers**: https://github.com/SuperInstance/superinstance-papers
- **SmartCRDT**: https://github.com/SuperInstance/SmartCRDT
- **hierarchical-memory**: https://github.com/SuperInstance/hierarchical-memory
- **SwarmOrchestration**: https://github.com/SuperInstance/SwarmOrchestration
- **CognitiveEngine**: https://github.com/SuperInstance/CognitiveEngine

---

## License

MIT License (matching autoresearch)
