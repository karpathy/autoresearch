# autoresearch

Autonomous research framework. One target, one metric, keep or discard, repeat forever.

Evolved from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and generalized beyond ML training.

## How it works

An AI agent modifies a target file, measures a metric, keeps improvements, discards regressions, and repeats autonomously. The framework provides structure for any domain: ML training, prompt engineering, performance tuning, configuration optimization, or anything with a measurable outcome.

Each research workflow defines: what file(s) to edit, how to measure success, and what constraints to respect. The agent handles everything else.

You can leave it running overnight. At 5 minutes per experiment, that's ~100 experiments while you sleep. The agent explores the design space, accumulates knowledge, and converges toward better solutions without human intervention.

The core loop is simple: edit → commit → run → measure → keep or discard. The agent maintains a research log, tracks what works, and builds on successful experiments. You wake up to a log of trials and (hopefully) a better solution.

## Quick start

```bash
# Create a new research workflow
python scaffold.py my-experiment

# Edit the workflow config
# Fill in: workflows/my-experiment/workflow.yaml
# Write your research program: workflows/my-experiment/program.md

# Point your AI agent at the workflow and let it go
```

## Repo structure

```
autoresearch/
├── agents/                   # Agent definitions (portable, tool-agnostic)
├── skills/                   # Skill definitions (portable, tool-agnostic)
├── workflows/
│   ├── _template/            # Template for new workflows
│   └── examples/
│       └── ml-training/      # Reference implementation
├── scaffold.py               # Create new workflows from template
├── AGENTS.md                 # Root agent context
└── README.md
```

## Core concepts

### workflow.yaml

Machine-readable manifest that defines the experiment structure:

```yaml
name: my-experiment
target: train.py              # File(s) the agent modifies
metric: val_loss              # What to minimize (or maximize)
run_command: python train.py  # How to execute the experiment
time_budget: 300              # Wall-clock seconds per run
```

Agents parse this to understand what they can change and how success is measured.

### program.md

Human-written research strategy. This is where you encode domain knowledge:

- What approaches to try (architectural changes, hyperparameter ranges, optimization strategies)
- What constraints to respect (no external dependencies, stay under memory budget, maintain compatibility)
- What to log and track (metrics, intermediate outputs, failure modes)
- How to interpret results (what "better" means, when to explore vs exploit)

This file is your primary interface to the research process. You iterate on `program.md` to guide the agent toward productive experiments.

### AGENTS.md

Root agent context file that describes repo structure, conventions, and guidelines. Agents read this on startup to understand how to navigate the codebase.

### The experiment loop

1. **Edit**: Agent modifies target file(s) based on research program
2. **Commit**: Changes are committed to git with descriptive message
3. **Run**: Experiment executes with fixed time/resource budget
4. **Measure**: Metric is extracted from run output
5. **Keep or discard**: If metric improved, keep commit; otherwise, revert

This loop repeats indefinitely. The agent accumulates successful changes and builds on them.

### The Artisan's Triad

Three fundamental modes of experimentation:

| Mode | Description | Examples |
|------|-------------|----------|
| **Additive** | Add new things | New layers, features, logging, optimizations |
| **Reductive** | Remove things | Delete unused code, simplify architecture, reduce params |
| **Reformative** | Reshape existing things | Refactor structure, reorganize, change approach |

The agent cycles through these modes to explore the design space comprehensively.

## Creating a workflow

**Step 1: Scaffold**

```bash
python scaffold.py my-experiment
```

This creates `workflows/my-experiment/` with template files.

**Step 2: Configure workflow.yaml**

Edit `workflows/my-experiment/workflow.yaml`:

```yaml
name: my-experiment
target: run.py
metric: throughput
run_command: python run.py --benchmark
time_budget: 60
```

**Step 3: Write program.md**

Document your research strategy in `workflows/my-experiment/program.md`:

- What are you optimizing?
- What's allowed to change?
- What constraints must be respected?
- What domain knowledge should guide experiments?

**Step 4: Run baseline**

Manually execute the baseline to verify setup:

```bash
cd workflows/my-experiment
python run.py
```

Record the baseline metric in your notes.

**Step 5: Start the loop**

Point your AI agent at `workflows/my-experiment/program.md` and let it begin autonomous research. The agent will:

- Read `workflow.yaml` to understand the experiment structure
- Follow `program.md` to guide its research strategy
- Execute the experiment loop autonomously
- Log all experiments and decisions

Check back periodically to review progress and refine `program.md` based on what you learn.

## Example: ML training

The `workflows/examples/ml-training/` directory contains the original autoresearch experiment: autonomous optimization of a small GPT model trained for 5 minutes per experiment.

This workflow demonstrates:

- Fixed time budget (5 minutes wall-clock)
- Single target file (`train.py` containing model, optimizer, training loop)
- Single metric (`val_bpb` - validation bits per byte, lower is better)
- Comprehensive research program covering architecture, hyperparameters, and optimization

The ML training example is a reference implementation showing how to structure a research workflow. It's production-tested (ran overnight, produced improvements) and serves as a template for other domains.

See [workflows/examples/ml-training/](workflows/examples/ml-training/) for full details.

## Design choices

**Single target file per workflow**

The agent modifies one primary file (or a small set of related files). This keeps the scope manageable, diffs reviewable, and changes atomic. Multi-file workflows are possible but discouraged; better to decompose into multiple workflows.

**Fixed time/resource budget**

Experiments run with a fixed time budget (e.g., 5 minutes), regardless of what changes the agent makes. This has two key benefits:

1. **Comparable experiments**: You can fairly compare architectures, batch sizes, and other changes because every experiment gets the same resources.
2. **Optimizes for your platform**: The agent finds the best solution for your specific hardware in the given time budget.

At 5 minutes per experiment, you get ~12 experiments/hour and ~100 experiments overnight.

**Machine-readable manifest**

`workflow.yaml` is structured YAML that agents can parse, not just prose. This eliminates ambiguity: the agent knows exactly what file to edit, how to run the experiment, and what metric to optimize.

**One source of truth**

Each workflow is self-contained. No scattered configs, no duplicated definitions. Everything the agent needs is in the workflow directory.

**Human-curated artifacts**

Agents propose experiments; humans promote the best results. You review the research log, extract insights, and decide what to productionize. The framework makes agents productive, but you remain in control of what ships.

## License

MIT

---

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
