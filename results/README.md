# Results & Outputs

All research results, experiments, and outputs stored here.

## 📋 Directory Structure

```
results/
├── experiments/              # All training experiments
│   ├── <date-tag>/          # Experiment grouped by tag/date
│   │   ├── results.tsv      # Tab-separated results
│   │   ├── run.log          # Training logs
│   │   ├── metrics.json     # Detailed metrics
│   │   └── checkpoints/     # Model checkpoints
│   └── archive/             # Old experiments (cold storage)
│
├── knowledge-graph/          # Wiki and semantic links
│   ├── murmur-exports/      # Exported from murmur wiki
│   ├── semantic-links.json  # Knowledge graph structure
│   └── fact-checks.json     # Validation results
│
├── podcast-content/          # Generated podcast episodes
│   ├── scripts/             # Episode scripts
│   └── metadata.json        # Episode metadata
│
└── cold-storage/             # Long-term archival
    ├── compressed/          # Compressed old data
    └── indices/             # Search indices
```

## 📊 Understanding Experiment Results

### results.tsv Format

Tab-separated file with one experiment per line:

```
commit   val_bpb   memory_gb   training_seconds   hyperparameters                  status
abc1234  1.042     42.5        285                LR=0.001,BS=16,WD=0.01          keep
abc1235  1.035     42.0        270                LR=0.01,BS=32,WD=0.01           keep
abc1236  0.998     44.0        260                LR=0.04,BS=32,WD=0.01           keep
abc1237  1.150     48.0        250                LR=0.1,BS=64,WD=0.01            discard
```

**Columns:**
- `commit` - Git hash of train.py version
- `val_bpb` - Validation bits per byte (lower = better)
- `memory_gb` - Peak GPU memory used
- `training_seconds` - Wall-clock seconds (should be ~285 for 5 min)
- `hyperparameters` - What was tested
- `status` - keep / discard / failed

### metrics.json Example

```json
{
  "experiment_id": "abc1236",
  "timestamp": "2026-03-17T14:32:15Z",
  "agent": "hyperparameter-specialist",
  "hyperparameters": {
    "learning_rate": 0.04,
    "batch_size": 32,
    "weight_decay": 0.01,
    "warmup_steps": 100
  },
  "results": {
    "val_bpb": 0.998,
    "train_loss": 1.234,
    "val_loss": 1.256,
    "peak_memory_gb": 44.0,
    "training_time_seconds": 260
  },
  "improvement": {
    "vs_baseline": 0.042,  # 4.2% improvement
    "vs_previous": 0.037,  # 3.7% improvement
    "status": "kept"
  },
  "decision_reasoning": "4% improvement justifies memory increase"
}
```

### run.log Format

```
Starting training run: experiment_abc1236
Device: NVIDIA H100 (80GB)
Model: GPT, depth=8, width=512
Training params: LR=0.04, BS=32, WD=0.01

Epoch 1/100: loss=2.345, val_bpb=1.234 (ETA: 4m 30s)
Epoch 2/100: loss=1.892, val_bpb=1.156 (ETA: 4m 15s)
...
Epoch 100/100: loss=1.234, val_bpb=0.998 (ETA: 0s)

Training complete!
Peak VRAM: 44GB
Total time: 285s (exactly 5 minutes)
Improvement: 4.2% vs baseline
Status: KEPT

Commit hash: abc1236
```

## 🔍 Analyzing Results

### View Recent Experiments

```bash
# Display results.tsv
cat results/experiments/*/results.tsv | head -20

# Best result
cat results/experiments/*/results.tsv | sort -k2 -n | head -1

# Recent experiments (most recent first)
ls -ltr results/experiments/*/results.tsv | tail -5
```

### Parse Metrics

```bash
# Extract best val_bpb
jq '.results.val_bpb' results/experiments/*/metrics.json | sort -n | head -1

# Find experiments by agent
jq '.agent' results/experiments/*/metrics.json | sort | uniq -c

# Get improvement statistics
jq '.improvement.vs_baseline' results/experiments/*/metrics.json | \
  awk '{sum+=$1; count++} END {print "Avg improvement:", sum/count*100 "%"}'
```

### Plot Results

```bash
# Using gnuplot
gnuplot << 'EOF'
set title "val_bpb vs Experiment Number"
set xlabel "Experiment"
set ylabel "val_bpb"
plot "results/experiments/results.tsv" using 1:2 with lines
EOF

# Or export to JSON and plot with Python/matplotlib
```

## 📁 Experiment Organization

Experiments are grouped by date and tag:

```
results/experiments/
├── 2026-03-17-hyperparameter-sweep-1/
│   ├── results.tsv
│   ├── metrics.json
│   └── checkpoints/
├── 2026-03-17-architecture-exploration/
│   ├── results.tsv
│   ├── metrics.json
│   └── checkpoints/
└── archive/  (old experiments, compressed)
    ├── 2026-01-*  (compressed)
```

### Querying Experiments

```bash
# Find all experiments for hyperparameter agent
find results/experiments -name metrics.json -exec \
  grep -l "hyperparameter-specialist" {} \;

# Find all "kept" experiments
jq '.improvement.status' results/experiments/*/metrics.json | \
  grep -l "kept"

# Get experiments that improved >5%
jq 'select(.improvement.vs_baseline > 0.05)' \
  results/experiments/*/metrics.json
```

## 🔄 Data Retention & Cold Storage

### Hot Storage (1 day)
```
results/experiments/<recent>/
├── results.tsv          # Full detail
├── metrics.json         # Complete metrics
├── run.log              # Full logs
└── checkpoints/         # Model checkpoints
```

### Warm Storage (14-30 days, compressed)
```
results/cold-storage/compressed/2026-03-*.tar.gz
├── results.tsv (summarized)
├── metrics-summary.json (key fields only)
└── references to published findings
```

### Cold Storage (180+ days, archival)
```
results/cold-storage/archive/2026-01-*/
├── summary.json         # Only essential fields
├── links-to-published   # Where findings were published
└── index-entry          # Searchable metadata
```

### Automatic Archival

```bash
# Manual archival to cold storage
ar archive --tier warm-to-cold

# Check archive status
ar archive --list

# Restore from archive
ar restore <experiment-id>
```

## 📝 Knowledge Graph Integration

### Export to Wiki (murmur)

Findings are auto-exported to murmur:

```bash
# Manual export
ar graph --export murmur --confidence 0.85

# View what was published
cat results/knowledge-graph/murmur-exports/*.md

# Check semantic links
jq '.' results/knowledge-graph/semantic-links.json | head -30
```

### Fact-Checking Results

```bash
# View validation results
cat results/knowledge-graph/fact-checks.json

# Example output:
{
  "claim": "Learning rate 0.04 is optimal",
  "status": "VERIFIED",
  "confidence": 0.987,
  "evidence_count": 4,
  "conflicting_findings": 0
}
```

## 🎙️ Podcast Content

### Generated Episode Scripts

```
results/podcast-content/scripts/
├── episode-001-hyperparameter-optimization.md
├── episode-002-architecture-breakthroughs.md
└── metadata.json
```

### Episode Structure

```markdown
# Episode 001: Hyperparameter Optimization

## Summary
This week we discovered optimal hyperparameters...

## Key Findings
- Learning rate 0.04 is 4.2% better than baseline
- Batch size 32 minimizes memory-accuracy tradeoff
- Warmup steps = 100 critical for stability

## Evidence
[Linked to experiment abc1236]
[Cited constraint-theory validation]
[Referenced related findings in knowledge graph]

## Show Notes
- Link to wiki: http://wiki.example.com/hyperparameter-optimization
- Live dashboard: http://localhost:5173/metrics
- Full results: results/experiments/2026-03-17-hyper-sweep-1/
```

## 🗂️ File Organization Best Practices

### For Your Own Analysis

```bash
# Copy results for analysis (don't modify originals)
mkdir -p ~/research-analysis
cp -r results/experiments ~/research-analysis/

# Work with copies
cd ~/research-analysis

# Keep originals pristine
```

### Version Control

Add to `.gitignore` (results are large):

```bash
echo "results/experiments/" >> .gitignore
echo "results/cold-storage/" >> .gitignore
echo "results/podcast-content/scripts/" >> .gitignore

# But keep track of experiment summaries
git add results/knowledge-graph/
```

## 📈 Common Analysis Tasks

### Find Best Hyperparameters

```bash
# Sort by val_bpb (lower is better)
cat results/experiments/*/results.tsv | sort -k2 -n | head -5

# Parse best hyperparameters
jq '.hyperparameters' results/experiments/*/metrics.json | sort | uniq -c
```

### Track Improvement Over Time

```bash
# Extract val_bpb for time series
jq '[.timestamp, .results.val_bpb]' \
  results/experiments/*/metrics.json | sort | head -20

# Or use ar metrics command
ar metrics --since 24h --graph val_bpb
```

### Compare Agents

```bash
# Group by agent, show average improvement
jq -r '[.agent, .improvement.vs_baseline] | @tsv' \
  results/experiments/*/metrics.json | \
  awk '{agent=$1; improvement=$2; \
    sum[agent]+=improvement; count[agent]++} \
    END {for (a in sum) \
    print a, sum[a]/count[a]*100 "%"}'
```

### Find Failed Experiments

```bash
# Experiments that crashed or failed
jq 'select(.improvement.status == "failed")' \
  results/experiments/*/metrics.json | jq '.experiment_id'

# Why did they fail?
jq '.failure_reason' results/experiments/*/metrics.json
```

## 🔐 Data Privacy & Backup

### Backup Results

```bash
# Tar and compress
tar czf results-backup-$(date +%Y%m%d).tar.gz results/

# Upload to cloud (e.g., AWS S3)
aws s3 cp results-backup-*.tar.gz s3://my-bucket/

# Or use configured S3 target
ar backup --destination s3 --encrypt
```

### Remove Sensitive Data

```bash
# Redact API keys from logs
ar sanitize results/

# Remove raw training data
ar prune results/ --keep-summary-only

# Apply GDPR retention
ar retention-policy --gdpr --delete-older-than 180d
```

## ❓ FAQ

**Q: Where do experiment files come from?**
A: Agents create them when running `ar start`

**Q: Can I edit results manually?**
A: Not recommended - breaks consistency. Use `ar validate` to check integrity.

**Q: How much disk space for 1000 experiments?**
A: ~50-100GB (full) or ~5-10GB (cold storage)

**Q: Can I restore deleted experiments?**
A: If in cold storage yes, otherwise no. Keep backups!

**Q: How long does archival take?**
A: ~1 hour for 1000 experiments

---

**See also:** [../README.md](../README.md), [../docker/README.md](../docker/README.md)
