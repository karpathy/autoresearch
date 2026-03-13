# Phase 2 Quick Reference Card

## 🎯 Bottom Line Results

| Metric | Value |
|--------|-------|
| **Best Performance** | **1.418567** (Focused Bayesian) |
| **Best Reproducible** | **1.420612** (Manual Trial 22) |
| **Baseline** | 1.451763 |
| **Improvement** | **2.3%** |
| **Total Trials** | 74 (10 hours autonomous) |

## 📊 Method Rankings

1. **🥇 Focused Bayesian**: 1.419 (100% success, sample-efficient)
2. **🥈 Focused Genetic**: 1.419 (100% success, robust validation)
3. **🥉 Manual Refined**: 1.421 (strategic, reproducible)
4. Genetic Unfocused: 1.431 (50% success)
5. Bayesian Unfocused: 1.489 (too exploratory)

## 🔧 Optimal Hyperparameters

### Production Config (cleanest values)
```python
EMBEDDING_LR = 0.60        # ↓8% vs baseline
MATRIX_LR = 0.08           # ↑100% vs baseline
WEIGHT_DECAY = 0.18        # ↓10% vs baseline
WARMDOWN_RATIO = 0.3       # ↓40% vs baseline
DEPTH = 4                  # unchanged
```

### Research Config (optimization best)
```python
EMBEDDING_LR = 0.5886      # ↓2% vs production
MATRIX_LR = 0.0857         # ↑7% vs production
WEIGHT_DECAY = 0.1667      # ↓7% vs production
WARMDOWN_RATIO = 0.358     # ↑19% vs production
```

## 💡 Key Insights

### ✅ What Worked
- **Hybrid approach**: Manual → Unfocused optimization → Focused optimization → Manual refinement
- **Narrowing search space**: Fixed optimal categoricals, ±20% continuous params
- **Cross-validation**: Multiple optimizers agreed on optimal region
- **Agent autonomy**: 10 hours of unsupervised experimentation

### ❌ What Didn't Work
- **Unfocused optimization**: Too broad search space wasted 20 trials
- **Direct config reproduction**: Training variance 0.5-0.7% between runs
- **Extreme values**: MATRIX_LR > 0.09, WEIGHT_DECAY < 0.17, EMBEDDING_LR < 0.58

## 📈 Sensitivity Analysis

| Parameter | Optimal Range | Sensitivity |
|-----------|---------------|-------------|
| MATRIX_LR | 0.07-0.09 | 🔴 High |
| EMBEDDING_LR | 0.58-0.65 | 🟡 Medium |
| WARMDOWN_RATIO | 0.25-0.40 | 🟡 Medium |
| WEIGHT_DECAY | 0.16-0.20 | 🟢 Low |
| DEPTH | 4 (fixed) | 🔴 Critical |

## 🚀 Recommended Workflow

For future hyperparameter optimization:

1. **Explore (10-15 trials)**: Manual, identify promising regions
2. **Narrow (5 min)**: Define focused search space ±20% around findings
3. **Optimize (15 trials)**: Run Bayesian on narrowed space
4. **Validate (10 trials)**: Run Genetic to cross-check
5. **Refine (5-10 trials)**: Manual fine-tuning of best configs

**Total time**: ~8 hours GPU, mostly autonomous

## 📝 Quick Start

To use Phase 2 findings:

```bash
# Apply best config to train.py
EMBEDDING_LR=0.60 MATRIX_LR=0.08 WEIGHT_DECAY=0.18 WARMDOWN_RATIO=0.3

# Or run focused optimization yourself
python run_optuna.py bayesian --n_trials 15 --output_dir experiments/my_run

# Parameters will be in focused range (edit run_optuna.py lines 24-32)
```

## 📚 Files

- `PHASE2_SUMMARY.md` - Full detailed report
- `phase2_progress.png` - Main results visualization
- `phase2_parameter_exploration.png` - Parameter space analysis
- `results.tsv` - Manual trial log
- `experiments/` - All optimization runs (Bayesian, Genetic)

---

**TL;DR**: Agent + Focused Optimization beat Manual-only by **0.5%**. Use hybrid workflow: explore manually → narrow search → optimize automatically → refine manually.
