# Phase 2: Hybrid Manual + Automated Hyperparameter Optimization

**Date**: March 11-12, 2026
**Status**: ✅ Complete
**Result**: Hybrid approach achieved **1.418567** (2.3% improvement over baseline)

---

## 🎯 Bottom Line Results

| Metric | Value |
|--------|-------|
| **Best Performance** | **1.418567** (Focused Bayesian Trial 9) |
| **Best Reproducible** | **1.420612** (Manual Trial 22) |
| **Baseline** | 1.451763 (Karpathy's H100 config) |
| **Improvement** | **2.3%** (0.033 bpb reduction) |
| **Total Trials** | 74 trials across 5 methods |
| **GPU Time** | ~10 hours autonomous |
| **Hardware** | RTX 2060 (6GB VRAM) |

### Method Performance Comparison

1. **🥇 Focused Bayesian**: 1.418567 (100% success, most sample-efficient)
2. **🥈 Focused Genetic**: 1.418820 (100% success, robust cross-validation)
3. **🥉 Manual Refined**: 1.420612 (strategic, reproducible)
4. Genetic Unfocused: 1.431 (50% success, too exploratory)
5. Bayesian Unfocused: 1.489 (broad search wasted trials)

---

## 🔧 Optimal Configuration

### Production Config (cleanest values)
```python
EMBEDDING_LR = 0.60        # ↓8% vs baseline (0.65)
MATRIX_LR = 0.08           # ↑100% vs baseline (0.04)
WEIGHT_DECAY = 0.18        # ↓10% vs baseline (0.20)
WARMDOWN_RATIO = 0.3       # ↓40% vs baseline (0.50)
DEPTH = 4                  # unchanged (memory constraint)
```
**Result**: 1.420612 (validated, reproducible)

### Research Config (optimization best)
```python
EMBEDDING_LR = 0.5886      # Bayesian Trial 9
MATRIX_LR = 0.0857
WEIGHT_DECAY = 0.1667
WARMDOWN_RATIO = 0.358
```
**Result**: 1.418567 (absolute best, minor training variance)

---

## 💡 Key Findings

### ✅ What Worked

**Hybrid approach wins**: Manual exploration → Focused optimization → Manual refinement
- Unfocused Bayesian: 1.489
- **Focused Bayesian: 1.419** (used manual insights to narrow search)
- **Improvement from focusing: 4.7%**

**Search space design is critical**:
- Fixed `DEPTH=4` (was exploring 2-6, mostly failed)
- Narrowed continuous params to ±20% of manual findings
- Result: 100% success rate vs 50% unfocused

**Cross-validation with multiple optimizers**:
- Bayesian and Genetic independently converged to same region
- Both found: `embedding_lr: 0.55-0.59`, `matrix_lr: 0.0806-0.0857`
- Increased confidence in findings

### ❌ What Didn't Work

- **Unfocused optimization**: Broad search wasted 20 trials exploring clearly suboptimal regions
- **Pure automation**: Tools without domain knowledge underperformed manual+tools hybrid
- **Extreme values**: MATRIX_LR > 0.09, WEIGHT_DECAY < 0.17, EMBEDDING_LR < 0.58 all degraded performance

---

## 📈 Parameter Sensitivity

| Parameter | Optimal Range | Sensitivity | Notes |
|-----------|---------------|-------------|-------|
| MATRIX_LR | 0.07-0.09 | 🔴 High | Most impactful, sweet spot at 0.08 |
| EMBEDDING_LR | 0.58-0.65 | 🟡 Medium | Plateau around 0.60 |
| WARMDOWN_RATIO | 0.25-0.40 | 🟡 Medium | 0.3 optimal |
| WEIGHT_DECAY | 0.16-0.20 | 🟢 Low | Gentle regularization effect |
| DEPTH | 4 (fixed) | 🔴 Critical | Memory constraint, 2/3/5/6 all worse |

---

## 🚀 Recommended Workflow

For future hyperparameter optimization on resource-constrained hardware:

1. **Explore** (10-15 manual trials, ~2 hours)
   - Systematic grid search of key parameters
   - Identify promising regions and failure modes

2. **Narrow** (5 minutes analysis)
   - Fix categorical variables to optimal values
   - Define focused continuous search space ±20% around manual findings

3. **Optimize** (15 Bayesian trials, ~3 hours)
   - Run TPE on narrowed search space
   - Most sample-efficient automated exploration

4. **Validate** (10 Genetic trials, ~2 hours)
   - Run CMA-ES for cross-validation
   - Verify Bayesian findings independent

5. **Refine** (5-10 manual trials, ~1 hour)
   - Test edge cases and round to clean values
   - Validate reproducibility

**Total**: ~8 hours GPU time, mostly autonomous after initial exploration

---

## 📊 Experimental Timeline

74 trials organized in 4 stages:

### Stage 1: Manual Exploration (Trials 1-16, ~3 hours)
- Systematic hyperparameter sweep
- **Best**: 1.423438 (Trial 12: EMBEDDING_LR=0.65)
- **Discoveries**: MATRIX_LR sweet spot 0.07-0.08, WARMDOWN_RATIO optimal at 0.3, DEPTH=4 clearly best

### Stage 2: Unfocused Optimization (Trials 17-36, ~3 hours)
- Bayesian unfocused: 1.489 (too exploratory)
- Genetic unfocused: 1.431 (50% crash rate)
- **Lesson**: Broad search without domain knowledge wastes trials

### Stage 3: Focused Optimization (Trials 37-66, ~5 hours) ⭐
- Narrowed search space based on Stage 1 findings
- Bayesian focused: **1.418567** (15 trials, 100% success)
- Genetic focused: **1.418820** (15 trials, 100% success)
- **Key**: Both optimizers independently converged to same region

### Stage 4: Manual Refinement (Trials 67-74, ~1 hour)
- Applied optimization insights to clean configs
- **Best**: 1.420612 (reproducible, rounded values)
- Validated findings with multiple reruns

---

## 📁 Phase 2 Files

### Documentation
- `PHASE2_SUMMARY.md` - Full 11KB detailed report with methodology and analysis
- `README.md` - This file (overview + results)

### Visualizations
- `analysis/phase2_progress.png` - All 74 trials, running best tracking
- `analysis/phase2_parameter_exploration.png` - Parameter space heatmaps
- `analysis/plot_phase2_results.py` - Plotting script (reproducible)

### Data
- `results.tsv` - Manual trial log (24 trials with descriptions)
- `experiments/bayesian_focused_*/` - Bayesian TPE results (15 trials)
- `experiments/bayesian_run_*/` - Bayesian unfocused (10 trials)
- `experiments/genetic_focused_*/` - Genetic CMA-ES results (15 trials)
- `experiments/genetic_run_*/` - Genetic unfocused (10 trials)

### Tools
- `run_optuna.py` - Optuna optimization runner (Bayesian/Genetic)
- `train_wrapper.py` - Training wrapper for optimization framework
- `optimization_tools.py` - Helper utilities
- `run_agent.py` - Agent orchestration script

---

## 🔬 Experimental Setup

### Baseline Configuration

Phase 2 started with **Karpathy's original hyperparameters** from the H100 config, with only memory-critical adjustments for RTX 2060 compatibility:

**Original hyperparameters (unchanged)**:
- `EMBEDDING_LR = 0.6`
- `UNEMBEDDING_LR = 0.004`
- `MATRIX_LR = 0.04`
- `SCALAR_LR = 0.5`
- `WEIGHT_DECAY = 0.2`
- `WARMDOWN_RATIO = 0.5`

**Memory adjustments only** (required for 6GB VRAM):
- `DEPTH = 4` (vs 8 original)
- `DEVICE_BATCH_SIZE = 4` (vs 128 original)
- `TOTAL_BATCH_SIZE = 2^16` (vs 2^19 original)
- `WINDOW_PATTERN = "L"` (vs "SSSL" original)

This ensures the agent does NOT benefit from Phase 1 discoveries (best config: val_bpb=1.371).

### Phase 1 Results (Hidden)

All Phase 1 results are archived in `.phase1_archive/` (hidden from agent):
- Agent method: 15 trials, best 1.421
- Bayesian (TPE): 10 trials, best 1.371 ⭐
- Genetic (CMA-ES): 10 trials, best 1.426

### New Tools Available

Two optimization skills are available to the agent:

#### `/bayesian-optimize`
Run Bayesian (TPE) optimization with Optuna
- Sample-efficient probabilistic search
- Good for limited trial budgets
- High exploration, finds novel configurations

#### `/genetic-optimize`
Run Genetic (CMA-ES) optimization with Optuna
- Evolutionary population-based search
- Adapts search distribution over time
- Good for smooth objective landscapes

### Optimization Tools Used

Two complementary optimization approaches:

#### Bayesian Optimization (TPE)
- **Focused run**: 15 trials, best 1.418567
- Sample-efficient probabilistic search
- Explores high-uncertainty regions intelligently
- Best for limited trial budgets

#### Genetic Optimization (CMA-ES)
- **Focused run**: 15 trials, best 1.418820
- Evolutionary population-based search
- Adapts covariance matrix over generations
- Good for smooth objective landscapes

Both optimizers independently converged to nearly identical optimal regions, validating the findings.

---

## 🎓 Lessons Learned

### Critical Success Factors

1. **Domain knowledge beats pure automation**
   - Manual exploration identified that DEPTH=4 was optimal
   - Focusing search space on this knowledge improved success rate from 50% → 100%

2. **Hybrid workflow is sample-efficient**
   - 16 manual trials identified promising region
   - 30 focused optimization trials refined to optimum
   - vs 50+ trials needed for unfocused exploration

3. **Cross-validation increases confidence**
   - Two independent optimizers agreed on optimal hyperparameters
   - Reduced risk of overfitting to single optimizer's biases

### Transferable Insights

- **For resource-constrained setups**: Manual exploration is essential—tools need guidance
- **For optimization problems**: Always validate findings with independent methods
- **For agent systems**: Strategic tool use > pure automation

---

## 📊 Comparison to Phase 1

Phase 2 used a **fair baseline** (Karpathy's H100 config) rather than building on Phase 1 results.

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Best result** | 1.371 | 1.419 |
| **Baseline** | ~1.45 | 1.452 |
| **Improvement** | 5.3% | 2.3% |
| **Methods** | 3 separate | 5 methods combined |
| **Total trials** | 35 | 74 |
| **Approach** | Pure methods | Hybrid workflow |

**Note**: Phase 2 focused on demonstrating hybrid methodology rather than absolute performance. Starting from Phase 1's best config (1.371) would have made fair comparison impossible.

---

## 🚀 Quick Start

Apply Phase 2 findings:

```bash
# Update train.py with optimal config
EMBEDDING_LR=0.60
MATRIX_LR=0.08
WEIGHT_DECAY=0.18
WARMDOWN_RATIO=0.3

# Or run focused optimization yourself
python phase2/run_optuna.py bayesian --n_trials 15

# View all results
cat phase2/results.tsv
python phase2/analysis/plot_phase2_results.py
```

For full methodology and detailed analysis, see `PHASE2_SUMMARY.md`.
