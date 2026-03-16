# Evaluating Experiment Results at Scale

Once you've run 50+ experiments, the results.tsv file becomes hard to read by hand. This guide covers practical approaches to separating real improvements from noise.

## Understanding val_bpb

val_bpb measures validation bits per byte. Lower is better. It tells you how efficiently the model compresses unseen text.

Small deltas matter. A drop from 0.998 to 0.995 looks tiny but represents a meaningful improvement in next token prediction quality. The evaluation in prepare.py is fixed and deterministic, so the only variable across runs is what you changed in train.py. That said, GPU nondeterminism (cuDNN, floating point ordering) still introduces some jitter.

## The noise floor

After enough experiments, improvements get small. A 0.001 val_bpb improvement on a 5 minute training run could be real signal or it could be jitter from initialization randomness.

You can estimate the noise floor from your own history. Take all consecutive keep pairs from results.tsv and compute the median of their absolute pairwise differences:

```bash
# Noise floor from consecutive keeps
awk -F'\t' '$4=="keep" {if(prev!="") print ($2>prev ? $2-prev : prev-$2); prev=$2}' results.tsv | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)]}'
```

The result is your noise floor. Anything smaller than this number is probably not a real improvement. Anything much larger probably is.

With fewer than 5 keeps the noise floor estimate is unreliable. In early runs, trust absolute val_bpb improvements of 0.003 or more and be skeptical of anything smaller.

## Pareto efficiency

val_bpb and memory_gb are the two numbers that matter. program.md says VRAM is a "soft constraint" and "some increase is acceptable for meaningful val_bpb gains." In practice this means you need to track the tradeoff.

A 0.001 val_bpb improvement that doubles your VRAM usage is not clearly better. A 0.005 improvement that adds 2 GB probably is.

Sort your keeps by val_bpb and check the memory cost:

```bash
# Keeps sorted by val_bpb, showing memory cost
awk -F'\t' '$4=="keep" {printf "%.6f\t%.1f GB\t%s\n", $2, $3, $5}' results.tsv | sort -n
```

Experiments on the lower left of a val_bpb vs memory_gb scatter are Pareto efficient. They give you the best val_bpb for their memory budget. Everything else paid more memory than the improvement justified.

If a row has lower val_bpb than the previous keep but also higher memory, it's a tradeoff. If it has both lower val_bpb and lower memory, it dominates. Those are your best experiments.

## Reading results.tsv at scale

At 100+ rows, eyeballing stops working. Some useful views:

**Keeps only, sorted by val_bpb:**
```bash
awk -F'\t' 'NR==1 || $4=="keep"' results.tsv | sort -t$'\t' -k2 -n
```

**Success rate per experiment category:**

If you use consistent description prefixes (like "architecture:", "optimizer:", "hyperparams:"), you can compute which categories are working:

```bash
awk -F'\t' 'NR>1 {split($5,a,":"); cat=a[1]; total[cat]++; if($4=="keep") keeps[cat]++} END {for(c in total) printf "%s: %d/%d (%.0f%%)\n", c, keeps[c]+0, total[c], (keeps[c]+0)/total[c]*100}' results.tsv | sort -t'(' -k2 -rn
```

**Crash rate:**
```bash
awk -F'\t' '$4=="crash"' results.tsv | wc -l
```

If crashes are above 20% of total runs, you're trying changes that are too aggressive. Scale back the magnitude of your modifications.

**Improvement rate over time:**
```bash
# Compare first half vs second half keep rates
awk -F'\t' 'NR>1 {total++; if($4=="keep") keeps++; if(total==int(TOTAL/2)) {printf "First half: %d/%d\n", keeps, total; keeps=0; total=0}}' TOTAL=$(awk 'END{print NR-1}' results.tsv) results.tsv
```

A declining keep rate is normal. It means the easy wins are gone and you're in the long tail of incremental improvements.

## When to trust an improvement

Use the noise floor as your ruler.

**Above 1.5x the noise floor:** probably real. Keep with confidence.

**Between 1.0x and 1.5x:** ambiguous. The simplicity criterion from program.md applies here. If the change is simple, lean toward keeping. If it added complexity, consider discarding.

**Below 1.0x:** probably jitter. Discard unless the change is a simplification. program.md explicitly says "removing something and getting equal or better results is a great outcome." A simplification that holds val_bpb steady is always worth keeping.

**Early runs (fewer than 5 keeps):** the noise floor itself isn't stable yet. Use absolute thresholds instead. A 0.003+ val_bpb improvement is likely real. Below 0.001 is likely noise. Between 0.001 and 0.003 is a judgment call where simplicity should break the tie.

## Useful one liners

All commands assume standard results.tsv format (tab separated, 5 columns: commit, val_bpb, memory_gb, status, description).

```bash
# Best val_bpb achieved
awk -F'\t' '$4=="keep" {print $2}' results.tsv | sort -n | head -1

# Total experiments, keeps, discards, crashes
awk -F'\t' 'NR>1 {total++; s[$4]++} END {printf "total: %d, keep: %d, discard: %d, crash: %d\n", total, s["keep"]+0, s["discard"]+0, s["crash"]+0}' results.tsv

# Memory range across keeps
awk -F'\t' '$4=="keep" {print $3}' results.tsv | sort -n | awk 'NR==1{min=$1} END{printf "%.1f to %.1f GB\n", min, $1}'

# Last 10 experiments
tail -10 results.tsv

# All keeps below a specific val_bpb threshold
awk -F'\t' '$4=="keep" && $2<0.995' results.tsv

# Biggest single improvement (largest drop between consecutive keeps)
awk -F'\t' '$4=="keep" {if(prev!="") {d=prev-$2; if(d>max){max=d; line=$0}} prev=$2} END{printf "%.6f improvement: %s\n", max, line}' results.tsv
```
