# Near-Misses

Near-misses: within ~0.01 of best (1.912682).

| Commit | val_bpb | Delta | Config | Notes |
|---|---|---|---|---|
| b1f96dd | 1.912736 | +0.000054 | EMA homeostatic alpha=0.005 | Too slow; try alpha=0.05 or batch-level stats |
| 4dc96da | 1.912861 | +0.000179 | Progressive sparsification 20%/10%/5% | Budget-neutral, no gain |
| 7d55f8e | 1.912832 | +0.000150 | PRUNE_TARGET=55% no MQA | Previous best before MQA |
| 8496b3f | 1.913173 | +0.000491 | sqrt pruning schedule | Linear is better |
| 507e975 | 1.913322 | +0.000640 | PRUNE_TARGET=65% | Over-pruned |

## Revisit candidates
- **Homeostatic plasticity**: retry with alpha=0.05 or instantaneous batch stats — EMA was too slow
- **Progressive sparsification**: try non-budget-neutral variants (more total sparsity to find new sweet spot)
