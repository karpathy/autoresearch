# Bio-Inspired Search Strategies for autoresearch

> A proposal for making the autoresearch loop more creative and sample-efficient
> by borrowing search policies that biology has already optimized over millions of years.

The current loop in [`program.md`](program.md) is a single-track greedy hill-climber:
mutate `train.py`, run for 5 minutes, keep if `val_bpb` dropped, otherwise revert.
That works, but it has known failure modes:

- **No memory across runs.** A failed idea can be re-tried; a near-miss is forgotten.
- **No exploration / exploitation balance.** The agent leans on whatever just worked
  until it stops working, then thrashes.
- **No global topology.** Search has no notion of *where* in the architecture/optimizer
  space it is — every step is local.
- **Single track.** Multi-GPU setups end up running uncoordinated copies of the same hill-climb.

The signature of this problem — a *sparse-reward, high-dimensional, non-convex*
search — is exactly what foraging predators, root systems, slime molds, and
hippocampal replay have been pressured to solve. The patterns below map cleanly
onto autoresearch with minimal code, and most can be added without touching
`prepare.py`.

The patterns are ordered from cheapest-to-add to most invasive.

---

## 1. Lévy-flight perturbations (zoology)

**Biology.** Sharks, albatrosses, fruit flies, deer, and human hunter-gatherers
forage with heavy-tailed step-length distributions: many short moves and rare
long jumps. Under sparse, patchy rewards Lévy walks dominate Brownian motion in
expected encounter rate ([Sims et al.; Lévy flight foraging hypothesis](https://pubmed.ncbi.nlm.nih.gov/25709823/)).
Modern metaheuristics (Manta Ray, Cuckoo Search, Prairie Dog) inject a Lévy
operator on top of an underlying local search to escape basins
([Survey of Lévy-flight metaheuristics](https://www.mdpi.com/2227-7390/10/15/2785)).

**Map to autoresearch.** Most experiments today are 1–2 line tweaks; they correspond
to "short moves." Add an explicit **long-jump probability** so the agent occasionally
proposes a *radical* mutation (swap optimizer family, change activation, replace
attention with a different mechanism). Reasonable schedule:

```
P(long_jump) = 0.10               # 10% of experiments
P(medium)    = 0.25               # 25% multi-hyperparam tweaks
P(short)     = 0.65               # 65% single-axis nudges
```

**Smallest implementation.** Add a paragraph to `program.md` instructing the
agent to roll a 0–1 die before choosing each next experiment and to size the
mutation accordingly. No code changes. Optional: track each step's
"jump size" in `results.tsv` as a 6th column.

---

## 2. Pheromone trail with evaporation (ant colony optimization)

**Biology.** Ants deposit pheromone on returning paths in proportion to path
quality; pheromone evaporates over time. The colony collectively converges on
short paths *while preserving exploration* because evaporation prevents lock-in
([Dorigo & Stützle, ACO book](https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf)).
Evaporation rate is the single most consequential ACO hyperparameter — too high
and the colony forgets, too low and it traps in local optima
([review: ACO with multi-evaporation](https://www.sciencedirect.com/science/article/abs/pii/S1568494625013146)).

**Map to autoresearch.** Maintain a small `pheromone.tsv` keyed by **search axis**
(LR, depth, head_dim, optimizer_family, activation, normalization, attention_window,
weight_decay, dropout, …). Each KEEP deposits +Δbpb on the axis(es) the
experiment touched; each DISCARD deposits a tiny negative; every iteration,
multiply all values by ρ ∈ [0.92, 0.97]. Sample the next axis with probability
proportional to pheromone (softmax over current values).

**Smallest implementation.** A 30-line helper script the agent reads/updates
between experiments. Optionally, encode it as a routine instruction in
`program.md` so the agent maintains the table itself in plain text — no code dependency.

---

## 3. Slime-mold reinforcement and pruning (Physarum polycephalum)

**Biology.** *P. polycephalum*, a single-celled organism, can reproduce the
Tokyo rail network in 26 hours by depositing a network of protoplasmic tubes
between food sources, then **thickening tubes that carry useful flow and pruning
the rest**. The dynamics have a clean mathematical model
([Tero et al. shortest-path proof](https://arxiv.org/abs/1106.0423);
[Slime Mould Algorithm](https://aliasgharheidari.com/SMA.html)).

**Map to autoresearch.** Treat each *contiguous lineage of KEEPs* as a tube:
the head of the tube is the current branch, the body is the chain of commits
that contributed. After every K experiments, **prune the weakest tube** —
i.e. abandon the lineage with the smallest cumulative Δbpb in the last
window — and re-seed from the strongest. This converts a single-track
hill-climb into an evolutionary tournament with *positive feedback on
working lineages* and *negative feedback on stagnant ones*. Pairs naturally
with #2 (the pheromone trail tells you *which axes* the surviving tube is
exploiting).

**Smallest implementation.** When running on N GPUs, give each its own
`autoresearch/<tag>-gpuN` branch and a shared `tubes.tsv`. Every K=20
experiments, drop the weakest branch and `git reset` it onto the head of
the strongest. No core code change.

---

## 4. Hippocampal replay + dopamine prediction error (neuroscience)

**Biology.** Between trials, the hippocampus replays recent trajectories,
biased toward those associated with reward-prediction errors
([Ambrose et al.; replay biased by RPE](https://www.nature.com/articles/s41467-025-65354-2)).
Dopamine encodes the temporal-difference reward-prediction error and gates
synaptic plasticity ([Niv 2009 review](https://www.pnas.org/doi/10.1073/pnas.1014269108);
[VTA dopamine drives hippocampal sharp-wave ripples](https://elifesciences.org/articles/99678)).
The posterior hippocampus invigorates exploration; the anterior body delivers
the global value signal that drives exploitation
([Loh et al., explore/exploit dilemma](https://www.nature.com/articles/s41467-020-18864-0)).

**Map to autoresearch.** Build a tiny **replay buffer** of the last K=5 KEEPs
plus their Δbpb. Before proposing each new experiment, the agent must:

1. Pick one buffered KEEP at random with probability ∝ Δbpb (dopaminergic gating
   — high-RPE trajectories get re-sampled).
2. Generate at least one *variant* of it (mutate one hyperparameter, transplant
   it into a different position in the architecture, scale it to a different
   model size, etc.).
3. Otherwise, propose a fresh idea from #1/#2.

Surprising wins (large |actual − predicted| Δbpb) get inflated weight in the
buffer, mimicking RPE-modulated consolidation.

**Smallest implementation.** A "Replay" section in `program.md` that
instructs the agent to maintain `keeps.tsv` with `predicted_delta` next to
`actual_delta`. The agent reads the top-K and proposes variants with explicit
prediction-error logging.

---

## 5. Inhibition of return / novelty bonus

**Biology.** After saccading to a target, primates exhibit *inhibition of return*
— a 200–500 ms suppression of attention to recently-visited locations. The
computational analogue is the *novelty exploration bonus* used in deep RL
agents like Random Network Distillation and Never Give Up
([Lilian Weng exploration survey](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/);
[Wittmann et al., novelty bonus](https://www.sciencedirect.com/science/article/abs/pii/S0028393209000190)).

**Map to autoresearch.** Maintain a **recency penalty** on each search axis.
After an axis is touched, its sampling weight is reduced for the next M=3
experiments regardless of outcome. Combined with #2's pheromone, the effective
sampling distribution is
`P(axis) ∝ pheromone(axis) · novelty(axis)` where
`novelty(axis) = max(0, 1 − recency_count(axis) / M)`. This stops the agent
from thrashing one corner of the search space and forces breadth.

**Smallest implementation.** Two extra columns in the pheromone table. Zero
code outside it.

---

## 6. Apical dominance and lateral roots (botany)

**Biology.** Plants concentrate growth resources at the apical (lead) meristem;
lateral buds are suppressed by auxin from the lead. When the lead is damaged
or competition increases, lateral buds break dormancy and become new leads.
Mature plants exhibit a balance between *one strong lead* and *many quiescent
laterals* — the laterals are a portfolio of fallback hypotheses.

**Map to autoresearch.** Each KEEP creates a **dormant lateral checkpoint** —
a saved branch state with a 1-line description of *why* it might still be
worth re-visiting. The lead branch keeps climbing. If the lead stalls (M
consecutive DISCARDs, e.g. M=8), break a lateral: pick the highest-promise
lateral, switch to it, and resume hill-climbing from there.

**Smallest implementation.** A `laterals.tsv` (commit, source_keep_idx, why_dormant)
maintained by the agent. The "stall break" rule is two sentences in `program.md`.

---

## 7. Cuckoo abandon-the-worst-nest (parallel exploration)

**Biology.** Cuckoo birds parasitize host nests; hosts that detect the
parasitic egg abandon the nest, which is the basis for [Cuckoo Search]
([Yang & Deb, applied to NoC mapping with Lévy flight](https://ieeexplore.ieee.org/document/9570345/)).
The optimizer maintains a population of solutions and periodically abandons a
fraction of the worst with new random ones — a hard exploration kick that
pairs well with Lévy-flight perturbations of the survivors.

**Map to autoresearch.** When N>1 agents are running in parallel, every K=50
experiments the **worst-performing agent** is forcibly re-seeded from the
best-performing agent's current state plus a Lévy long-jump (#1). This is
how a single human research group "rotates a researcher off a dead idea";
encoding it makes the swarm self-correcting.

**Smallest implementation.** A cron-style coordinator script that watches all
`autoresearch/<tag>-gpuN` branches and issues a `git reset --hard` on the
laggard once per K experiments.

---

## 8. Phyllotaxis golden-angle sampling (botany / number theory)

**Biology.** Plants arrange leaves and seeds at successive angles of φ⁻¹ ≈ 137.5°
because that angle minimizes overlap and maximizes light capture; equivalently,
it is the most *irrational* angle, producing the lowest-discrepancy 1D
distribution
([Strauss et al., light capture optimum](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16040);
[Mitchison; biophysical optimality](https://www.nature.com/articles/srep15358)).
The corresponding multi-dimensional generalization (R₂ generalized golden
ratio) is one of the lowest-discrepancy quasi-random sequences known
([Roberts: unreasonable effectiveness of quasirandom sequences](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)).

**Map to autoresearch.** When the agent decides to *sweep* a continuous
hyperparameter (LR, weight decay, embedding scale, gate temperature, …),
sample candidate values from a golden-ratio sequence on log-space rather
than uniform/random/grid. Empirically this covers space ~10–30% more
efficiently than random search at the budgets autoresearch operates at.

**Smallest implementation.** A 5-line helper:
```python
GOLDEN = 0.61803398875
def golden_iter(n, lo, hi, seed=0.5):
    x = seed
    out = []
    for _ in range(n):
        x = (x + GOLDEN) % 1.0
        out.append(lo * (hi / lo) ** x)        # log-space interpolation
    return out
```
The agent calls this whenever it decides to sweep a continuous knob.

---

## 9. Octopus distributed cognition (zoology, multi-agent)

**Biology.** Two-thirds of an octopus's neurons sit in its arms; each arm has
substantial autonomy, executes local reflexes, and only occasionally consults
the central brain. The colony of arms is *loosely* coupled — coordination is
expensive, so it's used sparingly. This is why an octopus can perform eight
different things in parallel and still appear coherent.

**Map to autoresearch.** On multi-GPU setups, give each agent a *strong default*
to act locally, *weak coupling* via a shared `results.tsv`, and a single
synchronization point per N=20 experiments to share laterals (#6) and update
the global pheromone table (#2). Avoid tight synchronization — it's the
single biggest reason multi-agent setups underperform `N` solo agents.

**Smallest implementation.** Already implicit in the existing branch-per-GPU
recipe; this just makes the synchronization cadence explicit.

---

## Suggested integration order

If only one is added: **#1 (Lévy)** — pure prompt change, zero infra, biggest
expected lift on a sparse-reward landscape.

If two: add **#5 (inhibition of return / novelty)** — also pure prompt + one
small table — to cure the thrashing failure mode.

If you have multi-GPU: add **#7 (cuckoo abandonment)** — it's where
parallelism actually starts to compound rather than just multiplying single-track
search.

If the agent is reaching the end of its idea pool: add **#4 (replay)** — it
turns the latest few wins into a renewable source of variation.

The other patterns are stronger but require more scaffolding — start with
the cheap ones and only add infra as the loop demonstrably stalls without it.

---

## Prior art and citations

**Lévy flight foraging**
- [The Lévy flight foraging hypothesis (review)](https://pubmed.ncbi.nlm.nih.gov/25709823/)
- [Survey of Lévy-flight metaheuristics, *Mathematics*, 2022](https://www.mdpi.com/2227-7390/10/15/2785)
- [Memoryless search under sparse rewards (IEEE)](https://ieeexplore.ieee.org/document/9172790/)
- [Manta Ray Foraging + chaotic Lévy + restart, *Sci. Rep.* 2025](https://www.nature.com/articles/s41598-025-25766-y)

**Slime mold / Physarum**
- [Tero et al., Physarum can compute shortest paths](https://arxiv.org/abs/1106.0423)
- [Slime Mould Algorithm (Li et al.)](https://aliasgharheidari.com/SMA.html)
- [Improved Physarum for shortest path, *Sci. World J.* 2014](https://onlinelibrary.wiley.com/doi/10.1155/2014/487069)

**Ant colony optimization**
- [Dorigo & Stützle, ACO book](https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf)
- [Multi-evaporation ACO ensemble (EPAnt)](https://www.sciencedirect.com/science/article/abs/pii/S1568494625013146)
- [Ant colony optimization (Wikipedia)](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)

**Hippocampal replay and dopamine RPE**
- [Replay biased by reward-prediction signals, *Nat. Comm.* 2025](https://www.nature.com/articles/s41467-025-65354-2)
- [Niv, dopamine RPE hypothesis review, *PNAS*](https://www.pnas.org/doi/10.1073/pnas.1014269108)
- [Posterior vs anterior hippocampus in explore/exploit, *Nat. Comm.* 2020](https://www.nature.com/articles/s41467-020-18864-0)
- [VTA dopamine gates hippocampal replay, *eLife* 2024](https://elifesciences.org/articles/99678)
- [Robotic model of reverse replay for RL (arXiv)](https://arxiv.org/pdf/2102.11914)

**Novelty bonus and inhibition of return**
- [Lilian Weng, Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)
- [The novelty exploration bonus, Wittmann et al., *Neuropsychologia* 2009](https://www.sciencedirect.com/science/article/abs/pii/S0028393209000190)
- [Exploration in deep RL: A survey, arXiv 2205.00824](https://arxiv.org/pdf/2205.00824)

**Phyllotaxis and quasi-random sampling**
- [Phyllotaxis: golden angle for light capture, *New Phytologist* 2020](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16040)
- [Biophysical optimality of the golden angle, *Sci. Rep.* 2015](https://www.nature.com/articles/srep15358)
- [Roberts: unreasonable effectiveness of quasi-random sequences](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
- [GoldenSequences.jl reference implementation](https://github.com/mschauer/GoldenSequences.jl)
- [Low-discrepancy sequence (Wikipedia)](https://en.wikipedia.org/wiki/Low-discrepancy_sequence)
