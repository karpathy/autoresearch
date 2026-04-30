# Physics-Inspired Search Strategies for autoresearch

> A companion to [`bioinspired.md`](bioinspired.md). Bio gives you *what to do*;
> physics gives you *whether the thing you just did was real*. Both are needed.

The autoresearch loop has a clean physical interpretation. The state is the
contents of `train.py`. The energy is `val_bpb` (lower = lower energy). The
moves are commits. The temperature is implicit — right now, `T = 0` (any
worse move is rejected outright). The agent is a particle doing greedy
descent on a noisy energy landscape. So we should ask the same questions a
physicist asks of any zero-temperature dynamics:

1. *How big is the noise we are descending into?*
2. *How does the dynamics behave at the noise floor?*
3. *What invariants must any honest update preserve?*
4. *What would a non-zero temperature change?*

These give us five concrete protocol changes for `program.md`, ordered by
how directly they protect against a known failure mode of greedy hill-climbs.

---

## 1. Measure the noise floor before you celebrate anything

> *"The first principle is that you must not fool yourself — and you are the
> easiest person to fool."* — Feynman, Cargo Cult Science (1974)

`val_bpb` is reported to six decimal places. That suggests precision the
metric does not have. Nothing in the loop is *exactly* deterministic:

- bf16 matmul reductions on CUDA have order-dependent floating-point error.
- the dataloader's best-fit packing is deterministic given the doc stream,
  but `gc.collect()` timing across runs can shift CUDA stream ordering.
- the FA3 kernel can dispatch differently across driver versions.

So the same `train.py`, run twice, gives slightly different `val_bpb`. Call
that variance `σ²_noise`. Any reported delta `|Δval_bpb| ≲ σ_noise` is
**indistinguishable from noise**. The greedy KEEP rule accepts these as wins.

**Back-of-envelope.** Cross-entropy on a 21M-token val shard, with per-token
loss std ~0.5 and ~3 bytes/token, gives an irreducible statistical floor
around `σ_stat ≈ 0.5 / sqrt(21e6) / log(2) / 3 ≈ 5e-5`. The CUDA-kernel
nondeterminism is empirically larger — typically a few × 1e-4. Without
measuring, treat **anything below ~3e-4** as suspect.

**Protocol.** At the start of every run, train the *current* baseline twice
with different `torch.manual_seed` values (or, cheaper, the same seed and
just relying on CUDA nondeterminism between runs). Define
`σ_noise = |val_bpb_run1 − val_bpb_run2|`. Reject any later "improvement"
smaller than `2 × σ_noise` as a discard, regardless of sign.

**Cost.** One extra run per session — pays for itself the first time it
prevents a noise-driven KEEP from anchoring the next 12 experiments.

---

## 2. Metropolis-Hastings: temperature > 0

The current accept/reject rule is the `T → 0` limit:

```
accept iff val_bpb_new < val_bpb_old
```

A finite-temperature version (Metropolis-Hastings):

```
ΔE = val_bpb_new − val_bpb_old           # could be negative
P(accept) = 1                            if ΔE < 0
P(accept) = exp(−ΔE / T(progress))       if ΔE ≥ 0
```

with a cooling schedule `T(progress)` that starts warm and drops to 0
toward the end of the session. This is exactly how
[simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
escapes local minima — you take the occasional uphill step, on purpose, to
reach a better basin you couldn't see from where you started.

**Calibrating `T`.** Pick `T(0)` so that a *typical* discard's `ΔE`
(`0.001` to `0.005` in BPB units) gets accepted with probability ~0.3 at
the start of the session and ~0 near the end. A geometric schedule
`T(p) = T_0 · (T_end / T_0)^p` with `T_0 ≈ 0.005` and `T_end ≈ 1e-5` works.
The flask-sized table:

| ΔE     | P(accept) at p=0.0 | at p=0.5 | at p=1.0 |
| ------ | ------------------ | -------- | -------- |
| 0.0005 | 0.90               | 0.21     | 0.00     |
| 0.0010 | 0.82               | 0.05     | 0.00     |
| 0.0050 | 0.37               | 0.00     | 0.00     |

**Why it matters.** Greedy + a deceptive landscape → trap. Lévy flights
(`bioinspired.md` #1) help by occasional long jumps; SA helps by
occasionally accepting *the result* of a short jump even when it looked
worse. The two operate on different parts of the dynamics and stack.

---

## 3. Fluctuation-Dissipation: read the curvature out of the discards

> *In equilibrium, the variance of fluctuations is proportional to the
> susceptibility — the system tells you its stiffness for free.*

Every discard you log is a measurement of the local landscape. After K
discards near the current state with deltas `δ₁, …, δ_K`, you have a
free, no-extra-cost estimate of local curvature:

```
σ_local = stdev(δ₁, …, δ_K)        # local "softness"
```

- **σ_local small (≲ 2 σ_noise):** the surface is locally flat. Short
  steps are wasted; take medium / long jumps next (`bioinspired.md` Lévy).
- **σ_local large:** you are on a steep slope. Reduce mutation size,
  step carefully along the gradient direction implied by the recent KEEPs.

**Protocol.** Every K=10 experiments, compute `σ_local` from the last K
deltas (the *signed* differences `val_bpb_new − val_bpb_old`, including
discards). Quantize the result into {flat, slope, cliff} and let the
mutation-size policy depend on the regime. Two extra lines in
`results.tsv` analysis, no code change to `train.py`.

This is the same idea behind classical [trust-region](https://en.wikipedia.org/wiki/Trust_region)
optimizers, derived in two lines from
[fluctuation-dissipation](https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem).

---

## 4. The Feynman ratchet: watch out for asymmetric reject rules

[The Feynman–Smoluchowski ratchet](https://en.wikipedia.org/wiki/Brownian_ratchet)
*looks* like it can extract work from thermal noise — a pawl that only
allows rotation in one direction. Feynman's lecture takes the device apart
and shows that, at uniform temperature, the pawl is itself buffeted by
noise and lifts at exactly the rate the wheel kicks it. Net work: zero.
Any apparent net rotation is an artefact of the asymmetry of the model,
not real work.

The autoresearch loop is full of opportunities to build accidental
ratchets:

- **"Any improvement is real, but only big regressions discard."** Asymmetric
  rejection thresholds turn the noise floor into a one-way valve and
  ratchet into the noise. Symmetric thresholds (`|Δ| < ε ⇒ no change`)
  fix it.
- **"It's easier to add capacity than to remove it."** If the agent has
  more comfortable mutations in the "make-it-bigger" direction, it
  ratchets toward larger models even when smaller ones are equivalent.
  The `program.md` `Simplicity criterion` is the explicit pawl-disabler.
- **"Failures don't get logged in detail; successes get analyzed."** This
  ratchets confirmation bias. Logging discards as carefully as keeps
  flattens the asymmetry.

**Protocol.** When proposing a change, also write down the *reverse* move
(the move that would undo it). If you can't construct the reverse cheaply,
your search has an asymmetric pawl somewhere and you should add the
reverse to the available-moves list explicitly.

---

## 5. Action principle: penalize description length

> *"Among all the paths a particle could take, the one it takes is the one
> that makes the action stationary."*

`val_bpb` measures fit. It does not measure complexity. Two experiments at
the same `val_bpb`, one of which adds 80 lines of optimizer hackery and one
of which adds 4, are *not* equivalent — the simpler one transfers, the
complex one is a Rube Goldberg machine that survived because the metric
has a flat direction along complexity.

Define a soft information-theoretic prior over a change:

```
description_length(change) ≈ size_of_diff_in_bytes
```

The accept rule becomes a free-energy comparison rather than a pure energy
one:

```
F = val_bpb + λ · description_length / (1 byte_per_unit)
accept iff F_new < F_old
```

with `λ` calibrated so a 100-byte diff is worth ~0.0005 BPB. This is the
exact form Hinton & Zemel used to derive the
[autoencoder bottleneck via MDL and Helmholtz free energy](https://proceedings.neurips.cc/paper/1993/hash/9e3cfc48eccf81a0d57663e129aef3cb-Abstract.html).

**Protocol.** A 6th column in `results.tsv`: `diff_bytes` (line count is
the cheap proxy; `git diff --shortstat HEAD^ HEAD | awk '{...}'`). Compare
`val_bpb + 5e-6 × diff_bytes` between candidate and current state. This
formalizes the existing English `Simplicity criterion` so two agents can
agree on it without rhetoric.

---

## 6. Renormalization: don't trust a small-scale win until a single mid-scale check

`DEPTH = 8` is a knob. The hyperparameters that win at d=8 don't generally
win at d=24. This is just renormalization-group flow: the relevant
operators at one scale are different from the relevant operators at
another. The agent should not declare a kept improvement transferable
without a single confirming run at a larger scale.

**Cost.** Roughly one out of every 20 keeps gets a single d=24 confirmation
run (~5 min if d=24 fits the time budget; otherwise compare relative
deltas). If the improvement vanishes at the bigger scale, downgrade the
keep to "scale-bound" and don't carry it forward as a foundation for new
keeps.

This is what [issue #226 (Principlex)](https://github.com/karpathy/autoresearch/issues/226)
is asking for in different language; the physics frame just gives a clean
name and a calibration knob (~5% of compute spent on scale-confirmation).

---

## 7. The "honest loss" check: track second-order signals

`val_bpb` is one number. Two runs at the same `val_bpb` can be very
different models — different gradient norms, different attention entropy,
different sensitivity to a single token flip. The current loop ignores
this; the agent treats `val_bpb` as a sufficient summary statistic.

In thermodynamic language: `val_bpb` is the energy *only*. There are
*entropy*-like degrees of freedom — gradient-norm variance, eigenvalue
spread of the Hessian, attention entropy across heads — that change the
free energy without changing the energy. Two well-known consequences:

- **A change that lowers `val_bpb` but raises gradient-norm variance** is
  trading robustness for fit. It will under-transfer (cf. #6).
- **A change that lowers `val_bpb` but concentrates attention entropy** is
  often a memorization artifact, not generalization.

**Protocol.** Capture three cheap auxiliary signals at the end of each
run (one extra log line each):

```
final_grad_norm:     <peak grad-norm at the last step>
attn_entropy_mean:   <mean across heads of -Σ p log p over the last batch>
loss_std_last_100:   <stdev of train_loss over the last 100 steps>
```

These are not part of the accept rule. They're a *consistency check* the
agent can read after a KEEP to flag suspect wins ("BPB went down but
gradient norm 5x'd — this win is on shaky ground; verify before building
on it"). Two-line additions to `train.py`, optional to log.

---

## 8. Path integrals over architectures (sketch)

The truly Feynman-style move is to abandon "the agent has a single current
state" and replace it with "the agent maintains a posterior over states
weighted by `exp(−val_bpb / T)`." Each experiment is a Monte Carlo sample;
the running posterior is the right thing to *propose from*. This is
[Neural Simulated Annealing](https://proceedings.mlr.press/v206/correia23a/correia23a.pdf)
done by hand — the agent's "memory" is the empirical Boltzmann
distribution over recent states.

This is the heaviest proposal here and it overlaps with `bioinspired.md`
patterns #2 (pheromone) and #4 (replay). Listing it for completeness.
Don't bother adopting it until #1–#5 are running and the loop has actually
saturated against them.

---

## Suggested integration order

The first five are *protocol changes only* — no Python edits, just
amendments to `program.md` plus a couple of extra columns in
`results.tsv`:

1. **#1 noise floor**: cheapest, prevents the largest class of false
   positives. Do this first.
2. **#4 ratchet check**: a one-line discipline ("write down the reverse
   move"); kills accidental confirmation bias.
3. **#3 fluctuation-dissipation**: free curvature estimate from existing
   discard log.
4. **#2 Metropolis acceptance**: more invasive (changes the accept rule)
   and only worth it once #1 is in place. Without #1, the temperature
   accepts noise; with #1, the temperature accepts *signal you couldn't
   see greedily*.
5. **#5 MDL prior**: turns the existing English `Simplicity criterion`
   into a quantity two agents can agree on.

Patterns #6 (RG) and #7 (honest loss) cost some compute but cure the
biggest *real* failure mode of the current loop — wins that don't transfer
or that are masking a worse model.

---

## Sources and further reading

- [Simulated Annealing (Wikipedia)](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Multi-objective simulated annealing for hyperparameter optimization, *PeerJ CS* 2021](https://peerj.com/articles/cs-338/)
- [Embedded hyper-parameter tuning by Simulated Annealing, arXiv 1906.01504](https://arxiv.org/abs/1906.01504)
- [Neural Simulated Annealing, *PMLR* 2023](https://proceedings.mlr.press/v206/correia23a/correia23a.pdf)
- [Brownian / Feynman–Smoluchowski ratchet (Wikipedia)](https://en.wikipedia.org/wiki/Brownian_ratchet)
- [Fluctuation-dissipation theorem (Wikipedia)](https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem)
- [Hinton & Zemel: Autoencoders, MDL, Helmholtz free energy, *NeurIPS* 1993](https://proceedings.neurips.cc/paper/1993/hash/9e3cfc48eccf81a0d57663e129aef3cb-Abstract.html)
- [The free-energy principle, *Nature Reviews Neurosci* 2010](https://www.nature.com/articles/nrn2787)
- [Feynman, *Cargo Cult Science* (Caltech 1974 commencement address)](https://calteches.library.caltech.edu/51/2/CargoCult.htm)
- [Trust region method (Wikipedia)](https://en.wikipedia.org/wiki/Trust_region)
