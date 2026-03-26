# overfit

Prevent fake progress.

The goal is real improvements, not benchmark tricks.

## Core rule

Do not trust a single run.

A lower metric is evidence, not proof.

## General behavior

Prefer robust gains over fragile gains.

Prefer simple changes over complex hacks.

Prefer changes with a clear mechanism.

Be skeptical of surprising improvements.

## Suspicious signals

Treat results as suspicious if:

- a large improvement appears from a tiny opaque change
- a gain disappears when rerun
- the improvement is extremely small
- code complexity increases significantly
- the change has no clear explanation
- the change resembles a benchmark-specific trick

## Validation rule

If a change looks promising:

Rerun the experiment once.

If the improvement disappears, discard it.

If the improvement holds, treat it as stronger evidence.

## Noise rule

Tiny gains are often noise.

If the improvement is extremely small:

Prefer simpler code.

Or rerun the experiment before accepting it.

## Complexity rule

Complexity is a cost.

Prefer:

- fewer lines
- simpler logic
- fewer special cases

A small gain from deleting code is very strong.

A small gain that adds messy code is weak.

## Mechanism rule

Before committing a meaningful change, ask:

Why should this help?

If the result contradicts the hypothesis, be cautious.

## Forbidden behaviors

Do not:

- exploit evaluation quirks
- hardcode dataset assumptions
- add benchmark-specific logic
- indirectly modify evaluation behavior

## Exploration rule

If improvements stop appearing:

Change direction.

Do not endlessly tune tiny hyperparameters.

## Final rule

Optimize for improvements that are likely to generalize, not just improvements that win one run.
