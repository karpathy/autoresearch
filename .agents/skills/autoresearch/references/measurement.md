# Measurement, goals, and evidence

## Comparability

The baseline and trials in an epoch share a frozen evaluator, metric contract, protected inputs, and evaluator fingerprint. If one changes intentionally, open a new epoch with a reason. Keep prior manifests and history intact; results across different epochs or fingerprints are not comparable.

Use the manifest's `parent` as the trial's comparison point. Read the guide's aggregate, worst-repeat, and parent-relative deltas together. The primary trial gate uses the configured repeat rule (`acceptance.statistic`: `"worst"` by default, or `"mean"` / `"median"` for noisy benchmarks; `acceptance.stop_on_first_failure` defaults to `true` for `"worst"` and is bypassed when aggregating across all repeats). Guard metrics must also hold. When `"worst"` is active, a better mean does not override a failed worst-repeat gate.

Guide signals summarize evidence-backed opportunities: primary improvement paired with a guard regression (`near_miss`), repeat spread (`high_variance`), per-case regressions on a specific repeat (`case_regression` with `regressed_case_ids`), semantic overlap with prior failed trials (`similar_hypothesis_rejected`), or recurring rejection (`repeated_rejection`, which after 3+ consecutive failures advises consulting `openresearch` (`orx discover`) for new algorithmic framing). Signals link to run IDs and help focus inspection; they are advisory, not extra acceptance gates. They require comparable contract fingerprints and epoch. Verify the linked result or case evidence before making a hypothesis.

## Goal criteria

Framework acceptance answers whether a trial beats its parent under the lab's configured repeat and guard rules. Goal success answers whether a declared external quality objective passed. These are separate results.

The optional `goal_criteria` block has exactly this shape:

```json
{
  "thresholds": {"quality": {"min": 0.9}, "latency_ms": {"max": 250}},
  "confirmation_repeats": 3
}
```

Each threshold names a metric and has at least one finite `min` or `max`; `confirmation_repeats` is a positive integer. This is the automatic goal assessment: every successful repeat of the selected accepted candidate must satisfy every threshold, and the completed repeat count must meet the confirmation count. It confirms repeats of the same candidate within one run; it does not imply independent model or full-run replications.

When thresholds come from an external artifact, record its path and hash alongside the lab for audit. That provenance is not part of `goal_criteria` and is not inferred or validated automatically.

Goal status is `unknown` when criteria or evidence are absent, `unconfirmed` when evidence is incomplete or is historical/offline/imported, `not_met` when any successful repeat violates a threshold, and `met` only with sufficient fresh-origin repeats passing all thresholds. Goal criteria do not change trial acceptance. The schema has no arbitrary veto field: only supported primary/guard metrics and explicit goal thresholds are automatic; do not infer a missing cross-gold or other external criterion from prose or report it as an enforced gate.

## Evidence classes

The recorded result origin is one of `fresh`, `offline_recompute`, `historical`, or `imported`. A `fresh` origin means the evaluator ran on the current candidate and contract; inspect request and usage logs to determine whether it called a provider. `offline_recompute`, `historical`, and `imported` do not establish a fresh production goal. The origin label is metadata, not a network sandbox. Recovery actions such as `case_resume` and native lab resume describe how work continued, not a substitute for inspecting each repeat's origin and attempt logs. Keep unknown historical usage separate from usage observed during recovery.

## Epochs

`baseline --supersede --reason TEXT` starts a new baseline epoch. `epoch open --reason TEXT --spec PATH` updates supported contract fields, refreshes fingerprints, and clears baseline pointers without rewriting `runs/`. Interrupted work must be resolved before opening an epoch.

Baseline validation modes describe comparability checks: `historical` verifies against comparable saved metric runs; `fresh` requires a hash-pinned known case. Do not change modes merely to bypass a mismatch.
