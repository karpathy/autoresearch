Project: toy-rule
Goal: lower mean_error of the toy rule without losing coverage
Metric: mean_error, min

## Project
project/ (rule.txt is the only editable file)

## Goal
Lower the mean absolute error of rule.txt on the fixed cases in cases.json.

## Metric and direction
mean_error, min. Guard: coverage, max.

## Goal criteria
not specified, exploratory only

## Evaluation
evaluator.py next to this brief. It is offline and deterministic, reads cases.json, and writes result.json.

## Constraints
evaluator.py, cases.json, spec.json and this brief do not change. Only the candidate's rule.txt is edited.

## Limits
no paid requests; no time or trial cap
