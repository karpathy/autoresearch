`prepare.py` populates this directory with editable working copies of the
deception2 baseline scripts.

The intended workflow is:

1. Seed `sandbox/feature_extractor.py` and `sandbox/multidataset_ood_xgb.py`
   from `/playpen-ssd/smerrill/deception2/src/`.
2. Let the autonomous loop edit only these sandbox copies.
3. Write all generated features, models, metrics, logs, and notebooks under
   `runs/<tag>/`.

The source dataset tree under `/playpen-ssd/smerrill/deception2/Dataset` is
treated as read-only input.
