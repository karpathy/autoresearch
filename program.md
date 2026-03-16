## Autoresearch Program

This repo is an autonomous research loop for training OOD models across datasets.
- The goal is to maximize use the features in /playpen-ssd/smerrill/autoresearch/temporal_window_features
- Each feature file corresponds to a dataset and an aggregation window.
- advisoraudit_window2_temporal means these features were trained on the "advisoraudit" dataset and aggregations were performed over a window of 2
- Your goal is to train a classification model on one dataset and test it on the other two.
   -  For example, you if you train on advisoraudit, you should test on gridworld and bs
   -  Note you should use the same features for testing as you do training (ie. if you use window 2 features for training you should use those for testing as well)
- Your objective is to predict deception rate.
   - Since this is a classification task call a deceptive example, one where the deception rate > 0.4 and not deceptive otherwise
   - You should try to maximize OOD AUROC (how well does your model do on environments it wasn't trained on)
   
## Setup
-  You should make some files to train models and evaluate on a test set
-  Choose models you see most appropriate for the task
   - You can, for instance, try XGBoostClassifiers, NN Classifiers, etc
-  You should also try to understand features and which are important for this task
   - Note we have way more features than examples, so we may not want to select all of them
   - We want to use these features only
   - Further, you are only allowed to use features which start with the string "before_" or "at_"
   - You are not allowed to perform PCA or anything like that
   - But you are allowed to scale/normalize features as you'd like

## Deliverables
- All outputs should be put in a new folder called "/playpen-ssd/smerrill/autoresearch/runs/YYYY-MM-DD"
- You can use this folder to create any temporary files you'd like, for instance it may be useful to cache models here
   - runs/YYYY-MM-DD/tmp
- You should report confusion matrices (runs/YYYY-MM-DD/cm.png)
   - Validation and OOD for the best model
- You should report AUROC curves  (runs/YYYY-MM-DD/auroc.png)
   - Validation and OOD for the best model
- You should report the best model (runs/YYYY-MM-DD/best_model.md)
   - What features did you use
   - How did you process features
   - What model type worked best
- You should report any key findings (runs/YYYY-MM-DD/key_findings.md)
   - What features worked best, and speculate why
   - What features worked well together


## Immutable inputs

Every file in:
/playpen-ssd/smerrill/autoresearch/temporal_window_features
is and must not be modified:

## Label definition

Regression target:

- `y_reg = deception_rate`

Classification target:

- `y_cls = 1 if deception_rate > threshold else 0`

The default threshold search is `0.4`.

This thresholded label definition is mandatory unless you deliberately change
the experiment design and document why.

## GPU rule

All experiments must run with only GPU 4 visible.


## Scoring and keep/discard policy

Train on one datset, then the other two are OOD datasets we want to compute metrics on.  Keep a change when it materially improves one objective without causing a clear
collapse in the other. Prefer simpler changes when scores are effectively tied.

If a run crashes, fix obvious issues and rerun. If the idea is fundamentally
bad, log it as a crash and move on.


## Additional requirements
- The dataset tree is read-only.
- Runs must be seeded and completely reproducable
- Never use GPUs other than GPU 4.
- You should only use the conda environment "deception"
