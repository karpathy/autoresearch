# Logs Directory

Detailed logs are kept locally for reproducibility but not committed to reduce repository size.

## What's Kept Local
- Individual trial logs (`trial_*.log`)
- Optuna study databases (`study.db`)
- Detailed training logs

## What's Committed
- Experiment summaries (`summary.json`)
- Results data (`results.jsonl`)  
- Main results table (`results.tsv`)

## Reproducing Results
Re-run experiments to generate logs:
```bash
cd phase2
python run_optuna.py bayesian --n_trials 15 --output_dir experiments/my_run
```
