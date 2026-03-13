# Experiment Code

Primary entry point: `scripts/run_experiments.py`

## Modes

- `--mode full`: report-grade training budget
- `--mode quick`: reduced training for validation

## Commands

From project root:

```bash
python code/scripts/run_experiments.py --mode full
python code/scripts/run_experiments.py --mode quick --no-plots
```

## Outputs

- `code/results_summary.json`
- `code/figures/*.png`

## Reproducibility

Use `bash scripts/reproduce_all.sh` to regenerate experiments and report in one command.
