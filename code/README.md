# Experiment Code

This folder contains the full experiment runner used in the report.

## Entry point

- `scripts/run_experiments.py`

## CLI options

- `--mode full|quick`:
  - `full` reproduces report-grade results
  - `quick` runs reduced epochs/trajectories for fast validation
- `--seed <int>`: set random seed (default `395`)
- `--no-plots`: skip figure generation

## Typical commands

From project root:

```bash
python code/scripts/run_experiments.py --mode full
python code/scripts/run_experiments.py --mode quick --no-plots
```

## Outputs

- `code/results_summary.json`: machine-readable metrics
- `code/figures/*.png`: generated plots

## Notes

- Script is CPU-only by design for portability.
- Results are deterministic for a fixed seed.
