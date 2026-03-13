# Experiment Code

## Module Structure

| File | Purpose |
|---|---|
| `models.py` | All `nn.Module` classes: `TimeMLP`, `StateMLP`, `ODEFunc`, `PotentialMLP` |
| `solvers.py` | RK4 integrator, Euler/RK4 rollouts (numpy + torch), metrics, energy helpers |
| `plotting.py` | Publication-quality figure generation for all experiments |
| `scripts/run_experiments.py` | Main experiment runner (imports from above modules) |

## Commands

From project root:

```bash
python code/scripts/run_experiments.py --mode full        # full training budget, save figures
python code/scripts/run_experiments.py --mode quick        # reduced epochs for validation
python code/scripts/run_experiments.py --mode quick --no-plots  # skip figure generation
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `full` | `quick` for fast validation, `full` for report-grade results |
| `--seed` | `395` | Random seed for numpy and torch |
| `--device` | `cpu` | Compute device |
| `--no-plots` | off | Skip figure generation |

## Outputs

- `results_summary.json`: all metrics, parameter counts, training times
- `figures/*.png`: 10 figures covering both systems
