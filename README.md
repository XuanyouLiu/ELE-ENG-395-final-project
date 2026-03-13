# Scientific ML for Nonlinear Dynamics

ELE_ENG 395 final project comparing data-driven and physics-informed models for learning nonlinear ODE dynamics from noisy trajectories.

## Key Results

| System | Model | State 1 RMSE | State 2 RMSE | Params | Time (s) |
|---|---|---|---|---|---|
| Van der Pol | Baseline MLP (2x32) | 0.053 (z) | 0.239 (v) | 1,153 | 0.4 |
| Van der Pol | Neural ODE | 0.188 (z) | 0.383 (v) | 4,482 | 23.7 |
| Pendulum | Baseline MLP | 0.496 (theta) | 1.511 (omega) | 4,482 | 2.7 |
| Pendulum | Neural ODE | 0.188 (theta) | 0.499 (omega) | 4,482 | 52.8 |
| Pendulum | **Structured LNN** | **0.006** (theta) | **0.017** (omega) | 8,513 | 12.1 |

The LNN reduces trajectory error by ~100x over black-box baselines and preserves energy conservation (drift std 0.002 vs 0.553 for Neural ODE).

## Scope

**Systems**: Van der Pol oscillator, simple pendulum

**Models**:
- Baseline MLP: direct regression (time to state, or one-step state map)
- Neural ODE: learned vector field with Euler rollout training
- Structured Lagrangian NN: learns potential energy V(theta), derives dynamics via Euler-Lagrange equations

**Additional analyses**:
- MLP capacity study (depth vs width, spectral bias)
- Learned potential vs true potential comparison
- Energy conservation analysis
- Training loss curves

## Quick Start

```bash
pip install -r code/requirements.txt
python code/scripts/run_experiments.py --mode full
```

Or reproduce everything (experiments + report PDF) in one command:

```bash
bash scripts/reproduce_all.sh
```

## Math Development

### Van der Pol oscillator

State and dynamics:

h(t) = [z(t), v(t)]^T,  dz/dt = v,  dv/dt = mu*(1 - z^2)*v - z,  mu = 1

Reference trajectories use RK4 with fixed step dt. Learning objectives:

1. **Time-regression MLP**: z_hat(t) = g_phi(t), trained with MSE loss on noisy observations.
2. **Neural ODE rollout**: h_{n+1} = h_n + dt * F_theta(h_n), trained with trajectory MSE loss on the full rollout.

### Simple pendulum

Equation of motion: d^2(theta)/dt^2 + (g/l)*sin(theta) = 0

First-order form: d(theta)/dt = omega,  d(omega)/dt = -(g/l)*sin(theta)

### Structured Lagrangian NN

For unit mass and length:

L(theta, omega) = (1/2)*omega^2 - V(theta)

A neural network learns V_W(theta). From the Euler-Lagrange equation:

d^2(theta)/dt^2_pred = -dV_W/d(theta)

Trained on finite-difference acceleration targets from simulated data.

### Metrics

RMSE(a, b) = sqrt(mean((a - b)^2)),  MAE(a, b) = mean(|a - b|)

Additionally: parameter count per model, training wall-clock time, and energy drift standard deviation for pendulum models.

## Reproducibility

### Environment

- Python 3.10+
- CPU execution (no GPU required)
- Dependencies: `code/requirements.txt` (numpy, torch, matplotlib)

### Determinism

The experiment runner sets NumPy and PyTorch seeds. Default seed is `395`.

### Commands

| Command | Purpose |
|---|---|
| `bash scripts/reproduce_all.sh` | Full reproduction: experiments + report PDF |
| `python code/scripts/run_experiments.py --mode full` | Run all experiments, save figures and JSON |
| `python code/scripts/run_experiments.py --mode quick --no-plots` | Fast validation (reduced epochs, no figures) |
| `cd report && pdflatex report.tex && pdflatex report.tex` | Build report PDF from LaTeX |

### Output Artifacts

- `code/results_summary.json`: all metrics in machine-readable format
- `code/figures/*.png`: 10 publication-quality figures
- `report/report.pdf`: 2-page final report

## Repository Layout

```
Final Project/
├── README.md
├── PROJECT_PLAN.md
├── finalProject.md              # assignment instructions
├── .gitignore
├── docs/
│   ├── MATH_DEVELOPMENT.md      # full equation derivations
│   └── REPRODUCIBILITY.md       # standalone reproduction guide
├── scripts/
│   └── reproduce_all.sh         # one-command full reproduction
├── code/
│   ├── requirements.txt
│   ├── models.py                # nn.Module definitions
│   ├── solvers.py               # RK4, rollout, metrics
│   ├── plotting.py              # figure generation
│   ├── results_summary.json
│   ├── figures/                  # generated PNG plots
│   └── scripts/
│       └── run_experiments.py   # main experiment runner
└── report/
    ├── report.tex
    └── report.pdf
```

## Generated Figures

The pipeline produces these figures:

1. `vdp_capacity.png`: MLP capacity study (depth/width comparison)
2. `vdp_model_comparison.png`: Baseline MLP vs Neural ODE (Euler and RK4)
3. `vdp_vector_field.png`: Learned vector field with sampled trajectories
4. `vdp_loss_curves.png`: Training loss curves for all Van der Pol models
5. `pendulum_model_comparison.png`: Three-model comparison on test trajectory
6. `pendulum_energy.png`: Energy conservation comparison (LNN vs Neural ODE)
7. `pendulum_learned_potential.png`: Learned V(theta) vs true -g/l*cos(theta)
8. `pendulum_loss_curves.png`: Training loss curves for all pendulum models
9. `overall_rmse_bar.png`: Cross-system RMSE bar chart (log scale)

## Extended Docs

Full standalone documents:
- `docs/MATH_DEVELOPMENT.md`: complete equation derivations with code mapping
- `docs/REPRODUCIBILITY.md`: step-by-step reproduction guide
