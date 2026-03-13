# Scientific ML for Nonlinear Dynamics

ELE_ENG 395 final project comparing data-driven and physics-informed models for learning nonlinear ODE dynamics from noisy trajectories.

## Key Results

| System | Model | State 1 RMSE | State 2 RMSE | Params | Time (s) |
|---|---|---|---|---|---|
| Van der Pol | Baseline MLP (2x32) | 0.053 ($z$) | 0.239 ($v$) | 1,153 | 0.4 |
| Van der Pol | Neural ODE | 0.188 ($z$) | 0.383 ($v$) | 4,482 | 23.7 |
| Pendulum | Baseline MLP | 0.496 ($\theta$) | 1.511 ($\omega$) | 4,482 | 2.7 |
| Pendulum | Neural ODE | 0.188 ($\theta$) | 0.499 ($\omega$) | 4,482 | 52.8 |
| Pendulum | **Structured LNN** | **0.006** ($\theta$) | **0.017** ($\omega$) | 8,513 | 12.1 |

The LNN reduces trajectory error by ~100x over black-box baselines and preserves energy conservation (drift std 0.002 vs 0.553 for Neural ODE).

## Scope

**Systems**: Van der Pol oscillator, simple pendulum

**Models**:
- Baseline MLP: direct regression (time to state, or one-step state map)
- Neural ODE: learned vector field with Euler rollout training
- Structured Lagrangian NN: learns potential energy $V(\theta)$, derives dynamics via Euler-Lagrange equations

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

State vector:

$$\mathbf{h}(t) = \begin{bmatrix} z(t) \\ v(t) \end{bmatrix}$$

Dynamics with $\mu = 1$:

$$\dot{z} = v, \qquad \dot{v} = \mu(1 - z^2)v - z$$

Reference trajectories are generated with RK4:

$$\mathbf{h}_{n+1} = \mathbf{h}_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Learning objectives:

1. **Time-regression MLP**:

$$\hat{z}(t) = g_\phi(t), \qquad \mathcal{L}_{\text{MLP}} = \frac{1}{N}\sum_i \|g_\phi(t_i) - z_i^{\text{noisy}}\|_2^2$$

2. **Neural ODE / ResNet rollout**:

$$\hat{\mathbf{h}}_{n+1} = \hat{\mathbf{h}}_n + \Delta t \, F_\theta(\hat{\mathbf{h}}_n), \qquad \mathcal{L}_{\text{NODE}} = \frac{1}{N}\sum_{n} \|\hat{\mathbf{h}}_n - \mathbf{h}_n^{\text{noisy}}\|_2^2$$

### Simple pendulum

Equation of motion:

$$\ddot{\theta} + \frac{g}{\ell}\sin\theta = 0$$

First-order form:

$$\dot{\theta} = \omega, \qquad \dot{\omega} = -\frac{g}{\ell}\sin\theta$$

### Structured Lagrangian NN

For unit mass and length ($m = \ell = 1$):

$$L(\theta, \omega) = \frac{1}{2}\omega^2 - V(\theta)$$

A neural network learns $V_W(\theta)$. From the Euler-Lagrange equation:

$$\ddot{\theta}_{\text{pred}} = -\frac{\mathrm{d}V_W}{\mathrm{d}\theta}$$

Training loss on finite-difference acceleration targets:

$$\mathcal{L}_{\text{LNN}} = \frac{1}{N}\sum_i \left(\ddot{\theta}_{\text{pred}}(\theta_i) - \ddot{\theta}_i^{\text{FD}}\right)^2$$

### Metrics

$$\mathrm{RMSE}(a, b) = \sqrt{\frac{1}{N}\sum_i (a_i - b_i)^2}, \qquad \mathrm{MAE}(a, b) = \frac{1}{N}\sum_i |a_i - b_i|$$

Reported metrics per model: trajectory RMSE/MAE, parameter count, training wall-clock time, and energy drift standard deviation (pendulum).

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
7. `pendulum_learned_potential.png`: Learned $V(\theta)$ vs true $-\frac{g}{\ell}\cos\theta$
8. `pendulum_loss_curves.png`: Training loss curves for all pendulum models
9. `overall_rmse_bar.png`: Cross-system RMSE bar chart (log scale)

## Extended Docs

Full standalone documents:
- `docs/MATH_DEVELOPMENT.md`: complete equation derivations with code mapping
- `docs/REPRODUCIBILITY.md`: step-by-step reproduction guide
