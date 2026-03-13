# Final Project: Scientific ML for Nonlinear Dynamics

ELE_ENG 395 final project comparing data-driven and physics-informed models on nonlinear ODE systems.

## Current Scope

- Systems: Van der Pol oscillator and simple pendulum
- Models:
  - Baseline MLP
  - Neural ODE (shared residual vector field)
  - Structured Lagrangian NN (pendulum)
- Outputs:
  - Figures in `code/figures/`
  - Metrics JSON in `code/results_summary.json`
  - 2-page report in `report/report.pdf`

## Reproducibility

1. Install dependencies:
   `pip install -r code/requirements.txt`
2. Run full experiment suite:
   `python code/scripts/run_experiments.py --mode full`
3. Fast sanity run:
   `python code/scripts/run_experiments.py --mode quick --no-plots`
4. Build report:
   `cd report && pdflatex report.tex`

## Repository Layout

```
Final Project/
├── README.md
├── PROJECT_PLAN.md
├── finalProject.md
├── code/
│   ├── requirements.txt
│   ├── results_summary.json
│   ├── figures/
│   └── scripts/
│       └── run_experiments.py
└── report/
    ├── report.tex
    └── report.pdf
```
