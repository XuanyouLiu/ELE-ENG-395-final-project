# Final Project: Scientific ML for Nonlinear Dynamics

ELE_ENG 395 final project comparing data-driven and physics-informed models on nonlinear ODE systems.

## This repository includes

- **All code**: end-to-end experiment pipeline in `code/scripts/run_experiments.py`
- **Math development**: derivations and equations in `docs/MATH_DEVELOPMENT.md`
- **Explanations**: project plan and report narrative in `PROJECT_PLAN.md` and `report/report.tex`
- **Clear structure**: separate `code/`, `docs/`, `report/`, and `scripts/`
- **Reproducibility**: exact rerun instructions in `docs/REPRODUCIBILITY.md` and one-command script

## Quick start

1. Install dependencies:
   `pip install -r code/requirements.txt`
2. Reproduce everything (experiments + report PDF):
   `bash scripts/reproduce_all.sh`
3. Fast smoke test:
   `python code/scripts/run_experiments.py --mode quick --no-plots`

## Core outputs

- `code/results_summary.json` (metrics)
- `code/figures/*.png` (figures)
- `report/report.pdf` (2-page final report)

## Repository layout

```
Final Project/
├── README.md
├── PROJECT_PLAN.md
├── finalProject.md
├── docs/
│   ├── MATH_DEVELOPMENT.md
│   └── REPRODUCIBILITY.md
├── scripts/
│   └── reproduce_all.sh
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
