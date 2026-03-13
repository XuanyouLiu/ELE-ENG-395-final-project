# Reproducibility Guide

## Environment

- Python 3.10+
- CPU execution (no GPU required)
- Dependencies in `code/requirements.txt`

Install:

```bash
pip install -r code/requirements.txt
```

## Determinism

The experiment runner sets NumPy and PyTorch seeds. Default seed is `395`.

## One-command reproduction

From project root:

```bash
bash scripts/reproduce_all.sh
```

This does:
1. Run full experiments (`--mode full`)
2. Regenerate `code/results_summary.json`
3. Regenerate figures in `code/figures/`
4. Compile LaTeX report into `report/report.pdf`

## Manual commands

```bash
python code/scripts/run_experiments.py --mode full
cd report && pdflatex report.tex && pdflatex report.tex
```

## Fast validation run

```bash
python code/scripts/run_experiments.py --mode quick --no-plots
```

## Output artifacts

- `code/results_summary.json` (all metrics, machine-readable)
- `code/figures/*.png` (plots used in report/slides)
- `report/report.pdf` (final 2-page report)

## Expected structure

- `README.md` project overview
- `docs/MATH_DEVELOPMENT.md` equations and model derivations
- `docs/REPRODUCIBILITY.md` exact steps to rerun
- `code/scripts/run_experiments.py` primary executable
