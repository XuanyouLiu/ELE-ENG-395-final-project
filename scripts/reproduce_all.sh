#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/2] Running full experiments"
python code/scripts/run_experiments.py --mode full

echo "[2/2] Building report"
cd report
pdflatex -interaction=nonstopmode report.tex >/dev/null
pdflatex -interaction=nonstopmode report.tex >/dev/null

echo "Done. Outputs: code/results_summary.json and report/report.pdf"
