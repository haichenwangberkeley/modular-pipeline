# Portability

This repository is intended to be runnable on a separate device without the
original analysis workspace.

## What Is Self-Contained

- `analysis/`: canonical physics workflow implementation
- `modular_pipeline/`: maskable orchestration and artifact state tracking
- `analysis/analysis.summary.json`: default analysis configuration
- `analysis/regions.yaml`: generated region configuration used by the workflow
- `scripts/`: HHXYY comparison and bootstrap helpers
- `pyproject.toml`: installable Python dependencies and console entry points
- `tests/`: portable regression tests for the modular orchestration layer

## What Must Be Supplied Externally

- ATLAS open-data ROOT files, passed with `--inputs /path/to/input-data`
- A Python environment with PyROOT/RooFit for full statistical stages
- Optional HHXYY/quickFit tools for the external cross-check backend

## Fresh-Machine Checklist

```bash
git clone git@github.com:haichenwangberkeley/modular-pipeline.git
cd modular-pipeline
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pytest -q
```

Verify the dataset and runtime:

```bash
hgg-preflight \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_preflight
```

Run:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_modular_full
```
