# Modular Pipeline

Self-contained modular H to gamma gamma analysis pipeline.

This repository contains both:

- `analysis/`: the canonical analysis implementation, including sample reading,
  selections, histogramming, RooFit modeling, significance, plotting, and reports
- `modular_pipeline/`: a maskable orchestrator and artifact-readiness tracker
  around the canonical analysis stages

The dataset is intentionally external. A separate device can run the pipeline as
long as it has the same dataset directory layout and a Python environment with
PyROOT/RooFit for the statistical stages.

## Dataset Layout

Point `--inputs` at a directory containing:

```text
input-data/
  data/
    *.root
  MC/
    *.root
```

The path does not need to be named `input`. Use the absolute path on the host
device, for example `--inputs /data/atlas-open-data-hgg`.

## Prerequisites

Installable Python dependencies are declared in `pyproject.toml`.

Full pipeline execution also requires CERN ROOT with PyROOT and RooFit:

```bash
python -c "import ROOT; print(ROOT.gROOT.GetVersion())"
```

The runtime prefers a repo-local `.rootenv/bin/python` if present. If not, it
uses the active Python interpreter, so a conda/micromamba environment with ROOT
also works.

The HHXYY/quickFit cross-check path is optional. If available, set:

```bash
export HHXYY_REFERENCE_ROOT=/path/to/hhxyy
export HHXYY_FITTING_ROOT=/path/to/hhxyy-codex/fitting
export HHXYY_QUICKFIT_SETUP=/path/to/quickfit/setup.sh
```

Without those, the pipeline uses the local PyROOT fallback for the significance
stage.

## Install

```bash
git clone git@github.com:haichenwangberkeley/modular-pipeline.git
cd modular-pipeline
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

For full RooFit execution, use the environment where `import ROOT` succeeds
instead of a plain virtualenv if needed.

## Quick Checks

```bash
modular-pipeline list-components --verbose
hgg-preflight \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_preflight
pytest -q
```

## Run

Full modular pipeline:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_modular_full
```

Mask components or groups:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_modular_no_plots \
  --mask plots,report
```

Inspect an existing output directory:

```bash
modular-pipeline inspect \
  --outputs outputs_modular_full \
  --write-state
```

## Artifacts

Every modular run writes:

```text
<outputs>/modular_pipeline_manifest.json
<outputs>/modular_pipeline_state.json
```

The manifest records what ran or was masked. The state file records component
artifact completeness, context hydration status, and which entry points are
ready from existing artifacts.

## Agent Docs

Start with:

- `modular_pipeline/AGENTS.md`
- `modular_pipeline/docs/OPERATIONS.md`
- `modular_pipeline/docs/RESUME_AND_MASKING.md`
- `modular_pipeline/docs/REPRODUCTION.md`
- `modular_pipeline/docs/PORTABILITY.md`
- `modular_pipeline/docs/FIT_SIGNIFICANCE_NOTES.md`
- `modular_pipeline/docs/OFFICIAL_BACKGROUND_SELECTION.md`
- `modular_pipeline/docs/TROUBLESHOOTING.md`

## Tests

```bash
pytest -q
```
