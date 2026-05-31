# Modular Pipeline

Maskable orchestration and artifact-readiness tracking for the H to gamma gamma
analysis pipeline.

This repository contains the `modular_pipeline` package extracted from the
analysis workspace.  It is intentionally a thin orchestration layer: it calls
the canonical stage functions from the host analysis package instead of
reimplementing physics logic.

## Host Requirement

Run this package from an analysis checkout that provides the `analysis` Python
package and the project ROOT environment.  In the original workspace this is:

```bash
cd /global/homes/h/haichen/disk/opendataanalysis/fix-stat-interpretation/pipeline-for-testing
export PYTHONPATH=$PWD
```

Then the package can be run in-place or installed editable.

## Commands

List components:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli list-components --verbose
```

Run the full modular pipeline:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_full
```

Inspect an existing output directory:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs outputs_modular_full \
  --write-state
```

Mask components or groups:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_no_plots \
  --mask plots,report
```

## Artifacts

Every modular run writes:

```text
<outputs>/modular_pipeline_manifest.json
<outputs>/modular_pipeline_state.json
```

The manifest records what ran or was masked.  The state file records component
artifact completeness, context hydration status, and which entry points are
ready from existing artifacts.

## Agent Docs

Start with:

- `modular_pipeline/AGENTS.md`
- `modular_pipeline/docs/OPERATIONS.md`
- `modular_pipeline/docs/RESUME_AND_MASKING.md`
- `modular_pipeline/docs/REPRODUCTION.md`
- `modular_pipeline/docs/FIT_SIGNIFICANCE_NOTES.md`
- `modular_pipeline/docs/TROUBLESHOOTING.md`

## Tests

The focused tests can be run from a host analysis checkout where `analysis` is
importable:

```bash
PYTHONPATH=$PWD pytest -q tests/test_modular_pipeline.py
```

