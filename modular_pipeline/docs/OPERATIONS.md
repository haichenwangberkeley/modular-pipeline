# Operations

This package is a maskable orchestration layer around the canonical analysis
pipeline.  It is not a replacement for the physics implementation in
`analysis/`.

## Setup

Run from the repository root:

```bash
cd modular-pipeline
pip install -e '.[dev]'
```

Use a Python environment where `import ROOT` succeeds for full RooFit stages:

```bash
modular-pipeline list-components --verbose
```

## Full Run

```bash
out="outputs_modular_full_$(date -u +%Y%m%dT%H%M%SZ)"
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs "$out"
```

The full run writes the same classes of outputs as:

```bash
hgg-analysis run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs <out>
```

The difference is that the modular run also writes:

```text
<out>/modular_pipeline_manifest.json
<out>/modular_pipeline_state.json
```

`modular_pipeline_state.json` is updated during the run, so if a long run is
interrupted it still records the completed artifact contracts and the entry
points that are plausibly available from disk.

## Component Listing

```bash
modular-pipeline list-components --verbose
```

This prints each component, its groups, required context keys, provided context
keys, and description.

## Inspect Existing Outputs

Before starting from the middle of a workflow, inspect the old output directory:

```bash
modular-pipeline inspect \
  --outputs outputs_modular_full_20260531T163358Z \
  --write-state
```

For the full JSON view:

```bash
modular-pipeline inspect \
  --outputs outputs_modular_full_20260531T163358Z \
  --json
```

Read `ready_entrypoints_from_artifacts` and the per-entrypoint `requirements`
before choosing a restart point.

## Masks

Mask by component:

```bash
--mask plots,report
```

Mask by group:

```bash
--mask plotting,reporting
```

Use `--strict-mask` when you want the runner to fail rather than skip a
component whose dependencies were removed by a mask:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_strict_no_samples \
  --mask samples \
  --strict-mask
```

## Common Masks

Run through fit/significance but skip plots and report packaging:

```bash
--mask plots,review_artifacts,report
```

Run only early configuration and preflight artifacts:

```bash
--mask samples,modeling,selections,stats,plotting,reporting,validation
```

Skip all statistical stages:

```bash
--mask stats
```

Skip downstream visual/reporting stages:

```bash
--mask plotting,reporting
```
