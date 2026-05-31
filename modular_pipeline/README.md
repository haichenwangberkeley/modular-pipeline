# Modular Maskable Pipeline

This directory contains a thin modular orchestrator around the existing
`analysis.pipeline` implementation.  The default unmasked run calls the same
stage functions in the same order as the canonical CLI, so it is intended to
reproduce the same artifacts and numerical results while making stage masking
explicit.

Agent-facing operating notes live in:

- `AGENTS.md`
- `docs/OPERATIONS.md`
- `docs/RESUME_AND_MASKING.md`
- `docs/REPRODUCTION.md`
- `docs/FIT_SIGNIFICANCE_NOTES.md`
- `docs/TROUBLESHOOTING.md`

List available components:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli list-components --verbose
```

Run the full pipeline:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_full
```

Mask one or more components:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_no_plots \
  --mask plots,report
```

Masks accept component names or groups.  Useful groups include `samples`,
`modeling`, `stats`, `plotting`, `validation`, and `reporting`.

Every run writes:

```text
<outputs>/modular_pipeline_manifest.json
<outputs>/modular_pipeline_state.json
```

The manifest records which components ran, which were explicitly masked, and
which were skipped because an earlier mask removed required in-memory context.
The state file records which artifacts are present and which component entry
points are ready from existing artifacts.

Inspect an existing output directory:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs outputs_modular_full_20260531T163358Z \
  --write-state
```
