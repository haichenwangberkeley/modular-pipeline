# Agent Guide: Modular Pipeline

This directory is a thin orchestration layer over the canonical analysis code in
`analysis/`.  It exists to make components maskable while preserving exact
results for an unmasked run.

## Prime Directive

Do not reimplement physics logic here.  The modular pipeline must call the same
stage functions as `analysis.pipeline.run_all_stages` unless there is an
explicit, reviewed reason to change behavior.  Exact reproduction depends on
sharing the canonical implementation.

## Environment

Run commands from the repository root:

```bash
cd /global/homes/h/haichen/disk/opendataanalysis/fix-stat-interpretation/pipeline-for-testing
export PYTHONPATH=$PWD
```

Use the project ROOT environment:

```bash
.rootenv/bin/python -m modular_pipeline.cli list-components --verbose
```

For tests, use the shell Python with `PYTHONPATH=$PWD` unless a ROOT-only probe
is needed:

```bash
PYTHONPATH=$PWD pytest -q
```

## Key Files

- `components.py`: component registry, dependency checks, masks, and manifest writing.
- `cli.py`: command line interface.
- `README.md`: short user-facing command examples.
- `docs/`: operational notes for agents.
- `<outputs>/modular_pipeline_manifest.json`: record of what ran, what was masked, and what was dependency-skipped.
- `<outputs>/modular_pipeline_state.json`: incremental artifact ledger and entrypoint-readiness map.

## Component Semantics

Components are ordered to mirror `analysis.pipeline.run_all_stages`:

1. `summary`
2. `runtime_contract`
3. `preflight`
4. `metadata`
5. `registry`
6. `strategy`
7. `partition`
8. `samples`
9. `templates`
10. `cutflow`
11. `fit`
12. `systematics`
13. `significance`
14. `plots`
15. `review_artifacts`
16. `report`

Each component declares:

- `requires`: in-memory context keys needed to run.
- `provides`: in-memory context keys created for later stages.
- `groups`: aliases that can be masked together.

If an upstream component is masked, downstream components with missing context
are skipped unless `--strict-mask` is used, in which case the run fails at the
first missing dependency.

## Artifact Ledger

Every run writes and updates:

```text
<outputs>/modular_pipeline_state.json
```

This state artifact answers:

- which component artifacts are complete, partial, missing, or masked;
- which context keys each component requires and provides;
- which existing artifacts can hydrate those context keys;
- which entry points are ready from artifacts;
- which entry points are blocked because a live context such as `fit_context` is not recoverable.

Inspect an old output directory before deciding where to enter:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs outputs_modular_full_20260531T163358Z \
  --write-state
```

## Starting Points

Start from the beginning with no mask:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_full
```

Start with a reduced workflow by masking components or groups:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_no_plots \
  --mask plots,report
```

Do not claim a masked run is a resumed middle-of-pipeline physics rerun unless
the required in-memory context was actually produced in that same process or
the state artifact reports the entry point as `ready_from_artifacts` and a
reviewed loader is used.  The current CLI reports artifact readiness but does
not yet implement `--start-at` hydration.  See `docs/RESUME_AND_MASKING.md`.

## Reproduction Reference

The unmasked modular run was checked against:

```text
outputs_full_localmatch5_20260531T062318Z
```

Validated modular output:

```text
outputs_modular_full_20260531T163358Z
```

The checked numerical outputs matched exactly for:

- local fit `mu_hat`
- local fit `mu_uncertainty`
- local fitted category yields
- HHXYY expected significance `mu_hat`
- HHXYY expected significance `z_discovery`

## Fit/Significance Lessons From Debugging

The local statistical setup only matched after these conditions were enforced:

- background normalizations `nbkg_*` float in Asimov fits;
- background shape parameters float in Asimov fits;
- signal-shape parameters are fixed from signal MC fits;
- combined Asimov data uses `RooDataHist`;
- local final signal uses HHXYY-style symmetric `sigmaCB`;
- local final Bernstein models use HHXYY-style fixed trailing coefficient;
- fixed Bernstein tail variables must be retained in Python object lists so RooFit does not see dangling objects;
- do not pass explicit `Range("full")` to the combined simultaneous extended fit;
- for the validated reproduction, local Bernstein coefficients are constrained positive for RooFit validity while HHXYY remains the central expected-significance backend.

## Verification Before Handoff

Run:

```bash
PYTHONPATH=$PWD pytest -q
```

For an unmasked reproduction run, compare:

```bash
PYTHONPATH=$PWD .rootenv/bin/python - <<'PY'
import json
from pathlib import Path

candidate = Path("outputs_modular_full_20260531T163358Z")
reference = Path("outputs_full_localmatch5_20260531T062318Z")
for rel in [
    "fit/FIT1/results.json",
    "fit/FIT1/significance_asimov.json",
]:
    c = json.loads((candidate / rel).read_text())
    r = json.loads((reference / rel).read_text())
    print(rel)
    for key in ["mu_hat", "mu_uncertainty", "q0", "z_discovery"]:
        if key in c or key in r:
            print(key, c.get(key), r.get(key))
PY
```
