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
cd modular-pipeline
pip install -e '.[dev]'
```

Use a Python environment where `import ROOT` succeeds for full pipeline runs:

```bash
modular-pipeline list-components --verbose
```

For tests, use the installed editable package unless a ROOT-only probe is
needed:

```bash
pytest -q
```

## Key Files

- `components.py`: component registry, dependency checks, masks, and manifest writing.
- `cli.py`: command line interface.
- `README.md`: short user-facing command examples.
- `docs/`: operational notes for agents, including `PORTABILITY.md` for fresh-machine setup.
- `docs/OFFICIAL_BACKGROUND_SELECTION.md`: optional HHXYY `bkgParamTool` background-selection cross-check.
- `<outputs>/modular_pipeline_manifest.json`: record of what ran, what was masked, and what was dependency-skipped.
- `<outputs>/modular_pipeline_state.json`: incremental artifact ledger and entrypoint-readiness map.
- `runs/ANALYSIS_TREE.yaml`: canonical analysis-version lineage tree across optimization rounds.
- `runs/ANALYSIS_TREE.md`: human-readable rendering of the analysis-version tree.
- `runs/<round_id>/ROUND_STATE.yaml`: canonical machine-readable optimization-round handoff artifact.
- `runs/<round_id>/ROUND_STATE.md`: human-readable rendering of the same round state.

## Optimization Handoff

For any follow-up optimization round, read `runs/ANALYSIS_TREE.yaml` first if
it exists. Use it to choose the parent likelihood version deliberately, then
read that node's `ROUND_STATE.yaml` before choosing the next scan or strategy
branch.

`ANALYSIS_TREE` is the canonical lineage map across optimization rounds. Each
node is a full HEP analysis likelihood version, and each edge records how a
child version descends from its parent. Do not infer lineage from directory
names or timestamps when `ANALYSIS_TREE` exists.

`ROUND_STATE` is the canonical handoff artifact between rounds. It is intended
to be consumed by both agents and human scientists, and it must drive the next
decision more than ad hoc recollection or isolated metric files.

Treat `ROUND_STATE.yaml` as the full analysis-state artifact for the selected
reference point, not as a short memo. A valid `ROUND_STATE` must contain:

- the analysis definition that was actually run;
- the full primary result set for the current reference point;
- the artifact catalog needed to locate detailed outputs;
- the interpretation and trajectory needed to choose the next step.

For HEP analysis, the critical object is the likelihood definition. Future
agents must assume that the analysis is effectively defined by the region
selection and the observable fitted in each region. `ROUND_STATE.md` must
therefore present this in human-readable form.

At minimum, the next-round planner should read:

- `runs/ANALYSIS_TREE.yaml`
- `runs/<round_id>/ROUND_STATE.yaml`
- the paths listed under `handoff_read_first`

Use `ROUND_STATE` to answer:

- what the current best known state is;
- what full analysis definition produced that state;
- what the likelihood definition is in plain language: fit regions, selections,
  and fitted observables;
- what the full primary result set is for that state;
- how the current result should be interpreted, not just what the raw metric is;
- whether the metric landscape looks like a sharp optimum or a broad plateau;
- how the current round fits into the trajectory of prior rounds;
- what next actions are currently recommended.

After completing a new optimization round, regenerate that round's
`ROUND_STATE.yaml` and update `runs/ANALYSIS_TREE.yaml` / `runs/ANALYSIS_TREE.md`.
The handoff is incomplete until both artifacts reflect the new likelihood
version.

Do not start a new optimization round from only `scan_results.*`,
`significance_asimov.json`, or a PDF note if a `ROUND_STATE` artifact exists.
Those files are supporting evidence, but `ROUND_STATE` is the primary round
summary and decision handoff.

If `ROUND_STATE` is missing `analysis_definition`, `analysis_results`,
`artifact_catalog`, or a human-readable likelihood-definition section in
`ROUND_STATE.md`, treat the round handoff as incomplete and fix/regenerate
`ROUND_STATE` before planning or executing the next optimization step. Do not
push that responsibility back to the user.

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
modular-pipeline inspect \
  --outputs outputs_modular_full_20260531T163358Z \
  --write-state
```

## Starting Points

Start from the beginning with no mask:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
  --outputs outputs_modular_full
```

Start with a reduced workflow by masking components or groups:

```bash
modular-pipeline run \
  --summary analysis/analysis.summary.json \
  --inputs /path/to/input-data \
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

The official HHXYY background-function selector is available as a separate
cross-check/substitution tool:

```bash
hgg-official-bkg-select parse \
  --bkg-model-dir outputs_modular_full/fit/FIT1/hhxyy_workspace/bkg_model \
  --out outputs_modular_full/fit/FIT1/official_bkg_selection.json
```

This must remain opt-in; do not overwrite the local selector artifacts merely
because official results are present.

## Verification Before Handoff

Run:

```bash
pytest -q
```

For an unmasked reproduction run, compare:

```bash
python - <<'PY'
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
