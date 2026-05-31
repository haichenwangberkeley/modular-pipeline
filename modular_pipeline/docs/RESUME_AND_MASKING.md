# Resume And Masking

Masking and resuming are related but not identical.

## What Masking Does

The modular runner executes components in order.  A mask marks named components
or groups as skipped.  If a later component needs in-memory context from a
masked component, the later component is skipped with a manifest reason such as:

```text
missing required context: processed_samples
```

This is deliberate.  It prevents an agent from accidentally running a stage with
stale or guessed inputs.

## What The Current CLI Does Not Do

The current CLI does not hydrate these in-memory objects from previous output
directories as a `--start-at` feature:

- `registry`
- `processed_samples`
- `fit_context`
- `plot_manifest`
- `cutflow_table`

Because of that, a CLI run cannot truly start at `fit` from only a previous
`outputs_*` directory.  It can inspect whether the artifacts are present, and it
can skip earlier components, but downstream components that require missing
context will be skipped or, with `--strict-mask`, will fail.

## What The State Ledger Does

Every run writes:

```text
<outputs>/modular_pipeline_state.json
```

You can also generate or refresh it for an old output directory:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs <outputs> \
  --write-state
```

The state ledger records:

- `components`: artifact status for each component;
- `contexts`: where each provided context key comes from and whether it has a documented hydration source;
- `entrypoints`: whether each component is ready from artifacts, blocked by missing artifacts, or blocked because it requires live context;
- `ready_entrypoints_from_artifacts`: short list of components whose disk inputs are present.

This makes midpoint planning auditable even before true resume execution is
implemented.

## How To Start From The Beginning

Use no mask:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_full
```

## How To Start From A Logical Middle Today

Use masks to stop unnecessary downstream work while still producing needed
upstream context in the same process.

For example, run through fit/significance but skip plots and reports:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli run \
  --summary analysis/analysis.summary.json \
  --inputs input \
  --outputs outputs_modular_stats_only \
  --mask plots,review_artifacts,report
```

This still runs samples/templates/cutflow/fit because those are needed to build
`fit_context`.

## How To Add True Artifact Resume Later

Add explicit hydration components instead of silent file reads inside physics
stages.  A safe design would include:

1. `load_registry`: reads `samples.registry.json`.
2. `load_processed_samples`: reconstructs the `processed_samples` structures from cache `.npz` files and registry metadata.
3. `load_fit_context`: either reconstructs a live RooFit context from saved artifacts or refuses if that cannot be done exactly.
4. `load_cutflow`: reads cutflow/yield artifacts into the expected table shape.
5. `load_plot_manifest`: reads `report/plots/manifest.json`.

Each hydration component should declare its `provides` keys just like normal
components, and the manifest should record that the context came from disk.

## Rule For Agents

Never claim exact middle-of-pipeline reproduction from a masked run unless the
manifest shows that the required context was produced or safely hydrated in that
same run.

Also check `modular_pipeline_state.json`; a safe future resume should require
the requested component to be listed as `ready_from_artifacts` and should record
which loader hydrated each context key.
