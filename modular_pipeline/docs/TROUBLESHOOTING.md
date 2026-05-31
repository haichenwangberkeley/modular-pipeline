# Troubleshooting

## PyROOT Warnings

This warning is common in the current environment:

```text
This distribution of ROOT is in alpha stage
```

It is noisy but not by itself a pipeline failure.

This warning is also common:

```text
CPyCppyy API not found
```

It has appeared in successful runs.  Treat it as environment noise unless it is
paired with an import failure or a crash.

## Fit Runs To Bounds

If local RooFit reports status 0/covQual 3 but returns:

```text
mu_hat = 5
nbkg_* at upper bounds
```

check:

- fixed Bernstein tail objects are retained;
- background shape parameters are floating except fixed tails;
- `Range("full")` is not passed to the simultaneous extended fit;
- the dataset is `RooDataHist`, not a weighted bin-center `RooDataSet`;
- background normalizations are not constant.

## HHXYY Missing Or Failing

Check:

```bash
PYTHONPATH=$PWD .rootenv/bin/python - <<'PY'
from analysis.stats.hhxyy_fitting_backend import is_atlas_env_available
print(is_atlas_env_available())
PY
```

If HHXYY is unavailable, the local fallback may run, but do not label it as the
central HHXYY quickFit expected-significance result.

## Masked Components

Inspect:

```bash
cat <outputs>/modular_pipeline_manifest.json
cat <outputs>/modular_pipeline_state.json
```

If a component is marked:

```text
status = masked
reason = missing required context: ...
```

then an upstream dependency was masked or not produced.  Either remove the mask
or add an explicit, reviewed hydration component.

## Which Entry Point Can I Use?

Run:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs <outputs>
```

If the desired component is `blocked_missing_artifacts`, rerun or hydrate the
upstream artifacts first.  If it is `blocked_requires_live_context`, the current
artifacts are not enough to recreate that stage exactly; rerun the provider of
the live context.

## Slow Full Runs

Full runs spend most of their time in sample/cache processing and external ROOT
subprocesses.  It is normal to see repeated PyROOT warnings before `fit/FIT1`
appears.

Progress checkpoints:

```bash
find <outputs> -maxdepth 2 -type d | sort
find <outputs>/fit/FIT1 -maxdepth 2 -type f | sort
```

## Minimal Health Checks

```bash
PYTHONPATH=$PWD pytest -q

PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli list-components --verbose
```
