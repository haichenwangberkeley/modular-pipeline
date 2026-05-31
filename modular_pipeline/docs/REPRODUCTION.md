# Reproduction

The modular pipeline is designed to reproduce the canonical pipeline by calling
the same stage functions in the same order.

For iterative work, first inspect the candidate output directory:

```bash
PYTHONPATH=$PWD .rootenv/bin/python -m modular_pipeline.cli inspect \
  --outputs <outputs> \
  --write-state
```

The resulting `modular_pipeline_state.json` tells an agent what has already
been produced and which entry points are ready from artifacts.

## Validated Reference

Reference output:

```text
outputs_full_localmatch5_20260531T062318Z
```

Validated modular output:

```text
outputs_modular_full_20260531T163358Z
```

The checked outputs matched exactly for:

- `fit/FIT1/results.json`
  - `mu_hat`
  - `mu_uncertainty`
  - `fit_status`
  - `cov_qual`
  - fitted category yields
- `fit/FIT1/significance_asimov.json`
  - `mu_hat`
  - `mu_uncertainty`
  - `q0`
  - `z_discovery`
  - `fit_driver`
  - `background_parameter_policy`

## Reproduce The Check

```bash
PYTHONPATH=$PWD .rootenv/bin/python - <<'PY'
import json
from pathlib import Path

candidate = Path("outputs_modular_full_20260531T163358Z")
reference = Path("outputs_full_localmatch5_20260531T062318Z")

checks = {
    "fit/FIT1/results.json": [
        "mu_hat",
        "mu_uncertainty",
        "fit_status",
        "cov_qual",
        "fitted_category_yields",
        "background_parameter_policy",
    ],
    "fit/FIT1/significance_asimov.json": [
        "mu_hat",
        "mu_uncertainty",
        "q0",
        "z_discovery",
        "fit_driver",
        "background_parameter_policy",
    ],
}

for rel, keys in checks.items():
    c = json.loads((candidate / rel).read_text())
    r = json.loads((reference / rel).read_text())
    print(rel)
    for key in keys:
        print(f"  {key}: {c.get(key)!r} == {r.get(key)!r}")
        assert c.get(key) == r.get(key), (rel, key, c.get(key), r.get(key))
print("matched")
PY
```

## Expected Key Values

Local RooFit measurement fit:

```text
mu_hat = 0.9844271627301904
mu_uncertainty = 0.14066948508662064
fit_status = 0
cov_qual = 3
background_parameter_policy = floating_shape_and_normalization
```

HHXYY quickFit expected significance:

```text
mu_hat = 0.9972894866687582
mu_uncertainty = 0.14017231625967103
q0 = 52.58783904789016
z_discovery = 7.251747309986067
fit_driver = hhxyy_fitting_quickfit
background_parameter_policy = floating_shape_and_normalization
```

## Why Exact Reproduction Can Break

Exact reproduction can break if an agent changes:

- stage order;
- runtime defaults in `analysis/analysis.summary.json`;
- sample selection or metadata resolution;
- background model mapping;
- RooFit fit options;
- Asimov construction semantics;
- HHXYY XML mapping;
- blinding or observed-data flags.

When debugging statistical changes, compare both local and HHXYY outputs.  A
local-only match is not enough for the central expected-significance claim.
