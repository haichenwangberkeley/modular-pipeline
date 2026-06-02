# Config-Driven Category Routing

The analysis now supports ordered first-match category assignment from validated routing YAML files. A category strategy can mix cut-based predicates and learned-score thresholds without editing Python, as long as the required observables already exist.

## What Is Configurable

Routing configs declare:

- stable category `id` values;
- human-readable `label` text;
- integer `priority` order;
- `required_inputs`;
- optional `eligible_when` predicates;
- required `select_when` predicates;
- optional `block_if_missing` fields for learned scores or derived inputs;
- diagnostic `reason` and `block_reason` text.

Production configs live in `configs/routing/`:

- `configs/routing/five_category_ptt.yaml`
- `configs/routing/section8_ads_bdt.yaml`

The non-production demo is `configs/routing/examples/mixed_strategy_demo.yaml`.

## Predicate Schema

Predicates are structured mappings, not Python expressions. Supported forms are:

```yaml
always: true

all:
  - {field: N_jets_30, op: ">=", value: 3}
  - {field: N_btag_25, op: ">=", value: 1}

any:
  - {field: N_lep, op: ">", value: 0}
  - {field: MET, op: ">", value: 80.0}

not:
  field: Z_ll_veto
  op: "=="
  value: 0

finite:
  field: BDT_ttH
```

Supported comparisons are `==`, `!=`, `>`, `>=`, `<`, and `<=`.

## Learned Scores And Blocking

If a category is topologically eligible but a declared `block_if_missing` score is absent or non-finite, the router assigns `blocked_missing_input` and prevents later fallback categories from claiming that event. This preserves the Section 8 BDT behavior where missing scores block eligible BDT-dependent categories.

## Creating A New Strategy

Create a YAML file with `routing.mode: ordered_first_match` and a `categories` list. Then override the selected analysis version's routing config:

```python
from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.config.versions import apply_analysis_version

cfg = apply_analysis_version(
    DEFAULT_RUNTIME,
    version_name="round1_5cat",
    routing_config="configs/routing/examples/mixed_strategy_demo.yaml",
)
```

For materialized NPZ observable tables, rerun only routing:

```bash
python -m analysis.routing.route_npz \
  --events /path/to/materialized_observables.npz \
  --routing-config configs/routing/section8_ads_bdt.yaml \
  --outputs /path/to/routed_outputs
```

The command writes `routed_categories.npz` with `assigned_category`, `assignment_reason`, `assignment_blocked`, and `category_label`.

## Out Of Scope

This router does not compute observables, train models, score models, build templates, or refactor fits. Object definitions, BDT features, BDT training, normalization, blinding, and statistical behavior remain hardcoded in their existing services.

The legacy Python routers remain available as reference implementations for parity tests and compatibility debugging while the config-backed engine becomes the normal runtime path.
