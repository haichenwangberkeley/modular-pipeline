# Observation Schema

Date: 2026-05-31

Observation files are written as `runs/<run_id>/observations.yaml`. They summarize behavior, not only the winning metric.

## Required Sections

### `configuration_changes`

- `changed_parameters`
- `rationale`
- `change_mode`: `quantitative_local` or `qualitative_strategy`

### `primary_metric_response`

- `metric`
- `baseline_value`
- `candidate_value`
- `absolute_change`
- `relative_change`
- `uncertainty`
- `meaningfulness`

### `yield_cutflow_response`

- `yield_changes`
- `cutflow_changes`
- `signal_background_pattern`
- `unexpected_empty_or_unstable_categories`

### `shape_response`

- `changed_distributions`
- `localized_or_global`
- `suspicious_structures`
- `binning_assessment`

### `fit_inference_response`

- `parameter_changes`
- `nuisance_changes`
- `goodness_of_fit_changes`
- `uncertainty_changes`
- `dominant_categories`

### `validation_response`

- `verifier_results`
- `warnings`
- `failed_checks`
- `artifacts_reused`
- `stages_rerun`
- `cache_concerns`

### `interpretation`

- `status`
- `direct_observations`
- `plausible_interpretation`
- `unresolved_uncertainty`
- `possible_implementation_issue`

Allowed interpretation statuses:

- `credible improvement`
- `credible deterioration`
- `negligible change`
- `trade-off`
- `suspicious improvement`
- `invalid candidate`
- `ambiguous result`
- `new qualitative opportunity`
- `possible implementation problem`
- `requires human review`
