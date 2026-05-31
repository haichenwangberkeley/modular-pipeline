# Run Registry Schema

Date: 2026-05-31

The run registry is an append-only JSON Lines file at `optimization_infra/runs.jsonl`. Each line is one run record.

## Required Fields

- `run_id`: unique run identifier.
- `timestamp`: ISO 8601 timestamp.
- `parent_run_id`: parent run ID or null.
- `branch_id`: branch identifier.
- `strategy_id`: strategy identifier.
- `run_type`: one of the supported run types below.
- `objective`: scientific or infrastructure objective.
- `configuration_snapshot_path`: path to immutable candidate configuration.
- `changed_parameters`: list of changed config paths and values.
- `reused_artifacts`: list of artifact IDs or stage IDs reused.
- `invalidated_artifacts`: list of artifact IDs or stage IDs invalidated.
- `stages_executed`: stages planned or executed.
- `stages_skipped`: stages skipped because reusable artifacts exist.
- `verifier_status`: aggregate verifier status.
- `cut_flow_path`: path or null.
- `yield_table_path`: path or null.
- `fit_output_path`: path or null.
- `metric_values`: mapping of metrics.
- `expected_significance`: number or null.
- `observed_significance`: number or null; allowed only when blinding policy permits.
- `runtime`: runtime summary or null.
- `warnings`: list of warnings.
- `failure_reason`: string or null.
- `human_or_agent_note`: string or null.
- `git_state`: mapping with commit and dirty state.
- `service_versions`: mapping of service IDs to versions.

## Supported Run Types

- `baseline`
- `single_candidate`
- `parameter_scan`
- `structured_scan`
- `qualitative_strategy_branch`
- `validation_only`
- `service_extension_validation`

## Append Rules

- Registry writes are append-only.
- Existing entries must not be rewritten by scan tools.
- A run ID should be unique; duplicate run IDs are invalid unless a human explicitly marks a repair record.
- Failed dry runs may be recorded if useful, but must set `failure_reason`.
