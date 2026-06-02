# Service Change Log

This changelog records durable executable-service changes. Future service modifications must update this file and `ledger/EXECUTABLE_SERVICES.yaml` atomically with the implementation change.

## Entry Template

```text
Date:
Author:
Service ID:
Old Version:
New Version:
Change Class:
Summary:
Motivation:
Files Changed:
Tests Run:
Baseline Configurations Tested:
Backward-Compatibility Assessment:
Affected Consumers:
Migration Note:
Approval Status:
```

## 2026-06-02

Date: 2026-06-02
Author: Codex
Service ID: event_processor_histogram_builder
Old Version: 0.1.0
New Version: 0.1.0
Change Class: Class B: Backward-compatible repair
Summary: Repaired the Section 8 observable-seam checkpoint by restoring immutable historical run provenance, moving generated BDT metadata defaults under the selected output directory, clarifying standalone versus modular external-input requirements, expanding observable-overlap tests, and adding a bounded Section 8 output comparator.
Motivation: Make the first observable-materialization checkpoint safe to merge without changing physics behavior, category routing, BDT features, cuts, split logic, normalization, blinding, model parameters, or statistical behavior.
Files Changed:

- `analysis/analysis_versions.json`
- `analysis/section8_ads/pipeline.py`
- `analysis/section8_ads/modular_adapter.py`
- `analysis/section8_ads/BDT_TRAINING_NOTES.md`
- `analysis/section8_ads/SECTION8_BDT_OPTIMIZATION.md`
- `docs/OBSERVABLE_LAYER_REFACTOR_PLAN.md`
- `tools/compare_section8_outputs.py`
- `tests/test_analysis_versions.py`
- `tests/test_compare_section8_outputs.py`
- `tests/test_section8_ads_observables.py`
- `ledger/GLOBAL_INVARIANTS.yaml`
- `ledger/EXECUTABLE_SERVICES.yaml`
- `ledger/CHANGELOG.md`
- `optimization_infra/runs.jsonl`

Tests Run: `python -m py_compile analysis/section8_ads/observables.py analysis/section8_ads/pipeline.py tools/compare_section8_outputs.py tests/test_section8_ads_observables.py`; `pytest -q tests/test_analysis_versions.py`; `pytest -q tests/test_section8_ads_observables.py`; `pytest -q tests/test_section8_ads_categories.py`; `pytest -q tests/test_section8_ads_bdt_training.py`; `pytest -q tests/test_section8_ads_loader.py`; `pytest -q tests/test_section8_ads_branch_map.py`; `pytest -q tests/test_section8_process_cutflow.py`; `pytest -q tests/test_compare_section8_outputs.py`; `pytest -q` (`47 passed`).
Baseline Configurations Tested: Synthetic unit fixtures only unless a real-data parity run is explicitly reported.
Backward-Compatibility Assessment: Existing CLI entrypoints, category routing, BDT training mechanics, tri-state assignment, output field names, and draft-schema non-authoritative status are preserved. Normal Section 8 BDT runs now write metadata beneath `<outputs>/metadata/` by default; central registry writes require explicit override paths.
Affected Consumers: Section 8 ADS pipeline, modular Section 8 adapter, future optimization agents, provenance auditors.
Migration Note: Historical rows in `optimization_infra/runs.jsonl` preserve original values. Path audits classify those rows as historical provenance rather than rewriting them.
Approval Status: No physics-policy approval required; no physics semantics changed.

Date: 2026-06-02
Author: Codex
Service ID: event_processor_histogram_builder
Old Version: 0.1.0
New Version: 0.1.0
Change Class: Class B: Backward-compatible service extension
Summary: Added an internal shared Section 8 observable builder and removed host-specific Section 8 paths from active analysis-version configuration and BDT metadata-writing logic.
Motivation: Create the first stable observable-materialization seam for future configuration-driven routing while preserving current physics semantics and portability.
Files Changed:

- `analysis/section8_ads/observables.py`
- `analysis/section8_ads/pipeline.py`
- `analysis/analysis_versions.json`
- `configs/schema_drafts/*.example.yaml`
- `docs/OBSERVABLE_LAYER_REFACTOR_PLAN.md`
- `tests/test_analysis_versions.py`
- `tests/test_section8_ads_observables.py`
- `tests/test_section8_ads_loader.py`
- `tests/test_section8_ads_branch_map.py`
- `ledger/GLOBAL_INVARIANTS.yaml`
- `ledger/EXECUTABLE_SERVICES.yaml`
- `ledger/CHANGELOG.md`

Tests Run: `python -m py_compile analysis/section8_ads/observables.py analysis/section8_ads/pipeline.py tests/test_section8_ads_observables.py`; `pytest -q tests/test_analysis_versions.py tests/test_section8_ads_observables.py tests/test_section8_ads_categories.py tests/test_section8_ads_bdt_training.py tests/test_section8_ads_loader.py tests/test_section8_ads_branch_map.py tests/test_section8_process_cutflow.py`; `pytest -q`.
Baseline Configurations Tested: Synthetic unit fixtures only; no external ROOT-data run.
Backward-Compatibility Assessment: Existing CLI entrypoints, category routing, BDT training mechanics, split policy, model parameters, cuts, blinding, and output field names are preserved. Section 8 external ADS and BDT artifact paths must be supplied explicitly by CLI overrides or direct function arguments.
Affected Consumers: Section 8 ADS pipeline, modular Section 8 adapter, future optimization agents.
Migration Note: Draft schema examples are non-authoritative and are not wired into runtime execution.
Approval Status: No physics-policy approval required; no physics semantics changed.

## 2026-05-31

Date: 2026-05-31
Author: Codex
Service ID: workspace_governance
Old Version: none
New Version: 0.1.0
Change Class: Governance documentation
Summary: Added initial architecture audit, durable-layer proposal, executable-service ledger, global invariants, compatibility policy, changelog template, and human-review list.
Motivation: Establish durable, accountable architecture before agent-driven optimization.
Files Changed:

- `WORKSPACE_ARCHITECTURE_AUDIT.md`
- `DURABLE_LAYER_PROPOSAL.md`
- `ledger/EXECUTABLE_SERVICES.yaml`
- `ledger/GLOBAL_INVARIANTS.yaml`
- `ledger/COMPATIBILITY.md`
- `ledger/CHANGELOG.md`
- `HUMAN_REVIEW_REQUIRED.md`

Tests Run: Documentation-only change; pytest should be run to confirm no repository side effects.
Baseline Configurations Tested: None.
Backward-Compatibility Assessment: Additive documentation and ledger files only; no existing code, configs, or generated artifacts changed.
Affected Consumers: Future agents and human reviewers.
Migration Note: None.
Approval Status: Not required for additive governance files.

## 2026-05-31 Optimization Infrastructure

Date: 2026-05-31
Author: Codex
Service ID: optimization_orchestrator
Old Version: none
New Version: 0.1.0
Change Class: Class C: New executable service
Summary: Added root-level optimization infrastructure for workflow-stage auditing, dependency-graph planning, artifact-reuse policy, run registry records, scan dry runs, observation summaries, scan synthesis, decision packets, and branching/control policies.
Motivation: Enable iterative, resumable, evidence-driven optimization without launching real optimization or changing scientific logic.
Files Changed:

- `optimization_infra/WORKFLOW_STAGE_AUDIT.md`
- `optimization_infra/workflow_graph.yaml`
- `optimization_infra/artifact_manifest_schema.yaml`
- `optimization_infra/ARTIFACT_REUSE_POLICY.md`
- `optimization_infra/RUN_REGISTRY_SCHEMA.md`
- `optimization_infra/runs.jsonl`
- `optimization_infra/scan_spec_schema.yaml`
- `optimization_infra/example_scan.yaml`
- `optimization_infra/OBSERVATION_SCHEMA.md`
- `optimization_infra/BRANCHING_POLICY.md`
- `optimization_infra/OPTIMIZATION_CONTROL_POLICY.md`
- `optimization_infra/__init__.py`
- `optimization_infra/plan_candidate_run.py`
- `optimization_infra/run_scan.py`
- `optimization_infra/summarize_scan.py`
- `optimization_infra/create_decision_packet.py`
- `runs/README.md`
- `tests/test_optimization_infra.py`
- `ledger/EXECUTABLE_SERVICES.yaml`
- `ledger/CHANGELOG.md`

Tests Run: `PYTHONPATH=<repo> python -m pytest -q tests`
Baseline Configurations Tested: None; infrastructure tests use synthetic fixtures only.
Backward-Compatibility Assessment: Additive root-level infrastructure only. Existing scientific code, physics-policy invariants, and modular-pipeline entrypoints are unchanged.
Affected Consumers: Future optimization agents, human reviewers, reasoning-agent handoff workflows.
Migration Note: None.
Approval Status: Not required for additive infrastructure; required before any future science-changing optimization or service execution runner is enabled.

## 2026-05-31 Round Versioning And Loop Driver

Date: 2026-05-31
Author: Codex
Service ID: optimization_orchestrator
Old Version: 0.1.0
New Version: 0.1.0
Change Class: Class B: Backward-compatible service extension
Summary: Added descriptive version anchors for optimization rounds and a bounded dry-run loop driver that preserves per-round reports and decision packets.
Motivation: Make each optimization round traceable to a descriptive version name, git state, compact evidence artifacts, and fixed-round loop control.
Files Changed:

- `optimization_infra/version_round.py`
- `optimization_infra/run_optimization_loop.py`
- `optimization_infra/loop_spec_schema.yaml`
- `optimization_infra/example_loop.yaml`
- `optimization_infra/RUN_REGISTRY_SCHEMA.md`
- `optimization_infra/OPTIMIZATION_CONTROL_POLICY.md`
- `optimization_infra/run_scan.py`
- `runs/README.md`
- `tests/test_optimization_infra.py`
- `ledger/EXECUTABLE_SERVICES.yaml`
- `ledger/CHANGELOG.md`

Tests Run: `PYTHONPATH=<repo> python -m pytest -q tests`
Baseline Configurations Tested: None; infrastructure tests use synthetic fixtures only.
Backward-Compatibility Assessment: Additive extension only. Existing scientific code, physics-policy invariants, and scan planning behavior remain compatible.
Affected Consumers: Future optimization agents, human reviewers, run-history inspection workflows.
Migration Note: None.
Approval Status: Not required for additive dry-run infrastructure; required before creating permanent git tags or executing scientific optimization loops.
