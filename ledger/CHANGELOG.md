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
