# Durable Layer Proposal

Date: 2026-05-31

This proposal adapts the existing workspace rather than moving files immediately. The current `modular-pipeline/analysis/` package remains the canonical executable implementation until service boundaries are migrated deliberately.

## Proposed Repository Shape

```text
newpipeline/
  physics_policy/
    PHYSICS_POLICY.md
    invariants.yaml

  workflow_contracts/
    analysis_contract.yaml
    verification_contract.yaml
    artifact_schemas/

  modular-pipeline/
    analysis/
      ... existing canonical implementation ...
    modular_pipeline/
      ... existing orchestration/tracking implementation ...
    tests/
      ... existing tests plus service compatibility tests ...

  configs/
    baseline/
      hgg_5cat_baseline.yaml
      analysis.summary.json
    optimization/
      scan_ranges.yaml
      candidate_template.yaml
    examples/

  ledger/
    EXECUTABLE_SERVICES.yaml
    GLOBAL_INVARIANTS.yaml
    CHANGELOG.md
    COMPATIBILITY.md

  verifiers/
    verify_run_artifacts.py
    verify_histograms.py
    verify_fit_outputs.py
    verify_blinding.py
    verify_baseline_comparison.py

  runs/
    <run_id>/
      config_snapshot/
      provenance.yaml
      outputs/
      verification_report.md

  experimental/
    smoothing_method_studies/
    documentation_asset_generation/

  generated/
    documentation/
    plots/
```

## Layer Mapping

### Layer 1: Physics Policy

Create `physics_policy/` for rules that automated optimization must not silently change.

Initial sources to mine:

- `modular-pipeline/analysis/analysis.summary.json`
- `modular-pipeline/analysis/config/load_summary.py`
- `modular-pipeline/analysis/selections/engine.py`
- `modular-pipeline/analysis/objects/photons.py`
- `modular-pipeline/analysis/objects/jets.py`
- `modular-pipeline/analysis/samples/registry.py`
- `modular-pipeline/modular_pipeline/AGENTS.md`
- `modular-pipeline/modular_pipeline/docs/FIT_SIGNIFICANCE_NOTES.md`

Policy topics to extract:

- 36.1 fb-1 central luminosity convention unless explicitly changed.
- Photon and jet object definitions.
- Diphoton mass fit range, signal window, and sideband semantics.
- Five-category H to gamma gamma signal-region interpretation.
- Nominal versus alternative sample policy.
- Event-weight formula and normalization convention.
- Blinding defaults and observed-significance approval rule.
- Statistical parameter policy for expected significance.

### Layer 2: Workflow Contracts

Create `workflow_contracts/` for required artifacts and pass/fail criteria independent of implementation.

Initial contracts should cover:

- Required run manifest and state artifacts.
- Required normalized configuration and provenance snapshot.
- Required sample registry and normalization table.
- Required cutflow and yield table.
- Required histogram/template bundle with bin edges and sumw2.
- Required fit outputs, fit provenance, and significance JSON.
- Required blinding summary.
- Required independent verification report before scientific comparison.

### Layer 3: Executable Services

Keep current implementations in place but treat the following as durable service boundaries:

- `modular_pipeline_orchestrator`: `modular-pipeline` CLI and component registry.
- `summary_normalizer`: summary/config loading and runtime default materialization.
- `preflight_verifier`: input and configuration preflight checks.
- `sample_registry`: metadata discovery, classification, nominal/alternative policy, normalization table.
- `event_processor_histogram_builder`: event reading, object building, selections, weights, cutflows, templates.
- `fit_builder`: RooFit model construction and measurement fit.
- `significance_calculator`: expected/observed significance with blinding enforcement.
- `background_selector_bridge`: optional official HHXYY background selector bridge.
- `plot_report_builder`: blinded plotting and report artifact generation.
- `artifact_tracker`: manifest/state inspection and readiness tracking.

Do not move these into `services/` immediately. First document interfaces and add compatibility tests. Later, if useful, create service directories that wrap the existing modules without rewriting physics logic.

### Layer 4: Analysis Configuration

Create `configs/` only after human review decides which embedded defaults are tunable. Likely configuration-only optimization fields:

- Category thresholds within approved policy bounds.
- Histogram bin width and plotting selections.
- Background model candidate set and selection metric.
- Fit ranges only if policy allows.
- Scan ranges and optimization budgets.
- Optional plotting/report masks.
- Resource settings such as `max_events`.

Do not treat object definitions, blinding rules, event-weight meaning, nominal-sample definitions, or statistical interpretation as ordinary optimization knobs.

## Migration Plan

1. Adopt the ledger and invariants added by this audit.
2. Add `physics_policy/PHYSICS_POLICY.md` and `workflow_contracts/*.yaml` without changing code behavior.
3. Add independent verifiers for existing output artifacts.
4. Add compatibility tests for the current baseline run and service entrypoints.
5. Extract tunable defaults from `DEFAULT_RUNTIME` into baseline config files, leaving policy defaults locked.
6. Teach the summary normalizer to read explicit config files while preserving existing JSON inputs.
7. Move exploratory utilities into `experimental/` only after preserving current paths or providing aliases.

## Proposed Handling of Existing Duplicates

- Keep `analysis/analysis.summary.json` as the preferred canonical config unless humans choose `Higgs-to-diphoton.json`.
- Preserve `analysis/Higgs-to-diphoton.json` as a compatibility alias until all docs and skill references are updated.
- Treat `analysis/pipeline.py` as the canonical sequential behavior reference and `modular_pipeline/components.py` as the preferred orchestration interface.
- Keep `analysis/ad_hoc_smoothing_method_study.py` experimental unless it becomes a tested service.
- Treat root-level documentation outputs and `analysis_documentation_assets/` as generated or experimental documentation support, not source-of-truth policy.

## Acceptance Criteria For Future Service Promotion

A candidate service should not be called durable until it has:

- A ledger entry.
- A documented CLI or Python interface.
- Declared inputs, outputs, invariants, and consumers.
- Tests or explicit compatibility checks.
- Versioned output schemas for generated artifacts.
- A verifier or independent acceptance check.
- A changelog process for behavior changes.

## Immediate No-Refactor Recommendation

The current code already runs through a modular orchestrator and has a clean git state. Do not move source files yet. The safe next increment is documentation, policy extraction, tests, and verifiers.
