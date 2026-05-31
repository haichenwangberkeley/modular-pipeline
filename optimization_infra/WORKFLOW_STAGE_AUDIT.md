# Workflow Stage Audit

Date: 2026-05-31

Scope: current runnable H to gamma gamma workflow in `modular-pipeline/analysis/` and `modular-pipeline/modular_pipeline/`.

## Summary

The current workflow is implemented as a canonical sequential pipeline in `modular-pipeline/analysis/pipeline.py` and a maskable orchestration graph in `modular-pipeline/modular_pipeline/components.py`. The modular component graph is the best available runnable-stage model, but several scientific sub-steps are still fused inside the `samples`, `templates`, `fit`, and `report` components.

For optimization infrastructure, the graph below uses current executable boundaries where possible and records finer invalidation semantics in metadata. This avoids rewriting scientific code while still allowing the planner to reason about partial reuse.

## Discovered Stages

| Stage ID | Name | Implementation | Inputs | Outputs | Config Dependencies | Consumers | Existing Service | Cacheable | Versioned Outputs | Verifier Exists | Missing Metadata | Ambiguities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `summary_normalization` | Summary Normalization | `analysis/config/load_summary.py`, modular `summary` component | Analysis summary JSON | `summary.normalized.json`, validation inventory/diagnostics/overlap policy | Full summary, runtime defaults, blinding defaults | All downstream stages | Yes, `summary_normalizer` | Yes | No explicit schema version | Partial schema validation | Config-subset hashes, schema version | Canonical summary path is unresolved. |
| `runtime_contract` | Runtime Contract Materialization | `analysis/runtime.py`, `analysis/report/artifacts.py`, modular `runtime_contract` component | Normalized summary | Runtime recovery, blinding summary, execution contract, generated `regions.yaml` | Runtime defaults, blinding, categories, inputs path, `max_events` | Preflight, reports, review | Yes, part of orchestration/report services | Yes | No explicit schema version | Partial generated artifacts | Artifact hashes, policy version | Whether source-tree `regions.yaml` is config or generated. |
| `preflight_validation` | Preflight Validation | `analysis/preflight.py` | Summary, inputs, outputs | `report/preflight_fact_check.json` | Summary, input layout | Human/agent gate, downstream run confidence | Yes, `preflight_verifier` | Yes | No explicit schema version | Yes, action emits check artifact | Pass/fail schema, thresholds | Some checks may need human-set thresholds. |
| `metadata_resolution` | Dataset Metadata Resolution | `analysis/samples/metadata.py` | `input-data/{data,MC}` ROOT files | Metadata rows, metadata resolution JSON | Input data paths | Sample registry | Candidate/existing service | Yes | No explicit schema version | Indirect | Input checksums, metadata source versions | `skills/metadata.csv` write path may be legacy. |
| `sample_registry` | Sample Registry And Nominal Selection | `analysis/samples/registry.py` | Metadata rows, inputs, normalized summary | `samples.registry.json`, `mc_sample_selection.json`, `norm_table.json` | Target luminosity, sample classification policy | Event processing, fit, reports | Yes, `sample_registry` | Yes | No explicit schema version | Partial via report artifacts | Config subset, sample file checksums | Nominal/alternative policy requires physics approval. |
| `background_strategy` | Background Strategy And CR/SR Map | `analysis/samples/strategy.py` | Registry, normalized summary | `samples.classification.json`, `background_modeling_strategy.json`, `cr_sr_constraint_map.json` | Background process policy, region definitions | Fit, report, review | Yes, part of `sample_registry` | Yes | No explicit schema version | No independent verifier | Strategy schema, policy hash | Data-driven background semantics need policy extraction. |
| `partitioning` | Region Partitioning | `analysis/selections/partitioning.py` | Normalized summary | `partition/partition_spec.json` | Categories, overlap policy, fit regions | Review, validation | Yes, workflow-contract producer | Yes | No explicit schema version | No independent verifier | Region graph hash | Overlap policy is partly generated. |
| `event_processing` | Event Extraction, Object Selection, Feature Computation, Category Assignment | `analysis/io/readers.py`, `analysis/objects/*.py`, `analysis/selections/engine.py`, `analysis/hists/histmaker.py` | Registry, ROOT files, runtime defaults | Per-sample processed event dictionaries, `cache/*.npz` | Tree name, object definitions, fit mass range, photon/jet selections, category rules, `max_events` | Templates, cutflow, fit, plots | Yes, `event_processor_histogram_builder` | Yes | No explicit schema version | No independent verifier | Branch list, cache schema, input file hashes | Fused stage; category threshold changes currently require rerunning processing. |
| `histogram_production` | Histogram Template Production | `analysis/hists/histmaker.py` | Processed samples | `hists/templates.json`, `hists/processed_samples.json` | Fit mass range, histogram bin width, categories | Fit, plots, reports | Yes, `event_processor_histogram_builder` | Yes | No explicit schema version | No independent verifier | Histogram schema version, bin coverage metadata | Fit-range reuse depends on histogram coverage, not currently declared. |
| `cutflow_yields` | Cutflow And Yield Tables | `analysis/report/artifacts.py` | Processed samples | `report/cutflow_table.json`, `report/yields_by_category.json` | Selection/category semantics | Reports, comparisons, observations | Yes, report artifact service | Yes | No explicit schema version | No independent verifier | Cutflow schema, category stability checks | Thresholds for suspicious yield changes need human review. |
| `fit_construction` | Statistical Model Construction And Fit | `analysis/stats/fit.py`, `analysis/stats/models.py`, optional `hhxyy_fitting_backend.py` | Processed samples, registry, normalized summary, templates | Fit workspace, fit results, model selection artifacts, fit provenance | Fit ranges, background model candidates, smoothing policy, statistical parameter policy | Significance, plots, reports | Yes, `fit_builder` | Partly | No explicit schema version | Fit status recorded; no independent hook | Baseline comparison test, workspace hash | Central backend versus cross-check role needs policy clarity. |
| `systematics_build` | Systematics Artifact Build | `analysis/stats/systematics.py` | Registry, normalized summary | `systematics.json`, provenance and mapping | Systematics mode/entries | Fit/report/review | Candidate/existing placeholder service | Yes | No explicit schema version | No independent verifier | Finality status, model version | Current mode is placeholder. |
| `significance` | Significance Calculation | `analysis/stats/significance.py`, optional HHXYY backend | Fit context, normalized summary | `significance_asimov.json`, `significance.json`, diagnostics | Blinding, statistical policy, backend availability | Optimization metric, reports | Yes, `significance_calculator` | Yes | No explicit schema version | Fit diagnostics recorded; no independent hook | Metric schema, backend compatibility | Observed significance requires explicit approval. |
| `plotting` | Blinded Plot Generation | `analysis/plotting/blinded_regions.py`, `analysis/plotting/hhxyy_fit_plots.py` | Processed samples, summary, fit context, cutflow | Plot payloads/manifests | Plotting selections, blinding policy | Reports, humans | Yes, `plot_report_builder` | Yes | No explicit schema version | Blinding summary; no independent image verifier | Plot manifest schema | Plot-style-only changes should not recompute science. |
| `review_artifacts` | Review And Validation Artifacts | `analysis/report/artifacts.py` | Registry, processed samples, fit context, plot manifest, policy defaults | Discrepancy, smoothing, effective-lumi, verification status, skill artifacts | Validation/report settings | Human review, reports | Yes, report/validation service | Yes | No explicit schema version | Partial, action-generated | Independent verifier separation | Some artifacts are report-oriented rather than validation gates. |
| `report_generation` | Report Generation And Handoff | `analysis/report/make_report.py`, `analysis/report/artifacts.py` | Summary, outputs, smoke outputs if present | Final report and handoff artifacts | Report settings, all upstream outputs | Humans/agents | Yes, `plot_report_builder` | Yes | No explicit schema version | No independent verifier | Report provenance | Root-level documentation pipeline overlaps. |

## Key Invalidation Observations

- Object-definition or event-weight changes invalidate nearly all downstream science artifacts.
- Category-threshold changes currently invalidate event processing because category assignment is fused with processing.
- Fit-range changes may reuse event caches if selected-event mass coverage is adequate, but the current cache does not explicitly declare coverage.
- Plot-style-only changes should invalidate plotting/report stages only.
- Blinding-policy changes are physics-policy changes and cannot be ordinary optimization changes.

## Missing Workflow Metadata

- Schema versions for all durable artifacts.
- Per-stage config-subset hashes.
- Input ROOT file checksums.
- Service version compatibility classes.
- Independent verifier commands separate from action services.
- Artifact coverage metadata, especially mass/histogram ranges.
- A first-class baseline run registry entry.
