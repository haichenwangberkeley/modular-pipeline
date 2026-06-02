# Observable Layer Refactor Plan

Date: 2026-06-02

This document records the first safe increment toward a configuration-driven observable layer. It is a planning and governance artifact; it does not make the draft schema files authoritative configuration.

## Current Architecture

The repository has two layers. `analysis/` is the canonical executable implementation for summary normalization, sample discovery, event processing, category assignment, histogram/template production, fitting, significance, plotting, and reports. `modular_pipeline/` is a maskable orchestrator over those canonical functions and must not reimplement physics logic.

The default analysis is the five-category `pTt`/VBF-enriched path in `analysis/hists/histmaker.py` and `analysis/selections/engine.py`. The Section 8 analysis version switches sample processing through `analysis/section8_ads/modular_adapter.py`, which calls `analysis/section8_ads/pipeline.py`, materializes Section 8 arrays, scores supplemental BDTs if artifacts are available, and then routes events through `analysis/section8_ads/categories.py`.

## Current Fusion

Event extraction, object building, observable calculation, BDT scoring, category assignment, and histogram production are currently fused in two places:

- `analysis/hists/histmaker.py` reads events, builds photons/jets, applies baseline cuts, assigns five categories, and appends histogram fields.
- `analysis/section8_ads/pipeline.py` reads events, builds the Section 8 diphoton view, computes object/event observables, prepares BDT-training candidates, writes BDT diagnostics, scores BDTs, assigns Section 8 categories, and writes validation reports.

The modular orchestrator is maskable, but durable reuse is still coarse because category routing consumes arrays produced directly by event-loop code rather than by a versioned observable-materialization stage.

## Duplicated Section 8 Calculations

Before this increment, the nominal Section 8 event path and the BDT-training-candidate path duplicated:

- branch requirements;
- loose photon filtering and two-leading-photon selection;
- diphoton mass, `pT_gammagamma`, `eta_gammagamma`, `pTt_gammagamma`, lead/sublead kinematics, and `ET/mgg` masks;
- sideband and signal-window flags;
- jet multiplicities at 25 and 30 GeV, central/forward counts, b-tag counts, and JVT diagnostics;
- selected-lepton counts;
- `MET_significance = MET / sqrt(H_T)`;
- leading-jet `pT`, `m_jj_30`, `abs_delta_eta_jj_30`, `pT_Hjj_30`, `deltaR_min_gamma_j`, `VBF_centrality`, `H_T`, `m_all_jets`, `delta_y_gammagamma_jj`, `cos_theta_star_gammagamma_jj`, and capped `abs_delta_phi_gammagamma_jj`;
- BDT training masks for ttH, VH, and VBF.

The nominal path additionally computes detailed `m_ll`, `Z_ll_veto`, `m_e_gamma_veto`, and `pT_lepton_plus_MET`. The BDT-training-candidate path intentionally keeps those detailed lepton-veto fields as placeholders while still computing `N_lep`.

## Observable-Producer Abstraction

The new internal seam is `analysis/section8_ads/observables.py`. It separates:

- diphoton event-view construction;
- Section 8 object/event observable materialization for an input event view;
- caller-specific masks and output assembly.

This is deliberately not a generic DSL. It preserves the existing masks, approximations, field names, and category engine while giving future work a stable place to introduce validated observable producers.

Future producers should have explicit inputs, output columns, version/hash metadata, approximation status, and invalidation rules. Candidate producer kinds are input branch, expression, Python plugin, and learned score.

## Training vs Score Materialization

Model training creates reusable artifacts: model files, feature manifests, training-audit hashes, backend versions, hyperparameters, and normalization diagnostics. Score materialization is a later inference step that consumes a compatible model artifact and writes an event-level score column such as `BDT_ttH`.

Downstream categorization should treat a learned score like any other derived observable once it is materialized. Boundary scans over score thresholds should not retrain the model. Hyperparameter or feature changes must invalidate training and all downstream score/category artifacts.

## Future Category Routing

The future category router should consume a table of event observables and an ordered first-match category declaration. It must preserve the Section 8 tri-state outcome:

- assigned category;
- no matching category;
- blocked because an earlier eligible category requires a missing classifier or derived input.

The router must also preserve category ordering, missing-input blocking, and diagnostic reasons. The current `analysis/section8_ads/categories.py` remains the behavior reference until compatibility tests approve a generic router.

## Future Stage Graph

```text
event_extraction
object_building
observable_materialization
model_training
model_scoring
category_assignment
template_production
fit_construction
significance
plotting
report_generation
```

## Artifact Types And Invalidation

- Event-extraction artifacts: input-file inventories, branch maps, event IDs. Invalidate when input files, tree names, branch maps, or event filters change.
- Object-building artifacts: photon, jet, lepton, and MET views. Invalidate when object definitions, calibration choices, or quality masks change.
- Observable-materialization artifacts: derived-column arrays and manifests. Invalidate when producer code, required inputs, constants, formulas, plugin versions, or approximation policies change.
- Model-training artifacts: training rows, audits, model files, feature manifests, backend versions, and hyperparameters. Invalidate when features, labels, sample roles, split policy, weights, backend, hyperparameters, or training selections change.
- Model-scoring artifacts: event-level score columns and score manifests. Invalidate when model artifacts, scoring code, input observable columns, or model compatibility hashes change.
- Category-assignment artifacts: category labels, blocked flags, and reasons. Invalidate when category order, boundaries, required inputs, missing-input policy, or consumed observables/scores change.
- Template-production artifacts: histograms and cache fields. Invalidate when categories, fitted observable, binning, weights, or selected event rows change.
- Fit/significance artifacts: workspaces, fit results, Asimov/observed significance. Invalidate when templates, fit model policy, blinding state, parameter policy, or backend changes.
- Plot/report artifacts: derived visual and narrative products. Invalidate when upstream artifacts or presentation configuration changes.

## Staged Migration Plan

1. Keep the current shared Section 8 observable builder internal and covered by synthetic tests.
2. Add producer manifests for existing Section 8 observables without changing execution.
3. Add a non-authoritative observable graph validator for draft configs.
4. Introduce score-materialization manifests for existing BDT scoring.
5. Build a generic category-router prototype behind tests that replay the current five-category and Section 8 outputs.
6. Promote only after side-by-side tests prove unchanged category labels, blocked states, cutflows, templates, and BDT diagnostics.
7. Move optimization scans to edit validated config files, with artifact reuse decided by declared invalidation rules.

## Risks And Compatibility

The highest risks are silent changes to event selection, photon isolation approximation, b-tag proxy, `MET_significance`, BDT feature values, category priority, missing-score blocking, deterministic splits, and training normalization. Existing CLI entrypoints, output fields, and category IDs must remain available. Draft schema examples must not be treated as production configuration.

## Human Physics Approval Required

Human approval is required before changing object definitions, numerical cuts, category order, sample roles, model features, BDT hyperparameters, split policies, normalization, blinding, fit/statistical behavior, or the interpretation of supplemental BDT scores. Approval is also required before promoting generated artifacts into authoritative configuration.
