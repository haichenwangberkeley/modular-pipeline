# Workspace Architecture Audit

Date: 2026-05-31

Scope: this repository plus the surrounding analysis workspace observed during the audit, including generated run outputs, documentation artifacts, and external input-data handles visible locally.

This audit is intentionally architectural. It does not optimize the analysis, refactor source code, delete files, or change physics behavior.

## Executive Summary

The workspace already contains a useful separation between canonical analysis code and orchestration:

- `modular-pipeline/analysis/` is the current canonical implementation for H to gamma gamma processing, sample classification, selections, histograms, RooFit/HHXYY-backed fitting, significance, plotting, reports, and artifact writing.
- `modular-pipeline/modular_pipeline/` is a maskable orchestration and artifact-readiness layer over the canonical stages.
- `modular-pipeline/analysis/analysis.summary.json` and `modular-pipeline/analysis/Higgs-to-diphoton.json` are analysis-definition inputs, but many mutable runtime defaults are still embedded in Python code.
- `outputs_modular_hgg_20260531T213216Z/`, `analysis_documentation_assets/`, and `ANALYSIS_DOCUMENTATION.*` are generated products and should not become authoritative sources.
- Several scientifically sensitive conventions are implemented in services but not yet captured as standalone physics-policy or workflow-contract artifacts.

The highest-value next step is to add durable governance around the existing implementation, then gradually migrate embedded defaults into explicit policy, contract, and configuration files.

## Classification Key

1. Physics-policy artifact
2. Workflow-contract artifact
3. Candidate executable service
4. Existing executable service
5. Analysis configuration
6. Experimental or exploratory code
7. Obsolete or duplicated implementation
8. Generated artifact
9. Missing capability
10. Ambiguous item requiring human clarification

## Artifact Inventory

### Workspace Root

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/` | 4 | Main analysis repository | Human/agent runs, CLI entrypoints | Contains package metadata, tests, `analysis/`, `modular_pipeline/` | Yes | Partly | Yes, narrow | With tests and ledger updates | No | Treat as canonical source tree. |
| `input-data/` | 1, 8 | External data/MC directory handle | Sample metadata, registry, event processing | README declares expected `data/` and `MC/` layout | Yes as input contract | Directory contract only | Preflight only | No ordinary edits | No | Treat as protected external input; record exact sample provenance per run. |
| `outputs_modular_hgg_20260531T213216Z/` | 8 | Prior pipeline output bundle | Documentation, comparison, inspection | Contains manifest, state, cache, fit, hist, report artifacts | Reusable as reference output | Output schemas not fully versioned | Partly via state/verification artifacts | No | No | Keep as generated reference; do not use as source of policy except by citation. |
| `reports/` | 8 | Report target directory | Report builder | Workspace directory created by runs | No | No | No | Yes, generated only | No | Keep generated reports out of durable layer unless selected as references. |
| `ANALYSIS_DOCUMENTATION.md` | 8, 10 | Human-readable analysis writeup | Humans | Paired with generated TeX/PDF/assets | Yes as narrative reference | No | No | Low-risk if regenerated | Duplicates output facts | Mark generated/derived; clarify whether it is authoritative. |
| `ANALYSIS_DOCUMENTATION.tex` | 8 | Generated TeX report source | PDF build | Paired `.aux`, `.log`, `.pdf` | No | No | No | Generated-only | Duplicates `.md`/outputs | Keep as generated artifact. |
| `ANALYSIS_DOCUMENTATION.pdf` | 8 | Rendered report | Humans | Derived from TeX | No | No | No | Generated-only | Duplicates docs | Keep as generated artifact. |
| `ANALYSIS_DOCUMENTATION.aux`, `ANALYSIS_DOCUMENTATION.log` | 8 | TeX build intermediates | TeX tooling | Build byproducts | No | No | No | Yes | No | Consider excluding from future source control. |
| `.TinyTeX/` | 8, 10 | Local TeX runtime | Documentation rendering | Runtime tool installation under workspace | Reusable locally | Toolchain interface | No | Not as analysis code | No | Clarify whether toolchains belong outside repo/workspace. |

### Generated Documentation Assets

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `analysis_documentation_assets/generate_assets.py` | 3, 6 | Generates documentation plots/assets from outputs | Documentation workflow | Sits outside package, reads generated artifacts | Potentially | No | No | Yes, if scoped | Overlaps plotting/report services | Decide whether to fold into report-generation service or keep experimental. |
| `analysis_documentation_assets/*.png` | 8 | Rendered documentation figures | `ANALYSIS_DOCUMENTATION.*` | Static image assets | No | No | No | Generated-only | Derived from outputs | Keep generated. |
| `analysis_documentation_assets/derived_category_metrics.json` | 8 | Derived metrics for documentation figures | Asset generator/docs | JSON derived from outputs | No | No | No | Generated-only | Derived from fit/report outputs | Keep generated. |

### Package and Entry Points

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/pyproject.toml` | 2, 4 | Package metadata and CLI entrypoint registry | Installer, users, agents | Declares scripts and dependencies | Yes | Yes | Indirect | Carefully | `setup.py` metadata | Treat as interface contract for CLI names. |
| `modular-pipeline/setup.py` | 7, 10 | Legacy packaging shim | Installer fallback | Coexists with `pyproject.toml` | Low | No | No | Yes, but not now | Duplicates packaging | Clarify if needed; avoid changing during audit. |
| `modular-pipeline/README.md` | 2 | User-facing run contract | Humans/agents | Documents dataset layout, commands, artifacts | Yes | Text contract | No | Yes with review | Overlaps docs | Promote key commands/contracts into workflow docs. |
| `modular-pipeline/.gitignore` | 2 | Repository hygiene policy | Git | Ignore rules | Yes | Yes | N/A | Yes | No | Update later for generated docs/outputs if desired. |

### Analysis Configuration and Physics Policy Sources

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/analysis/analysis.summary.json` | 1, 5 | Canonical analysis summary: metadata, objectives, regions, fit setup | `hgg-analysis`, `modular-pipeline`, summary loader | README uses it as run input | Yes | JSON shape lightly validated | Yes, schema smoke | Physics edits require approval | Same content as `Higgs-to-diphoton.json` | Split stable physics policy from tunable optimization config. |
| `modular-pipeline/analysis/Higgs-to-diphoton.json` | 7, 10 | Duplicate analysis summary | Skill/runtime examples, possibly agents | Content matches `analysis.summary.json` in current audit | Yes if alias | Same as above | Indirect | Avoid until clarified | Duplicates `analysis.summary.json` | Choose one canonical path; keep alias/migration if consumers depend on it. |
| `modular-pipeline/analysis/regions.yaml` | 8, 10 | Generated categories/runtime defaults from summary normalization | Some tools/humans | `write_regions_yaml` writes this path during runs | Maybe | YAML shape implicit | No | Generated-only | Duplicates normalized summary | Move to generated outputs or clarify if checked-in config is intentional. |
| `modular-pipeline/analysis/config/load_summary.py` | 1, 3, 4, 5 | Summary normalization and default runtime policy injection | Pipeline and CLI | Contains `DEFAULT_RUNTIME`, blinding, mass windows, selection defaults | Yes | Function/CLI | Indirect | Carefully | Duplicates desired config layer | Extract defaults into policy/config files after approval. |
| `modular-pipeline/analysis/config/summary_schema.py` | 2, 4 | Validates required summary shape and references | Summary loader/tests | Required top-level keys and ID checks | Yes | Function | Indirect | Yes with tests | No | Promote to explicit schema service/contract. |

### Canonical Executable Services in `analysis/`

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/analysis/cli.py` | 4 | Legacy/canonical `hgg-analysis` CLI | `pyproject` script | Exposes bootstrap, preflight, run | Yes | CLI | Indirect | With compatibility review | Overlaps modular CLI | Preserve; treat as compatibility entrypoint. |
| `modular-pipeline/analysis/pipeline.py` | 4, 7 | Monolithic canonical stage runner and shared helpers | `hgg-analysis`, `modular_pipeline.components` imports helpers | AGENTS says modular must call same stage functions | Yes | Python function | Indirect | Carefully | Stage logic duplicated by components sequence | Keep as canonical behavior reference; reduce duplication later via shared stage registry. |
| `modular-pipeline/analysis/preflight.py` | 4 | Input/config/environment preflight checks | CLI, modular component | Dedicated script entrypoint | Yes | CLI/function | No direct test | Yes with tests | No | Promote as verifier service. |
| `modular-pipeline/analysis/common.py` | 4 | Shared IO/path/hash utilities | Most services | Imported widely | Yes | Function API | Indirect | Carefully | No | Treat as support service library. |
| `modular-pipeline/analysis/runtime.py` | 4 | Runtime recovery artifact writing | Pipeline | Writes recovery JSON | Yes | Function | Indirect | Yes with tests | No | Include in provenance service boundary. |
| `modular-pipeline/analysis/io/readers.py` | 4 | ROOT/uproot event reading and diagnostics | Histmaker/sample processing | Required branch handling | Yes | Function API | Indirect | Carefully | No | Candidate dataset-reader service with branch contract. |
| `modular-pipeline/analysis/objects/photons.py` | 1, 4 | Photon object construction and selection | Histmaker | Implements tight photon and kinematic cuts | Yes | Function API | Indirect | Physics approval needed for semantics | No | Treat object definitions as policy-backed service. |
| `modular-pipeline/analysis/objects/jets.py` | 1, 4 | Jet object construction and kinematics | Histmaker | Implements jet selection and VBF features | Yes | Function API | Indirect | Physics approval needed for semantics | No | Treat object definitions as policy-backed service. |
| `modular-pipeline/analysis/selections/engine.py` | 1, 4 | Category assignment and mass-window masks | Histmaker, plotting/fits | Defines `CATEGORY_ORDER`, SR logic, sidebands | Yes | Function/constants | Indirect | Physics approval needed | No | Make category semantics explicit in physics policy. |
| `modular-pipeline/analysis/selections/partitioning.py` | 2, 4 | Region partition specification | Pipeline/modular component | Writes partition artifacts | Yes | CLI/function | Indirect | Yes with tests | No | Promote as workflow-contract producer/verifier. |
| `modular-pipeline/analysis/samples/metadata.py` | 4 | Build metadata rows from samples | Registry, metadata CLI | DSID/generator metadata resolution | Yes | CLI/function | Indirect | Carefully | No | Candidate sample-metadata service. |
| `modular-pipeline/analysis/samples/registry.py` | 1, 4 | Sample classification, nominal/alternative selection, normalization inputs | Pipeline | Contains signal/background token rules and nominal policy | Yes | CLI/function/output JSON | Indirect | Physics approval for rules | No | Promote sample-registry service; extract policy. |
| `modular-pipeline/analysis/samples/strategy.py` | 1, 4 | Background modeling strategy and CR/SR map | Pipeline | Builds classification/strategy artifacts | Yes | Function | Indirect | Physics approval for semantics | No | Promote strategy-contract service. |
| `modular-pipeline/analysis/hists/histmaker.py` | 1, 4 | Event weights, object selection, sample processing, hist templates, cutflow | Pipeline, `hgg-histmaker` | Central event-loop and weighting implementation | Yes | CLI/function | Indirect | Carefully with tests | No | Promote high-priority executable service. |
| `modular-pipeline/analysis/stats/models.py` | 1, 4 | Statistical model helpers | Fit/significance | RooFit model definitions | Yes | Function API | Indirect | Very carefully | No | Promote stats-model service with compatibility tests. |
| `modular-pipeline/analysis/stats/fit.py` | 1, 4 | RooFit model construction and fit outputs | Pipeline, `hgg-fit` | Large central fit implementation | Yes | CLI/function/output artifacts | Indirect | Very carefully | No | Promote fit service; require baseline reproducibility tests. |
| `modular-pipeline/analysis/stats/significance.py` | 1, 4 | Expected/observed significance and blinding handling | Pipeline | Large stats/significance implementation | Yes | Function/output artifacts | Indirect | Very carefully | No | Promote significance service; lock blinding invariants. |
| `modular-pipeline/analysis/stats/hhxyy_fitting_backend.py` | 3, 4 | Optional HHXYY/quickFit backend integration | Fit/significance | Backend module and scripts | Yes | Function/config/env | Indirect | Carefully | Local PyROOT fallback | Document backend-substitution contract. |
| `modular-pipeline/analysis/stats/official_bkg_selector.py` | 4 | Optional official HHXYY background selector bridge | `hgg-official-bkg-select`, tests | Has explicit tests | Yes | CLI/function | Yes | Yes with tests | Complements local selector | Keep opt-in; ledger as background-selector bridge. |
| `modular-pipeline/analysis/stats/systematics.py` | 3, 4, 10 | Build systematics artifacts | Pipeline | Runtime defaults say `mode: placeholder` | Potentially | CLI/function | Indirect | Yes, but clarify physics | Placeholder | Human review needed before treating as final science. |
| `modular-pipeline/analysis/plotting/blinded_regions.py` | 1, 4 | Blinded plots and plotting manifest | Pipeline | Blinding-sensitive plotting | Yes | Function/output artifacts | Indirect | Carefully | No | Promote plotting service plus verifier. |
| `modular-pipeline/analysis/plotting/hhxyy_fit_plots.py` | 3, 4 | HHXYY fit plotting utility | CLI | Separate script entrypoint | Yes | CLI | No direct test | Yes with tests | Overlaps plotting | Decide if separate service or subcommand. |
| `modular-pipeline/analysis/report/artifacts.py` | 2, 4 | Cutflow/yields/validation/report JSON artifact writers | Pipeline | Writes many workflow artifacts | Yes | Function/output paths | Indirect | Carefully | Some contract logic mixed with reporting | Split verifiers from report generators over time. |
| `modular-pipeline/analysis/report/make_report.py` | 4 | Final report generation | Pipeline | Builds report | Yes | Function | Indirect | Yes with visual/content checks | Docs assets overlap | Promote report generator service. |
| `modular-pipeline/analysis/ad_hoc_smoothing_method_study.py` | 6 | Exploratory smoothing-method study | `hgg-smoothing-study` | Name says ad hoc; standalone large script | Maybe | CLI | No | Yes, experimental | Related to smoothing policy | Keep under `experimental/` unless formalized. |

### Modular Orchestration Layer

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/modular_pipeline/components.py` | 2, 4 | Component registry, masks, context dependencies, manifest writing | `modular-pipeline` CLI | Defines 16 ordered components | Yes | Python/manifest | Yes, narrow | Carefully | Mirrors `analysis.pipeline.run_all_stages` | Treat as workflow contract and orchestration service. |
| `modular-pipeline/modular_pipeline/cli.py` | 4 | `modular-pipeline` command line interface | Users/agents | Lists, inspects, runs components | Yes | CLI | Yes, narrow | With compatibility review | Overlaps `hgg-analysis` | Keep as preferred orchestration entrypoint. |
| `modular-pipeline/modular_pipeline/tracking.py` | 2, 4 | Artifact readiness/state inspection | CLI and components | Writes `modular_pipeline_state.json` | Yes | Output schema implicit | Yes, narrow | Carefully | No | Promote tracking/provenance service; version state schema. |
| `modular-pipeline/modular_pipeline/AGENTS.md` | 1, 2 | Agent operational policy | Future agents | Contains prime directive, component semantics, fit lessons | Yes | Human contract | No | Requires review | Overlaps docs/notes | Mine into physics policy and workflow contracts. |
| `modular-pipeline/modular_pipeline/README.md` | 2 | Local package docs | Humans | Documentation file | Yes | Human contract | No | Yes | Overlaps top README | Keep concise. |
| `modular-pipeline/modular_pipeline/docs/*.md` | 1, 2 | Operational, portability, reproduction, fit, troubleshooting, background-selection docs | Agents/humans | Document run contracts and statistical lessons | Yes | Human contract | No | With review | Some overlap with AGENTS/README | Promote policy/contract content to formal layer. |

### Tests and Scripts

| Path | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `modular-pipeline/tests/test_modular_pipeline.py` | 2, 4 | Unit tests for masks, components, readiness | CI/local pytest | Existing tests | Yes | Test contract | Yes | Yes | No | Expand to cover service ledger guarantees. |
| `modular-pipeline/tests/test_official_bkg_selector.py` | 2, 4 | Unit tests for HHXYY background selector bridge | CI/local pytest | Existing tests | Yes | Test contract | Yes | Yes | No | Keep as compatibility tests for selector bridge. |
| `modular-pipeline/scripts/bootstrap_hhxyy_quickfit.sh` | 3 | HHXYY quickFit setup helper | Humans/agents | Script under `scripts/` | Maybe | Shell interface | No | Carefully | Env docs overlap | Keep as setup utility; document optional dependency. |
| `modular-pipeline/scripts/compare_hhxyy_fit_parameters.py` | 3 | Compare local and HHXYY fit parameters | Debug/reproduction | Standalone comparison script | Maybe | CLI-ish | No | Yes | Related to verifier need | Candidate verifier after interface cleanup. |
| `modular-pipeline/scripts/tabulate_local_hhxyy_fit_parameters.py` | 3 | Tabulate fit parameters | Debug/reproduction | Standalone script | Maybe | CLI-ish | No | Yes | Related to comparison script | Candidate verifier/report utility. |

### Generated Run Output Bundle

| Path Pattern | Classification | Purpose | Current Consumers | Evidence | Reusable | Stable Interface | Tested | Safe To Modify | Duplicates | Recommended Action |
|---|---:|---|---|---|---|---|---|---|---|---|
| `outputs_modular_hgg_20260531T213216Z/modular_pipeline_manifest.json` | 8, 2 | Run manifest | Inspectors/humans | Written by modular runner | Yes as provenance | Implicit | Yes, narrow | No | No | Version schema in future runs. |
| `outputs_modular_hgg_20260531T213216Z/modular_pipeline_state.json` | 8, 2 | Component artifact readiness state | Inspectors/humans | Written by tracking | Yes as provenance | Implicit | Yes, narrow | No | No | Version schema and preserve per run. |
| `outputs_modular_hgg_20260531T213216Z/summary.normalized.json` | 8 | Normalized config snapshot | Downstream output readers | Generated from summary/defaults | Yes for provenance | Implicit | No | No | Duplicates config/defaults | Treat as immutable run snapshot. |
| `outputs_modular_hgg_20260531T213216Z/validation/*.json` | 8, 2 | Inventory, diagnostics, overlap policy | Reviewers/verifiers | Generated validation outputs | Yes | Implicit | No | No | Derived from summary | Define verifier contract. |
| `outputs_modular_hgg_20260531T213216Z/samples.*.json`, `normalization/*.json` | 8 | Sample registry/classification/normalization artifacts | Fit, reports, review | Generated by sample services | Yes as provenance | Implicit | No | No | Derived from input samples | Version schema. |
| `outputs_modular_hgg_20260531T213216Z/cache/*.npz` | 8 | Processed event caches | Fit/plots/debug | Generated per sample | Maybe | Numpy array keys implicit | No | No | Derived from ROOT inputs | Do not edit; document cache schema. |
| `outputs_modular_hgg_20260531T213216Z/hists/*.json` | 8 | Histogram/template outputs | Fit/report | Generated by histmaker | Yes | Implicit | No | No | Derived from processed samples | Version schema and verifier. |
| `outputs_modular_hgg_20260531T213216Z/fit/**` | 8 | RooFit/HHXYY fit, workspace, significance artifacts | Review/report | Generated statistical outputs | Yes as benchmark | Implicit | Partly | No | Derived from analysis | Preserve immutable; use as baseline reference. |
| `outputs_modular_hgg_20260531T213216Z/report/*.json` | 8, 2 | Cutflow, yields, blinding, execution and review artifacts | Humans/verifiers | Generated report/contract bundle | Yes | Implicit | No | No | Derived | Define contract schemas. |

## Missing Capabilities

| Capability | Classification | Evidence | Recommended Action |
|---|---:|---|---|
| Standalone `physics_policy/PHYSICS_POLICY.md` and machine-readable invariants | 9 | Physics rules are embedded in summary JSON, docs, and Python defaults | Create policy artifacts before allowing automated optimization to change semantics. |
| Formal workflow-contract YAML schemas | 9 | Artifacts are produced, but schemas/pass criteria are implicit | Create `workflow_contracts/analysis_contract.yaml` and verifier contract. |
| Versioned service interfaces | 9 | CLI names exist; per-service inputs/outputs are not ledgered | Use `ledger/EXECUTABLE_SERVICES.yaml` as initial source of truth. |
| Output schema versioning | 9 | JSON artifacts have implicit keys | Add schema_version/provenance fields to future outputs. |
| Configuration-only optimization config area | 9 | Tunables live inside `DEFAULT_RUNTIME` in Python | Create `configs/baseline/` and `configs/optimization/` after policy review. |
| Dedicated verifier scripts separate from action services | 9 | Some verification artifacts are written by action/report code | Add `verifiers/` with independent checks. |
| Compatibility/migration policy | 9 | No explicit change classes before this audit | Use `ledger/COMPATIBILITY.md`. |
| Comprehensive tests for event weighting, selections, fit compatibility, blinding | 9 | Existing tests cover orchestration and official selector only | Add focused tests before service changes. |

## Ambiguous Items Requiring Human Clarification

| Item | Classification | Why Ambiguous | Question |
|---|---:|---|---|
| `analysis/Higgs-to-diphoton.json` vs `analysis/analysis.summary.json` | 10 | They appear duplicate, but skill docs reference `Higgs-to-diphoton.json` while README references `analysis.summary.json` | Which path is canonical for future configs? |
| Checked-in `analysis/regions.yaml` | 10 | It is generated by runtime contract but exists in source tree | Should it be source config or generated output only? |
| `DEFAULT_RUNTIME` in `load_summary.py` | 10 | Contains both physics policy and tunable optimization parameters | Which values are immutable physics policy versus tunable config? |
| Systematics mode `placeholder` | 10 | Systematics artifacts exist but may not be final scientific treatment | What systematic model is acceptable for optimization comparisons? |
| Observed significance unblinding flag | 1, 10 | Code supports unblinding with CLI flag; default blocks observed significance | Who can approve observed significance runs and under what conditions? |
| `.TinyTeX/` under workspace | 10 | Toolchain appears local/generated | Should local toolchains be kept outside analysis workspace? |

## Safety Assessment

Most code under `analysis/` is scientifically sensitive. Automated agents should default to configuration-only changes until the physics-policy and workflow-contract layers are explicit. The riskiest files to edit without review are:

- `analysis/hists/histmaker.py`
- `analysis/objects/photons.py`
- `analysis/objects/jets.py`
- `analysis/selections/engine.py`
- `analysis/samples/registry.py`
- `analysis/stats/fit.py`
- `analysis/stats/significance.py`
- `analysis/stats/models.py`
- `analysis/config/load_summary.py`

The least risky near-term edits are additive governance files, schema/verifier additions, and tests that lock current behavior.
