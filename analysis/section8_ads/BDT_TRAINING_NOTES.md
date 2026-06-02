# Section 8 ADS BDT Training Notes For Future Agents

This note records the current working interpretation of the supplemental BDT-training information for the Section 8 H -> gamma gamma reconstruction categories. It is meant for future coding agents implementing the supplemental BDTs. Treat it as local analysis guidance, not as an official ATLAS model description.

## Scope

The Section 8 pipeline currently has placeholder support for training three supplemental BDTs:

- `BDT_ttH`
- `BDT_VH`
- `BDT_VBF`

The first full run did not train usable models because the training samples were too sparse or class-imbalanced. The next implementation pass should focus on constructing the correct BDT-specific training samples before changing model technology or optimizing score boundaries.

## Common Diphoton Preselection

All BDT training samples should start from events satisfying the common diphoton selection used before Section 8 categorization:

- At least two photon candidates.
- Photon transverse energy greater than 25 GeV.
- Photon `abs(eta) < 2.37`.
- Exclude photons in the calorimeter transition region, `1.37 < abs(eta) < 1.52`.
- Loose photon identification for the initial candidate selection.
- Select the two highest-ET photons as the diphoton candidate.
- Leading photon `ET / m_gammagamma > 0.35`.
- Subleading photon `ET / m_gammagamma > 0.25`.
- Tight photon identification for the nominal signal-region photons.
- Photon isolation for the nominal signal-region photons.
- Diphoton mass range `105 < m_gammagamma < 160 GeV`.

For jet-based category definitions and BDT variables, use jets with `pT > 30 GeV` unless a category explicitly states otherwise.

## Control-Sample Convention

For the data-driven reducible backgrounds, keep all non-photon selections identical to the relevant BDT application preselection, but reverse the photon identification or isolation requirement for at least one of the two selected photons.

In practical terms, agents should build anti-ID/anti-isolation data control samples from the same diphoton candidate structure:

- Select the two photon candidates by the same kinematic and ranking rules.
- Keep the same mass window and BDT-specific jet/lepton preselection.
- Require at least one of the two selected photons to fail either the photon ID requirement or the photon isolation requirement.
- Keep this control sample separate from the nominal tight-and-isolated diphoton sample.

This convention is the intended meaning here for the multijet, gamma-plus-jet, and dijet data-control samples. The names describe different reducible-background compositions, but the implementation handle is the same reversal of photon ID or isolation while preserving the rest of the selection.

If the ntuples expose separate loose/tight ID and isolation flags, implement the reversal explicitly from those flags. If only precomputed nominal photon flags are available, record the limitation in the classifier training report and implementation decisions.

## `BDT_ttH`

Purpose: identify hadronic `ttH` signal against gluon-fusion Higgs and reducible multijet-like backgrounds.

Training signal:

- Simulated `ttH` events.

Training backgrounds:

- Simulated `ggH` events.
- Data-driven multijet control sample.

Application and training preselection:

- Common diphoton preselection.
- No prompt leptons.
- At least three jets.
- At least one b-tagged jet.

Data-driven multijet control sample:

- Same selections as above, except at least one of the two selected photons may fail photon identification or photon isolation.
- More precisely: keep the diphoton and hadronic `ttH` preselection otherwise identical, but require at least one selected photon to fail either ID or isolation.

Features:

- `H_T`: scalar sum of selected-jet transverse momenta.
- `m_all_jets`: invariant mass of the system formed from all selected jets.
- `N_jets`: total selected-jet multiplicity.
- `N_central_jets`: selected jets with `abs(eta) < 2.5`.
- `N_btag`: selected b-tagged jet multiplicity.

Known implementation caution:

- A previous full run produced many `ttH` rows but very few background rows because the data-driven multijet control sample had not yet been implemented. Do not treat that result as evidence that the BDT is impossible to train.

## `BDT_VH`

Purpose: identify hadronic vector-boson associated Higgs production.

Training signal:

- Simulated `VH` events, including available `WH`, `ZH`, and `ggZH` samples.

Training backgrounds:

- Simulated Higgs events from all production modes other than `VH`.
- Simulated diphoton events.
- Gamma-plus-jet data control samples.
- Dijet data control samples.

Application and training preselection:

- Common diphoton preselection.
- At least two jets.
- Use the two leading jets for dijet quantities.
- `60 < m_jj < 120 GeV`.

Data-driven gamma-plus-jet and dijet control samples:

- Same VBF-style control-sample convention: keep the `VH` hadronic preselection identical, but require at least one selected photon to fail ID or isolation.
- Preserve these rows as data-driven reducible-background rows for training.

Features:

- `m_jj`: dijet invariant mass.
- `pTt_gammagamma`: diphoton momentum component transverse to the diphoton thrust axis in the transverse plane.
- `delta_y_gammagamma_jj`: rapidity difference between the diphoton and dijet systems.
- `cos_theta_star_gammagamma_jj`: Collins-Soper-frame angle variable used for the combined diphoton-dijet system.

Known implementation caution:

- A previous full run used mostly `VH` signal and prompt diphoton MC, without gamma-plus-jet or dijet data controls. That is not the intended final training definition and produced an unhealthy class balance.

## `BDT_VBF`

Purpose: identify VBF Higgs production against gluon-fusion Higgs and nonresonant diphoton-like backgrounds.

Training signal:

- Simulated `VBF` Higgs events.

Training backgrounds:

- Simulated `ggH` events.
- Simulated diphoton events.
- Gamma-plus-jet data control samples.
- Dijet data control samples.

Application and training preselection:

- Common diphoton preselection.
- At least two hadronic jets.
- Use the two leading jets.
- `abs(delta_eta_jj) > 2`.
- Diphoton centrality relative to the two leading jets less than 5.
- Split the selected sample into two regions before applying or optimizing BDT boundaries:
  - low-`pT_Hjj`: `pT_Hjj < 25 GeV`
  - high-`pT_Hjj`: `pT_Hjj > 25 GeV`

Data-driven gamma-plus-jet and dijet control samples:

- Keep the VBF preselection identical, but require at least one of the two selected photons to fail photon ID or isolation.
- This is expected to yield substantially more background statistics than the nominal tight-and-isolated diphoton sample.

Features:

- `m_jj`: invariant mass of the two leading jets.
- `abs(delta_eta_jj)`: absolute pseudorapidity separation of the two leading jets.
- `pTt_gammagamma`: diphoton transverse-thrust momentum component.
- `abs(delta_phi_gammagamma_jj)`: absolute azimuthal separation between diphoton and dijet systems.
- Capping/binning rule for `abs(delta_phi_gammagamma_jj)`: events above `2.94` should not use additional shape information; implement this as a capped value or equivalent final-bin treatment.
- `deltaR_min_gamma_j`: minimum angular separation between either selected photon and either leading jet.
- `VBF_centrality`: `abs(eta_gammagamma - 0.5 * (eta_j1 + eta_j2))`.

Known implementation caution:

- A previous full run produced no usable `VBF` signal training rows under the current mask. Future agents should inspect whether the mask is too restrictive, whether the `VBF` sample mapping is incomplete, or whether the existing baseline excluded the relevant rows before training.

## Training Statistics Audit Before Fitting

Before fitting any BDT, produce a training-sample audit table for each classifier. The audit should include:

- Number of candidate events before the BDT-specific preselection.
- Number after common diphoton preselection.
- Number after BDT-specific jet/lepton requirements.
- Number with all required feature values finite.
- Number of signal rows.
- Number of each background component.
- Weighted and unweighted yields.
- Data-control yield from the reversed-ID/isolation sample.
- Train/validation/test split counts for signal and each background component.

Do not proceed to boundary optimization unless the corresponding BDT has finite scores for both signal and background-like events.

## Recommended Implementation Order

1. Implement explicit BDT-specific masks for `BDT_ttH`, `BDT_VH`, and `BDT_VBF`.
2. Implement anti-ID/anti-isolation photon-control masks from available photon ID and isolation branches.
3. Add the data-driven control samples to the training-row builder.
4. Emit the training statistics audit before model fitting.
5. Only then fit BDTs and score events.
6. Run boundary optimization only after at least one classifier produces finite scores.

## Implemented CLI And Artifacts

The Section 8 CLI exposes the BDT workflow through:

```bash
hgg-section8 \
  --ads /path/to/atlas_hgg_36fb_section8_ads.json \
  --inputs /path/to/input-data \
  --outputs /path/to/outputs \
  --prepare-bdt-training \
  --train-bdts \
  --score-bdts \
  --optimize-boundaries
```

The BDT-specific outputs are:

- `bdt_training_sample_audit.json/.csv/.md`: per-classifier training counts by sample, process, photon region, finite-feature status, and deterministic split.
- `bdt_background_normalization.json/.csv/.md`: per-component normalization from the full mass range into the `123-127 GeV` peak window.
- `bdt_mass_correlation_report.json/.md`: feature-vs-`m_gammagamma` correlation checks for continuum/control backgrounds.
- `classifier_training_report.json/.md`: XGBoost model status, row counts, AUC metrics, and blocking reasons.
- `classifiers/<BDT>.pkl`: Python pickle used by the local scorer.
- `classifiers/<BDT>.json`: XGBoost-native model artifact.
- `classifiers/<BDT>_manifest.json`: feature list, backend parameters, hashes, and model paths.
- `bdt_boundary_optimization.json/.md`: score-boundary scan result or a precise blocked status.

The implementation uses XGBoost with fixed seed `20260601`. Training weights are normalized in two stages. First, each component is normalized to its expected contribution in the `123-127 GeV` peak window. Continuum backgrounds use the full `105-160 GeV` mass range with an exponential-transfer estimate where possible and a `1/11` fallback otherwise. Second, total signal and background training weights are balanced inside each BDT. Raw weighted yields are still preserved in the audit.

Future agents should use `--reuse-bdt-artifacts <outputs-dir>` only when the ADS hash, branch-map hash, feature config, sample list, backend version, and selection-policy hash are compatible with the run they want to score.

## Open Questions And Risks

- The exact separation between gamma-plus-jet and dijet data controls may not be recoverable from the current high-level ntuples. If not, combine them into one reducible-background control sample and document the approximation.
- The official ATLAS BDT implementation, hyperparameters, preprocessing, and event weighting are still not specified here.
- The current b-tagging proxy is `jet_btag_quantile >= 4`; verify this before final interpretation of top-associated categories.
- The current `MET_significance` approximation is `MET / sqrt(HT)`, unrelated to BDT training but relevant to neighboring category migration.
