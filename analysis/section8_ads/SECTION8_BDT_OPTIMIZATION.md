# Section 8 BDT Optimization Notes

This document explains how future agents should treat the Section 8 supplemental BDT artifacts during modular analysis optimization.

## Optimization Contract

The BDT workflow is an optimization-aware service extension, not an official ATLAS classifier reproduction. A BDT training run must preserve enough metadata for later agents to decide whether artifacts can be reused or must be regenerated.

Reusable artifacts must include:

- ADS configuration hash.
- Branch-map hash.
- Sample-list hash.
- Feature-manifest hash.
- Selection-policy hash.
- XGBoost backend name and version.
- XGBoost hyperparameters.
- Training sample audit hash.

If any of these change, retrain the affected BDTs before comparing category yields or boundary scans.

## Standard Run Shape

Use the Section 8 CLI in staged mode:

```bash
PYTHONPATH=/path/to/modular-pipeline \
python \
-m analysis.section8_ads.pipeline \
  --ads /path/to/atlas_hgg_36fb_section8_ads.json \
  --inputs /path/to/input-data \
  --outputs /path/to/outputs \
  --prepare-bdt-training \
  --train-bdts \
  --score-bdts \
  --optimize-boundaries
```

For a quick validation pass, add `--max-events N`. Do not interpret BDT performance or boundary optimization from a small-event smoke run.

## Continuous Optimization Loop

1. Start from a validated baseline output directory.
2. Modify only one BDT training policy or hyperparameter family at a time.
3. Rebuild the BDT training sample audit.
4. Train and score BDTs.
5. Compare training rows, peak-window normalization factors, mass-correlation warnings, AUC metrics, BDT score distributions, category migration, blocked categories, and expected-significance proxy.
6. Preserve the generated run metadata. The default location is output-local: `<outputs>/metadata/runs.jsonl` and `<outputs>/metadata/runs/<run_id>/observations.yaml`.

Stop and escalate if a model trains with suspiciously high AUC, if any classifier has missing signal or background, or if category migration improves the proxy metric while producing unstable or empty physics categories.

Use `--metadata-registry` and `--metadata-runs-dir` only when intentionally appending to a central campaign registry. Historical rows already committed in `optimization_infra/runs.jsonl` are immutable provenance records and should not be rewritten to remove original machine-specific paths.

## Background Normalization

Continuum backgrounds may use the full `105-160 GeV` training mass range for statistics, but their effective component weight is normalized to the expected `123-127 GeV` contribution. The implementation first tries an exponential transfer factor from the classifier-specific eligible rows. If that fit is not stable, it uses the documented fallback factor `1/11`.

Resonant Higgs backgrounds use their direct weighted yield in `123-127 GeV`, with a full-range fallback only when a small validation run has no peak-window rows.

After component normalization, the trainer balances the total signal and background class weight for XGBoost fitting while preserving relative background component fractions.

## Boundary Optimization Guardrails

Boundary optimization is meaningful only when nominal analysis events have finite BDT scores. If a classifier has no finite scores, keep the ADS seed boundaries and record the blocked status.

Optimized boundaries are local supplemental choices. Reports must always show the ADS seed boundaries next to optimized values.

## Observable Builder Compatibility

Section 8 nominal processing and BDT-training-candidate preparation now share `analysis.section8_ads.observables` for diphoton, object-derived, event-derived, and training-mask observables. This is an internal compatibility seam only. It does not change category routing, BDT training mechanics, boundary optimization, or the physics-policy meaning of any variable.
