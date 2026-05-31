# Official HHXYY Background Selection

The modular pipeline has a local background-selection path and an optional
official HHXYY cross-check path. The official path is implemented as a sidecar
tool, so it does not overwrite the current local selector or its artifacts.

## What The Official Procedure Does

HHXYY uses `fitting/bkgParamTool` and its `checkBkgModel` executable in
`spuriousSig` mode. For each category it:

1. reads a background template histogram,
2. reads the signal PDF from a RooWorkspace,
3. scans signal-plus-background fits over the configured Higgs-mass scan points,
4. records the maximum spurious-signal quantities,
5. accepts models using the official thresholds, and
6. ranks accepted candidates using the official sorting rule.

The default HHXYY candidate list is:

```text
ExpPoly2 ExpPoly3 Exponential Pow Bern3 Bern4 Bern5
```

The default thresholds mirrored by this repository are:

```text
Selection.MaxSignalOverError: 0.20
Selection.MaxSignalOverRef: 0.10
Selection.MaxOneSigmaSignalOverError: 0.20
Selection.MaxTwoSigmaSignalOverError: 0.20
Selection.MinChiSquarePvalue: 0.00
```

## Tool

The entry point is:

```bash
hgg-official-bkg-select
```

It has three modes.

### Parse Existing Official Results

Use this when another stage, HHXYY, or a previous run already produced
`cat*/results.txt` files:

```bash
hgg-official-bkg-select parse \
  --bkg-model-dir outputs_modular_full/fit/FIT1/hhxyy_workspace/bkg_model \
  --out outputs_modular_full/fit/FIT1/official_bkg_selection.json \
  --csv outputs_modular_full/fit/FIT1/official_bkg_selection.csv
```

This writes a standalone official-selection artifact. The local background
choice remains unchanged.

### Generate Official Configs

Use this when the inputs are available but the official configs have not been
materialized yet:

```bash
hgg-official-bkg-select make-configs \
  --out-dir outputs_modular_full/fit/FIT1/official_bkg_configs \
  --dataset-file outputs_modular_full/fit/FIT1/hhxyy_workspace/hists/yyjets_myy_categories.root \
  --signal-pdf-file outputs_modular_full/fit/FIT1/hhxyy_workspace/hists/signal.root \
  --signal-yields outputs_modular_full/fit/FIT1/hhxyy_workspace/hists/category_yield.yaml \
  --dataset-histogram-template 'category{cat}'
```

The category-yield file must contain either a top-level category mapping or a
`signal:` mapping, for example:

```yaml
signal:
  category_0: 66.96
  category_1: 235.86
```

### Run The Official Binary

Use this when HHXYY `checkBkgModel` is built and available:

```bash
hgg-official-bkg-select run \
  --check-bkg-model /path/to/hhxyy/fitting/bkgParamTool/build/checkBkgModel \
  --config-dir outputs_modular_full/fit/FIT1/official_bkg_configs \
  --out-dir outputs_modular_full/fit/FIT1/official_bkg_model
```

Then parse the new `official_bkg_model` directory with the `parse` command.

## Substitution Contract

The parser records both the official model name and any known local equivalent:

```text
Bern3 -> bernstein2
Bern4 -> bernstein3
Bern5 -> bernstein4
Exponential -> exponential
```

`ExpPoly2`, `ExpPoly3`, and `Pow` are valid HHXYY/XML candidates but do not have
a guaranteed local PyROOT equivalent in the current selector. Treat those as
HHXYY-backend substitutions unless a local implementation is explicitly added.

The substitution is deliberately not automatic. A downstream stage should read
`official_bkg_selection.json`, check `selected_hhxyy_xml_model` and
`selected_local_equivalent`, and then opt in to using that model.

## Artifacts

Recommended artifact names:

```text
fit/FIT1/official_bkg_selection.json
fit/FIT1/official_bkg_selection.csv
fit/FIT1/official_bkg_configs/spurious_cat*.config
fit/FIT1/official_bkg_model/cat*/results.txt
```

These names keep the official cross-check separate from the current local
background-selection artifacts, which makes iterative work and mid-workflow
resume safer.
