# Fit And Significance Notes

This note records the debugging lessons that matter for future agents.

## Parameter Policy

For the validated expected-fit/significance path:

- signal shape parameters are fixed from signal MC fits;
- background normalizations float;
- background shape parameters float;
- expected significance uses signal-plus-background Asimov pseudo-data;
- central expected significance comes from HHXYY quickFit when available.

Do not silently freeze `nbkg_*` or background shape parameters in an Asimov fit.

## Local RooFit Fixes That Matter

The local RooFit setup only stopped producing runaway yields after these fixes:

- use `RooDataHist` for combined binned Asimov data;
- remove explicit `Range("full")` from the simultaneous extended fit;
- retain fixed trailing Bernstein coefficients in Python lists;
- keep fixed trailing Bernstein coefficients constant while floating the other background parameters;
- use HHXYY-style local final signal parameterization with symmetric `sigmaCB`;
- map local Bernstein degree consistently with HHXYY, including the fixed tail coefficient.

The failure symptom before the fix was:

```text
mu -> 5.0
nbkg_* -> artificial upper bounds
fit_status = 0
cov_qual = 3
```

This is dangerous because RooFit can report a numerically successful fit while
the physics result is obviously wrong.

## Bernstein Mapping

HHXYY-style Bernstein models use a fixed trailing coefficient:

```text
bernstein2 -> {p1, p2, 1}
bernstein3 -> {p1, p2, p3, 1}
bernstein4 -> {p1, p2, p3, p4, 1}
```

The local PyROOT implementation must retain the fixed tail object.  Otherwise
RooFit can see a dangling object and either crash or fit nonsense.

## Validated Numbers

The validated unmasked modular output:

```text
outputs_modular_full_20260531T163358Z
```

matched:

```text
outputs_full_localmatch5_20260531T062318Z
```

for the checked local and HHXYY statistical outputs.

## What Not To Do

- Do not treat local `results.json` alone as the final expected-significance result.
- Do not use fixed-background Asimov formulas after the policy says background floats.
- Do not use a weighted bin-center `RooDataSet` for the combined Asimov fit.
- Do not compare NLL absolute values between local RooFit and HHXYY; compare POI, uncertainty, status/covQual, yields, and declared significance artifacts.
- Do not claim a masked run validates a stage that was dependency-skipped.

