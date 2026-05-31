#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CATEGORY_INDEX = {
    "central_high_ptt": 0,
    "central_low_ptt": 1,
    "rest_high_ptt": 2,
    "rest_low_ptt": 3,
    "two_jet_vbf_enriched": 4,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.12g}"


def rel_delta(new: float | None, ref: float | None) -> float | None:
    if new is None or ref is None or ref == 0.0:
        return None
    return (float(new) - float(ref)) / float(ref)


def quickfit_tree(path: Path) -> dict[str, float]:
    import ROOT

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open quickFit output: {path}")
    try:
        tree = root_file.Get("nllscan")
        if not tree or tree.GetEntries() < 1:
            raise RuntimeError(f"Missing nllscan in {path}")
        tree.GetEntry(0)
        values: dict[str, float] = {}
        for branch in tree.GetListOfBranches():
            name = branch.GetName()
            values[name] = float(getattr(tree, name))
        fit_result = root_file.Get("fitResult")
        if fit_result:
            values["_fitResult_status"] = float(fit_result.status())
            values["_fitResult_covQual"] = float(fit_result.covQual())
            poi = fit_result.floatParsFinal().find("mu")
            if poi:
                values["mu_err"] = float(poi.getError())
        return values
    finally:
        root_file.Close()


def selected_candidate(scan: dict[str, Any], category: str, selected_model: str) -> dict[str, Any] | None:
    candidates = scan.get("categories", {}).get(category, {}).get("candidates", [])
    for candidate in candidates:
        if candidate.get("model") == selected_model:
            return candidate
    return None


def local_coefficients(choice: dict[str, Any], category: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for name, value in choice.get("sideband_param_snapshot", {}).items():
        match = re.match(r"c(\d+)_side_", name)
        if match:
            values[int(match.group(1))] = float(value)
    return values


def local_exponential_slope(choice: dict[str, Any]) -> float | None:
    for name, value in choice.get("sideband_param_snapshot", {}).items():
        if name.startswith("tau_side_"):
            return float(value)
    return None


def quickfit_background_coeffs(values: dict[str, float], category: str) -> dict[int, float]:
    idx = CATEGORY_INDEX[category]
    coeffs: dict[int, float] = {}
    for name, value in values.items():
        match = re.fullmatch(r"p(\d+)_category_(\d+)", name)
        if not match:
            continue
        if int(match.group(2)) == idx:
            # p1 is the first Bernstein coefficient in the XML list.
            coeffs[int(match.group(1)) - 1] = float(value)
    return coeffs


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "section",
        "category",
        "parameter",
        "local_model",
        "hhxyy_model",
        "local_value",
        "hhxyy_free_value",
        "hhxyy_mu0_value",
        "free_minus_local",
        "free_rel_delta",
        "mu0_minus_local",
        "mu0_rel_delta",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-output", default="outputs_full_updatedstats_20260530T054703Z")
    parser.add_argument("--hhxyy-output", default="outputs_full_hhxyyplots_20260530T155036Z")
    parser.add_argument("--fit-id", default="FIT1")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    local_fit_dir = Path(args.local_output) / "fit" / args.fit_id
    hhxyy_fit_dir = Path(args.hhxyy_output) / "fit" / args.fit_id
    out_dir = Path(args.out_dir) if args.out_dir else hhxyy_fit_dir / "parameter_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_results = read_json(local_fit_dir / "results.json")
    local_sig = read_json(local_fit_dir / "significance_asimov.json")
    local_choice = read_json(local_fit_dir / "background_pdf_choice.json")
    local_scan = read_json(local_fit_dir / "background_pdf_scan.json")
    hhxyy_sig = read_json(hhxyy_fit_dir / "significance_asimov.json")
    manifest = read_json(hhxyy_fit_dir / "hhxyy_workspace" / "manifest.json")
    categories = manifest.get("categories") or local_results["categories"]

    qf_dir = hhxyy_fit_dir / "hhxyy_workspace" / "fitting" / "fit"
    qf_free = quickfit_tree(qf_dir / "bestfit_mu_asimovData_1.root")
    qf_mu0 = quickfit_tree(qf_dir / "bestfit_mu0_asimovData_1.root")

    rows: list[dict[str, Any]] = []

    summary_pairs = [
        ("mu_hat", local_sig.get("mu_hat"), qf_free.get("mu"), qf_mu0.get("mu"), "free fit POI"),
        ("mu_uncertainty", local_sig.get("mu_uncertainty"), qf_free.get("mu_err"), None, "local is fixed-background curvature; quickFit is fitResult error"),
        ("twice_nll_free", local_sig.get("twice_nll_free"), 2.0 * qf_free.get("nll", float("nan")), None, ""),
        ("twice_nll_mu0", local_sig.get("twice_nll_mu0"), None, 2.0 * qf_mu0.get("nll", float("nan")), ""),
        ("q0", local_sig.get("q0"), hhxyy_sig.get("q0"), None, ""),
        ("z_discovery", local_sig.get("z_discovery"), hhxyy_sig.get("z_discovery"), None, ""),
        ("fit_status", local_sig.get("fit_status_free"), qf_free.get("status"), qf_mu0.get("status"), "quickFit tree status"),
        ("cov_qual", local_sig.get("cov_qual_free"), qf_free.get("_fitResult_covQual"), qf_mu0.get("_fitResult_covQual"), "fitResult covariance quality"),
    ]
    for parameter, local_value, free_value, mu0_value, note in summary_pairs:
        rows.append(
            {
                "section": "summary",
                "category": "combined",
                "parameter": parameter,
                "local_value": local_value,
                "hhxyy_free_value": free_value,
                "hhxyy_mu0_value": mu0_value,
                "free_minus_local": (float(free_value) - float(local_value)) if free_value is not None and local_value is not None else None,
                "free_rel_delta": rel_delta(free_value, local_value),
                "mu0_minus_local": (float(mu0_value) - float(local_value)) if mu0_value is not None and local_value is not None else None,
                "mu0_rel_delta": rel_delta(mu0_value, local_value),
                "note": note,
            }
        )

    for category in categories:
        idx = CATEGORY_INDEX[category]
        choice = local_choice["categories"][category]
        local_model = choice["selected_model"]
        hhxyy_model = {
            "exponential": "Exponential",
            "bernstein2": "Bern2",
            "bernstein3": "Bern3",
            "bernstein4": "Bern4",
        }.get(local_model, local_model)

        local_signal = float(local_results["expected_signal_yields"][category])
        local_nbkg = float(local_results["fitted_category_yields"][category]["background"])
        local_total = local_nbkg + local_signal
        free_nbkg = qf_free.get(f"nbkg_category_{idx}")
        mu0_nbkg = qf_mu0.get(f"nbkg_category_{idx}")
        free_total = free_nbkg + qf_free.get("mu", 0.0) * local_signal if free_nbkg is not None else None
        mu0_total = mu0_nbkg
        free_signal_contribution = qf_free.get("mu", 0.0) * local_signal
        mu0_signal_contribution = 0.0

        normalization_rows = [
            ("signal_yield_nominal", local_signal, local_signal, local_signal, "fixed nominal signal yield before multiplying by mu"),
            ("signal_contribution_mu_times_yield", local_signal, free_signal_contribution, mu0_signal_contribution, "actual S contribution under each fitted/fixed mu"),
            ("background_normalization_nbkg", local_nbkg, free_nbkg, mu0_nbkg, "full-range background yield/normalization"),
            ("total_splusb_yield", local_total, free_total, mu0_total, "local fixed S+B total vs quickFit postfit totals"),
            ("sideband_scale_factor", choice.get("sideband_scale_factor"), None, None, "local sideband template scale factor, not a quickFit POI"),
            ("observed_sideband_count", choice.get("observed_data_sideband_count"), None, None, ""),
            ("template_sideband_before_scaling", choice.get("template_sideband_yield_before_scaling"), None, None, ""),
        ]
        for parameter, local_value, free_value, mu0_value, note in normalization_rows:
            rows.append(
                {
                    "section": "normalization",
                    "category": category,
                    "parameter": parameter,
                    "local_model": local_model,
                    "hhxyy_model": hhxyy_model,
                    "local_value": local_value,
                    "hhxyy_free_value": free_value,
                    "hhxyy_mu0_value": mu0_value,
                    "free_minus_local": (float(free_value) - float(local_value)) if free_value is not None and local_value is not None else None,
                    "free_rel_delta": rel_delta(free_value, local_value),
                    "mu0_minus_local": (float(mu0_value) - float(local_value)) if mu0_value is not None and local_value is not None else None,
                    "mu0_rel_delta": rel_delta(mu0_value, local_value),
                    "note": note,
                }
            )

        candidate = selected_candidate(local_scan, category, local_model) or {}
        for parameter in ("n_spur", "sigma_nsig", "r_spur", "sideband_fit_status", "sideband_cov_qual", "sideband_aic"):
            rows.append(
                {
                    "section": "local_background_selection",
                    "category": category,
                    "parameter": parameter,
                    "local_model": local_model,
                    "hhxyy_model": hhxyy_model,
                    "local_value": candidate.get(parameter),
                    "note": "local model-selection diagnostic; not present in quickFit output",
                }
            )

        if local_model == "exponential":
            local_slope = local_exponential_slope(choice)
            free_slope = qf_free.get(f"p1_category_{idx}")
            mu0_slope = qf_mu0.get(f"p1_category_{idx}")
            rows.append(
                {
                    "section": "background_shape",
                    "category": category,
                    "parameter": "exponential_slope",
                    "local_model": local_model,
                    "hhxyy_model": hhxyy_model,
                    "local_value": local_slope,
                    "hhxyy_free_value": free_slope,
                    "hhxyy_mu0_value": mu0_slope,
                    "free_minus_local": (free_slope - local_slope) if free_slope is not None and local_slope is not None else None,
                    "free_rel_delta": rel_delta(free_slope, local_slope),
                    "mu0_minus_local": (mu0_slope - local_slope) if mu0_slope is not None and local_slope is not None else None,
                    "mu0_rel_delta": rel_delta(mu0_slope, local_slope),
                    "note": "comparable slope; constant exponent offsets are absorbed by normalization",
                }
            )
        else:
            local_coeffs = local_coefficients(choice, category)
            free_coeffs = quickfit_background_coeffs(qf_free, category)
            mu0_coeffs = quickfit_background_coeffs(qf_mu0, category)
            local_degree = max(local_coeffs) if local_coeffs else -1
            hhxyy_degree = max(free_coeffs) + 1 if free_coeffs else local_degree
            local_last = local_coeffs.get(local_degree) if local_degree >= 0 else None
            max_degree = max(local_degree, hhxyy_degree)
            for coeff_idx in range(max_degree + 1):
                local_raw = local_coeffs.get(coeff_idx)
                local_norm = local_raw / local_last if local_raw is not None and local_last not in (None, 0.0) else None
                free_value = free_coeffs.get(coeff_idx, 1.0 if coeff_idx == hhxyy_degree else None)
                mu0_value = mu0_coeffs.get(coeff_idx, 1.0 if coeff_idx == hhxyy_degree else None)
                rows.append(
                    {
                        "section": "background_shape",
                        "category": category,
                        "parameter": f"bernstein_coeff_{coeff_idx}",
                        "local_model": f"{local_model} degree {local_degree}",
                        "hhxyy_model": f"{hhxyy_model} degree {hhxyy_degree}",
                        "local_value": local_norm,
                        "hhxyy_free_value": free_value,
                        "hhxyy_mu0_value": mu0_value,
                        "free_minus_local": (float(free_value) - float(local_norm)) if free_value is not None and local_norm is not None else None,
                        "free_rel_delta": rel_delta(free_value, local_norm),
                        "mu0_minus_local": (float(mu0_value) - float(local_norm)) if mu0_value is not None and local_norm is not None else None,
                        "mu0_rel_delta": rel_delta(mu0_value, local_norm),
                        "note": (
                            f"local raw coefficient={fmt(local_raw)}; values shown are local coefficients normalized to the last local coefficient"
                            if local_raw is not None
                            else "coefficient exists only in HHXYY XML basis"
                        ),
                    }
                )

    csv_path = out_dir / "hhxyy_vs_local_fit_parameters.csv"
    md_path = out_dir / "hhxyy_vs_local_fit_parameters.md"
    json_path = out_dir / "hhxyy_vs_local_fit_parameters.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    summary_rows = [row for row in rows if row["section"] == "summary"]
    norm_rows = [
        row
        for row in rows
        if row["section"] == "normalization"
        and row["parameter"]
        in {"signal_yield_nominal", "signal_contribution_mu_times_yield", "background_normalization_nbkg", "total_splusb_yield"}
    ]
    shape_rows = [row for row in rows if row["section"] == "background_shape"]

    notes = [
        "# HHXYY vs Local Fit Parameter Comparison",
        "",
        f"- Local/current source: `{local_fit_dir}`",
        f"- HHXYY quickFit source: `{hhxyy_fit_dir}`",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        "",
        "## Summary",
        markdown_table(summary_rows, ["parameter", "local_value", "hhxyy_free_value", "hhxyy_mu0_value", "free_minus_local", "note"]),
        "",
        "## Normalizations",
        markdown_table(norm_rows, ["category", "parameter", "local_model", "hhxyy_model", "local_value", "hhxyy_free_value", "hhxyy_mu0_value", "free_rel_delta", "note"]),
        "",
        "## Background Shape Parameters",
        "For Bernstein PDFs, local coefficients are shown normalized to the last local coefficient, matching the fixed trailing coefficient convention in the HHXYY XML basis.",
        "",
        markdown_table(shape_rows, ["category", "parameter", "local_model", "hhxyy_model", "local_value", "hhxyy_free_value", "hhxyy_mu0_value", "note"]),
        "",
    ]
    md_path.write_text("\n".join(notes))

    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
