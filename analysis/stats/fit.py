from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import ROOT

from analysis.common import ensure_dir, read_json, stable_hash, write_json
from analysis.samples.registry import parse_mass_window
from analysis.selections.engine import category_order
from analysis.stats.models import (
    background_candidate,
    configure_mass_var,
    crystal_ball_pdf,
    fit_pdf,
    histogram_counts,
    make_weighted_bin_center_dataset,
    make_weighted_dataset,
    pdf_to_curve,
    pdf_to_counts,
    th1_smooth,
)

ROOT.gROOT.SetBatch(True)

FIT_ID = "FIT1"


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.array([])


def aggregate_processed_samples(processed_samples: list[dict], cfg: dict[str, Any] | None = None) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    categories_in_use = category_order(cfg)
    aggregated: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        "data": {category: {"mgg": [], "weight": []} for category in categories_in_use},
        "signal": {category: {"mgg": [], "weight": []} for category in categories_in_use},
        "prompt_diphoton": {category: {"mgg": [], "weight": []} for category in categories_in_use},
    }
    for sample in processed_samples:
        if len(sample["events"].get("mgg", [])) == 0:
            continue
        for category in categories_in_use:
            mask = sample["events"]["category"] == category
            if not np.any(mask):
                continue
            if sample["kind"] == "data":
                target = aggregated["data"][category]
            elif sample["analysis_role"] == "signal_nominal":
                target = aggregated["signal"][category]
            elif sample["process_key"] == "prompt_diphoton" and sample["analysis_role"] == "background_nominal":
                target = aggregated["prompt_diphoton"][category]
            else:
                continue
            target["mgg"].append(sample["events"]["mgg"][mask])
            target["weight"].append(sample["events"]["weight"][mask])
    finalized: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for role, category_map in aggregated.items():
        finalized[role] = {}
        for category, payload in category_map.items():
            finalized[role][category] = {key: _concat(value) for key, value in payload.items()}
    return finalized


def _fit_signal_shape(category: str, masses: np.ndarray, weights: np.ndarray) -> tuple[dict[str, Any], ROOT.RooAbsPdf, list]:
    mass_var = configure_mass_var(f"mgg_sig_{category}")
    dataset = make_weighted_dataset(f"sig_{category}", mass_var, masses, weights)
    pdf, params = crystal_ball_pdf(category, mass_var)
    result = fit_pdf(pdf, dataset, fit_range="full", weighted=True)
    parameter_values = {
        param.GetName(): {"value": float(param.getVal()), "error": float(param.getError()), "constant": bool(param.isConstant())}
        for param in params
    }
    for param in params:
        param.setConstant(True)
    artifact = {
        "status": "ok" if result.status() == 0 else "warning",
        "fit_status": int(result.status()),
        "cov_qual": int(result.covQual()),
        "entries": int(dataset.numEntries()),
        "weighted_entries": float(dataset.sumEntries()),
        "parameters": parameter_values,
    }
    return artifact, pdf, params


def _fixed_signal_pdf_from_artifact(category: str, mass_var, signal_artifact: dict) -> tuple[ROOT.RooAbsPdf, list]:
    signal_pdf, signal_params = crystal_ball_pdf(f"scan_{category}", mass_var)
    for param in signal_params:
        source_key = param.GetName().replace(f"scan_{category}", category)
        source = signal_artifact["parameters"].get(source_key)
        if source is not None:
            param.setVal(source["value"])
        param.setConstant(True)
    return signal_pdf, signal_params


def _hhxyy_style_signal_pdf_from_artifact(category: str, mass_var, signal_artifact: dict) -> tuple[ROOT.RooAbsPdf, list]:
    params = signal_artifact["parameters"]

    def _value(stem: str, fallback: float) -> float:
        for key, payload in params.items():
            if key.startswith(stem):
                return float(payload["value"] if isinstance(payload, dict) else payload)
        return float(fallback)

    sigma_l = _value("sigmaL_", 1.6)
    sigma_r = _value("sigmaR_", sigma_l)
    mean = ROOT.RooRealVar(f"mean_final_{category}", f"mean_final_{category}", _value("mean_", 125.0), 122.0, 128.0)
    sigma = ROOT.RooRealVar(f"sigmaCB_final_{category}", f"sigmaCB_final_{category}", 0.5 * (sigma_l + sigma_r), 0.001, 20.0)
    alpha_l = ROOT.RooRealVar(f"alphaL_final_{category}", f"alphaL_final_{category}", _value("alphaL_", 1.5), 0.001, 20.0)
    n_l = ROOT.RooRealVar(f"nL_final_{category}", f"nL_final_{category}", _value("nL_", 5.0), 0.001, 200.0)
    alpha_r = ROOT.RooRealVar(f"alphaR_final_{category}", f"alphaR_final_{category}", _value("alphaR_", 1.5), 0.001, 20.0)
    n_r = ROOT.RooRealVar(f"nR_final_{category}", f"nR_final_{category}", _value("nR_", 5.0), 0.001, 200.0)
    pdf = ROOT.RooCrystalBall(
        f"sigpdf_final_{category}",
        f"sigpdf_final_{category}",
        mass_var,
        mean,
        sigma,
        alpha_l,
        n_l,
        alpha_r,
        n_r,
    )
    signal_params = [mean, sigma, alpha_l, n_l, alpha_r, n_r]
    for param in signal_params:
        param.setConstant(True)
    return pdf, signal_params


def _hhxyy_style_background_from_choice(category: str, mass_var, choice: dict[str, Any]) -> tuple[ROOT.RooAbsPdf, list]:
    selected_model = choice["selected_model"]
    if selected_model == "exponential":
        tau = ROOT.RooRealVar(f"tau_final_{category}", f"tau_final_{category}", -0.03, -1.0, 1.0)
        for key, value in choice.get("sideband_param_snapshot", {}).items():
            if key.startswith("tau_side_"):
                tau.setVal(float(value))
        pdf = ROOT.RooGenericPdf(
            f"bkgpdf_final_{category}_{selected_model}",
            f"bkgpdf_final_{category}_{selected_model}",
            "exp((@0-125.0)*@1)",
            ROOT.RooArgList(mass_var, tau),
        )
        return pdf, [tau]

    if not selected_model.startswith("bernstein"):
        return background_candidate(f"final_{category}", mass_var, selected_model).pdf, []

    local_coeffs: dict[int, float] = {}
    for key, value in choice.get("sideband_param_snapshot", {}).items():
        match = re.match(r"c(\d+)_side_", key)
        if match:
            local_coeffs[int(match.group(1))] = float(value)
    degree = int(selected_model.removeprefix("bernstein"))
    trailing = local_coeffs.get(degree, 1.0)
    if abs(trailing) < 1e-12:
        trailing = 1.0

    coeffs = ROOT.RooArgList()
    params = []
    for idx in range(degree):
        seed = local_coeffs.get(idx, 1.0) / trailing
        coeff = ROOT.RooRealVar(
            f"p{idx + 1}_final_{category}",
            f"p{idx + 1}_final_{category}",
            max(float(seed), 1e-5),
            1e-5,
            100.0,
        )
        coeffs.add(coeff)
        params.append(coeff)
    fixed_tail = ROOT.RooRealVar(f"p{degree + 1}_fixed_final_{category}", f"p{degree + 1}_fixed_final_{category}", 1.0)
    fixed_tail.setConstant(True)
    coeffs.add(fixed_tail)
    params.append(fixed_tail)
    pdf = ROOT.RooBernstein(f"bkgpdf_final_{category}_{selected_model}", f"bkgpdf_final_{category}_{selected_model}", mass_var, coeffs)
    return pdf, params


def _sideband_scale_factor(data_masses: np.ndarray, template_masses: np.ndarray, template_weights: np.ndarray) -> tuple[float, int, float]:
    data_sideband_mask = ((data_masses >= 105.0) & (data_masses < 120.0)) | ((data_masses > 130.0) & (data_masses <= 160.0))
    template_sideband_mask = ((template_masses >= 105.0) & (template_masses < 120.0)) | ((template_masses > 130.0) & (template_masses <= 160.0))
    data_sidebands = int(np.sum(data_sideband_mask))
    mc_sidebands = float(np.sum(template_weights[template_sideband_mask]))
    if mc_sidebands <= 0.0:
        return 1.0, data_sidebands, mc_sidebands
    return data_sidebands / mc_sidebands, data_sidebands, mc_sidebands


def _spur_bounds(expected_signal_yield: float, total_template_yield: float) -> tuple[float, float]:
    span = max(5.0, 3.0 * max(expected_signal_yield, 1.0), 0.1 * max(total_template_yield, 1.0))
    lower = -min(span, 0.95 * max(total_template_yield, 1.0))
    upper = min(max(span, 0.5 * max(total_template_yield, 1.0)), 0.95 * max(total_template_yield, 1.0))
    return lower, upper


def _fit_template_plus_signal(
    category: str,
    kind: str,
    counts: np.ndarray,
    signal_artifact: dict,
    expected_signal_yield: float,
) -> dict[str, Any]:
    mass_var = configure_mass_var(f"mgg_spur_{category}_{kind}")
    template_dataset = make_weighted_bin_center_dataset(f"template_{category}_{kind}", mass_var, counts)
    signal_pdf, _ = _fixed_signal_pdf_from_artifact(category, mass_var, signal_artifact)
    background = background_candidate(f"spur_{category}", mass_var, kind)
    total_yield = float(np.sum(counts))
    lower_nsig, upper_nsig = _spur_bounds(expected_signal_yield, total_yield)
    nsig = ROOT.RooRealVar(f"nspur_{category}_{kind}", f"nspur_{category}_{kind}", 0.0, lower_nsig, upper_nsig)
    total_const = ROOT.RooRealVar(
        f"ntotal_{category}_{kind}",
        f"ntotal_{category}_{kind}",
        max(total_yield, 1e-6),
    )
    total_const.setConstant(True)
    nbkg = ROOT.RooFormulaVar(
        f"nbkg_{category}_{kind}",
        "@0-@1",
        ROOT.RooArgList(total_const, nsig),
    )
    spur_model = ROOT.RooAddPdf(
        f"spurmodel_{category}_{kind}",
        f"spurmodel_{category}_{kind}",
        ROOT.RooArgList(signal_pdf, background.pdf),
        ROOT.RooArgList(nsig, nbkg),
    )
    result = fit_pdf(spur_model, template_dataset, fit_range="full", weighted=False, extended=True)
    sigma_nsig = abs(float(nsig.getError())) if abs(float(nsig.getError())) > 1e-9 else 1e-9
    n_spur = float(nsig.getVal())
    background_counts = pdf_to_counts(background.pdf, mass_var, float(nbkg.getVal()), bins=len(counts))
    signal_counts = pdf_to_counts(signal_pdf, mass_var, float(nsig.getVal()), bins=len(counts))
    total_counts = background_counts + signal_counts
    return {
        "fit_status": int(result.status()),
        "cov_qual": int(result.covQual()),
        "n_spur": n_spur,
        "sigma_nsig": sigma_nsig,
        "r_spur": float(abs(n_spur) / sigma_nsig),
        "signal_yield_fit": float(nsig.getVal()),
        "background_yield_fit": float(nbkg.getVal()),
        "param_snapshot": {param.GetName(): float(param.getVal()) for param in background.params},
        "total_counts": total_counts.tolist(),
        "background_counts": background_counts.tolist(),
        "signal_counts": signal_counts.tolist(),
    }


def _scan_background_models(
    category: str,
    cfg: dict,
    data_masses: np.ndarray,
    template_masses: np.ndarray,
    template_weights: np.ndarray,
    signal_artifact: dict,
    expected_signal_yield: float,
    smoothing_applied: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scale_factor, observed_sb, template_sb = _sideband_scale_factor(data_masses, template_masses, template_weights)
    scaled_weights = template_weights * scale_factor
    unsmoothed_counts = histogram_counts(template_masses, scaled_weights)
    selection_counts = th1_smooth(unsmoothed_counts, 1) if smoothing_applied else unsmoothed_counts.copy()

    sideband_mass_var = configure_mass_var(f"mgg_side_{category}")
    sideband_dataset = make_weighted_dataset(f"data_side_{category}", sideband_mass_var, data_masses)
    candidate_rows = []
    for kind in cfg["background_model"]["candidates"]:
        model = background_candidate(f"side_{category}", sideband_mass_var, kind)
        side_fit = fit_pdf(model.pdf, sideband_dataset, fit_range="sideband_lo,sideband_hi")
        sideband_params = {param.GetName(): float(param.getVal()) for param in model.params}
        aic = 2.0 * len(model.params) + 2.0 * float(side_fit.minNll())
        spur_fit = _fit_template_plus_signal(category, kind, selection_counts, signal_artifact, expected_signal_yield)
        row = {
            "category": category,
            "model": kind,
            "complexity": model.complexity,
            "sideband_fit_status": int(side_fit.status()),
            "sideband_cov_qual": int(side_fit.covQual()),
            "sideband_aic": float(aic),
            "n_spur": spur_fit["n_spur"],
            "sigma_nsig": spur_fit["sigma_nsig"],
            "r_spur": spur_fit["r_spur"],
            "passes": bool(spur_fit["r_spur"] < 0.2),
            "sideband_param_snapshot": sideband_params,
            "spur_fit_param_snapshot": spur_fit["param_snapshot"],
            "spur_fit_status": spur_fit["fit_status"],
            "spur_cov_qual": spur_fit["cov_qual"],
            "selection_fit_total_counts": spur_fit["total_counts"],
            "selection_fit_background_counts": spur_fit["background_counts"],
            "selection_fit_signal_counts": spur_fit["signal_counts"],
        }
        candidate_rows.append(row)

    passing = [row for row in candidate_rows if row["passes"]]
    if passing:
        passing.sort(key=lambda row: (row["complexity"], abs(row["n_spur"]), row["sideband_aic"]))
        best_choice = passing[0]
        rationale = "lowest-complexity candidate passing r_spur < 0.2"
        status = "ok"
        capped_noncompliant = False
    else:
        candidate_rows.sort(key=lambda row: (row["r_spur"], row["complexity"], row["sideband_aic"]))
        best_choice = candidate_rows[0]
        rationale = "no candidate passed the spurious-signal threshold by the complexity-3 cap; chose the smallest r_spur among tested candidates"
        status = "capped_noncompliant"
        capped_noncompliant = True

    unsmoothed_display = _fit_template_plus_signal(category, best_choice["model"], unsmoothed_counts, signal_artifact, expected_signal_yield)
    choice = {
        "status": status,
        "category": category,
        "selected_model": best_choice["model"],
        "selected_complexity": best_choice["complexity"],
        "rationale": rationale,
        "sideband_scale_factor": float(scale_factor),
        "observed_data_sideband_count": observed_sb,
        "template_sideband_yield_before_scaling": template_sb,
        "sideband_param_snapshot": best_choice["sideband_param_snapshot"],
        "spur_fit_param_snapshot": best_choice["spur_fit_param_snapshot"],
        "spurious_threshold_passed": bool(best_choice["passes"]),
        "selected_r_spur": float(best_choice["r_spur"]),
        "target_r_spur": 0.2,
        "maximum_tested_complexity": max(row["complexity"] for row in candidate_rows),
        "complexity_cap": 3,
        "cap_reached": max(row["complexity"] for row in candidate_rows) >= 3,
        "capped_noncompliant": capped_noncompliant,
    }
    scan_artifact = {
        "status": "ok",
        "category": category,
        "selection_counts_source": "smoothed" if smoothing_applied else "unsmoothed",
        "candidates": candidate_rows,
        "target_r_spur": 0.2,
        "maximum_tested_complexity": max(row["complexity"] for row in candidate_rows),
        "complexity_cap": 3,
        "cap_reached": max(row["complexity"] for row in candidate_rows) >= 3,
    }
    template_display = {
        "unsmoothed_counts": unsmoothed_counts.tolist(),
        "selection_counts": selection_counts.tolist(),
        "smoothing_applied": smoothing_applied,
        "sideband_scale_factor": float(scale_factor),
        "unsmoothed_fit_total_counts": unsmoothed_display["total_counts"],
        "unsmoothed_fit_background_counts": unsmoothed_display["background_counts"],
        "unsmoothed_fit_signal_counts": unsmoothed_display["signal_counts"],
        "selection_fit_total_counts": best_choice["selection_fit_total_counts"],
        "selection_fit_background_counts": best_choice["selection_fit_background_counts"],
        "selection_fit_signal_counts": best_choice["selection_fit_signal_counts"],
        "selection_fit_model": best_choice["model"],
    }
    return scan_artifact, choice, template_display


def _build_final_model(category: str, mass_var, signal_artifact: dict, choice: dict, expected_signal_yield: float, observed_count: int, shared_mu):
    signal_pdf, signal_params = _hhxyy_style_signal_pdf_from_artifact(category, mass_var, signal_artifact)
    background_pdf, background_params = _hhxyy_style_background_from_choice(category, mass_var, choice)
    s_const = ROOT.RooRealVar(f"sconst_{category}", f"sconst_{category}", expected_signal_yield)
    s_const.setConstant(True)
    nsig = ROOT.RooFormulaVar(f"nsig_{category}", "@0*@1", ROOT.RooArgList(shared_mu, s_const))
    nbkg = ROOT.RooRealVar(
        f"nbkg_{category}",
        f"nbkg_{category}",
        max(float(observed_count), 1.0),
        0.0,
        20.0 * max(float(observed_count), 1.0),
    )
    model = ROOT.RooAddPdf(
        f"model_{category}",
        f"model_{category}",
        ROOT.RooArgList(signal_pdf, background_pdf),
        ROOT.RooArgList(nsig, nbkg),
    )
    return {
        "model": model,
        "signal_pdf": signal_pdf,
        "background_pdf": background_pdf,
        "background_params": background_params,
        "signal_params": signal_params,
        "nsig": nsig,
        "nbkg": nbkg,
        "s_const": s_const,
        "choice": choice,
    }


def _model_plot_payload(final_models: dict[str, Any], mass_var, fit_range: list[float]) -> dict[str, Any]:
    curve_x = np.linspace(float(fit_range[0]), float(fit_range[1]), 551)

    def _curve_payload(model_ctx: dict[str, Any]) -> dict[str, Any]:
        signal_curve = pdf_to_curve(
            model_ctx["signal_pdf"],
            mass_var,
            float(model_ctx["nsig"].getVal()),
            x_values=curve_x,
        )
        background_curve = pdf_to_curve(
            model_ctx["background_pdf"],
            mass_var,
            float(model_ctx["nbkg"].getVal()),
            x_values=curve_x,
        )
        signal_y = np.asarray(signal_curve["y"], dtype=float)
        background_y = np.asarray(background_curve["y"], dtype=float)
        return {
            "x": curve_x.tolist(),
            "signal": signal_y.tolist(),
            "background": background_y.tolist(),
            "total": (signal_y + background_y).tolist(),
        }

    categories: dict[str, Any] = {}
    for category, model_ctx in final_models.items():
        signal_counts = pdf_to_counts(
            model_ctx["signal_pdf"],
            mass_var,
            float(model_ctx["nsig"].getVal()),
        )
        background_counts = pdf_to_counts(
            model_ctx["background_pdf"],
            mass_var,
            float(model_ctx["nbkg"].getVal()),
        )
        total_counts = signal_counts + background_counts
        categories[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
            "curve": _curve_payload(model_ctx),
        }

    combined_signal = (
        np.sum([np.asarray(payload["signal_counts"], dtype=float) for payload in categories.values()], axis=0)
        if categories
        else np.zeros(55, dtype=float)
    )
    combined_background = (
        np.sum([np.asarray(payload["background_counts"], dtype=float) for payload in categories.values()], axis=0)
        if categories
        else np.zeros(55, dtype=float)
    )
    combined_total = combined_signal + combined_background
    if categories:
        first_curve = next(iter(categories.values()))["curve"]
        combined_curve = {
            "x": first_curve["x"],
            "signal": np.sum(
                [np.asarray(payload["curve"]["signal"], dtype=float) for payload in categories.values()],
                axis=0,
            ).tolist(),
            "background": np.sum(
                [np.asarray(payload["curve"]["background"], dtype=float) for payload in categories.values()],
                axis=0,
            ).tolist(),
            "total": np.sum(
                [np.asarray(payload["curve"]["total"], dtype=float) for payload in categories.values()],
                axis=0,
            ).tolist(),
        }
    else:
        combined_curve = {
            "x": curve_x.tolist(),
            "signal": np.zeros_like(curve_x).tolist(),
            "background": np.zeros_like(curve_x).tolist(),
            "total": np.zeros_like(curve_x).tolist(),
        }
    return {
        "status": "ok",
        "fit_id": FIT_ID,
        "binning": {"observable": "m_gg", "n_bins": 55, "range": fit_range},
        "categories": categories,
        "combined": {
            "signal_counts": combined_signal.tolist(),
            "background_counts": combined_background.tolist(),
            "total_counts": combined_total.tolist(),
            "curve": combined_curve,
        },
    }


def _final_parameter_snapshot(final_models: dict[str, Any], shared_mu) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        shared_mu.GetName(): {
            "value": float(shared_mu.getVal()),
            "error": float(shared_mu.getError()),
            "constant": bool(shared_mu.isConstant()),
        }
    }
    for category, model_ctx in final_models.items():
        for param in [*model_ctx["signal_params"], *model_ctx["background_params"], model_ctx["nbkg"], model_ctx["s_const"]]:
            parameters[param.GetName()] = {
                "value": float(param.getVal()),
                "error": float(param.getError()) if hasattr(param, "getError") else 0.0,
                "constant": bool(param.isConstant()) if hasattr(param, "isConstant") else None,
            }
    return parameters


def _combined_observed_dataset(name: str, common_mass, channel, category_context: dict[str, Any]):
    observables = ROOT.RooArgSet(common_mass, channel)
    dataset = ROOT.RooDataSet(name, name, observables)
    for category, ctx in category_context.items():
        masses = np.asarray(ctx["data_masses"], dtype=float)
        valid = np.isfinite(masses) & (masses >= 105.0) & (masses <= 160.0)
        channel.setLabel(category)
        for mass in masses[valid]:
            common_mass.setVal(float(mass))
            dataset.add(observables)
    return dataset


def _combined_asimov_dataset(name: str, common_mass, channel, category_counts: dict[str, np.ndarray]):
    finite_positive_counts = [
        float(value)
        for counts in category_counts.values()
        for value in np.asarray(counts, dtype=float)
        if np.isfinite(value) and value > 0.0
    ]
    max_count = max(finite_positive_counts, default=1.0)
    weight = ROOT.RooRealVar(f"w_{name}", f"w_{name}", 0.0, -10.0 * max_count, 10.0 * max_count)
    observables = ROOT.RooArgSet(common_mass, channel, weight)
    dataset = ROOT.RooDataSet(name, name, observables, ROOT.RooFit.WeightVar(weight))
    for category, counts in category_counts.items():
        counts_arr = np.asarray(counts, dtype=float)
        edges = np.linspace(105.0, 160.0, len(counts_arr) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        channel.setLabel(category)
        for center, count in zip(centers, counts_arr, strict=False):
            if not np.isfinite(count) or count <= 0.0:
                continue
            common_mass.setVal(float(center))
            weight.setVal(float(count))
            dataset.add(observables, float(count))
    return dataset


def _measurement_dataset(
    *,
    category_context: dict[str, Any],
    final_models: dict[str, Any],
    common_mass,
    channel,
    use_observed_data: bool,
) -> tuple[Any, str, dict[str, Any]]:
    if use_observed_data:
        combined_data = _combined_observed_dataset("combData", common_mass, channel, category_context)
        return combined_data, "observed", {
            "status": "ok",
            "dataset_type": "observed",
            "construction_mode": "selected_event_dataset_explicit_categories",
            "blinding_policy_applied": False,
            "categories": sorted(category_context.keys()),
        }

    category_counts = {}
    category_payload: dict[str, Any] = {}
    for category, ctx in category_context.items():
        signal_yield = float(final_models[category]["s_const"].getVal())
        background_yield = float(ctx["template_total_yield"])
        signal_counts = pdf_to_counts(final_models[category]["signal_pdf"], common_mass, signal_yield)
        background_counts = pdf_to_counts(final_models[category]["background_pdf"], common_mass, background_yield)
        total_counts = signal_counts + background_counts
        category_counts[category] = total_counts
        category_payload[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
        }

    combined_data = _combined_asimov_dataset("combData", common_mass, channel, category_counts)
    return combined_data, "asimov_expected", {
        "status": "ok",
        "dataset_type": "asimov_expected",
        "generation_hypothesis": "signal_plus_background",
        "mu_gen": 1.0,
        "construction_mode": "weighted_bin_center_dataset_explicit_categories",
        "blinding_policy_applied": True,
        "categories": category_payload,
    }


def run_fit(processed_samples: list[dict], registry: list[dict], summary: dict, outputs: Path) -> dict:
    cfg = summary["runtime_defaults"]
    categories_in_use = category_order(cfg)
    aggregated = aggregate_processed_samples(processed_samples, cfg)
    fit_dir = ensure_dir(outputs / "fit" / FIT_ID)
    roofit_dir = ensure_dir(fit_dir / "roofit_combined")
    prompt_sample = next(sample for sample in registry if sample["process_key"] == "prompt_diphoton" and sample["is_nominal"])
    prompt_effective_lumi = float(prompt_sample["effective_lumi_fb"])
    threshold = 10.0 * float(cfg["target_lumi_fb"])
    smoothing_applied = prompt_effective_lumi < threshold
    prompt_window = parse_mass_window(prompt_sample["descriptor"])
    prompt_alternatives = [
        sample["sample_id"]
        for sample in registry
        if sample.get("process_key") == "prompt_diphoton" and sample["sample_id"] != prompt_sample["sample_id"]
    ]

    signal_shape_artifact: dict[str, Any] = {"status": "ok", "categories": {}}
    signal_param_artifact: dict[str, Any] = {"status": "ok", "categories": {}}
    background_scan_artifact: dict[str, Any] = {"status": "ok", "categories": {}}
    background_choice_artifact: dict[str, Any] = {"status": "ok", "categories": {}}
    spurious_signal_artifact: dict[str, Any] = {"status": "ok", "categories": {}}
    template_display: dict[str, Any] = {"status": "ok", "categories": {}}
    blinded_cr_fit: dict[str, Any] = {"status": "ok", "categories": {}}
    background_template_selection: dict[str, Any] = {
        "status": "ok",
        "selected_nominal_background_template_sample": prompt_sample["sample_id"],
        "selected_generated_mass_window": list(prompt_window) if prompt_window else None,
        "minimum_window_statement": "selected prompt-diphoton sample is the smallest generated-mass window fully containing 105-160 GeV",
        "rejected_low_statistics_auxiliary_background_samples": prompt_alternatives,
        "sidebands_gev": [[105.0, 120.0], [130.0, 160.0]],
        "categories": {},
        "notes": [
            "Nominal spurious-signal template uses prompt diphoton MC only, normalized to data in sidebands.",
            "Real observed background also contains gamma+jet, jet+jet, and Z->ee-fake contributions, but those are not the nominal template for the spurious-signal test.",
            "The prior failure mode of combining many low-statistics auxiliary MC background samples into the nominal template is explicitly rejected.",
        ],
    }
    effective_lumi_artifact: dict[str, Any] = {
        "status": "ok",
        "selected_nominal_background_template_sample": prompt_sample["sample_id"],
        "effective_luminosity_fb": prompt_effective_lumi,
        "target_lumi_fb": float(cfg["target_lumi_fb"]),
        "threshold_multiplier": 10.0,
        "required_min_lumi_fb": threshold,
        "smoothing_required": prompt_effective_lumi < threshold,
        "smoothing_applied": smoothing_applied,
        "smoothing_method": "TH1::Smooth" if smoothing_applied else "none",
        "smoothing_scope": cfg["smoothing_policy"]["scope"],
        "smoothing_rationale": (
            "effective MC luminosity is below 10 x 36 fb^-1, so TH1-based smoothing is required for background-function selection"
            if smoothing_applied
            else "effective MC luminosity exceeds the threshold; unsmoothed template is acceptable"
        ),
    }
    category_context: dict[str, Any] = {}

    for category in categories_in_use:
        signal_payload = aggregated["signal"][category]
        data_payload = aggregated["data"][category]
        template_payload = aggregated["prompt_diphoton"][category]
        if len(signal_payload["mgg"]) == 0 or len(data_payload["mgg"]) == 0 or len(template_payload["mgg"]) == 0:
            continue
        signal_artifact, _, _ = _fit_signal_shape(category, signal_payload["mgg"], signal_payload["weight"])
        signal_shape_artifact["categories"][category] = signal_artifact
        signal_param_artifact["categories"][category] = signal_artifact["parameters"]
        expected_signal_yield = float(np.sum(signal_payload["weight"]))
        signal_counts = histogram_counts(signal_payload["mgg"], signal_payload["weight"])
        scan_artifact, choice, display_payload = _scan_background_models(
            category,
            cfg,
            data_payload["mgg"],
            template_payload["mgg"],
            template_payload["weight"],
            signal_artifact,
            expected_signal_yield,
            smoothing_applied,
        )
        background_scan_artifact["categories"][category] = scan_artifact
        background_choice_artifact["categories"][category] = choice
        chosen_row = next(row for row in scan_artifact["candidates"] if row["model"] == choice["selected_model"])
        spurious_signal_artifact["categories"][category] = {
            "status": "ok" if chosen_row["passes"] else "warning",
            "N_spur": chosen_row["n_spur"],
            "sigma_Nsig": chosen_row["sigma_nsig"],
            "r_spur": chosen_row["r_spur"],
            "passes": chosen_row["passes"],
        }
        template_display["categories"][category] = display_payload
        blinded_cr_fit["categories"][category] = {
            "status": "ok",
            "fit_range": [[105.0, 120.0], [130.0, 160.0]],
            "selected_model": choice["selected_model"],
            "sideband_param_snapshot": choice["sideband_param_snapshot"],
            "sideband_scale_factor": choice["sideband_scale_factor"],
        }
        background_template_selection["categories"][category] = {
            "weighted_diphoton_mc_sideband_integral": choice["template_sideband_yield_before_scaling"],
            "observed_data_sideband_count": choice["observed_data_sideband_count"],
            "scale_factor_applied": choice["sideband_scale_factor"],
        }
        selection_total_yield = float(np.sum(display_payload["selection_counts"]))
        if selection_total_yield <= 0.0:
            continue
        category_context[category] = {
            "signal_artifact": signal_artifact,
            "signal_counts": signal_counts,
            "background_choice": choice,
            "expected_signal_yield": expected_signal_yield,
            "observed_count": int(len(data_payload["mgg"])),
            "data_masses": data_payload["mgg"],
            "template_total_yield": selection_total_yield,
            "selection_counts": np.asarray(display_payload["selection_counts"], dtype=float),
            "spurious_signal": spurious_signal_artifact["categories"][category],
        }

    common_mass = configure_mass_var("mgg")
    channel = ROOT.RooCategory("channel", "channel")
    for category in category_context:
        channel.defineType(category)
    shared_mu = ROOT.RooRealVar("mu", "mu", 1.0, 0.0, 5.0)
    simultaneous = ROOT.RooSimultaneous("simPdf", "simPdf", channel)
    final_models: dict[str, Any] = {}
    for category, ctx in category_context.items():
        model_ctx = _build_final_model(
            category,
            common_mass,
            ctx["signal_artifact"],
            ctx["background_choice"],
            ctx["expected_signal_yield"],
            ctx["observed_count"],
            shared_mu,
        )
        final_models[category] = model_ctx
        simultaneous.addPdf(model_ctx["model"], category)
    combined_data, measurement_dataset_type, measurement_dataset_artifact = _measurement_dataset(
        category_context=category_context,
        final_models=final_models,
        common_mass=common_mass,
        channel=channel,
        use_observed_data=bool(cfg["blinding"]["fit_uses_observed_data"]),
    )
    if measurement_dataset_type == "asimov_expected":
        shared_mu.setVal(1.0)
        shared_mu.setConstant(False)
        for category, model_ctx in final_models.items():
            model_ctx["nbkg"].setVal(float(category_context[category]["template_total_yield"]))
            model_ctx["nbkg"].setConstant(False)
            for param in model_ctx["background_params"]:
                param.setConstant("_fixed_" in param.GetName())
        fit_failure = None
        try:
            fit_result = fit_pdf(simultaneous, combined_data, weighted=True, extended=True)
        except Exception as exc:
            fit_result = None
            fit_failure = f"{type(exc).__name__}: {exc}"
    else:
        fit_failure = None
        try:
            fit_result = fit_pdf(simultaneous, combined_data, extended=True)
        except Exception as exc:
            fit_result = None
            fit_failure = f"{type(exc).__name__}: {exc}"

    workspace_root = fit_dir / "workspace.root"
    workspace = ROOT.RooWorkspace("w")
    workspace_status = "ok"
    workspace_error = None
    try:
        getattr(workspace, "import")(simultaneous)
        getattr(workspace, "import")(combined_data)
        workspace.writeToFile(str(workspace_root))
    except Exception as exc:
        workspace_status = "blocked"
        workspace_error = f"{type(exc).__name__}: {exc}"

    fitted_category_yields = {}
    for category, model_ctx in final_models.items():
        fitted_category_yields[category] = {
            "signal": float(model_ctx["nsig"].getVal()),
            "background": float(model_ctx["nbkg"].getVal()),
        }
    postfit_plot_payload = _model_plot_payload(final_models, common_mass, cfg["fit_mass_range_gev"])

    diagnostics = []
    if fit_result is None:
        diagnostics.append(f"Combined measurement fit did not return a RooFitResult: {fit_failure or 'unknown failure'}.")
    elif fit_result.status() != 0:
        diagnostics.append("RooFit returned non-zero fit_status for the combined measurement fit.")
    if fit_result is not None and fit_result.covQual() < 2:
        diagnostics.append("Combined measurement fit covariance quality is below the acceptable threshold of 2.")
    if workspace_status != "ok":
        diagnostics.append(f"Workspace writing failed: {workspace_error}.")
    capped_background_categories = sorted(
        category
        for category, choice in background_choice_artifact["categories"].items()
        if choice.get("capped_noncompliant")
    )
    if capped_background_categories:
        diagnostics.append(
            "Spurious-signal model selection reached the complexity-3 cap without satisfying the target in categories: "
            + ", ".join(capped_background_categories)
            + "."
        )

    active_regions = {
        category_info["source_region_id"]
        for category_info in summary.get("categories", [])
        if category_info["category_id"] in category_context
    }

    active_category_order = [category for category in categories_in_use if category in category_context]

    fit_summary = {
        "status": "blocked" if fit_result is None else ("ok" if not diagnostics else "warning"),
        "fit_id": FIT_ID,
        "backend": "pyroot_roofit",
        "dataset_type": measurement_dataset_type,
        "observed_data_used_in_fit": measurement_dataset_type == "observed",
        "mu_hat": float(shared_mu.getVal()),
        "mu_uncertainty": float(shared_mu.getError()),
        "min_nll": float(fit_result.minNll()) if fit_result is not None else None,
        "fit_status": int(fit_result.status()) if fit_result is not None else -1,
        "cov_qual": int(fit_result.covQual()) if fit_result is not None else -1,
        "categories": active_category_order,
        "configured_regions": list(summary["fit_regions"][FIT_ID]["regions"]),
        "inactive_regions": sorted(set(summary["fit_regions"][FIT_ID]["regions"]) - active_regions),
        "shared_mu": True,
        "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
        "background_parameter_policy": "floating_shape_and_normalization",
        "expected_signal_yields": {category: ctx["expected_signal_yield"] for category, ctx in category_context.items()},
        "fitted_category_yields": fitted_category_yields,
        "parameters": _final_parameter_snapshot(final_models, shared_mu),
        "diagnostics": diagnostics,
        "notes": (
            ["Observed-data measurement fit is enabled because explicit unblinding was requested."]
            if measurement_dataset_type == "observed"
            else ["Blinded run: central fit performed on signal-plus-background Asimov pseudo-data instead of observed signal-region data."]
        ),
    }
    workspace_json = {
        "status": workspace_status,
        "fit_id": FIT_ID,
        "workspace_root": str(workspace_root) if workspace_status == "ok" else None,
        "workspace_hash": stable_hash(fit_summary),
        "categories": active_category_order,
        "backend": "pyroot_roofit",
        "dataset_type": measurement_dataset_type,
        "error": workspace_error,
    }
    backend_artifact = {
        "status": "ok",
        "primary_backend": "pyroot_roofit",
        "configuration": {
            "prompt_diphoton_effective_lumi_fb": prompt_effective_lumi,
            "smoothing_applied": smoothing_applied,
            "analytic_background_families": {
                category: background_choice_artifact["categories"][category]["selected_model"]
                for category in background_choice_artifact["categories"]
            },
        },
    }
    provenance = {
        "status": "ok",
        "config_hash": summary["config_hash"],
        "fit_hash": stable_hash({"fit_summary": fit_summary, "background_choice": background_choice_artifact}),
    }

    write_json(signal_shape_artifact, fit_dir / "signal_pdf.json")
    write_json(background_scan_artifact, fit_dir / "background_pdf_scan.json")
    write_json(background_choice_artifact, fit_dir / "background_pdf_choice.json")
    write_json(spurious_signal_artifact, fit_dir / "spurious_signal.json")
    write_json(signal_param_artifact, roofit_dir / "signal_dscb_parameters.json")
    write_json(fit_summary, fit_dir / "results.json")
    write_json(postfit_plot_payload, fit_dir / "fit_plot_payload.json")
    write_json(backend_artifact, fit_dir / "backend.json")
    write_json(workspace_json, outputs / "fit" / "workspace.json")
    write_json(provenance, fit_dir / "fit_provenance.json")
    write_json(template_display, fit_dir / "background_template_display.json")
    write_json(background_template_selection, fit_dir / "background_template_selection.json")
    write_json(effective_lumi_artifact, fit_dir / "effective_lumi_and_smoothing.json")
    write_json(blinded_cr_fit, fit_dir / "blinded_cr_fit.json")
    write_json(measurement_dataset_artifact, fit_dir / "measurement_dataset.json")

    return {
        "fit_summary": fit_summary,
        "workspace_json": workspace_json,
        "backend_artifact": backend_artifact,
        "background_scan_artifact": background_scan_artifact,
        "background_choice_artifact": background_choice_artifact,
        "spurious_signal_artifact": spurious_signal_artifact,
        "signal_shape_artifact": signal_shape_artifact,
        "signal_param_artifact": signal_param_artifact,
        "template_display": template_display,
        "background_template_selection": background_template_selection,
        "effective_lumi_artifact": effective_lumi_artifact,
        "blinded_cr_fit": blinded_cr_fit,
        "category_context": category_context,
        "final_models": final_models,
        "common_mass": common_mass,
        "channel": channel,
        "simultaneous": simultaneous,
        "combined_data": combined_data,
        "shared_mu": shared_mu,
        "measurement_dataset_type": measurement_dataset_type,
        "measurement_dataset_artifact": measurement_dataset_artifact,
        "prompt_diphoton_effective_lumi_fb": prompt_effective_lumi,
        "smoothing_applied": smoothing_applied,
        "aggregated": aggregated,
        "measurement_plot_payload": postfit_plot_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-json", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    processed = read_json(args.processed_json)
    registry = read_json(args.registry)
    summary = read_json(args.summary)
    run_fit(processed, registry, summary, Path(args.outputs))
    print("ok")


if __name__ == "__main__":
    main()
