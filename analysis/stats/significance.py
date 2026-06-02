from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import ROOT

from analysis.common import write_json
from analysis.stats.fit import FIT_ID
from analysis.stats.models import asimov_fixed_background_profile, pdf_to_curve, pdf_to_counts
from analysis.stats.hhxyy_fitting_backend import (
    is_atlas_env_available,
    prepare_hhxyy_workspace,
    run_hhxyy_significance,
)

ROOT.gROOT.SetBatch(True)


def _set_generation_snapshot(fit_context: dict) -> None:
    for category, model_ctx in fit_context["final_models"].items():
        choice = fit_context["category_context"][category]["background_choice"]
        for param in model_ctx["background_params"]:
            source_key = param.GetName().replace(f"final_{category}", f"side_{category}")
            if source_key in choice["sideband_param_snapshot"]:
                param.setVal(choice["sideband_param_snapshot"][source_key])
            param.setConstant("_fixed_" in param.GetName())
        model_ctx["nbkg"].setVal(float(fit_context["category_context"][category]["template_total_yield"]))
        model_ctx["nbkg"].setConstant(False)


def _snapshot_fit_state(fit_context: dict) -> dict[str, Any]:
    shared_mu = fit_context["shared_mu"]
    snapshot = {
        "shared_mu": {
            "value": float(shared_mu.getVal()),
            "error": float(shared_mu.getError()),
            "constant": bool(shared_mu.isConstant()),
        },
        "categories": {},
    }
    for category, model_ctx in fit_context["final_models"].items():
        snapshot["categories"][category] = {
            "nbkg": {
                "value": float(model_ctx["nbkg"].getVal()),
                "error": float(model_ctx["nbkg"].getError()),
                "constant": bool(model_ctx["nbkg"].isConstant()),
            },
            "background_params": {
                param.GetName(): {
                    "value": float(param.getVal()),
                    "error": float(param.getError()),
                    "constant": bool(param.isConstant()),
                }
                for param in model_ctx["background_params"]
            },
        }
    return snapshot


def _restore_fit_state(fit_context: dict, snapshot: dict[str, Any]) -> None:
    shared_mu = fit_context["shared_mu"]
    shared_mu.setVal(snapshot["shared_mu"]["value"])
    shared_mu.setConstant(snapshot["shared_mu"]["constant"])
    shared_mu.setError(snapshot["shared_mu"]["error"])

    for category, model_ctx in fit_context["final_models"].items():
        category_snapshot = snapshot["categories"].get(category, {})
        nbkg_snapshot = category_snapshot.get("nbkg")
        if nbkg_snapshot is not None:
            model_ctx["nbkg"].setVal(nbkg_snapshot["value"])
            model_ctx["nbkg"].setConstant(nbkg_snapshot["constant"])
            model_ctx["nbkg"].setError(nbkg_snapshot["error"])
        for param in model_ctx["background_params"]:
            param_snapshot = category_snapshot.get("background_params", {}).get(param.GetName())
            if param_snapshot is None:
                continue
            param.setVal(param_snapshot["value"])
            param.setConstant(param_snapshot["constant"])
            param.setError(param_snapshot["error"])


def _capture_model_counts(fit_context: dict) -> dict[str, dict[str, list[float]]]:
    category_payload: dict[str, dict[str, list[float]]] = {}
    fit_range = fit_context.get("fit_summary", {}).get("fit_range", [105.0, 160.0])
    curve_x = np.linspace(float(fit_range[0]), float(fit_range[1]), 551)
    for category, model_ctx in fit_context["final_models"].items():
        signal_counts = pdf_to_counts(
            model_ctx["signal_pdf"],
            fit_context["common_mass"],
            float(model_ctx["nsig"].getVal()),
        )
        background_counts = pdf_to_counts(
            model_ctx["background_pdf"],
            fit_context["common_mass"],
            float(model_ctx["nbkg"].getVal()),
        )
        total_counts = signal_counts + background_counts
        signal_curve = pdf_to_curve(
            model_ctx["signal_pdf"],
            fit_context["common_mass"],
            float(model_ctx["nsig"].getVal()),
            x_values=curve_x,
        )
        background_curve = pdf_to_curve(
            model_ctx["background_pdf"],
            fit_context["common_mass"],
            float(model_ctx["nbkg"].getVal()),
            x_values=curve_x,
        )
        signal_y = np.asarray(signal_curve["y"], dtype=float)
        background_y = np.asarray(background_curve["y"], dtype=float)
        category_payload[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
            "curve": {
                "x": curve_x.tolist(),
                "signal": signal_y.tolist(),
                "background": background_y.tolist(),
                "total": (signal_y + background_y).tolist(),
            },
        }
    return category_payload


def _combined_counts(category_payload: dict[str, dict[str, list[float]]]) -> dict[str, list[float]]:
    if not category_payload:
        zeros = [0.0] * 55
        return {
            "signal_counts": zeros,
            "background_counts": zeros,
            "total_counts": zeros,
        }
    signal = [
        sum(float(payload["signal_counts"][idx]) for payload in category_payload.values())
        for idx in range(55)
    ]
    background = [
        sum(float(payload["background_counts"][idx]) for payload in category_payload.values())
        for idx in range(55)
    ]
    total = [signal[idx] + background[idx] for idx in range(55)]
    combined = {
        "signal_counts": signal,
        "background_counts": background,
        "total_counts": total,
    }
    curve_payloads = [payload.get("curve") for payload in category_payload.values() if payload.get("curve")]
    if curve_payloads:
        combined["curve"] = {
            "x": curve_payloads[0]["x"],
            "signal": np.sum([np.asarray(curve["signal"], dtype=float) for curve in curve_payloads], axis=0).tolist(),
            "background": np.sum([np.asarray(curve["background"], dtype=float) for curve in curve_payloads], axis=0).tolist(),
            "total": np.sum([np.asarray(curve["total"], dtype=float) for curve in curve_payloads], axis=0).tolist(),
        }
    return combined


def _asimov_plot_payload(
    *,
    fit_context: dict,
    asimov_payload: dict[str, Any],
    fit_range: list[float],
    mu_hat: float,
    mu_uncertainty: float,
    free_result,
    mu0_result,
    free_fit_counts: dict[str, dict[str, list[float]]],
    mu0_fit_counts: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    categories = {}
    for category, payload in asimov_payload.items():
        categories[category] = {
            "asimov_counts": payload["total_counts"],
            "generation_signal_counts": payload["signal_counts"],
            "generation_background_counts": payload["background_counts"],
            "free_fit": free_fit_counts[category],
            "mu0_fit": mu0_fit_counts[category],
        }

    combined_asimov = {
        "signal_counts": [
            sum(float(payload["signal_counts"][idx]) for payload in asimov_payload.values())
            for idx in range(55)
        ] if asimov_payload else [0.0] * 55,
        "background_counts": [
            sum(float(payload["background_counts"][idx]) for payload in asimov_payload.values())
            for idx in range(55)
        ] if asimov_payload else [0.0] * 55,
    }
    combined_asimov["total_counts"] = [
        combined_asimov["signal_counts"][idx] + combined_asimov["background_counts"][idx]
        for idx in range(55)
    ]

    return {
        "status": "ok",
        "fit_id": FIT_ID,
        "dataset_type": "asimov",
        "generation_hypothesis": "signal_plus_background",
        "mu_gen": 1.0,
        "binning": {"observable": "m_gg", "n_bins": 55, "range": fit_range},
        "categories": categories,
        "combined": {
            "asimov_counts": combined_asimov["total_counts"],
            "generation_signal_counts": combined_asimov["signal_counts"],
            "generation_background_counts": combined_asimov["background_counts"],
            "free_fit": _combined_counts(free_fit_counts),
            "mu0_fit": _combined_counts(mu0_fit_counts),
        },
        "free_fit": {
            "mu_hat": mu_hat,
            "mu_uncertainty": mu_uncertainty,
            "fit_status": int(free_result.status()),
            "cov_qual": int(free_result.covQual()),
        },
        "mu0_fit": {
            "mu_fixed": 0.0,
            "fit_status": int(mu0_result.status()),
            "cov_qual": int(mu0_result.covQual()),
        },
    }


def _fixed_background_asimov_fallback(
    fit_context: dict,
    *,
    fit_range: list[float],
    signal_window: list[float],
    blinding: dict[str, Any],
    background_parameter_source: dict[str, Any],
    error: str,
    hhxyy_manifest: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _set_generation_snapshot(fit_context)
    common_mass = fit_context["common_mass"]
    category_payload: dict[str, Any] = {}
    free_fit_counts: dict[str, Any] = {}
    mu0_fit_counts: dict[str, Any] = {}
    for category, model_ctx in fit_context["final_models"].items():
        signal_yield = float(model_ctx["s_const"].getVal())
        background_yield = float(fit_context["category_context"][category]["template_total_yield"])
        signal_counts = pdf_to_counts(model_ctx["signal_pdf"], common_mass, signal_yield)
        background_counts = pdf_to_counts(model_ctx["background_pdf"], common_mass, background_yield)
        total_counts = signal_counts + background_counts
        category_payload[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
        }
        free_fit_counts[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
        }
        mu0_fit_counts[category] = {
            "signal_counts": [0.0] * len(background_counts),
            "background_counts": background_counts.tolist(),
            "total_counts": background_counts.tolist(),
        }

    combined_signal = (
        np.sum([np.asarray(payload["signal_counts"], dtype=float) for payload in category_payload.values()], axis=0)
        if category_payload
        else np.zeros(55, dtype=float)
    )
    combined_background = (
        np.sum([np.asarray(payload["background_counts"], dtype=float) for payload in category_payload.values()], axis=0)
        if category_payload
        else np.zeros(55, dtype=float)
    )
    profile = asimov_fixed_background_profile(combined_signal, combined_background, mu=1.0)
    diagnostics = [
        "Local RooFit profile-likelihood significance failed; wrote fixed-background Asimov fallback.",
        error,
    ]
    asimov_artifact = {
        "fit_id": FIT_ID,
        "status": "warning",
        "dataset_type": "asimov",
        "generation_hypothesis": "signal_plus_background",
        "mu_gen": 1.0,
        "backend": "analytic_fixed_background_fallback",
        "atlas_env_available": is_atlas_env_available(),
        "fit_driver": "fixed_background_asimov_proxy_after_roofit_failure",
        "poi_name": "signal_strength_mu",
        "mu_hat": profile["mu_hat"],
        "mu_uncertainty": profile["mu_uncertainty"],
        "twice_nll_mu0": profile["twice_nll_mu0"],
        "twice_nll_free": profile["twice_nll_free"],
        "q0": profile["q0"],
        "z_discovery": profile["z_discovery"],
        "fit_range": fit_range,
        "background_parameter_source": background_parameter_source,
        "asimov_source": "weighted_bin_center_dataset_explicit_categories_failed_to_profile",
        "observed_significance_allowed": bool(blinding["observed_significance_allowed"]),
        "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
        "background_parameter_policy": "fixed_shape_and_normalization_fallback",
        "asimov_profile_method": "fixed_background_counting_proxy",
        "fisher_information_mu": profile["fisher_information_mu"],
        "hhxyy_workspace_manifest": hhxyy_manifest,
        "categories": fit_context["fit_summary"]["categories"],
        "shared_mu": True,
        "fit_status_free": -1,
        "fit_status_mu0": -1,
        "cov_qual_free": -1,
        "cov_qual_mu0": -1,
        "diagnostics": diagnostics,
    }
    construction_artifact = {
        "fit_id": FIT_ID,
        "status": "warning",
        "dataset_type": "asimov",
        "generation_range": fit_range,
        "blind_window_in_observed_data": signal_window,
        "construction_mode": "fixed_background_asimov_proxy",
        "hhxyy_equivalent_workspace": hhxyy_manifest,
        "binning": {"observable": "m_gg", "n_bins": 55, "range": fit_range},
        "weighted_dataset_object_type": "not_used_fallback_counts",
        "fixed_generation_inputs": [
            "signal yield normalized to MC prediction",
            "signal DSCB shape parameters from the signal-MC fit",
            "background PDF parameters from the mu=0 sideband-data fit snapshot",
        ],
        "floating_fit_parameters": [],
        "fixed_fit_parameters": [
            "signal shape parameters from the signal-MC fit",
            "background shape and normalization",
        ],
        "category_payload": category_payload,
        "diagnostics": diagnostics,
    }
    plot_payload = {
        "status": "warning",
        "fit_id": FIT_ID,
        "dataset_type": "asimov",
        "generation_hypothesis": "signal_plus_background",
        "mu_gen": 1.0,
        "binning": {"observable": "m_gg", "n_bins": 55, "range": fit_range},
        "categories": {
            category: {
                "asimov_counts": payload["total_counts"],
                "generation_signal_counts": payload["signal_counts"],
                "generation_background_counts": payload["background_counts"],
                "free_fit": free_fit_counts[category],
                "mu0_fit": mu0_fit_counts[category],
            }
            for category, payload in category_payload.items()
        },
        "combined": {
            "asimov_counts": (combined_signal + combined_background).tolist(),
            "generation_signal_counts": combined_signal.tolist(),
            "generation_background_counts": combined_background.tolist(),
            "free_fit": _combined_counts(free_fit_counts),
            "mu0_fit": _combined_counts(mu0_fit_counts),
        },
        "free_fit": {
            "mu_hat": profile["mu_hat"],
            "mu_uncertainty": profile["mu_uncertainty"],
            "fit_status": -1,
            "cov_qual": -1,
        },
        "mu0_fit": {
            "mu_fixed": 0.0,
            "fit_status": -1,
            "cov_qual": -1,
        },
        "diagnostics": diagnostics,
    }
    return asimov_artifact, construction_artifact, plot_payload


def _asimov_dataset(fit_context: dict):
    _set_generation_snapshot(fit_context)
    common_mass = fit_context["common_mass"]
    channel = fit_context["channel"]
    shared_mu = fit_context["shared_mu"]
    shared_mu.setVal(1.0)
    shared_mu.setConstant(False)

    category_counts = {}
    category_payload = {}
    for category, model_ctx in fit_context["final_models"].items():
        signal_yield = float(model_ctx["s_const"].getVal())
        background_yield = float(fit_context["category_context"][category]["template_total_yield"])
        signal_counts = pdf_to_counts(model_ctx["signal_pdf"], common_mass, signal_yield)
        background_counts = pdf_to_counts(model_ctx["background_pdf"], common_mass, background_yield)
        total_counts = signal_counts + background_counts
        category_counts[category] = total_counts
        category_payload[category] = {
            "signal_counts": signal_counts.tolist(),
            "background_counts": background_counts.tolist(),
            "total_counts": total_counts.tolist(),
        }
    combined = _weighted_category_dataset("asimovData", common_mass, channel, category_counts)
    return combined, category_payload


def _weighted_category_dataset(name: str, common_mass, channel, category_counts: dict[str, np.ndarray]):
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


def _free_parameter_policy(observed_significance_allowed: bool) -> dict[str, Any]:
    return {
        "observed_significance_allowed": bool(observed_significance_allowed),
        "explicit_unblinding_required": True,
        "explicit_unblinding_performed": bool(observed_significance_allowed),
        "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
        "signal_strength_parameter": "mu",
        "background_parameter_policy": "floating_shape_and_normalization",
        "background_parameter_source": "mu=0 sideband data-fit snapshot from background_pdf_choice.json",
    }


def _background_parameter_source(summary: dict, fit_dir: Path) -> dict[str, Any]:
    return {
        "source_artifact": str(fit_dir / "background_pdf_choice.json"),
        "fit_hypothesis": "mu=0 background-only sideband data fit",
        "sideband_ranges_gev": summary["runtime_defaults"]["sidebands_gev"],
    }


def _local_asimov_plot_payload_for_current_model(
    fit_context: dict,
    *,
    fit_range: list[float],
    mu_hat: float,
    mu_uncertainty: float,
) -> dict[str, Any]:
    measurement_snapshot = _snapshot_fit_state(fit_context)
    asimov_data, asimov_payload = _asimov_dataset(fit_context)
    shared_mu = fit_context["shared_mu"]

    shared_mu.setVal(1.0)
    free_result = _fit_with_mu(fit_context, asimov_data, mu_value=None)
    free_fit_counts = _capture_model_counts(fit_context)

    mu0_result = _fit_with_mu(fit_context, asimov_data, mu_value=0.0)
    mu0_fit_counts = _capture_model_counts(fit_context)
    shared_mu.setConstant(False)

    plot_payload = _asimov_plot_payload(
        fit_context=fit_context,
        asimov_payload=asimov_payload,
        fit_range=fit_range,
        mu_hat=mu_hat,
        mu_uncertainty=mu_uncertainty,
        free_result=free_result,
        mu0_result=mu0_result,
        free_fit_counts=free_fit_counts,
        mu0_fit_counts=mu0_fit_counts,
    )
    _restore_fit_state(fit_context, measurement_snapshot)
    return plot_payload


def _fit_with_mu(fit_context: dict, dataset, *, mu_value: float | None):
    shared_mu = fit_context["shared_mu"]
    if mu_value is None:
        shared_mu.setConstant(False)
    else:
        shared_mu.setVal(float(mu_value))
        shared_mu.setConstant(True)
    result = fit_context["simultaneous"].fitTo(
        dataset,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(1),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Extended(True),
    )
    try:
        result.status()
    except Exception as exc:
        raise RuntimeError("RooFit did not return a valid fit result") from exc
    return result


def _blocked_observed_artifact(
    *,
    fit_context: dict,
    fit_range: list[float],
    signal_window: list[float],
    blinding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fit_id": FIT_ID,
        "status": "blocked",
        "dataset_type": "observed",
        "backend": "pyroot_roofit",
        "poi_name": "signal_strength_mu",
        "observed_significance_allowed": bool(blinding["observed_significance_allowed"]),
        "fit_range": fit_range,
        "blind_window_in_observed_data": signal_window,
        "categories": fit_context["fit_summary"]["categories"],
        "shared_mu": True,
        "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
        "background_parameter_policy": "not_applicable_observed_significance_blocked_by_blinding",
        "error": "Observed significance is disabled by blinding policy; central claims use Asimov expected significance only.",
    }


def _observed_significance(
    *,
    fit_context: dict,
    fit_range: list[float],
    signal_window: list[float],
    blinding: dict[str, Any],
    background_parameter_source: dict[str, Any],
) -> dict[str, Any]:
    measurement_snapshot = _snapshot_fit_state(fit_context)
    shared_mu = fit_context["shared_mu"]
    shared_mu.setVal(1.0)

    free_result = _fit_with_mu(fit_context, fit_context["combined_data"], mu_value=None)
    twice_nll_free = 2.0 * float(free_result.minNll())
    mu_hat = float(shared_mu.getVal())
    mu_uncertainty = float(shared_mu.getError())

    mu0_result = _fit_with_mu(fit_context, fit_context["combined_data"], mu_value=0.0)
    twice_nll_mu0 = 2.0 * float(mu0_result.minNll())
    q0_raw = max(twice_nll_mu0 - twice_nll_free, 0.0)
    q0 = q0_raw if mu_hat > 0.0 else 0.0
    z_discovery = math.sqrt(q0)
    shared_mu.setConstant(False)

    diagnostics = []
    if free_result.status() != 0:
        diagnostics.append("Free-mu observed significance fit returned a non-zero RooFit status.")
    if mu0_result.status() != 0:
        diagnostics.append("Mu=0 observed significance fit returned a non-zero RooFit status.")
    if free_result.covQual() < 2:
        diagnostics.append("Free-mu observed significance fit covariance quality is below the acceptable threshold of 2.")
    if mu0_result.covQual() < 2:
        diagnostics.append("Mu=0 observed significance fit covariance quality is below the acceptable threshold of 2.")
    if mu_hat <= 0.0:
        diagnostics.append("Best-fit signal strength is non-positive, so the one-sided discovery test statistic is clipped to q0 = 0.")
    status = "ok" if not diagnostics else "warning"

    artifact = {
        "fit_id": FIT_ID,
        "status": status,
        "dataset_type": "observed",
        "backend": "pyroot_roofit",
        "poi_name": "signal_strength_mu",
        "mu_hat": mu_hat,
        "mu_uncertainty": mu_uncertainty,
        "twice_nll_mu0": twice_nll_mu0,
        "twice_nll_free": twice_nll_free,
        "q0": q0,
        "z_discovery": z_discovery,
        "fit_range": fit_range,
        "blind_window_in_observed_data": signal_window if blinding["plot_signal_window"] else None,
        "observed_significance_allowed": bool(blinding["observed_significance_allowed"]),
        "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
        "background_parameter_policy": "floating_shape_and_normalization",
        "background_parameter_source": background_parameter_source,
        "categories": fit_context["fit_summary"]["categories"],
        "shared_mu": True,
        "fit_status_free": int(free_result.status()),
        "fit_status_mu0": int(mu0_result.status()),
        "cov_qual_free": int(free_result.covQual()),
        "cov_qual_mu0": int(mu0_result.covQual()),
        "diagnostics": diagnostics,
    }
    _restore_fit_state(fit_context, measurement_snapshot)
    return artifact


def run_significance(fit_context: dict, summary: dict, outputs: Path) -> dict[str, Any]:
    fit_dir = outputs / "fit" / FIT_ID
    fit_range = summary["runtime_defaults"]["fit_mass_range_gev"]
    signal_window = summary["runtime_defaults"]["signal_window_gev"]
    blinding = summary["runtime_defaults"]["blinding"]
    policy = _free_parameter_policy(bool(blinding["observed_significance_allowed"]))
    write_json(policy, fit_dir / "significance_parameter_policy.json")
    background_parameter_source = _background_parameter_source(summary, fit_dir)

    if blinding["observed_significance_allowed"]:
        observed_artifact = _observed_significance(
            fit_context=fit_context,
            fit_range=fit_range,
            signal_window=signal_window,
            blinding=blinding,
            background_parameter_source=background_parameter_source,
        )
    else:
        observed_artifact = _blocked_observed_artifact(
            fit_context=fit_context,
            fit_range=fit_range,
            signal_window=signal_window,
            blinding=blinding,
        )
    write_json(observed_artifact, fit_dir / "significance.json")

    # ------------------------------------------------------------------
    # Asimov expected significance
    #   Primary path : hhxyy-fitting HistFactory + quickFit (ATLAS env)
    #   Fallback path: local RooFit profile likelihood with floating background
    # ------------------------------------------------------------------
    hhxyy_manifest = None
    hhxyy_available = is_atlas_env_available()
    if hhxyy_available:
        try:
            asimov_artifact = run_hhxyy_significance(
                category_context=fit_context["category_context"],
                fit_dir=fit_dir,
                summary=summary,
                category_order=fit_context["fit_summary"]["categories"],
            )
            construction_artifact = asimov_artifact.pop("_construction", None) or {
                "backend": "pyroot_roofit",
                "fit_driver": "hhxyy_fitting_quickfit",
                "atlas_env_available": True,
                "generation_range": fit_range,
            }
            plot_payload = _local_asimov_plot_payload_for_current_model(
                fit_context,
                fit_range=fit_range,
                mu_hat=float(asimov_artifact["mu_hat"]),
                mu_uncertainty=float(asimov_artifact["mu_uncertainty"]),
            )
            fit_context["asimov_plot_payload"] = plot_payload
            write_json(asimov_artifact, fit_dir / "significance_asimov.json")
            write_json(construction_artifact, fit_dir / "significance_asimov_construction.json")
            write_json(plot_payload, fit_dir / "significance_asimov_plot_payload.json")
            return {
                "observed": observed_artifact,
                "asimov": asimov_artifact,
                "construction": construction_artifact,
                "plot_payload": plot_payload,
                "policy": policy,
            }
        except Exception as exc:
            # If hhxyy path fails for any reason, fall through to analytic path
            import warnings
            warnings.warn(
                f"hhxyy-fitting significance failed ({exc}); "
                "falling back to the local RooFit profile-likelihood fit.",
                stacklevel=2,
            )
    else:
        try:
            hhxyy_manifest = prepare_hhxyy_workspace(
                category_context=fit_context["category_context"],
                fit_dir=fit_dir,
                summary=summary,
                category_order=fit_context["fit_summary"]["categories"],
            )
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Could not prepare HHXYY-equivalent workspace artifacts ({exc}); continuing with local RooFit path.",
                stacklevel=2,
            )

    measurement_snapshot = _snapshot_fit_state(fit_context)
    try:
        asimov_data, asimov_payload = _asimov_dataset(fit_context)
        shared_mu = fit_context["shared_mu"]

        shared_mu.setVal(1.0)
        free_result = _fit_with_mu(fit_context, asimov_data, mu_value=None)
        twice_nll_free = 2.0 * float(free_result.minNll())
        mu_hat = float(shared_mu.getVal())
        mu_uncertainty = float(shared_mu.getError())
        free_fit_counts = _capture_model_counts(fit_context)

        mu0_result = _fit_with_mu(fit_context, asimov_data, mu_value=0.0)
        twice_nll_mu0 = 2.0 * float(mu0_result.minNll())
        q0_raw = max(twice_nll_mu0 - twice_nll_free, 0.0)
        q0 = q0_raw if mu_hat > 0.0 else 0.0
        z_discovery = math.sqrt(q0)
        mu0_fit_counts = _capture_model_counts(fit_context)
        shared_mu.setConstant(False)

        diagnostics = []
        if free_result.status() != 0:
            diagnostics.append("Free-mu Asimov significance fit returned a non-zero RooFit status.")
        if mu0_result.status() != 0:
            diagnostics.append("Mu=0 Asimov significance fit returned a non-zero RooFit status.")
        if free_result.covQual() < 2:
            diagnostics.append("Free-mu Asimov significance fit covariance quality is below the acceptable threshold of 2.")
        if mu0_result.covQual() < 2:
            diagnostics.append("Mu=0 Asimov significance fit covariance quality is below the acceptable threshold of 2.")
        if mu_hat <= 0.0:
            diagnostics.append("Best-fit Asimov signal strength is non-positive, so the one-sided discovery test statistic is clipped to q0 = 0.")
        status = "ok" if not diagnostics else "warning"
        asimov_artifact = {
            "fit_id": FIT_ID,
            "status": status,
            "dataset_type": "asimov",
            "generation_hypothesis": "signal_plus_background",
            "mu_gen": 1.0,
            "backend": "pyroot_roofit",
            "atlas_env_available": hhxyy_available,
            "fit_driver": "local_pyroot_hhxyy_equivalent_fallback",
            "poi_name": "signal_strength_mu",
            "mu_hat": mu_hat,
            "mu_uncertainty": mu_uncertainty,
            "twice_nll_mu0": twice_nll_mu0,
            "twice_nll_free": twice_nll_free,
            "q0": q0,
            "z_discovery": z_discovery,
            "fit_range": fit_range,
            "background_parameter_source": background_parameter_source,
            "asimov_source": "weighted_bin_center_dataset_explicit_categories",
            "observed_significance_allowed": bool(blinding["observed_significance_allowed"]),
            "signal_shape_parameter_policy": "fixed_from_signal_mc_fit",
            "background_parameter_policy": "floating_shape_and_normalization",
            "asimov_profile_method": "profile_likelihood_with_floating_background_nuisances",
            "fisher_information_mu": (
                1.0 / (mu_uncertainty * mu_uncertainty)
                if math.isfinite(mu_uncertainty) and mu_uncertainty > 0.0
                else None
            ),
            "hhxyy_workspace_manifest": hhxyy_manifest,
            "categories": fit_context["fit_summary"]["categories"],
            "shared_mu": True,
            "fit_status_free": int(free_result.status()),
            "fit_status_mu0": int(mu0_result.status()),
            "cov_qual_free": int(free_result.covQual()),
            "cov_qual_mu0": int(mu0_result.covQual()),
            "diagnostics": diagnostics,
        }
        construction_artifact = {
            "fit_id": FIT_ID,
            "status": "ok",
            "dataset_type": "asimov",
            "generation_range": fit_range,
            "blind_window_in_observed_data": summary["runtime_defaults"]["signal_window_gev"],
            "construction_mode": "weighted_bin_center_dataset_explicit_categories",
            "hhxyy_equivalent_workspace": hhxyy_manifest,
            "binning": {"observable": "m_gg", "n_bins": 55, "range": fit_range},
            "weighted_dataset_object_type": "RooDataSet",
            "fixed_generation_inputs": [
                "signal yield normalized to MC prediction",
                "signal DSCB shape parameters from the signal-MC fit",
                "background PDF parameters from the mu=0 sideband-data fit snapshot",
            ],
            "floating_fit_parameters": [
                "signal strength mu in the free fit",
                "background normalization in each category",
                "background shape parameters in each category",
            ],
            "fixed_fit_parameters": [
                "signal shape parameters from the signal-MC fit",
                "signal strength mu=0 only in the background-only hypothesis fit",
            ],
            "category_payload": asimov_payload,
        }
        plot_payload = _asimov_plot_payload(
            fit_context=fit_context,
            asimov_payload=asimov_payload,
            fit_range=fit_range,
            mu_hat=mu_hat,
            mu_uncertainty=mu_uncertainty,
            free_result=free_result,
            mu0_result=mu0_result,
            free_fit_counts=free_fit_counts,
            mu0_fit_counts=mu0_fit_counts,
        )
    except Exception as exc:
        asimov_artifact, construction_artifact, plot_payload = _fixed_background_asimov_fallback(
            fit_context,
            fit_range=fit_range,
            signal_window=signal_window,
            blinding=blinding,
            background_parameter_source=background_parameter_source,
            error=f"{type(exc).__name__}: {exc}",
            hhxyy_manifest=hhxyy_manifest,
        )
    finally:
        _restore_fit_state(fit_context, measurement_snapshot)
    fit_context["asimov_plot_payload"] = plot_payload

    write_json(asimov_artifact, fit_dir / "significance_asimov.json")
    write_json(construction_artifact, fit_dir / "significance_asimov_construction.json")
    write_json(plot_payload, fit_dir / "significance_asimov_plot_payload.json")
    return {
        "observed": observed_artifact,
        "asimov": asimov_artifact,
        "construction": construction_artifact,
        "plot_payload": plot_payload,
        "policy": policy,
    }
