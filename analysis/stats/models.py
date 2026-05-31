from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

MASS_RANGE_GEV = (105.0, 160.0)
MASS_EPSILON_GEV = 1e-6


@dataclass
class CandidateModel:
    name: str
    pdf: ROOT.RooAbsPdf
    params: list
    complexity: int


def configure_mass_var(name: str = "mgg") -> ROOT.RooRealVar:
    mass = ROOT.RooRealVar(name, name, MASS_RANGE_GEV[0], MASS_RANGE_GEV[1])
    mass.setRange("full", MASS_RANGE_GEV[0], MASS_RANGE_GEV[1])
    mass.setRange("sideband_lo", MASS_RANGE_GEV[0], 120.0)
    mass.setRange("sideband_hi", 130.0, MASS_RANGE_GEV[1])
    mass.setRange("signal", 120.0, 130.0)
    mass.setBins(55)
    return mass


def sanitize_mass_inputs(masses: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    masses_arr = np.asarray(masses, dtype=float).reshape(-1)
    valid = np.isfinite(masses_arr)
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        valid &= np.isfinite(weights_arr)
    else:
        weights_arr = None

    masses_arr = masses_arr[valid]
    if weights_arr is not None:
        weights_arr = weights_arr[valid]

    in_window = (masses_arr >= MASS_RANGE_GEV[0] - MASS_EPSILON_GEV) & (masses_arr <= MASS_RANGE_GEV[1] + MASS_EPSILON_GEV)
    masses_arr = np.clip(masses_arr[in_window], MASS_RANGE_GEV[0] + MASS_EPSILON_GEV, MASS_RANGE_GEV[1] - MASS_EPSILON_GEV)
    if weights_arr is not None:
        weights_arr = weights_arr[in_window]

    return masses_arr, weights_arr


def make_weighted_dataset(name: str, mass_var: ROOT.RooRealVar, masses: np.ndarray, weights: np.ndarray | None = None):
    masses, weights = sanitize_mass_inputs(masses, weights)
    if weights is None:
        return ROOT.RooDataSet.from_numpy({mass_var.GetName(): masses}, [mass_var])
    max_abs_weight = max(float(np.max(np.abs(weights))) if len(weights) else 1.0, 1.0)
    weight_var = ROOT.RooRealVar(f"w_{name}", f"w_{name}", 0.0, -10.0 * max_abs_weight, 10.0 * max_abs_weight)
    return ROOT.RooDataSet.from_numpy(
        {mass_var.GetName(): masses, weight_var.GetName(): weights},
        [mass_var, weight_var],
        weight_name=weight_var.GetName(),
    )


def make_weighted_bin_center_dataset(name: str, mass_var: ROOT.RooRealVar, counts: np.ndarray):
    bins = len(counts)
    edges = np.linspace(105.0, 160.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return make_weighted_dataset(name, mass_var, centers, np.asarray(counts, dtype=float))


def make_datahist(name: str, mass_var: ROOT.RooRealVar, counts: np.ndarray, bins: int = 55):
    counts = np.asarray(counts, dtype=float)
    return ROOT.RooDataHist.from_numpy(
        counts,
        [mass_var],
        bins=[bins],
        ranges=[(105.0, 160.0)],
        weights_squared_sum=counts,
        name=name,
    )


def make_binned_counts_datahist(name: str, mass_var: ROOT.RooRealVar, counts: np.ndarray):
    return make_datahist(name, mass_var, counts, bins=len(counts))


def asimov_fixed_background_profile(signal_counts: np.ndarray, background_counts: np.ndarray, mu: float = 1.0) -> dict[str, float]:
    signal = np.asarray(signal_counts, dtype=float)
    background = np.asarray(background_counts, dtype=float)
    expected = np.clip(background + mu * signal, 1e-12, None)
    background_safe = np.clip(background, 1e-12, None)
    twice_nll_free = 2.0 * float(np.sum(expected - expected * np.log(expected)))
    twice_nll_mu0 = 2.0 * float(np.sum(background_safe - expected * np.log(background_safe)))
    q0 = max(twice_nll_mu0 - twice_nll_free, 0.0)
    information = float(np.sum(np.square(signal) / expected))
    mu_uncertainty = 1.0 / np.sqrt(information) if information > 0.0 else float("inf")
    return {
        "mu_hat": float(mu),
        "mu_uncertainty": float(mu_uncertainty),
        "twice_nll_free": twice_nll_free,
        "twice_nll_mu0": twice_nll_mu0,
        "q0": float(q0),
        "z_discovery": float(np.sqrt(q0)),
        "fisher_information_mu": information,
    }


def crystal_ball_pdf(prefix: str, mass_var: ROOT.RooRealVar):
    mean = ROOT.RooRealVar(f"mean_{prefix}", f"mean_{prefix}", 125.0, 122.0, 128.0)
    sigma_l = ROOT.RooRealVar(f"sigmaL_{prefix}", f"sigmaL_{prefix}", 1.6, 0.5, 4.0)
    sigma_r = ROOT.RooRealVar(f"sigmaR_{prefix}", f"sigmaR_{prefix}", 1.8, 0.5, 4.5)
    alpha_l = ROOT.RooRealVar(f"alphaL_{prefix}", f"alphaL_{prefix}", 1.4, 0.4, 5.0)
    n_l = ROOT.RooRealVar(f"nL_{prefix}", f"nL_{prefix}", 3.0, 0.5, 20.0)
    alpha_r = ROOT.RooRealVar(f"alphaR_{prefix}", f"alphaR_{prefix}", 1.8, 0.4, 5.0)
    n_r = ROOT.RooRealVar(f"nR_{prefix}", f"nR_{prefix}", 3.0, 0.5, 20.0)
    pdf = ROOT.RooCrystalBall(
        f"sigpdf_{prefix}",
        f"sigpdf_{prefix}",
        mass_var,
        mean,
        sigma_l,
        sigma_r,
        alpha_l,
        n_l,
        alpha_r,
        n_r,
    )
    params = [mean, sigma_l, sigma_r, alpha_l, n_l, alpha_r, n_r]
    return pdf, params


def freeze_parameters(params: list) -> None:
    for param in params:
        param.setConstant(True)


def background_candidate(prefix: str, mass_var: ROOT.RooRealVar, kind: str) -> CandidateModel:
    if kind == "exponential":
        tau = ROOT.RooRealVar(f"tau_{prefix}", f"tau_{prefix}", -0.03, -1.0, -1e-4)
        pdf = ROOT.RooExponential(f"bkgpdf_{prefix}_{kind}", f"bkgpdf_{prefix}_{kind}", mass_var, tau)
        return CandidateModel(kind, pdf, [tau], 1)
    order = 2 if kind == "bernstein2" else 3
    coeffs = ROOT.RooArgList()
    params = []
    for idx in range(order + 1):
        coeff = ROOT.RooRealVar(f"c{idx}_{prefix}_{kind}", f"c{idx}_{prefix}_{kind}", 0.6 + 0.15 * idx, 1e-5, 20.0)
        coeffs.add(coeff)
        params.append(coeff)
    pdf = ROOT.RooBernstein(f"bkgpdf_{prefix}_{kind}", f"bkgpdf_{prefix}_{kind}", mass_var, coeffs)
    return CandidateModel(kind, pdf, params, order)


def fit_pdf(pdf, dataset, *, fit_range: str | None = None, weighted: bool | None = None, extended: bool = False):
    def _fit_args(strategy: int) -> list:
        args = [
            ROOT.RooFit.Save(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Strategy(strategy),
            ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
            ROOT.RooFit.Offset(True),
            ROOT.RooFit.PrintEvalErrors(-1),
            ROOT.RooFit.Hesse(False),
            ROOT.RooFit.Minos(False),
        ]
        if fit_range:
            args.append(ROOT.RooFit.Range(fit_range))
        if weighted is not None:
            args.append(ROOT.RooFit.SumW2Error(bool(weighted)))
        if extended:
            args.append(ROOT.RooFit.Extended(True))
        return args

    result = pdf.fitTo(dataset, *_fit_args(1))
    if result.status() != 0 or result.covQual() < 2:
        result = pdf.fitTo(dataset, *_fit_args(2))
    return result


def th1_smooth(counts: np.ndarray, smooth_times: int = 1) -> np.ndarray:
    hist = ROOT.TH1D(ROOT.TUUID().AsString(), "template_smooth", len(counts), MASS_RANGE_GEV[0], MASS_RANGE_GEV[1])
    for idx, value in enumerate(np.asarray(counts, dtype=float), start=1):
        hist.SetBinContent(idx, float(value))
    original_integral = hist.Integral()
    hist.Smooth(smooth_times)
    smoothed = np.array([hist.GetBinContent(idx) for idx in range(1, len(counts) + 1)], dtype=float)
    if smoothed.sum() > 0.0 and original_integral > 0.0:
        smoothed *= original_integral / smoothed.sum()
    return smoothed


def histogram_counts(masses: np.ndarray, weights: np.ndarray | None = None, bins: int = 55) -> np.ndarray:
    masses_arr, weights_arr = sanitize_mass_inputs(masses, weights)
    counts, _ = np.histogram(masses_arr, bins=bins, range=MASS_RANGE_GEV, weights=weights_arr)
    return counts.astype(float)


def _parameter_map(pdf, mass_var: ROOT.RooRealVar) -> dict[str, float]:
    params = pdf.getVariables()
    payload: dict[str, float] = {}
    for arg in params:
        if hasattr(arg, "getVal") and arg.GetName() != mass_var.GetName():
            payload[arg.GetName()] = float(arg.getVal())
    return payload


def _counts_for_exponential(params: dict[str, float], centers: np.ndarray) -> np.ndarray:
    tau = next(value for name, value in params.items() if name.startswith("tau_"))
    return np.exp(tau * centers)


def _counts_for_bernstein(params: dict[str, float], centers: np.ndarray) -> np.ndarray:
    coeff_items = sorted(
        (
            (int(name.split("_", 1)[0][1:]), value)
            for name, value in params.items()
            if name.startswith("c") and "_" in name and name[1].isdigit()
        ),
        key=lambda item: item[0],
    )
    coeffs = [value for _, value in coeff_items]
    order = len(coeffs) - 1
    scaled = (centers - MASS_RANGE_GEV[0]) / (MASS_RANGE_GEV[1] - MASS_RANGE_GEV[0])
    values = np.zeros_like(centers, dtype=float)
    for idx, coeff in enumerate(coeffs):
        values += coeff * math.comb(order, idx) * np.power(scaled, idx) * np.power(1.0 - scaled, order - idx)
    return values


def _counts_for_crystal_ball(params: dict[str, float], centers: np.ndarray) -> np.ndarray:
    mean = next(value for name, value in params.items() if name.startswith("mean_"))
    sigma_l = next(value for name, value in params.items() if name.startswith("sigmaL_"))
    sigma_r = next(value for name, value in params.items() if name.startswith("sigmaR_"))
    alpha_l = next(value for name, value in params.items() if name.startswith("alphaL_"))
    alpha_r = next(value for name, value in params.items() if name.startswith("alphaR_"))
    n_l = next(value for name, value in params.items() if name.startswith("nL_"))
    n_r = next(value for name, value in params.items() if name.startswith("nR_"))

    values = np.zeros_like(centers, dtype=float)
    left = centers < mean
    right = ~left
    t = np.zeros_like(centers, dtype=float)
    t[left] = (centers[left] - mean) / max(sigma_l, 1e-6)
    t[right] = (centers[right] - mean) / max(sigma_r, 1e-6)

    core_left = left & (t >= -alpha_l)
    tail_left = left & ~core_left
    core_right = right & (t <= alpha_r)
    tail_right = right & ~core_right

    values[core_left | core_right] = np.exp(-0.5 * np.square(t[core_left | core_right]))

    alpha_l = max(alpha_l, 1e-6)
    alpha_r = max(alpha_r, 1e-6)
    n_l = max(n_l, 1e-6)
    n_r = max(n_r, 1e-6)

    a_l = math.exp(-0.5 * alpha_l * alpha_l) * math.pow(n_l / alpha_l, n_l)
    b_l = n_l / alpha_l - alpha_l
    values[tail_left] = a_l / np.power(b_l - t[tail_left], n_l)

    a_r = math.exp(-0.5 * alpha_r * alpha_r) * math.pow(n_r / alpha_r, n_r)
    b_r = n_r / alpha_r - alpha_r
    values[tail_right] = a_r / np.power(b_r + t[tail_right], n_r)
    return values


def pdf_to_counts(pdf, mass_var: ROOT.RooRealVar, yield_value: float, bins: int = 55) -> np.ndarray:
    message_service = ROOT.RooMsgService.instance()
    message_service.setGlobalKillBelow(ROOT.RooFit.FATAL)
    try:
        edges = np.linspace(MASS_RANGE_GEV[0], MASS_RANGE_GEV[1], bins + 1)
        norm_set = ROOT.RooArgSet(mass_var)
        values = np.zeros(bins, dtype=float)
        for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            range_name = f"bin_{pdf.GetName()}_{idx}"
            mass_var.setRange(range_name, float(low), float(high))
            integral = pdf.createIntegral(
                norm_set,
                ROOT.RooFit.NormSet(norm_set),
                ROOT.RooFit.Range(range_name),
            )
            values[idx] = float(integral.getVal())
    finally:
        message_service.setGlobalKillBelow(ROOT.RooFit.WARNING)
    values = np.clip(values, 0.0, None)
    if values.sum() > 0.0:
        values *= float(yield_value) / values.sum()
    return values


def _trapz(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def pdf_to_curve(
    pdf,
    mass_var: ROOT.RooRealVar,
    yield_value: float,
    *,
    x_values: np.ndarray | None = None,
    bin_width: float = 1.0,
    normalize_regions: list[tuple[float, float]] | None = None,
) -> dict[str, list[float]]:
    """Evaluate a RooFit PDF as a continuous events-per-bin-width curve."""

    x = (
        np.asarray(x_values, dtype=float)
        if x_values is not None
        else np.linspace(MASS_RANGE_GEV[0], MASS_RANGE_GEV[1], 551)
    )
    norm_set = ROOT.RooArgSet(mass_var)
    raw = np.zeros_like(x, dtype=float)
    message_service = ROOT.RooMsgService.instance()
    message_service.setGlobalKillBelow(ROOT.RooFit.FATAL)
    try:
        for idx, value in enumerate(x):
            mass_var.setVal(float(value))
            raw[idx] = max(float(pdf.getVal(norm_set)), 0.0)
    finally:
        message_service.setGlobalKillBelow(ROOT.RooFit.WARNING)

    if normalize_regions:
        norm_area = 0.0
        for low, high in normalize_regions:
            mask = (x >= float(low)) & (x <= float(high))
            if np.count_nonzero(mask) >= 2:
                norm_area += _trapz(raw[mask], x[mask])
    else:
        norm_area = _trapz(raw, x)

    y = np.zeros_like(raw, dtype=float)
    if norm_area > 0.0 and np.isfinite(norm_area):
        y = raw * (float(yield_value) / norm_area) * float(bin_width)
    return {"x": x.tolist(), "y": y.tolist()}
