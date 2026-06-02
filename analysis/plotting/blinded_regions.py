from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import ensure_dir, write_json
from analysis.stats.models import background_candidate, pdf_to_curve

PLOT_COLORS = {
    "prompt_diphoton": "#f4a261",
    "signal": "#d62828",
    "fit": "#1d3557",
    "data": "#000000",
    "uncertainty": "#9aa0a6",
    "smoothed": "#2a9d8f",
}

LUMI_LABEL = r"$\sqrt{s}=13$ TeV, $L=36.1\ \mathrm{fb}^{-1}$"


def _save_figure(fig: plt.Figure, out_base: Path) -> list[str]:
    ensure_dir(out_base.parent)
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path), str(png_path)]


def _group_events(processed_samples: list[dict], role: str, category: str | None = None) -> dict[str, np.ndarray]:
    payload: dict[str, list[np.ndarray]] = {}
    for sample in processed_samples:
        include = False
        if role == "data" and sample["kind"] == "data":
            include = True
        elif role == "signal" and sample["analysis_role"] == "signal_nominal":
            include = True
        elif role == "prompt_diphoton" and sample["analysis_role"] == "background_nominal" and sample["process_key"] == "prompt_diphoton":
            include = True
        if not include or len(sample["events"].get("mgg", [])) == 0:
            continue
        mask = np.ones(len(sample["events"]["mgg"]), dtype=bool)
        if category is not None:
            mask &= sample["events"]["category"] == category
        if not np.any(mask):
            continue
        for key, values in sample["events"].items():
            payload.setdefault(key, []).append(values[mask])
    out: dict[str, np.ndarray] = {}
    for key, chunks in payload.items():
        out[key] = np.concatenate(chunks) if chunks else np.array([])
    return out


def _apply_blinding(data_counts: np.ndarray, data_errors: np.ndarray, bins: np.ndarray, blind_window: list[float] | None):
    if blind_window is None:
        return data_counts, data_errors, np.ones(len(data_counts), dtype=bool)
    centers = 0.5 * (bins[:-1] + bins[1:])
    visible = (centers < blind_window[0]) | (centers > blind_window[1])
    blinded_counts = data_counts.astype(float).copy()
    blinded_errors = data_errors.astype(float).copy()
    blinded_counts[~visible] = np.nan
    blinded_errors[~visible] = np.nan
    return blinded_counts, blinded_errors, visible


def _ordered_legend(ax: plt.Axes, labels_in_order: list[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    ordered = [index_by_label[label] for label in labels_in_order if label in index_by_label]
    ordered.extend(idx for idx in range(len(labels)) if idx not in ordered)
    ax.legend([handles[idx] for idx in ordered], [labels[idx] for idx in ordered], loc="upper right", fontsize=9)


def _model_stat_uncertainty(model_counts: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(model_counts, 0.0, None))


def _fit_range_curve_x(bins: np.ndarray) -> np.ndarray:
    return np.linspace(float(bins[0]), float(bins[-1]), 551)


def _curve_arrays(curve: dict[str, Any], component: str = "total") -> tuple[np.ndarray, np.ndarray] | None:
    if not curve:
        return None
    try:
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve[component], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if x.size < 2 or x.size != y.size or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    return x, y


def _component_curve_payload(model_ctx: dict[str, Any], mass_var, *, x_values: np.ndarray) -> dict[str, Any]:
    signal_curve = pdf_to_curve(
        model_ctx["signal_pdf"],
        mass_var,
        float(model_ctx["nsig"].getVal()),
        x_values=x_values,
    )
    background_curve = pdf_to_curve(
        model_ctx["background_pdf"],
        mass_var,
        float(model_ctx["nbkg"].getVal()),
        x_values=x_values,
    )
    signal_y = np.asarray(signal_curve["y"], dtype=float)
    background_y = np.asarray(background_curve["y"], dtype=float)
    return {
        "x": x_values.tolist(),
        "signal": signal_y.tolist(),
        "background": background_y.tolist(),
        "total": (signal_y + background_y).tolist(),
    }


def _combine_curve_payloads(curves: list[dict[str, Any]]) -> dict[str, Any] | None:
    parsed = [_curve_arrays(curve, "total") for curve in curves]
    if not parsed or any(item is None for item in parsed):
        return None
    x = parsed[0][0]
    components = {}
    for component in ("signal", "background", "total"):
        arrays = []
        for curve in curves:
            curve_arrays = _curve_arrays(curve, component)
            if curve_arrays is None:
                return None
            arrays.append(curve_arrays[1])
        components[component] = np.sum(arrays, axis=0).tolist()
    return {"x": x.tolist(), **components}


def _interpolate_curve(curve_x: np.ndarray, curve_y: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.interp(centers, curve_x, curve_y, left=np.nan, right=np.nan)


def _analytic_curve_plot(
    *,
    observed_counts: np.ndarray,
    observed_errors: np.ndarray,
    observed_label: str,
    model_curve: dict[str, Any],
    model_component: str,
    model_label: str,
    title: str,
    out_base: Path,
    blind_window: list[float] | None,
    ratio_label: str,
    ratio_ylim: tuple[float, float],
    extra_curves: list[tuple[dict[str, Any], str, str, str]] | None = None,
) -> list[str]:
    bins = np.linspace(105.0, 160.0, len(observed_counts) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    display_counts, display_errors, visible = _apply_blinding(observed_counts, observed_errors, bins, blind_window)
    curve_arrays = _curve_arrays(model_curve, model_component)
    if curve_arrays is None:
        fallback = np.asarray(model_curve.get("total_counts", observed_counts), dtype=float)
        return _counts_fit_plot(
            observed_counts=observed_counts,
            observed_errors=observed_errors,
            observed_label=observed_label,
            model_counts=fallback,
            model_label=model_label,
            title=title,
            out_base=out_base,
            blind_window=blind_window,
            ratio_label=ratio_label,
            ratio_ylim=ratio_ylim,
        )
    curve_x, curve_y = curve_arrays
    center_model = _interpolate_curve(curve_x, curve_y, centers)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(curve_x, curve_y, color=PLOT_COLORS["fit"], linewidth=2.2, label=model_label)
    for curve, component, label, color in extra_curves or []:
        arrays = _curve_arrays(curve, component)
        if arrays is None:
            continue
        ax.plot(arrays[0], arrays[1], color=color, linewidth=1.8, linestyle="--", label=label)
    ax.errorbar(
        centers[visible],
        display_counts[visible],
        yerr=display_errors[visible],
        fmt="o",
        color=PLOT_COLORS["data"],
        markersize=4,
        label=observed_label,
    )
    if blind_window is not None:
        ax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)
    ax.set_xlim(105.0, 160.0)
    ymax = max(
        np.nanmax(display_counts) if np.isfinite(display_counts).any() else 0.0,
        np.nanmax(curve_y) if np.isfinite(curve_y).any() else 0.0,
        1.0,
    )
    ax.set_ylim(0.0, 1.45 * ymax)
    ax.set_ylabel("Events / 1 GeV")
    _ordered_legend(ax, [observed_label, model_label])
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ratio = np.divide(observed_counts, center_model, out=np.full_like(center_model, np.nan, dtype=float), where=center_model > 0.0)
    ratio_err = np.divide(observed_errors, center_model, out=np.full_like(center_model, np.nan, dtype=float), where=center_model > 0.0)
    valid = visible & np.isfinite(ratio)
    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.errorbar(centers[valid], ratio[valid], yerr=ratio_err[valid], fmt="o", color=PLOT_COLORS["data"], markersize=4)
    if blind_window is not None:
        rax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel(ratio_label)
    rax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    return _save_figure(fig, out_base)


def _apply_background_snapshot(background_pdf, params: list, snapshot: dict[str, float]) -> None:
    coeff_values = [
        value
        for name, value in sorted(
            snapshot.items(),
            key=lambda item: int(item[0].split("_", 1)[0][1:]) if item[0].startswith("c") and item[0][1].isdigit() else -1,
        )
        if name.startswith("c") and "_" in name and name[1].isdigit()
    ]
    coeff_idx = 0
    tau_values = [value for name, value in snapshot.items() if name.startswith("tau_")]
    for param in params:
        name = param.GetName()
        if name.startswith("tau_") and tau_values:
            param.setVal(float(tau_values[0]))
        elif name.startswith("c") and "_" in name and name[1].isdigit() and coeff_idx < len(coeff_values):
            param.setVal(float(coeff_values[coeff_idx]))
            coeff_idx += 1


def _spurious_fit_curve(
    *,
    category: str,
    selected_model: str,
    snapshot: dict[str, float],
    signal_pdf,
    mass_var,
    signal_yield: float,
    background_yield: float,
    x_values: np.ndarray,
) -> dict[str, Any]:
    background = background_candidate(f"plot_spur_{category}", mass_var, selected_model)
    _apply_background_snapshot(background.pdf, background.params, snapshot)
    signal_curve = pdf_to_curve(signal_pdf, mass_var, signal_yield, x_values=x_values)
    background_curve = pdf_to_curve(background.pdf, mass_var, background_yield, x_values=x_values)
    signal_y = np.asarray(signal_curve["y"], dtype=float)
    background_y = np.asarray(background_curve["y"], dtype=float)
    return {
        "x": x_values.tolist(),
        "signal": signal_y.tolist(),
        "background": background_y.tolist(),
        "total": (signal_y + background_y).tolist(),
    }


def _ratio_plot(
    *,
    data_values: np.ndarray,
    mc_background_values: np.ndarray,
    mc_background_weights: np.ndarray,
    signal_values: np.ndarray,
    signal_weights: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_base: Path,
    blind_window: list[float] | None = None,
) -> list[str]:
    data_counts, _ = np.histogram(data_values, bins=bins)
    data_errors = np.sqrt(data_counts)
    bkg_counts, _ = np.histogram(mc_background_values, bins=bins, weights=mc_background_weights)
    bkg_sumw2, _ = np.histogram(mc_background_values, bins=bins, weights=mc_background_weights**2)
    sig_counts, _ = np.histogram(signal_values, bins=bins, weights=signal_weights)
    sig_sumw2, _ = np.histogram(signal_values, bins=bins, weights=signal_weights**2)
    total_counts = bkg_counts + sig_counts
    total_unc = np.sqrt(bkg_sumw2 + sig_sumw2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    display_counts, display_errors, visible = _apply_blinding(data_counts, data_errors, bins, blind_window)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    ax.bar(centers, bkg_counts, width=widths, color=PLOT_COLORS["prompt_diphoton"], alpha=0.65, label="Prompt diphoton", align="center")
    ax.step(bins, np.r_[sig_counts, sig_counts[-1]], where="post", color=PLOT_COLORS["signal"], linewidth=2.0, label="Signal")
    ax.fill_between(
        bins,
        np.r_[total_counts - total_unc, (total_counts - total_unc)[-1]],
        np.r_[total_counts + total_unc, (total_counts + total_unc)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
        label="MC stat. unc.",
    )
    ax.errorbar(centers[visible], display_counts[visible], yerr=display_errors[visible], fmt="o", color=PLOT_COLORS["data"], markersize=4, label="Data")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ymax = max(np.nanmax(display_counts) if np.isfinite(display_counts).any() else 0.0, np.max(total_counts + total_unc) if len(total_counts) else 0.0)
    ax.set_ylim(0.0, 1.45 * max(ymax, 1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)
    if blind_window is not None:
        ax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)

    ratio = np.full_like(total_counts, np.nan, dtype=float)
    ratio_err = np.full_like(total_counts, np.nan, dtype=float)
    valid = (total_counts > 0.0) & visible
    ratio[valid] = data_counts[valid] / total_counts[valid]
    ratio_err[valid] = data_errors[valid] / total_counts[valid]
    band = np.full_like(total_counts, np.nan, dtype=float)
    band[total_counts > 0.0] = total_unc[total_counts > 0.0] / total_counts[total_counts > 0.0]

    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.fill_between(
        bins,
        np.r_[1.0 - band, (1.0 - band)[-1]],
        np.r_[1.0 + band, (1.0 + band)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
    )
    rax.errorbar(centers[valid], ratio[valid], yerr=ratio_err[valid], fmt="o", color=PLOT_COLORS["data"], markersize=4)
    if blind_window is not None:
        rax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(xlabel)
    return _save_figure(fig, out_base)


def _template_fit_plot(
    *,
    counts: np.ndarray,
    fit_counts: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_base: Path,
    model_curve: dict[str, Any] | None = None,
    model_label: str = "Selected analytic fit",
) -> list[str]:
    bins = np.linspace(105.0, 160.0, len(counts) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    curve_arrays = _curve_arrays(model_curve or {}, "total")
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color=PLOT_COLORS["data"], label="Template")
    if curve_arrays is None:
        ax.step(bins, np.r_[fit_counts, fit_counts[-1]], where="post", color=PLOT_COLORS["fit"], linewidth=2.0, label="Selected fit")
        denominator = np.asarray(fit_counts, dtype=float)
    else:
        curve_x, curve_y = curve_arrays
        ax.plot(curve_x, curve_y, color=PLOT_COLORS["fit"], linewidth=2.2, label=model_label)
        denominator = _interpolate_curve(curve_x, curve_y, centers)
    ax.set_xlim(105.0, 160.0)
    ax.set_ylim(0.0, 1.45 * max(np.max(counts) if len(counts) else 0.0, np.nanmax(denominator) if len(denominator) else 0.0, 1.0))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ratio = np.divide(counts, denominator, out=np.full_like(counts, np.nan, dtype=float), where=np.asarray(denominator) > 0.0)
    ratio_err = np.divide(np.sqrt(np.clip(counts, 0.0, None)), denominator, out=np.full_like(counts, np.nan, dtype=float), where=np.asarray(denominator) > 0.0)
    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o", color=PLOT_COLORS["data"], markersize=4)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Tpl / fit")
    rax.set_xlabel(xlabel)
    return _save_figure(fig, out_base)


def _weighted_histogram_uncertainty(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, scale: float = 1.0) -> np.ndarray:
    sumw2, _ = np.histogram(values, bins=bins, weights=np.asarray(weights, dtype=float) ** 2)
    return np.sqrt(sumw2) * abs(scale)


def _smoothing_overlay_plot(
    *,
    unsmoothed_counts: np.ndarray,
    unsmoothed_unc: np.ndarray,
    smoothed_counts: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_base: Path,
) -> list[str]:
    bins = np.linspace(105.0, 160.0, len(unsmoothed_counts) + 1)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    ax.step(
        bins,
        np.r_[unsmoothed_counts, unsmoothed_counts[-1]],
        where="post",
        color=PLOT_COLORS["prompt_diphoton"],
        linewidth=2.0,
        label="Unsmoothed template",
    )
    ax.fill_between(
        bins,
        np.r_[np.clip(unsmoothed_counts - unsmoothed_unc, 0.0, None), np.clip(unsmoothed_counts - unsmoothed_unc, 0.0, None)[-1]],
        np.r_[unsmoothed_counts + unsmoothed_unc, (unsmoothed_counts + unsmoothed_unc)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
        label="Unsmoothed stat. unc.",
    )
    ax.step(
        bins,
        np.r_[smoothed_counts, smoothed_counts[-1]],
        where="post",
        color=PLOT_COLORS["smoothed"],
        linewidth=2.0,
        label="Smoothed template",
    )
    ymax = max(
        np.max(unsmoothed_counts + unsmoothed_unc) if len(unsmoothed_counts) else 0.0,
        np.max(smoothed_counts) if len(smoothed_counts) else 0.0,
        1.0,
    )
    ax.set_xlim(105.0, 160.0)
    ax.set_ylim(0.0, 1.45 * ymax)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ratio = np.divide(smoothed_counts, unsmoothed_counts, out=np.full_like(smoothed_counts, np.nan, dtype=float), where=unsmoothed_counts > 0.0)
    ratio_band = np.divide(unsmoothed_unc, unsmoothed_counts, out=np.full_like(unsmoothed_unc, np.nan, dtype=float), where=unsmoothed_counts > 0.0)
    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.fill_between(
        bins,
        np.r_[1.0 - ratio_band, (1.0 - ratio_band)[-1]],
        np.r_[1.0 + ratio_band, (1.0 + ratio_band)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
    )
    rax.step(
        bins,
        np.r_[ratio, ratio[-1]],
        where="post",
        color=PLOT_COLORS["smoothed"],
        linewidth=2.0,
    )
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Sm. / unsm.")
    rax.set_xlabel(xlabel)
    return _save_figure(fig, out_base)


def _signal_shape_plot(category: str, masses: np.ndarray, weights: np.ndarray, signal_pdf, mass_var, yield_value: float, out_base: Path) -> list[str]:
    bins = np.linspace(105.0, 160.0, 56)
    counts, _ = np.histogram(masses, bins=bins, weights=weights)
    curve = pdf_to_curve(signal_pdf, mass_var, yield_value, x_values=_fit_range_curve_x(bins))
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color=PLOT_COLORS["data"], label="Signal MC")
    curve_arrays = _curve_arrays({"x": curve["x"], "total": curve["y"]}, "total")
    if curve_arrays is not None:
        ax.plot(curve_arrays[0], curve_arrays[1], color=PLOT_COLORS["signal"], linewidth=2.2, label="DSCB analytic PDF")
    ax.set_xlim(105.0, 160.0)
    curve_y = np.asarray(curve["y"], dtype=float)
    ax.set_ylim(0.0, 1.45 * max(np.max(counts) if len(counts) else 0.0, np.max(curve_y) if len(curve_y) else 0.0, 1.0))
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel("Weighted events / 1 GeV")
    _ordered_legend(ax, ["Signal MC", "DSCB analytic PDF"])
    ax.text(0.02, 0.95, f"Signal shape: {category}", transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)
    return _save_figure(fig, out_base)


def _counts_fit_plot(
    *,
    observed_counts: np.ndarray,
    observed_errors: np.ndarray,
    observed_label: str,
    model_counts: np.ndarray,
    model_label: str,
    title: str,
    out_base: Path,
    blind_window: list[float] | None,
    ratio_label: str,
    ratio_ylim: tuple[float, float],
) -> list[str]:
    centers = np.linspace(105.5, 159.5, len(model_counts))
    bins = np.linspace(105.0, 160.0, len(model_counts) + 1)
    display_counts, display_errors, visible = _apply_blinding(observed_counts, observed_errors, bins, blind_window)
    model_unc = _model_stat_uncertainty(model_counts)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    ax.fill_between(
        bins,
        np.r_[np.clip(model_counts - model_unc, 0.0, None), np.clip(model_counts - model_unc, 0.0, None)[-1]],
        np.r_[model_counts + model_unc, (model_counts + model_unc)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
        label="Model stat. unc.",
    )
    ax.step(bins, np.r_[model_counts, model_counts[-1]], where="post", color=PLOT_COLORS["fit"], linewidth=2.0, label=model_label)
    ax.errorbar(
        centers[visible],
        display_counts[visible],
        yerr=display_errors[visible],
        fmt="o",
        color=PLOT_COLORS["data"],
        markersize=4,
        label=observed_label,
    )
    if blind_window is not None:
        ax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)
    ax.set_xlim(105.0, 160.0)
    ax.set_ylim(0.0, 1.45 * max(np.nanmax(display_counts) if np.isfinite(display_counts).any() else 0.0, np.max(model_counts), 1.0))
    ax.set_ylabel("Events / 1 GeV")
    _ordered_legend(ax, [observed_label, model_label, "Model stat. unc."])
    ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ratio = np.divide(observed_counts, model_counts, out=np.full_like(model_counts, np.nan, dtype=float), where=model_counts > 0.0)
    ratio_err = np.divide(observed_errors, model_counts, out=np.full_like(model_counts, np.nan, dtype=float), where=model_counts > 0.0)
    ratio_band = np.divide(model_unc, model_counts, out=np.full_like(model_counts, np.nan, dtype=float), where=model_counts > 0.0)
    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.fill_between(
        bins,
        np.r_[1.0 - ratio_band, (1.0 - ratio_band)[-1]],
        np.r_[1.0 + ratio_band, (1.0 + ratio_band)[-1]],
        step="post",
        color=PLOT_COLORS["uncertainty"],
        alpha=0.35,
    )
    rax.errorbar(centers[visible], ratio[visible], yerr=ratio_err[visible], fmt="o", color=PLOT_COLORS["data"], markersize=4)
    if blind_window is not None:
        rax.axvspan(blind_window[0], blind_window[1], color="#bbbbbb", alpha=0.18)
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel(ratio_label)
    rax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    return _save_figure(fig, out_base)


def _fit_result_plot(
    *,
    category: str,
    data_masses: np.ndarray,
    model_counts: np.ndarray,
    bins: np.ndarray,
    out_base: Path,
    blind_window: list[float],
    model_curve: dict[str, Any] | None = None,
) -> list[str]:
    data_counts, _ = np.histogram(data_masses, bins=bins)
    data_errors = np.sqrt(data_counts)
    if model_curve:
        return _analytic_curve_plot(
            observed_counts=data_counts.astype(float),
            observed_errors=data_errors.astype(float),
            observed_label="Data",
            model_curve=model_curve,
            model_component="total",
            model_label="Analytic S+B fit",
            title=f"Post-fit mass spectrum: {category}",
            out_base=out_base,
            blind_window=blind_window,
            ratio_label="Data / fit",
            ratio_ylim=(0.5, 1.5),
        )
    return _counts_fit_plot(
        observed_counts=data_counts.astype(float),
        observed_errors=data_errors.astype(float),
        observed_label="Data",
        model_counts=np.asarray(model_counts, dtype=float),
        model_label="Post-fit model",
        title=f"Post-fit mass spectrum: {category}",
        out_base=out_base,
        blind_window=blind_window,
        ratio_label="Data / fit",
        ratio_ylim=(0.5, 1.5),
    )


def _asimov_fit_plot(
    *,
    category: str,
    asimov_counts: np.ndarray,
    fit_counts: np.ndarray,
    fit_label: str,
    out_base: Path,
    ratio_ylim: tuple[float, float],
    model_curve: dict[str, Any] | None = None,
    model_component: str = "total",
) -> list[str]:
    zeros = np.zeros_like(asimov_counts, dtype=float)
    if model_curve:
        return _analytic_curve_plot(
            observed_counts=np.asarray(asimov_counts, dtype=float),
            observed_errors=zeros,
            observed_label="Asimov S+B pseudo-data",
            model_curve=model_curve,
            model_component=model_component,
            model_label=fit_label,
            title=f"Asimov significance fit: {category}",
            out_base=out_base,
            blind_window=None,
            ratio_label="Asimov / fit",
            ratio_ylim=ratio_ylim,
        )
    return _counts_fit_plot(
        observed_counts=np.asarray(asimov_counts, dtype=float),
        observed_errors=zeros,
        observed_label="Asimov S+B pseudo-data",
        model_counts=np.asarray(fit_counts, dtype=float),
        model_label=fit_label,
        title=f"Asimov significance fit: {category}",
        out_base=out_base,
        blind_window=None,
        ratio_label="Asimov / fit",
        ratio_ylim=ratio_ylim,
    )


def _sideband_fit_plot(
    *,
    category: str,
    data_masses: np.ndarray,
    background_pdf,
    mass_var,
    sideband_yield: float,
    bins: np.ndarray,
    out_base: Path,
    blind_window: list[float],
) -> list[str]:
    data_counts, _ = np.histogram(data_masses, bins=bins)
    data_errors = np.sqrt(data_counts)
    x_values = _fit_range_curve_x(bins)
    sidebands = [(float(bins[0]), float(blind_window[0])), (float(blind_window[1]), float(bins[-1]))]
    curve = pdf_to_curve(
        background_pdf,
        mass_var,
        float(sideband_yield),
        x_values=x_values,
        normalize_regions=sidebands,
    )
    model_curve = {
        "x": curve["x"],
        "signal": np.zeros(len(curve["x"]), dtype=float).tolist(),
        "background": curve["y"],
        "total": curve["y"],
    }
    return _analytic_curve_plot(
        observed_counts=data_counts.astype(float),
        observed_errors=data_errors.astype(float),
        observed_label="Blinded data",
        model_curve=model_curve,
        model_component="background",
        model_label="Analytic background PDF fit to sidebands",
        title=f"Sideband background fit: {category}",
        out_base=out_base,
        blind_window=blind_window,
        ratio_label="Data / B",
        ratio_ylim=(0.5, 1.5),
    )


def _cutflow_plot(cutflow_table: dict[str, Any], out_base: Path) -> list[str]:
    steps = list(cutflow_table["aggregated"].keys())
    x = np.arange(len(steps))
    data_counts = np.array([cutflow_table["aggregated"][step]["data_unweighted"] for step in steps], dtype=float)
    prompt_counts = np.array([cutflow_table["aggregated"][step]["prompt_diphoton_weighted"] for step in steps], dtype=float)
    signal_counts = np.array([cutflow_table["aggregated"][step]["signal_weighted"] for step in steps], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.25
    ax.bar(x - width, data_counts, width=width, color=PLOT_COLORS["data"], alpha=0.75, label="Data")
    ax.bar(x, prompt_counts, width=width, color=PLOT_COLORS["prompt_diphoton"], alpha=0.75, label="Prompt diphoton")
    ax.bar(x + width, signal_counts, width=width, color=PLOT_COLORS["signal"], alpha=0.75, label="Signal")
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=20, ha="right")
    ax.text(0.02, 0.95, "cut flow", transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    return _save_figure(fig, out_base)


def generate_plots(processed_samples: list[dict], summary: dict, fit_context: dict, outputs: Path, cutflow_table: dict[str, Any]) -> dict[str, Any]:
    plot_root = ensure_dir(outputs / "report" / "plots")
    object_dir = ensure_dir(plot_root / "objects")
    event_dir = ensure_dir(plot_root / "events")
    category_dir = ensure_dir(plot_root / "categories")
    control_dir = ensure_dir(plot_root / "control_regions")
    fit_dir = ensure_dir(plot_root / "fits")
    asimov_dir = ensure_dir(plot_root / "asimov_fits")
    signal_dir = ensure_dir(plot_root / "signal_shape")
    smoothing_dir = ensure_dir(plot_root / "smoothing_sb_fit")

    data_all = _group_events(processed_samples, "data")
    signal_all = _group_events(processed_samples, "signal")
    prompt_all = _group_events(processed_samples, "prompt_diphoton")
    blind_window = (
        summary["runtime_defaults"]["signal_window_gev"]
        if summary["runtime_defaults"]["blinding"]["plot_signal_window"]
        else None
    )
    measurement_plot_payload = fit_context.get("measurement_plot_payload", {"categories": {}, "combined": {"total_counts": [0.0] * 55}})
    asimov_plot_payload = fit_context.get("asimov_plot_payload", {"categories": {}, "combined": {}})

    manifest: dict[str, Any] = {
        "status": "ok",
        "fit_visualization_semantics": {
            "pdf_rendering": "analytic_roofit_pdf_curves",
            "data_rendering": "binned_points",
            "blinded_window_gev": blind_window,
            "sideband_fit_ranges_gev": [[105.0, 120.0], [130.0, 160.0]],
        },
        "plot_groups": {
            "objects": {},
            "events": {},
            "categories": {},
            "control_regions_prefit": {},
            "control_regions_postfit": {},
            "signal_shape": {},
            "smoothing_sb_fit": {},
            "fits": {},
            "asimov_fits": {"free_fit": {}, "mu0_fit": {}},
        },
    }

    object_specs = [
        ("photon_pt_leading", data_all.get("lead_pt", np.array([])), prompt_all.get("lead_pt", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("lead_pt", np.array([])), signal_all.get("weight", np.array([])), np.linspace(25.0, 250.0, 16), r"Leading photon $p_T$ [GeV]", "Events"),
        ("photon_pt_subleading", data_all.get("sublead_pt", np.array([])), prompt_all.get("sublead_pt", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("sublead_pt", np.array([])), signal_all.get("weight", np.array([])), np.linspace(25.0, 200.0, 15), r"Subleading photon $p_T$ [GeV]", "Events"),
        ("photon_eta_leading", data_all.get("lead_eta", np.array([])), prompt_all.get("lead_eta", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("lead_eta", np.array([])), signal_all.get("weight", np.array([])), np.linspace(-2.4, 2.4, 13), r"Leading photon $\eta$", "Events"),
        ("photon_eta_subleading", data_all.get("sublead_eta", np.array([])), prompt_all.get("sublead_eta", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("sublead_eta", np.array([])), signal_all.get("weight", np.array([])), np.linspace(-2.4, 2.4, 13), r"Subleading photon $\eta$", "Events"),
    ]
    for name, data_values, bkg_values, bkg_weights, sig_values, sig_weights, bins, xlabel, ylabel in object_specs:
        manifest["plot_groups"]["objects"][name] = _ratio_plot(
            data_values=data_values,
            mc_background_values=bkg_values,
            mc_background_weights=bkg_weights,
            signal_values=sig_values,
            signal_weights=sig_weights,
            bins=bins,
            xlabel=xlabel,
            ylabel=ylabel,
            title=name.replace("_", " "),
            out_base=object_dir / name,
        )

    event_specs = [
        ("diphoton_mass_preselection", data_all.get("mgg", np.array([])), prompt_all.get("mgg", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("mgg", np.array([])), signal_all.get("weight", np.array([])), np.linspace(105.0, 160.0, 56), r"$m_{\gamma\gamma}$ [GeV]", "Events / 1 GeV", blind_window),
        ("diphoton_pt", data_all.get("ptt", np.array([])), prompt_all.get("ptt", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("ptt", np.array([])), signal_all.get("weight", np.array([])), np.linspace(0.0, 200.0, 21), r"$p_{Tt}$ [GeV]", "Events", None),
        ("diphoton_deltaR", data_all.get("delta_r", np.array([])), prompt_all.get("delta_r", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("delta_r", np.array([])), signal_all.get("weight", np.array([])), np.linspace(0.0, 6.0, 13), r"$\Delta R(\gamma_1, \gamma_2)$", "Events", None),
        ("photon_multiplicity", data_all.get("photon_multiplicity", np.array([])), prompt_all.get("photon_multiplicity", np.array([])), prompt_all.get("weight", np.array([])), signal_all.get("photon_multiplicity", np.array([])), signal_all.get("weight", np.array([])), np.arange(1.5, 5.6, 1.0), "Photon multiplicity", "Events", None),
    ]
    for name, data_values, bkg_values, bkg_weights, sig_values, sig_weights, bins, xlabel, ylabel, local_blind in event_specs:
        manifest["plot_groups"]["events"][name] = _ratio_plot(
            data_values=data_values,
            mc_background_values=bkg_values,
            mc_background_weights=bkg_weights,
            signal_values=sig_values,
            signal_weights=sig_weights,
            bins=bins,
            xlabel=xlabel,
            ylabel=ylabel,
            title=name.replace("_", " "),
            out_base=event_dir / name,
            blind_window=local_blind,
        )

    manifest["plot_groups"]["events"]["cutflow_plot"] = _cutflow_plot(cutflow_table, event_dir / "cutflow_plot")

    active_categories = fit_context["fit_summary"]["categories"]
    bins = np.linspace(105.0, 160.0, 56)
    for category in active_categories:
        data_cat = _group_events(processed_samples, "data", category)
        signal_cat = _group_events(processed_samples, "signal", category)
        prompt_cat = _group_events(processed_samples, "prompt_diphoton", category)
        manifest["plot_groups"]["categories"][category] = _ratio_plot(
            data_values=data_cat.get("mgg", np.array([])),
            mc_background_values=prompt_cat.get("mgg", np.array([])),
            mc_background_weights=prompt_cat.get("weight", np.array([])),
            signal_values=signal_cat.get("mgg", np.array([])),
            signal_weights=signal_cat.get("weight", np.array([])),
            bins=bins,
            xlabel=r"$m_{\gamma\gamma}$ [GeV]",
            ylabel="Events / 1 GeV",
            title=f"Category validation: {category}",
            out_base=category_dir / f"diphoton_mass_category_{category}",
            blind_window=blind_window,
        )
        sideband_yield = fit_context["category_context"][category]["background_choice"]["observed_data_sideband_count"]
        manifest["plot_groups"]["control_regions_prefit"][category] = _sideband_fit_plot(
            category=category,
            data_masses=fit_context["category_context"][category]["data_masses"],
            background_pdf=fit_context["final_models"][category]["background_pdf"],
            mass_var=fit_context["common_mass"],
            sideband_yield=float(sideband_yield),
            bins=bins,
            out_base=control_dir / f"prefit_sidebands_{category}",
            blind_window=blind_window or [120.0, 130.0],
        )

        signal_payload = fit_context["aggregated"]["signal"][category]
        manifest["plot_groups"]["signal_shape"][category] = _signal_shape_plot(
            category,
            signal_payload["mgg"],
            signal_payload["weight"],
            fit_context["final_models"][category]["signal_pdf"],
            fit_context["common_mass"],
            fit_context["category_context"][category]["expected_signal_yield"],
            signal_dir / f"signal_mgg_{category}",
        )

        display = fit_context["template_display"]["categories"][category]
        selected_model = fit_context["category_context"][category]["background_choice"]["selected_model"]
        selected_row = next(
            row
            for row in fit_context["background_scan_artifact"]["categories"][category]["candidates"]
            if row["model"] == selected_model
        )
        selection_counts = np.asarray(display["selection_counts"], dtype=float)
        selection_signal_yield = float(selected_row["n_spur"])
        selection_background_yield = float(np.sum(selection_counts) - selection_signal_yield)
        selection_curve = _spurious_fit_curve(
            category=category,
            selected_model=selected_model,
            snapshot=selected_row["spur_fit_param_snapshot"],
            signal_pdf=fit_context["final_models"][category]["signal_pdf"],
            mass_var=fit_context["common_mass"],
            signal_yield=selection_signal_yield,
            background_yield=selection_background_yield,
            x_values=_fit_range_curve_x(bins),
        )
        smoothing_entry = {
            "unsmoothed_template": _template_fit_plot(
                counts=np.asarray(display["unsmoothed_counts"], dtype=float),
                fit_counts=np.asarray(display["unsmoothed_fit_total_counts"], dtype=float),
                xlabel=r"$m_{\gamma\gamma}$ [GeV]",
                ylabel="Events / 1 GeV",
                title=f"Unsmoothed sideband-normalized template: {category}",
                out_base=smoothing_dir / f"unsmoothed_template_{category}",
            ),
            "selected_spurious_fit": _template_fit_plot(
                counts=selection_counts,
                fit_counts=np.asarray(display["selection_fit_total_counts"], dtype=float),
                xlabel=r"$m_{\gamma\gamma}$ [GeV]",
                ylabel="Events / 1 GeV",
                title=f"Selected spurious-signal fit: {category}",
                out_base=smoothing_dir / f"selected_spurious_fit_{category}",
                model_curve=selection_curve,
                model_label="Analytic B+spurious-signal fit",
            ),
        }
        if fit_context["smoothing_applied"]:
            unsmoothed_counts = np.asarray(display["unsmoothed_counts"], dtype=float)
            smoothed_counts = np.asarray(display["selection_counts"], dtype=float)
            sideband_scale_factor = float(display["sideband_scale_factor"])
            unsmoothed_unc = _weighted_histogram_uncertainty(
                prompt_cat.get("mgg", np.array([])),
                prompt_cat.get("weight", np.array([])),
                bins,
                scale=sideband_scale_factor,
            )
            smoothing_entry["smoothing_effect_overlay"] = _smoothing_overlay_plot(
                unsmoothed_counts=unsmoothed_counts,
                unsmoothed_unc=unsmoothed_unc,
                smoothed_counts=smoothed_counts,
                xlabel=r"$m_{\gamma\gamma}$ [GeV]",
                ylabel="Events / 1 GeV",
                title=f"Smoothing effect: {category}",
                out_base=smoothing_dir / f"smoothing_effect_overlay_{category}",
            )
            smoothing_entry["smoothed_selection_fit"] = _template_fit_plot(
                counts=smoothed_counts,
                fit_counts=np.asarray(display["selection_fit_total_counts"], dtype=float),
                xlabel=r"$m_{\gamma\gamma}$ [GeV]",
                ylabel="Events / 1 GeV",
                title=f"Smoothed template fit: {category}",
                out_base=smoothing_dir / f"smoothed_template_fit_{category}",
                model_curve=selection_curve,
                model_label="Analytic B+spurious-signal fit",
            )
        manifest["plot_groups"]["smoothing_sb_fit"][category] = smoothing_entry

        model_counts = np.asarray(
            measurement_plot_payload["categories"].get(category, {}).get("total_counts", [0.0] * 55),
            dtype=float,
        )
        model_curve = _component_curve_payload(
            fit_context["final_models"][category],
            fit_context["common_mass"],
            x_values=_fit_range_curve_x(bins),
        )
        manifest["plot_groups"]["fits"][category] = _fit_result_plot(
            category=category,
            data_masses=fit_context["category_context"][category]["data_masses"],
            model_counts=model_counts,
            bins=bins,
            out_base=fit_dir / f"diphoton_mass_fit_{category}",
            blind_window=blind_window,
            model_curve=model_curve,
        )
        manifest["plot_groups"]["control_regions_postfit"][category] = _fit_result_plot(
            category=category,
            data_masses=fit_context["category_context"][category]["data_masses"],
            model_counts=model_counts,
            bins=bins,
            out_base=control_dir / f"postfit_sidebands_{category}",
            blind_window=blind_window,
            model_curve=model_curve,
        )
        if category in asimov_plot_payload.get("categories", {}):
            category_asimov = asimov_plot_payload["categories"][category]
            manifest["plot_groups"]["asimov_fits"]["free_fit"][category] = _asimov_fit_plot(
                category=category,
                asimov_counts=np.asarray(category_asimov["asimov_counts"], dtype=float),
                fit_counts=np.asarray(category_asimov["free_fit"]["total_counts"], dtype=float),
                fit_label="Analytic S+B fit",
                out_base=asimov_dir / f"diphoton_mass_asimov_free_fit_{category}",
                ratio_ylim=(0.5, 1.5),
                model_curve=category_asimov["free_fit"].get("curve"),
            )
            manifest["plot_groups"]["asimov_fits"]["mu0_fit"][category] = _asimov_fit_plot(
                category=category,
                asimov_counts=np.asarray(category_asimov["asimov_counts"], dtype=float),
                fit_counts=np.asarray(category_asimov["mu0_fit"]["total_counts"], dtype=float),
                fit_label="Analytic B-only fit",
                out_base=asimov_dir / f"diphoton_mass_asimov_mu0_fit_{category}",
                ratio_ylim=(0.5, 2.0),
                model_curve=category_asimov["mu0_fit"].get("curve"),
                model_component="background",
            )

    combined_data = np.concatenate([fit_context["category_context"][category]["data_masses"] for category in active_categories]) if active_categories else np.array([])
    combined_model = np.asarray(measurement_plot_payload.get("combined", {}).get("total_counts", [0.0] * 55), dtype=float)
    combined_curve = _combine_curve_payloads(
        [
            _component_curve_payload(fit_context["final_models"][category], fit_context["common_mass"], x_values=_fit_range_curve_x(bins))
            for category in active_categories
        ]
    )
    manifest["plot_groups"]["fits"]["combined"] = _fit_result_plot(
        category="combined",
        data_masses=combined_data,
        model_counts=combined_model,
        bins=bins,
        out_base=fit_dir / "diphoton_mass_fit",
        blind_window=blind_window,
        model_curve=combined_curve,
    )
    if asimov_plot_payload.get("combined"):
        manifest["plot_groups"]["asimov_fits"]["free_fit"]["combined"] = _asimov_fit_plot(
            category="combined",
            asimov_counts=np.asarray(asimov_plot_payload["combined"]["asimov_counts"], dtype=float),
            fit_counts=np.asarray(asimov_plot_payload["combined"]["free_fit"]["total_counts"], dtype=float),
            fit_label="Analytic S+B fit",
            out_base=asimov_dir / "diphoton_mass_asimov_free_fit",
            ratio_ylim=(0.5, 1.5),
            model_curve=asimov_plot_payload["combined"]["free_fit"].get("curve"),
        )
        manifest["plot_groups"]["asimov_fits"]["mu0_fit"]["combined"] = _asimov_fit_plot(
            category="combined",
            asimov_counts=np.asarray(asimov_plot_payload["combined"]["asimov_counts"], dtype=float),
            fit_counts=np.asarray(asimov_plot_payload["combined"]["mu0_fit"]["total_counts"], dtype=float),
            fit_label="Analytic B-only fit",
            out_base=asimov_dir / "diphoton_mass_asimov_mu0_fit",
            ratio_ylim=(0.5, 2.0),
            model_curve=asimov_plot_payload["combined"]["mu0_fit"].get("curve"),
            model_component="background",
        )

    write_json(manifest, plot_root / "manifest.json")
    write_json(
        {
            "status": "ok",
            "categories": manifest["plot_groups"]["smoothing_sb_fit"],
        },
        smoothing_dir / "mc_template_sb_fit_manifest.json",
    )
    write_json(
        {
            "status": "ok",
            "categories": manifest["plot_groups"]["signal_shape"],
        },
        signal_dir / "signal_shape_manifest.json",
    )
    write_json(
        {
            "status": "ok",
            "plot_groups": manifest["plot_groups"]["asimov_fits"],
        },
        asimov_dir / "asimov_fit_manifest.json",
    )
    return manifest
