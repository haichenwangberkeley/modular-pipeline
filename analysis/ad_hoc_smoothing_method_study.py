from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from analysis.common import ensure_dir, read_json, write_json, write_text
from analysis.plotting.blinded_regions import LUMI_LABEL, _weighted_histogram_uncertainty
from analysis.selections.engine import CATEGORY_ORDER
from analysis.stats.fit import _fit_template_plus_signal, _sideband_scale_factor, aggregate_processed_samples
from analysis.stats.models import background_candidate, configure_mass_var, fit_pdf, histogram_counts, make_weighted_dataset, th1_smooth

ROOT.gROOT.SetBatch(True)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    kind: str
    smoother: Callable[[np.ndarray], np.ndarray]


def _save_figure(fig: plt.Figure, out_base: Path) -> list[str]:
    ensure_dir(out_base.parent)
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf_path), str(png_path)]


def _load_processed_samples(outputs: Path) -> list[dict[str, Any]]:
    processed_manifest = read_json(outputs / "hists" / "processed_samples.json")
    processed_samples: list[dict[str, Any]] = []
    for item in processed_manifest["samples"]:
        arr = np.load(item["cache_path"], allow_pickle=True)
        events = {key: arr[key] for key in arr.files}
        if "category" in events:
            events["category"] = events["category"].astype(str)
        processed_samples.append(
            {
                "sample_id": item["sample_id"],
                "process_key": item["process_key"],
                "kind": item["kind"],
                "analysis_role": item["analysis_role"],
                "cutflow": item["cutflow"],
                "events": events,
                "cache_path": item.get("cache_path"),
            }
        )
    return processed_samples


def _preserve_integral(values: np.ndarray, target_sum: float) -> np.ndarray:
    smoothed = np.clip(np.asarray(values, dtype=float), 0.0, None)
    current_sum = float(np.sum(smoothed))
    if current_sum > 0.0 and target_sum > 0.0:
        smoothed *= float(target_sum) / current_sum
    return smoothed


def _markov_smooth(counts: np.ndarray, aver_window: int) -> np.ndarray:
    payload = array("d", np.asarray(counts, dtype=float).tolist())
    result = ROOT.TSpectrum().SmoothMarkov(payload, len(payload), aver_window)
    if result:
        raise RuntimeError(f"TSpectrum::SmoothMarkov failed: {result}")
    return _preserve_integral(np.asarray(payload, dtype=float), float(np.sum(counts)))


def _kernel_smooth(counts: np.ndarray, bandwidth: float) -> np.ndarray:
    centers = np.linspace(105.5, 159.5, len(counts)).astype("float64")
    x_values = array("d", centers.tolist())
    y_values = array("d", np.asarray(counts, dtype=float).tolist())
    graph = ROOT.TGraph(len(centers), x_values, y_values)
    smoother = ROOT.TGraphSmooth("smoothing_study")
    out = smoother.SmoothKern(graph, "normal", bandwidth, len(centers), x_values)
    smoothed = np.array([out.Eval(float(center)) for center in centers], dtype=float)
    return _preserve_integral(smoothed, float(np.sum(counts)))


def _gaussian_smooth(counts: np.ndarray, sigma: float) -> np.ndarray:
    return _preserve_integral(gaussian_filter1d(np.asarray(counts, dtype=float), sigma=sigma, mode="nearest"), float(np.sum(counts)))


def _savgol_smooth(counts: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    return _preserve_integral(
        savgol_filter(np.asarray(counts, dtype=float), window_length=window_length, polyorder=polyorder, mode="interp"),
        float(np.sum(counts)),
    )


def _method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("none", "Unsmoothed", "baseline", lambda counts: np.asarray(counts, dtype=float).copy()),
        MethodSpec("th1_smooth_1", "TH1::Smooth x1", "root_th1", lambda counts: th1_smooth(counts, 1)),
        MethodSpec("th1_smooth_2", "TH1::Smooth x2", "root_th1", lambda counts: th1_smooth(counts, 2)),
        MethodSpec("th1_smooth_5", "TH1::Smooth x5", "root_th1", lambda counts: th1_smooth(counts, 5)),
        MethodSpec("tspectrum_markov_w3", "TSpectrum Markov w=3", "root_markov", lambda counts: _markov_smooth(counts, 3)),
        MethodSpec("tgraph_kern_bw1p0", "TGraphSmooth kernel bw=1.0", "root_graph", lambda counts: _kernel_smooth(counts, 1.0)),
        MethodSpec("scipy_gaussian_sigma1", "SciPy Gaussian sigma=1", "scipy", lambda counts: _gaussian_smooth(counts, 1.0)),
        MethodSpec("scipy_savgol_7_2", "SciPy SavGol 7/2", "scipy", lambda counts: _savgol_smooth(counts, 7, 2)),
    ]


def _sideband_rows(category: str, data_masses: np.ndarray, candidate_models: list[str]) -> dict[str, dict[str, Any]]:
    sideband_mass_var = configure_mass_var(f"mgg_side_study_{category}")
    sideband_dataset = make_weighted_dataset(f"data_side_study_{category}", sideband_mass_var, data_masses)
    rows: dict[str, dict[str, Any]] = {}
    for kind in candidate_models:
        model = background_candidate(f"side_study_{category}", sideband_mass_var, kind)
        side_fit = fit_pdf(model.pdf, sideband_dataset, fit_range="sideband_lo,sideband_hi")
        rows[kind] = {
            "complexity": model.complexity,
            "sideband_fit_status": int(side_fit.status()),
            "sideband_cov_qual": int(side_fit.covQual()),
            "sideband_aic": float(2.0 * len(model.params) + 2.0 * float(side_fit.minNll())),
            "sideband_param_snapshot": {param.GetName(): float(param.getVal()) for param in model.params},
        }
    return rows


def _select_candidate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    passing = [row for row in rows if row["passes"]]
    if passing:
        passing.sort(key=lambda row: (row["complexity"], abs(row["n_spur"]), row["sideband_aic"]))
        return passing[0], "lowest-complexity candidate passing r_spur < 0.2"
    ranked = sorted(rows, key=lambda row: (row["r_spur"], row["complexity"], row["sideband_aic"]))
    return ranked[0], "no candidate passed the spurious-signal threshold; chose the smallest r_spur"


def _comparison_plot(
    *,
    category: str,
    unsmoothed_counts: np.ndarray,
    unsmoothed_unc: np.ndarray,
    smoothed_by_method: dict[str, np.ndarray],
    labels: dict[str, str],
    out_base: Path,
) -> list[str]:
    bins = np.linspace(105.0, 160.0, len(unsmoothed_counts) + 1)
    colors = {
        "th1_smooth_1": "#1d3557",
        "th1_smooth_2": "#457b9d",
        "th1_smooth_5": "#2a9d8f",
        "tspectrum_markov_w3": "#e76f51",
        "tgraph_kern_bw1p0": "#6a4c93",
        "scipy_gaussian_sigma1": "#f4a261",
        "scipy_savgol_7_2": "#d62828",
    }
    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    ax.step(bins, np.r_[unsmoothed_counts, unsmoothed_counts[-1]], where="post", color="#000000", linewidth=2.2, label="Unsmoothed")
    ax.fill_between(
        bins,
        np.r_[np.clip(unsmoothed_counts - unsmoothed_unc, 0.0, None), np.clip(unsmoothed_counts - unsmoothed_unc, 0.0, None)[-1]],
        np.r_[unsmoothed_counts + unsmoothed_unc, (unsmoothed_counts + unsmoothed_unc)[-1]],
        step="post",
        color="#b0b0b0",
        alpha=0.35,
        label="Unsmoothed stat. unc.",
    )
    ymax = float(np.max(unsmoothed_counts + unsmoothed_unc)) if len(unsmoothed_counts) else 1.0
    ratio_values = []
    for key, counts in smoothed_by_method.items():
        if key == "none":
            continue
        ymax = max(ymax, float(np.max(counts)) if len(counts) else 0.0)
        ax.step(bins, np.r_[counts, counts[-1]], where="post", color=colors[key], linewidth=1.8, label=labels[key])
        ratio = np.divide(counts, unsmoothed_counts, out=np.full_like(counts, np.nan, dtype=float), where=unsmoothed_counts > 0.0)
        ratio_values.append(ratio[np.isfinite(ratio)])
        rax.step(bins, np.r_[ratio, ratio[-1]], where="post", color=colors[key], linewidth=1.6)
    ax.set_xlim(105.0, 160.0)
    ax.set_ylim(0.0, 1.45 * max(ymax, 1.0))
    ax.set_ylabel("Events / 1 GeV")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.text(0.02, 0.95, f"Smoothing-method comparison: {category}", transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.text(0.02, 0.87, LUMI_LABEL, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    ratio_band = np.divide(unsmoothed_unc, unsmoothed_counts, out=np.full_like(unsmoothed_unc, np.nan, dtype=float), where=unsmoothed_counts > 0.0)
    rax.axhline(1.0, color="#555555", linewidth=1.2)
    rax.fill_between(
        bins,
        np.r_[1.0 - ratio_band, (1.0 - ratio_band)[-1]],
        np.r_[1.0 + ratio_band, (1.0 + ratio_band)[-1]],
        step="post",
        color="#b0b0b0",
        alpha=0.35,
    )
    finite_ratio = np.concatenate(ratio_values) if ratio_values else np.array([1.0])
    finite_ratio = finite_ratio[np.isfinite(finite_ratio)]
    ratio_min = float(np.min(finite_ratio)) if len(finite_ratio) else 1.0
    ratio_max = float(np.max(finite_ratio)) if len(finite_ratio) else 1.0
    lower = max(0.0, min(0.5, ratio_min - 0.1 * max(ratio_max - ratio_min, 0.1)))
    upper = max(1.5, min(5.0, ratio_max + 0.1 * max(ratio_max - ratio_min, 0.1)))
    rax.set_ylim(lower, upper)
    rax.set_ylabel("Method / unsm.")
    rax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    return _save_figure(fig, out_base)


def _heatmap_plot(
    *,
    categories: list[str],
    methods: list[MethodSpec],
    values: np.ndarray,
    out_base: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(2.5, float(np.nanmax(values))))
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([method.label for method in methods], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title(r"Best selected $r_{\mathrm{spur}}$ by smoothing method")
    for row in range(len(categories)):
        for col in range(len(methods)):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Selected $r_{\mathrm{spur}}$")
    return _save_figure(fig, out_base)


def _build_markdown_summary(summary_payload: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Ad Hoc Smoothing Method Study",
        "",
        f"- Source template sample: `{summary_payload['source_template_sample']}`",
        "- Selection metric: choose the lowest-complexity candidate with `r_spur < 0.2`; otherwise choose the smallest `r_spur`.",
        "- Candidate background functions: " + ", ".join(summary_payload["candidate_models"]),
        "",
    ]
    for category in summary_payload["categories"]:
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Method | Selected model | r_spur | N_spur | sigma_Nsig | Pass |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for method in summary_payload["methods"]:
            payload = summary_payload["results"][category][method["key"]]
            lines.append(
                f"| {method['label']} | `{payload['selected_model']}` | {payload['r_spur']:.3f} | {payload['n_spur']:.3f} | {payload['sigma_nsig']:.3f} | `{payload['passes']}` |"
            )
        lines.append("")
    write_text("\n".join(lines) + "\n", out_path)


def run_study(outputs: Path, fit_id: str) -> dict[str, Any]:
    study_dir = ensure_dir(outputs / "report" / "ad_hoc_smoothing_method_study")
    plot_dir = ensure_dir(outputs / "report" / "plots" / "ad_hoc_smoothing_method_study")

    summary = read_json(outputs / "summary.normalized.json")
    signal_pdf = read_json(outputs / "fit" / fit_id / "signal_pdf.json")
    fit_results = read_json(outputs / "fit" / fit_id / "results.json")
    sample_selection = read_json(outputs / "report" / "mc_sample_selection.json")

    processed_samples = _load_processed_samples(outputs)
    aggregated = aggregate_processed_samples(processed_samples)
    candidate_models = list(summary["runtime_defaults"]["background_model"]["candidates"])
    methods = _method_specs()
    method_labels = {method.key: method.label for method in methods}

    summary_payload: dict[str, Any] = {
        "status": "ok",
        "fit_id": fit_id,
        "source_template_sample": sample_selection["diphoton_template_policy"]["selected_nominal_background_template_sample"],
        "candidate_models": candidate_models,
        "categories": fit_results["categories"],
        "methods": [{"key": method.key, "label": method.label, "kind": method.kind} for method in methods],
        "results": {},
        "plots": {"comparisons": {}, "heatmap": []},
    }

    heatmap_values = np.zeros((len(fit_results["categories"]), len(methods)), dtype=float)

    for category_index, category in enumerate(fit_results["categories"]):
        data_payload = aggregated["data"][category]
        prompt_payload = aggregated["prompt_diphoton"][category]
        data_masses = np.asarray(data_payload["mgg"], dtype=float)
        template_masses = np.asarray(prompt_payload["mgg"], dtype=float)
        template_weights = np.asarray(prompt_payload["weight"], dtype=float)
        scale_factor, observed_sb, template_sb = _sideband_scale_factor(data_masses, template_masses, template_weights)
        scaled_weights = template_weights * scale_factor
        unsmoothed_counts = histogram_counts(template_masses, scaled_weights)
        bins = np.linspace(105.0, 160.0, len(unsmoothed_counts) + 1)
        unsmoothed_unc = _weighted_histogram_uncertainty(template_masses, template_weights, bins, scale=scale_factor)
        method_counts = {method.key: method.smoother(unsmoothed_counts) for method in methods}
        sideband_rows = _sideband_rows(category, data_masses, candidate_models)
        expected_signal_yield = float(fit_results["expected_signal_yields"][category])
        signal_artifact = signal_pdf["categories"][category]

        summary_payload["results"][category] = {}
        for method_index, method in enumerate(methods):
            candidate_rows = []
            for kind in candidate_models:
                sideband = sideband_rows[kind]
                spur_fit = _fit_template_plus_signal(category, kind, method_counts[method.key], signal_artifact, expected_signal_yield)
                candidate_rows.append(
                    {
                        "model": kind,
                        "complexity": sideband["complexity"],
                        "sideband_fit_status": sideband["sideband_fit_status"],
                        "sideband_cov_qual": sideband["sideband_cov_qual"],
                        "sideband_aic": sideband["sideband_aic"],
                        "n_spur": spur_fit["n_spur"],
                        "sigma_nsig": spur_fit["sigma_nsig"],
                        "r_spur": spur_fit["r_spur"],
                        "passes": bool(spur_fit["r_spur"] < 0.2),
                    }
                )
            selected, rationale = _select_candidate(candidate_rows)
            summary_payload["results"][category][method.key] = {
                "label": method.label,
                "kind": method.kind,
                "selected_model": selected["model"],
                "selected_complexity": selected["complexity"],
                "rationale": rationale,
                "r_spur": selected["r_spur"],
                "n_spur": selected["n_spur"],
                "sigma_nsig": selected["sigma_nsig"],
                "passes": bool(selected["passes"]),
                "sideband_scale_factor": float(scale_factor),
                "observed_data_sideband_count": int(observed_sb),
                "template_sideband_yield_before_scaling": float(template_sb),
                "candidate_rows": candidate_rows,
            }
            heatmap_values[category_index, method_index] = float(selected["r_spur"])

        comparison_paths = _comparison_plot(
            category=category,
            unsmoothed_counts=unsmoothed_counts,
            unsmoothed_unc=unsmoothed_unc,
            smoothed_by_method=method_counts,
            labels=method_labels,
            out_base=plot_dir / f"smoothing_method_comparison_{category}",
        )
        summary_payload["plots"]["comparisons"][category] = comparison_paths

    heatmap_paths = _heatmap_plot(
        categories=fit_results["categories"],
        methods=methods,
        values=heatmap_values,
        out_base=plot_dir / "best_rspur_heatmap",
    )
    summary_payload["plots"]["heatmap"] = heatmap_paths

    write_json(summary_payload, study_dir / "summary.json")
    _build_markdown_summary(summary_payload, study_dir / "summary.md")
    write_json(
        {
            "status": "ok",
            "plots": summary_payload["plots"],
        },
        study_dir / "manifest.json",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--fit-id", default="FIT1")
    args = parser.parse_args()
    payload = run_study(Path(args.outputs), args.fit_id)
    print(
        {
            "status": payload["status"],
            "study_dir": str(Path(args.outputs) / "report" / "ad_hoc_smoothing_method_study"),
            "plot_dir": str(Path(args.outputs) / "report" / "plots" / "ad_hoc_smoothing_method_study"),
        }
    )


if __name__ == "__main__":
    main()
