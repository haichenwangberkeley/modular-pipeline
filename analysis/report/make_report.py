from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis.common import read_json, write_json, write_text
from analysis.stats.fit import FIT_ID


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _select_png(paths: list[str]) -> str:
    for path in paths:
        if path.endswith(".png"):
            return path
    return paths[0]


def _image_block(report_dir: Path, image_path: str, alt: str, caption: str) -> str:
    rel_path = os.path.relpath(image_path, report_dir)
    return f"![{alt}]({rel_path})\n\n*Caption:* {caption}"


def _category_label(category: str) -> str:
    return category.replace("_", " ")


def _registry_summary(registry: list[dict[str, Any]]) -> tuple[str, str]:
    data_samples = [sample["sample_id"] for sample in registry if sample["kind"] == "data"]
    mc_samples = [sample for sample in registry if sample["kind"] != "data"]
    generators = sorted({sample["generator"] for sample in mc_samples if sample.get("generator")})
    simulations = sorted({sample["simulation_config"] for sample in mc_samples if sample.get("simulation_config")})
    dataset_text = (
        f"The run uses {len(data_samples)} data ROOT samples spanning the open-data periods "
        f"{', '.join(data_samples[:4])}{' ...' if len(data_samples) > 4 else ''} and {len(mc_samples)} MC samples."
    )
    mc_text = (
        f"Nominal and alternative MC inputs cover generators {', '.join(generators) if generators else 'not recorded'} "
        f"with simulation configurations {', '.join(simulations) if simulations else 'not recorded'}."
    )
    return dataset_text, mc_text


def _build_report_text(
    *,
    report_dir: Path,
    summary: dict[str, Any],
    fit_result: dict[str, Any],
    asimov: dict[str, Any],
    observed: dict[str, Any],
    sample_selection: dict[str, Any],
    discrepancy: dict[str, Any],
    checkpoint: dict[str, Any],
    skill_extraction: dict[str, Any],
    cutflow: dict[str, Any],
    yields: dict[str, Any],
    smoothing: dict[str, Any],
    smoothing_prov: dict[str, Any],
    mc_lumi: dict[str, Any],
    blinding: dict[str, Any],
    plot_manifest: dict[str, Any],
    registry: list[dict[str, Any]],
    background_choice: dict[str, Any],
) -> str:
    category_rows = [
        [
            category,
            payload["data_entries"],
            f"{payload['prompt_diphoton_yield']:.2f}",
            f"{payload['signal_yield']:.2f}",
        ]
        for category, payload in yields["categories"].items()
    ]
    cutflow_rows = [
        [
            step,
            payload["data_unweighted"],
            f"{payload['prompt_diphoton_weighted']:.2f}",
            f"{payload['signal_weighted']:.2f}",
        ]
        for step, payload in cutflow["aggregated"].items()
    ]

    dataset_text, mc_text = _registry_summary(registry)
    active_categories = fit_result["categories"]
    representative_category = active_categories[0] if active_categories else None
    prompt_nominal = sample_selection["selected_nominal_samples"]["prompt_diphoton"][0]
    prompt_alternatives = sample_selection["alternative_samples"].get("prompt_diphoton", [])
    signal_window = summary["runtime_defaults"]["signal_window_gev"]
    blinded_plots = bool(blinding["plot_signal_window"])
    signal_window_label = f"{signal_window[0]:.0f}-{signal_window[1]:.0f} GeV"
    capped_categories = [
        category
        for category, payload in background_choice["categories"].items()
        if payload.get("capped_noncompliant")
    ]
    fit_dataset_type = fit_result.get("dataset_type", "observed")

    if observed["status"] in {"ok", "warning"}:
        introduction_summary = (
            f"This run executes the five-category ATLAS open-data Higgs-to-diphoton measurement defined in `{summary['source_summary']}` "
            f"with a PyROOT/RooFit primary backend. The central measurement fit returns `mu = {fit_result['mu_hat']:.3f} +/- {fit_result['mu_uncertainty']:.3f}`, "
            f"the observed discovery significance from data is `Z = {observed['z_discovery']:.3f}` with `q0 = {observed['q0']:.3f}`, "
            f"and the Asimov expected sensitivity is `Z = {asimov['z_discovery']:.3f}` with `q0 = {asimov['q0']:.3f}`."
        )
        introduction_policy = "This run was explicitly unblinded, so observed and expected significance are both reported."
        fit_dataset_text = "observed selected events over the full 105-160 GeV range"
        fit_mu_text = f"Best-fit `mu`: `{fit_result['mu_hat']:.3f} +/- {fit_result['mu_uncertainty']:.3f}`"
        fit_range_text = (
            f"The fit range is `{summary['runtime_defaults']['fit_mass_range_gev'][0]:.0f}-{summary['runtime_defaults']['fit_mass_range_gev'][1]:.0f} GeV` "
            f"with plots shown across the full signal region, including `{signal_window_label}`."
        )
        blinding_policy_text = "observed data are shown across the full fit range and observed significance is enabled for this explicitly unblinded run"
        observed_stat_line = f"- Observed significance: `Z = {observed['z_discovery']:.3f}`"
        observed_q0_line = f"- Observed `q0`: `{observed['q0']:.3f}`"
        observed_diag_line = f"- Observed significance diagnostics: {', '.join(observed.get('diagnostics', []) or ['none'])}"
        preselection_caption = (
            "Unblinded preselection diphoton-mass spectrum. Data points are shown across the full 105-160 GeV fit range, including the former signal window."
        )
        prefit_caption = (
            "Sideband background fit for {category}. The chosen analytic background PDF is fit in 105-120 GeV and 130-160 GeV, then overlaid with the data."
        )
        postfit_caption = (
            "Post-fit category mass spectrum for {category}. The full observed spectrum is compared with the fitted analytic signal-plus-background expectation."
        )
    else:
        introduction_summary = (
            f"This run executes the five-category ATLAS open-data Higgs-to-diphoton measurement defined in `{summary['source_summary']}` with a PyROOT/RooFit primary backend. "
            f"Observed signal-region data remain blinded, so the central fit uses signal-plus-background Asimov pseudo-data and returns `mu = {fit_result['mu_hat']:.3f} +/- {fit_result['mu_uncertainty']:.3f}` as an expected-only reference. "
            f"The blinded expected discovery sensitivity from the same Asimov construction is `Z = {asimov['z_discovery']:.3f}` with `q0 = {asimov['q0']:.3f}`."
        )
        introduction_policy = f"Observed signal-strength and observed significance remain `{observed['status']}` because explicit unblinding was not requested. The report therefore keeps observed and expected statistics strictly separated."
        fit_dataset_text = "signal-plus-background Asimov pseudo-data over the full 105-160 GeV range"
        fit_mu_text = f"Expected-only fit `mu`: `{fit_result['mu_hat']:.3f} +/- {fit_result['mu_uncertainty']:.3f}`"
        fit_range_text = (
            f"The fit range is `{summary['runtime_defaults']['fit_mass_range_gev'][0]:.0f}-{summary['runtime_defaults']['fit_mass_range_gev'][1]:.0f} GeV` "
            f"with blinded plotting in `{signal_window_label}`."
        )
        blinding_policy_text = "observed data are masked in the signal window for plots, and observed significance is disabled"
        observed_stat_line = "- Observed significance: blocked by blinding policy"
        observed_q0_line = "- Observed `q0`: not applicable"
        observed_diag_line = "- Observed significance diagnostics: not applicable"
        preselection_caption = (
            "Blinded preselection diphoton-mass spectrum. Data points in the 120-130 GeV signal window are masked, while the sidebands remain visible for validation of the continuum-background description."
        )
        prefit_caption = (
            "Blinded sideband background fit for {category}. The 120-130 GeV window is masked, while the chosen analytic background PDF is fit in 105-120 GeV and 130-160 GeV and overlaid with the visible data."
        )
        postfit_caption = (
            "Post-fit category mass spectrum for {category}. Sideband data are compared with the fitted analytic signal-plus-background expectation, while the blinded window remains masked."
        )

    embedded_blocks = [
        _image_block(
            report_dir,
            _select_png(plot_manifest["plot_groups"]["events"]["diphoton_mass_preselection"]),
            "Preselection diphoton mass",
            preselection_caption,
        ),
        _image_block(
            report_dir,
            _select_png(plot_manifest["plot_groups"]["events"]["cutflow_plot"]),
            "Cut flow",
            "Cut-flow comparison for data, prompt-diphoton MC, and nominal signal MC. This figure documents how the event selection contracts the sample before the category assignment.",
        ),
    ]

    for category in active_categories:
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(plot_manifest["plot_groups"]["control_regions_prefit"][category]),
                f"Prefit sidebands {category}",
                prefit_caption.format(category=_category_label(category)),
            )
        )
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(plot_manifest["plot_groups"]["fits"][category]),
                f"Postfit mass {category}",
                postfit_caption.format(category=_category_label(category)),
            )
        )
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(plot_manifest["plot_groups"]["signal_shape"][category]),
                f"Signal shape {category}",
                f"Signal-shape validation for {_category_label(category)}. The weighted signal MC distribution is overlaid with the fitted analytic double-sided Crystal Ball PDF used downstream in the combined likelihood.",
            )
        )

    if representative_category is not None:
        smoothing_group = plot_manifest["plot_groups"]["smoothing_sb_fit"][representative_category]
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(smoothing_group["unsmoothed_template"]),
                f"Unsmoothed template {representative_category}",
                f"Nominal unsmoothed sideband-normalized prompt-diphoton template for {_category_label(representative_category)}. This plot documents the provenance template before any smoothing-based function-selection step.",
            )
        )
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(smoothing_group["selected_spurious_fit"]),
                f"Selected spurious fit {representative_category}",
                f"Selected analytic background-function-plus-signal fit used for the spurious-signal study in {_category_label(representative_category)}. This figure makes the fitted template choice auditable.",
            )
        )
        if smoothing.get("required") and "smoothing_effect_overlay" in smoothing_group:
            embedded_blocks.append(
                _image_block(
                    report_dir,
                    _select_png(smoothing_group["smoothing_effect_overlay"]),
                    f"Smoothing overlay {representative_category}",
                    f"Unsmoothed-versus-smoothed prompt-diphoton template overlay for {_category_label(representative_category)}. The ratio panel shows the direct effect of the mandatory TH1-based smoothing step.",
                )
            )

    if plot_manifest["plot_groups"].get("asimov_fits", {}).get("free_fit", {}).get("combined"):
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(plot_manifest["plot_groups"]["asimov_fits"]["free_fit"]["combined"]),
                "Asimov free-mu fit combined",
                "Combined binned Asimov signal-plus-background pseudo-data overlaid with the analytic signal-plus-background PDF from the free-mu significance fit over the full 105-160 GeV range.",
            )
        )
    if plot_manifest["plot_groups"].get("asimov_fits", {}).get("mu0_fit", {}).get("combined"):
        embedded_blocks.append(
            _image_block(
                report_dir,
                _select_png(plot_manifest["plot_groups"]["asimov_fits"]["mu0_fit"]["combined"]),
                "Asimov mu0 fit combined",
                "Combined binned Asimov signal-plus-background pseudo-data overlaid with the analytic background-only PDF for the mu=0 conditional significance test.",
            )
        )

    report_text = f"""# H->gammagamma Analysis Report

## Introduction

{introduction_summary}

{introduction_policy}

## Dataset Description

{dataset_text}

{mc_text}

- Experiment: {summary["analysis_metadata"]["experiment"]}
- Analysis name: {summary["analysis_metadata"]["analysis_name"]}
- Center-of-mass energy: {summary["analysis_metadata"]["energy"]}
- Target luminosity: {summary["analysis_metadata"]["luminosity"]}
- Primary backend: `pyroot_roofit`

## Object Definitions And Event Selection

- Photons must satisfy `pT > {summary["runtime_defaults"]["photon_selection"]["pt_min_gev"]:.0f} GeV`, `|eta| < {summary["runtime_defaults"]["photon_selection"]["abs_eta_max"]:.2f}`, crack veto `{summary["runtime_defaults"]["photon_selection"]["eta_crack"][0]:.2f} < |eta| < {summary["runtime_defaults"]["photon_selection"]["eta_crack"][1]:.2f}`, and tight ID/isolation.
- The diphoton selection requires `pT_lead / m_gg >= {summary["runtime_defaults"]["photon_selection"]["leading_pt_over_mgg_min"]:.2f}` and `pT_sublead / m_gg >= {summary["runtime_defaults"]["photon_selection"]["subleading_pt_over_mgg_min"]:.2f}`.
- Jets are reconstructed with `pT > {summary["runtime_defaults"]["jet_selection"]["pt_min_gev"]:.0f} GeV` and `|eta| < {summary["runtime_defaults"]["jet_selection"]["abs_eta_max"]:.1f}`.
- {fit_range_text}

## Signal, Control, And Blinding Regions

- Signal regions: {", ".join(region["signal_region_id"] for region in summary["signal_regions"])}
- Control region: {summary["control_regions"][0]["control_region_id"]} with sideband validation in `105-120 GeV` and `130-160 GeV`
- Active fit categories in this run: {", ".join(active_categories)}
- Inactive configured regions: {", ".join(fit_result.get("inactive_regions", [])) if fit_result.get("inactive_regions") else "none"}
- Blinding policy: {blinding_policy_text}

## Nominal Samples And Background Strategy

The nominal signal samples were chosen by process-key matching and generator preference. The nominal prompt-diphoton sample used for the spurious-signal workflow is `{prompt_nominal}`, corresponding to the minimum generated-mass window that fully contains `105-160 GeV`.

The prompt-diphoton template is normalized to observed data in the sidebands `105-120 GeV` and `130-160 GeV`. This keeps the nominal spurious-signal template anchored to diphoton MC while acknowledging that the observed continuum background also contains `gamma+jet`, `jet+jet`, and `Z->ee` fake-photon contributions.

The effective prompt-diphoton MC luminosity recorded in `outputs/report/mc_effective_lumi_check.json` is `{mc_lumi["per_process_effective_lumi_fb"]["prompt_diphoton_spurious_template"]:.3f} fb^-1`, compared with the required threshold `{mc_lumi["required_min_lumi_fb"]:.1f} fb^-1`. The smoothing gate status is `{smoothing["status"]}` with observed method `{smoothing_prov["method"]}`.

## Distribution Plots

{chr(10).join(embedded_blocks)}

## Cut Flow And Category Yields

{_table(["Step", "Data entries", "Prompt diphoton yield", "Signal yield"], cutflow_rows)}

{_table(["Category", "Data entries", "Prompt diphoton yield", "Signal yield"], category_rows)}

## Statistical Interpretation

- Fit status: `{fit_result["status"]}` (`fit_status={fit_result["fit_status"]}`, `cov_qual={fit_result["cov_qual"]}`)
- Fit dataset: `{fit_dataset_type}` ({fit_dataset_text})
- Shared POI across categories: `{fit_result["shared_mu"]}`
- {fit_mu_text}
- {observed_stat_line.removeprefix("- ")}
- {observed_q0_line.removeprefix("- ")}
- Asimov expected significance: `Z = {asimov["z_discovery"]:.3f}`
- Asimov `q0`: `{asimov["q0"]:.3f}`
- Asimov generation hypothesis: `mu_gen = {asimov["mu_gen"]}` with dataset type `{asimov["dataset_type"]}`
- Fit diagnostics: {", ".join(fit_result.get("diagnostics", []) or ["none"])}
- {observed_diag_line.removeprefix("- ")}
- Significance diagnostics: {", ".join(asimov.get("diagnostics", []) or ["none"])}

## Governance And Validation

- Data-MC discrepancy status: `{discrepancy["status"]}`
- Skill extraction status: `{skill_extraction["status"]}`
- Skill checkpoint status: `{checkpoint["status"]}`
- Plot groups produced: {", ".join(sorted(plot_manifest["plot_groups"].keys()))}
- Capped spurious-signal outcome categories: {", ".join(capped_categories) if capped_categories else "none"}

## Appendix: Nominal Sample Selection Rationale

- Selected nominal prompt-diphoton sample: `{prompt_nominal}`
- Alternative prompt-diphoton samples excluded from the nominal template: {", ".join(prompt_alternatives) if prompt_alternatives else "none"}
- Signal/background role assignment follows `outputs/report/mc_sample_selection.json` and excludes low-statistics auxiliary backgrounds from the nominal spurious-signal template.

## Summary

The pipeline completed bootstrap, selection, histogramming, RooFit modeling, expected-significance evaluation, plotting, discrepancy auditing, and report generation for the Higgs-to-diphoton workflow. Final handoff readiness remains governed by the machine-readable review and enforcement artifacts written alongside this report.
"""
    return report_text


def build_report(summary: dict, outputs: Path, reports_dir: Path) -> dict[str, str]:
    fit_result = read_json(outputs / "fit" / FIT_ID / "results.json")
    asimov = read_json(outputs / "fit" / FIT_ID / "significance_asimov.json")
    observed = read_json(outputs / "fit" / FIT_ID / "significance.json")
    sample_selection = read_json(outputs / "report" / "mc_sample_selection.json")
    discrepancy = read_json(outputs / "report" / "data_mc_discrepancy_audit.json")
    checkpoint = read_json(outputs / "report" / "skill_checkpoint_status.json")
    skill_extraction = read_json(outputs / "report" / "skill_extraction_summary.json")
    cutflow = read_json(outputs / "report" / "cutflow_table.json")
    yields = read_json(outputs / "report" / "yields_by_category.json")
    smoothing = read_json(outputs / "report" / "background_template_smoothing_check.json")
    smoothing_prov = read_json(outputs / "report" / "background_template_smoothing_provenance.json")
    mc_lumi = read_json(outputs / "report" / "mc_effective_lumi_check.json")
    blinding = read_json(outputs / "report" / "blinding_summary.json")
    plot_manifest = read_json(outputs / "report" / "plots" / "manifest.json")
    registry = read_json(outputs / "samples.registry.json")
    background_choice = read_json(outputs / "fit" / FIT_ID / "background_pdf_choice.json")

    report_path = outputs / "report" / "report.md"
    final_path = reports_dir / "final_analysis_report.md"
    dated_path = reports_dir / f"final_analysis_report_{datetime.now().strftime('%Y%m%d')}.md"

    report_text = _build_report_text(
        report_dir=report_path.parent,
        summary=summary,
        fit_result=fit_result,
        asimov=asimov,
        observed=observed,
        sample_selection=sample_selection,
        discrepancy=discrepancy,
        checkpoint=checkpoint,
        skill_extraction=skill_extraction,
        cutflow=cutflow,
        yields=yields,
        smoothing=smoothing,
        smoothing_prov=smoothing_prov,
        mc_lumi=mc_lumi,
        blinding=blinding,
        plot_manifest=plot_manifest,
        registry=registry,
        background_choice=background_choice,
    )
    final_text = _build_report_text(
        report_dir=final_path.parent,
        summary=summary,
        fit_result=fit_result,
        asimov=asimov,
        observed=observed,
        sample_selection=sample_selection,
        discrepancy=discrepancy,
        checkpoint=checkpoint,
        skill_extraction=skill_extraction,
        cutflow=cutflow,
        yields=yields,
        smoothing=smoothing,
        smoothing_prov=smoothing_prov,
        mc_lumi=mc_lumi,
        blinding=blinding,
        plot_manifest=plot_manifest,
        registry=registry,
        background_choice=background_choice,
    )
    dated_text = _build_report_text(
        report_dir=dated_path.parent,
        summary=summary,
        fit_result=fit_result,
        asimov=asimov,
        observed=observed,
        sample_selection=sample_selection,
        discrepancy=discrepancy,
        checkpoint=checkpoint,
        skill_extraction=skill_extraction,
        cutflow=cutflow,
        yields=yields,
        smoothing=smoothing,
        smoothing_prov=smoothing_prov,
        mc_lumi=mc_lumi,
        blinding=blinding,
        plot_manifest=plot_manifest,
        registry=registry,
        background_choice=background_choice,
    )

    write_text(report_text, report_path)
    write_text(final_text, final_path)
    write_text(dated_text, dated_path)

    artifact_inventory = {
        "status": "ok",
        "report_paths": [str(report_path), str(final_path), str(dated_path)],
        "key_artifacts": {
            "fit_results": str(outputs / "fit" / FIT_ID / "results.json"),
            "fit_plot_payload": str(outputs / "fit" / FIT_ID / "fit_plot_payload.json"),
            "significance": str(outputs / "fit" / FIT_ID / "significance.json"),
            "significance_asimov": str(outputs / "fit" / FIT_ID / "significance_asimov.json"),
            "significance_asimov_plot_payload": str(outputs / "fit" / FIT_ID / "significance_asimov_plot_payload.json"),
            "background_choice": str(outputs / "fit" / FIT_ID / "background_pdf_choice.json"),
            "plot_manifest": str(outputs / "report" / "plots" / "manifest.json"),
            "smoothing_check": str(outputs / "report" / "background_template_smoothing_check.json"),
            "mc_effective_lumi_check": str(outputs / "report" / "mc_effective_lumi_check.json"),
        },
        "plot_groups": plot_manifest["plot_groups"],
    }
    write_json(artifact_inventory, outputs / "report" / "artifact_link_inventory.json")

    return {
        "report": str(report_path),
        "final_report": str(final_path),
        "dated_report": str(dated_path),
    }
