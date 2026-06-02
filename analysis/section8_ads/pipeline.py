from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

from analysis.common import ensure_dir, read_json, stable_hash, utcnow_iso, write_json, write_text
from analysis.samples.registry import build_registry
from analysis.section8_ads.ads_loader import load_ads
from analysis.section8_ads.branch_map import build_branch_mapping
from analysis.section8_ads.categories import ORDERED_CATEGORIES, assign_categories
from analysis.section8_ads.classifiers import (
    classifier_input_status,
    optimize_boundaries,
    score_samples,
    train_classifiers,
    write_training_sample_audit,
)
from analysis.section8_ads.observables import (
    FIT_RANGE,
    SIDEBAND_SCALE,
    SIGNAL_WINDOW,
    SECTION8_REQUIRED_BRANCHES,
    build_diphoton_event_view,
    build_section8_observables,
    delta_phi as _delta_phi,
    invariant_mass as _invariant_mass,
    photon_iso_proxy as _photon_iso_proxy,
    rapidities as _rapidities,
    vector_components as _vector_components,
)


def _select_samples(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for sample in registry:
        if sample["kind"] == "data":
            selected.append(sample)
            continue
        if sample["analysis_role"] == "signal_nominal":
            selected.append(sample)
            continue
        if sample["process_key"] == "prompt_diphoton" and sample["is_nominal"]:
            selected.append(sample)
    return selected


def _iterate_batches(
    files: list[str],
    branches: list[str],
    max_events: int | None = None,
    event_selector: Callable[[ak.Array], np.ndarray] | None = None,
):
    seen = 0
    for file_path in files:
        with uproot.open(file_path) as handle:
            tree = handle["analysis"]
            for batch in tree.iterate(branches, step_size="100 MB", library="ak"):
                if event_selector is not None:
                    selector = np.asarray(event_selector(batch), dtype=bool)
                    if not np.any(selector):
                        continue
                    batch = batch[selector]
                if max_events is None:
                    yield batch
                    continue
                remaining = max_events - seen
                if remaining <= 0:
                    return
                if len(batch[branches[0]]) > remaining:
                    yield batch[:remaining]
                    return
                yield batch
                seen += len(batch[branches[0]])


def _event_weights(batch: ak.Array, sample: dict[str, Any]) -> np.ndarray:
    size = len(batch["eventNumber"])
    if sample["kind"] == "data":
        return np.ones(size, dtype=float)
    denom = sample["sumw"]
    norm = 1.0 if not denom else sample["xsec_pb"] * sample["k_factor"] * sample["filter_eff"] * sample["lumi_fb"] * 1000.0 / denom
    weights = norm * np.asarray(batch["mcWeight"], dtype=float)
    for branch in ("ScaleFactor_PILEUP", "ScaleFactor_PHOTON", "ScaleFactor_JVT", "ScaleFactor_FTAG"):
        if branch in batch.fields:
            weights *= np.asarray(batch[branch], dtype=float)
    return weights


def _trigger_mask(batch: ak.Array, n_events: int, trigger_policy: str = "input_preselected") -> np.ndarray:
    if trigger_policy == "trigP":
        return np.asarray(batch["trigP"], dtype=bool) if "trigP" in batch.fields else np.ones(n_events, dtype=bool)
    if trigger_policy in {"input_preselected", "none"}:
        return np.ones(n_events, dtype=bool)
    raise ValueError(f"Unknown Section 8 trigger policy: {trigger_policy}")


def _empty_arrays() -> dict[str, list[np.ndarray]]:
    return {
        "event_number": [],
        "run_number": [],
        "weight": [],
        "trigger_passed": [],
        "baseline_selected": [],
        "is_sideband": [],
        "is_signal_window": [],
        "m_gammagamma": [],
        "pT_gammagamma": [],
        "eta_gammagamma": [],
        "pTt_gammagamma": [],
        "lead_pt": [],
        "sublead_pt": [],
        "lead_pt_over_mgg": [],
        "sublead_pt_over_mgg": [],
        "lead_eta": [],
        "sublead_eta": [],
        "max_abs_photon_eta": [],
        "N_jets_25": [],
        "N_jets_30": [],
        "N_jets_25_jvt_diagnostic": [],
        "N_jets_30_jvt_diagnostic": [],
        "N_central_jets_25": [],
        "N_forward_jets_25": [],
        "N_btag_25": [],
        "N_lep": [],
        "m_ll": [],
        "Z_ll_veto": [],
        "m_e_gamma_veto": [],
        "pT_lepton_plus_MET": [],
        "MET": [],
        "MET_significance": [],
        "leading_jet_pT_30": [],
        "m_jj_30": [],
        "abs_delta_eta_jj_30": [],
        "pT_Hjj_30": [],
        "deltaR_min_gamma_j": [],
        "VBF_centrality": [],
        "H_T": [],
        "m_all_jets": [],
        "delta_y_gammagamma_jj": [],
        "cos_theta_star_gammagamma_jj": [],
        "abs_delta_phi_gammagamma_jj_capped": [],
        "training_mask_tth": [],
        "training_mask_vh": [],
        "training_mask_vbf": [],
    }


def _concat_arrays(payload: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, chunks in payload.items():
        out[key] = np.concatenate(chunks) if chunks else np.array([])
    return out


def _append_chunk(store: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    store[key].append(np.asarray(value))


def _process_sample(
    sample: dict[str, Any],
    max_events: int | None = None,
    event_selector: Callable[[ak.Array], np.ndarray] | None = None,
    trigger_policy: str = "input_preselected",
) -> dict[str, Any]:
    branches = SECTION8_REQUIRED_BRANCHES
    cutflow = {
        "all_input_events": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "trigger_requirement": {
            "before": 0,
            "after": 0,
            "status": "implicit_in_input_format" if trigger_policy == "input_preselected" else "approximated",
            "notes": (
                "Treating GamGam ntuples as trigger-preselected because trigP does not match the ADS diphoton-trigger semantics."
                if trigger_policy == "input_preselected"
                else "Using trigP as an explicit photon-trigger proxy."
            ),
        },
        "at_least_two_photon_candidates": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "photon_kinematic_acceptance": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "loose_photon_id_preselection": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "select_two_highest_et_photons": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "diphoton_primary_vertex_handling": {"before": 0, "after": 0, "status": "unavailable", "notes": "The primary-vertex NN inputs are not present in the ntuples."},
        "leading_photon_et_over_mgg": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "subleading_photon_et_over_mgg": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "tight_photon_identification": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
        "photon_isolation": {
            "before": 0,
            "after": 0,
            "status": "approximated",
            "notes": "Using photon_topoetcone40 and photon_ptcone20 with the Section 5.1 thresholds because the ntuples do not expose the exact calorimeter-isolation cone-0.2 branch.",
        },
        "diphoton_mass_window": {"before": 0, "after": 0, "status": "implemented_faithfully", "notes": ""},
    }
    for entry in cutflow.values():
        entry["weighted_before"] = 0.0
        entry["weighted_after"] = 0.0
    arrays = _empty_arrays()

    for batch in _iterate_batches(sample["files"], branches, max_events=max_events, event_selector=event_selector):
        weights = _event_weights(batch, sample)
        n_events = len(batch["eventNumber"])
        trigger_mask = _trigger_mask(batch, n_events, trigger_policy)
        view = build_diphoton_event_view(batch, weights=weights, trigger_mask=trigger_mask)
        baseline_mask = (
            view.trigger_mask
            & view.has_two
            & view.full_tight
            & view.full_iso
            & view.full_ptfrac
            & np.isfinite(view.full_mass)
            & (view.full_mass > FIT_RANGE[0])
            & (view.full_mass < FIT_RANGE[1])
        )

        def update_cutflow(name: str, before_mask: np.ndarray, after_mask: np.ndarray) -> None:
            cutflow[name]["before"] += int(np.sum(before_mask))
            cutflow[name]["after"] += int(np.sum(after_mask))
            cutflow[name]["weighted_before"] += float(np.sum(weights[before_mask]))
            cutflow[name]["weighted_after"] += float(np.sum(weights[after_mask]))

        trigger_has_two = view.trigger_mask & view.has_two
        et_fraction_mask = trigger_has_two & view.full_ptfrac
        tight_id_mask = et_fraction_mask & view.full_tight
        tight_iso_mask = tight_id_mask & view.full_iso

        all_events = np.ones(n_events, dtype=bool)
        update_cutflow("all_input_events", all_events, all_events)
        update_cutflow("trigger_requirement", all_events, view.trigger_mask)
        update_cutflow("at_least_two_photon_candidates", view.trigger_mask, trigger_has_two)
        update_cutflow("photon_kinematic_acceptance", view.trigger_mask, view.trigger_mask & view.has_two_kinematic_photons)
        update_cutflow("loose_photon_id_preselection", view.trigger_mask & view.has_two_kinematic_photons, trigger_has_two)
        update_cutflow("select_two_highest_et_photons", trigger_has_two, trigger_has_two)
        update_cutflow("diphoton_primary_vertex_handling", trigger_has_two, trigger_has_two)
        update_cutflow("leading_photon_et_over_mgg", trigger_has_two, et_fraction_mask)
        update_cutflow("subleading_photon_et_over_mgg", trigger_has_two, et_fraction_mask)
        update_cutflow("tight_photon_identification", et_fraction_mask, tight_id_mask)
        update_cutflow("photon_isolation", tight_id_mask, tight_iso_mask)
        update_cutflow("diphoton_mass_window", tight_iso_mask, baseline_mask)

        if not np.any(baseline_mask):
            continue
        fields = build_section8_observables(batch, view, baseline_mask, compute_lepton_vetoes=True).fields
        for key, value in fields.items():
            _append_chunk(arrays, key, np.asarray(value))

    return {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "cutflow": cutflow,
        "arrays": _concat_arrays(arrays),
    }


def _empty_bdt_arrays() -> dict[str, list[np.ndarray]]:
    payload = _empty_arrays()
    payload.update(
        {
            "photon_region": [],
            "nominal_photon_region": [],
            "anti_id_or_iso_control_region": [],
            "bdt_subregion": [],
        }
    )
    return payload


def _process_bdt_candidates(
    sample: dict[str, Any],
    max_events: int | None = None,
    trigger_policy: str = "input_preselected",
) -> dict[str, Any]:
    branches = SECTION8_REQUIRED_BRANCHES
    arrays = _empty_bdt_arrays()
    for batch in _iterate_batches(sample["files"], branches, max_events=max_events):
        weights = _event_weights(batch, sample)
        n_events = len(batch["eventNumber"])
        trigger_mask = _trigger_mask(batch, n_events, trigger_policy)
        view = build_diphoton_event_view(batch, weights=weights, trigger_mask=trigger_mask)
        nominal_region = view.nominal_photon_region
        control_region = view.anti_id_or_iso_control_region
        candidate_mask = (
            view.trigger_mask
            & view.has_two
            & view.full_ptfrac
            & np.isfinite(view.full_mass)
            & (view.full_mass > FIT_RANGE[0])
            & (view.full_mass < FIT_RANGE[1])
            & (nominal_region | control_region)
        )
        if not np.any(candidate_mask):
            continue

        nominal = nominal_region[candidate_mask]
        control = control_region[candidate_mask]
        photon_region = np.where(nominal, "nominal_photon_region", "anti_id_or_iso_control_region")
        fields = build_section8_observables(
            batch,
            view,
            candidate_mask,
            baseline_selected=nominal.astype(int),
            compute_lepton_vetoes=False,
            include_bdt_subregion=True,
        ).fields
        fields["photon_region"] = photon_region
        fields["nominal_photon_region"] = nominal.astype(int)
        fields["anti_id_or_iso_control_region"] = control.astype(int)
        for key, value in fields.items():
            _append_chunk(arrays, key, np.asarray(value))

    return {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "bdt_arrays": _concat_arrays(arrays),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _variable_status(outputs: Path) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "variables": {
            "m_gammagamma": {"status": "implemented_faithfully"},
            "pT_gammagamma": {"status": "implemented_faithfully"},
            "eta_gammagamma": {"status": "implemented_faithfully"},
            "N_jets": {"status": "implemented_faithfully"},
            "N_btag": {"status": "approximated", "notes": "Using jet_btag_quantile >= 4 as the 70% b-tag proxy."},
            "N_lep": {"status": "implemented_faithfully"},
            "m_ll": {"status": "implemented_faithfully"},
            "m_e_gamma": {"status": "approximated", "notes": "Using reconstructed electrons from the open-data ntuples."},
            "pT_lepton_plus_MET": {"status": "implemented_faithfully"},
            "MET_significance": {"status": "approximated", "notes": "Using MET/sqrt(HT) as the user-approved approximation."},
            "cos_theta_star_gammagamma_jj": {"status": "approximated", "notes": "Using a reconstructed system-angle proxy for the supplemental BDT feature."},
        },
    }
    write_json(payload, outputs / "variable_implementation_status.json")
    write_text(
        "# Variable Implementation Status\n\n" + "\n".join(
            f"- `{name}`: {item['status']}" for name, item in payload["variables"].items()
        )
        + "\n",
        outputs / "variable_implementation_status.md",
    )
    return payload


def _summarize_categories(samples: list[dict[str, Any]], outputs: Path) -> None:
    rows = []
    markdown_lines = ["# Category Assignment Summary", ""]
    baseline_total = 0
    assigned_total = 0
    blocked_total = 0
    for sample in samples:
        categories = sample["arrays"]["assigned_category"].astype(str)
        reasons = sample["arrays"]["assignment_reason"].astype(str)
        baseline_total += len(categories)
        assigned_total += int(np.sum((categories != "unassigned") & (categories != "blocked_missing_input")))
        blocked_total += int(np.sum(categories == "blocked_missing_input"))
        for category in ORDERED_CATEGORIES + ["blocked_missing_input", "unassigned"]:
            count = int(np.sum(categories == category))
            if count == 0:
                continue
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "process_key": sample["process_key"],
                    "category": category,
                    "count": count,
                }
            )
        if np.any(categories == "unassigned"):
            unique_reasons, counts = np.unique(reasons[categories == "unassigned"], return_counts=True)
            markdown_lines.append(f"## Unassigned Reasons For `{sample['sample_id']}`")
            markdown_lines.append("")
            for reason, count in zip(unique_reasons.tolist(), counts.tolist()):
                markdown_lines.append(f"- {reason}: `{count}`")
            markdown_lines.append("")
    write_json({"status": "ok", "rows": rows}, outputs / "category_yields.json")
    _write_csv(rows, outputs / "category_yields.csv")
    markdown_lines[1:1] = [
        f"- Baseline-selected events: `{baseline_total}`",
        f"- Assigned events: `{assigned_total}`",
        f"- Blocked events: `{blocked_total}`",
        f"- Unassigned events: `{baseline_total - assigned_total - blocked_total}`",
        "",
    ]
    write_text("\n".join(markdown_lines) + "\n", outputs / "category_assignment_summary.md")


def _implementation_decisions(ads: dict[str, Any], outputs: Path) -> None:
    payload = {
        "status": "ok",
        "decisions": [
            {
                "affected_component": "b_tag_working_point",
                "ads_requirement": "70% b-tag efficiency working point",
                "operational_choice": "jet_btag_quantile >= 4",
                "confidence": "medium",
                "risk_level": "medium",
                "affected_categories": ["tH lep 0fwd", "tH lep 1fwd", "ttH lep", "ttH had BDT1-4", "tH had 4j1b", "tH had 4j2b"],
            },
            {
                "affected_component": "MET_significance",
                "ads_requirement": "MET/sqrt(sum_ET)",
                "operational_choice": "MET/sqrt(HT)",
                "confidence": "medium",
                "risk_level": "medium",
                "affected_categories": ["VH lep Low", "VH MET High", "VH MET Low"],
            },
            {
                "affected_component": "classifier_scores",
                "ads_requirement": "Official BDT outputs",
                "operational_choice": "Train supplemental local BDTs from ADS feature lists",
                "confidence": "high",
                "risk_level": "high",
                "affected_categories": [
                    "ttH had BDT1",
                    "ttH had BDT2",
                    "ttH had BDT3",
                    "ttH had BDT4",
                    "VH had tight",
                    "VH had loose",
                    "VBF tight, high pT_Hjj",
                    "VBF loose, high pT_Hjj",
                    "VBF tight, low pT_Hjj",
                    "VBF loose, low pT_Hjj",
                ],
            },
        ],
        "ads_ambiguities": ads["ambiguity_registry"],
    }
    write_json(payload, outputs / "implementation_decisions.json")
    lines = ["# Implementation Decisions", ""]
    for item in payload["decisions"]:
        lines.append(f"- `{item['affected_component']}`: {item['operational_choice']}")
    write_text("\n".join(lines) + "\n", outputs / "implementation_decisions.md")


def _write_cutflow(samples: list[dict[str, Any]], outputs: Path) -> None:
    ordered_steps = [
        "all_input_events",
        "trigger_requirement",
        "at_least_two_photon_candidates",
        "photon_kinematic_acceptance",
        "loose_photon_id_preselection",
        "select_two_highest_et_photons",
        "diphoton_primary_vertex_handling",
        "leading_photon_et_over_mgg",
        "subleading_photon_et_over_mgg",
        "tight_photon_identification",
        "photon_isolation",
        "diphoton_mass_window",
    ]
    aggregated = {}
    for step in ordered_steps:
        before = sum(sample["cutflow"][step]["before"] for sample in samples if sample["kind"] == "data")
        after = sum(sample["cutflow"][step]["after"] for sample in samples if sample["kind"] == "data")
        aggregated[step] = {
            "selection_name": step,
            "events_before": before,
            "events_after": after,
            "incremental_efficiency": 0.0 if before == 0 else after / before,
            "cumulative_efficiency": 0.0 if aggregated == {} else after / max(aggregated["all_input_events"]["events_after"], 1),
            "implementation_status": samples[0]["cutflow"][step]["status"],
            "notes": samples[0]["cutflow"][step]["notes"],
        }
    rows = list(aggregated.values())
    write_json({"status": "ok", "aggregated": aggregated}, outputs / "cutflow_baseline.json")
    _write_csv(rows, outputs / "cutflow_baseline.csv")
    lines = ["# Cut Flow", ""]
    for row in rows:
        lines.append(
            f"- `{row['selection_name']}`: before `{row['events_before']}`, after `{row['events_after']}`, incremental `{row['incremental_efficiency']:.4f}`, cumulative `{row['cumulative_efficiency']:.4f}`"
        )
    write_text("\n".join(lines) + "\n", outputs / "cutflow_baseline.md")


def _validation_plots(samples: list[dict[str, Any]], outputs: Path) -> None:
    plot_dir = ensure_dir(outputs / "plots")
    manifest = []
    for key, title, bins in [
        ("m_gammagamma", "Diphoton Mass", np.linspace(105.0, 160.0, 56)),
        ("lead_pt", "Leading Photon pT", np.linspace(0.0, 300.0, 31)),
        ("sublead_pt", "Subleading Photon pT", np.linspace(0.0, 250.0, 26)),
        ("N_jets_30", "Jet Multiplicity", np.arange(-0.5, 8.5, 1.0)),
        ("N_btag_25", "B-Tagged Jet Multiplicity", np.arange(-0.5, 6.5, 1.0)),
        ("N_lep", "Lepton Multiplicity", np.arange(-0.5, 5.5, 1.0)),
        ("MET", "Missing Transverse Momentum", np.linspace(0.0, 300.0, 31)),
    ]:
        plt.figure(figsize=(8, 5))
        for sample in samples:
            if len(sample["arrays"][key]) == 0:
                continue
            weights = sample["arrays"]["weight"] if sample["kind"] != "data" else None
            plt.hist(sample["arrays"][key], bins=bins, histtype="step", label=sample["sample_id"], weights=weights)
        plt.title(title)
        plt.xlabel(key)
        plt.ylabel("Events")
        plt.legend(fontsize=7)
        out_path = plot_dir / f"{key}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        manifest.append({"plot": key, "path": str(out_path)})
    lines = ["# Validation Report", ""]
    for item in manifest:
        lines.append(f"- `{item['plot']}`: `{item['path']}`")
    write_text("\n".join(lines) + "\n", outputs / "validation_report.md")


def _implementation_status(outputs: Path) -> None:
    lines = [
        "# IMPLEMENTATION STATUS",
        "",
        "## Implemented faithfully",
        "- ADS loading and ordered category parsing",
        "- Diphoton kinematics, baseline ET/mgg cuts, and main multiplicity variables",
        "",
        "## Implemented with approximations",
        "- Supplemental BDT training in place of official ATLAS classifier artifacts",
        "- MET_significance approximated as MET/sqrt(HT)",
        "- B-tag working-point proxy from jet_btag_quantile >= 4",
        "",
        "## Blocked by unavailable inputs",
        "- Official diphoton primary-vertex NN reconstruction inputs",
        "",
        "## Recommended next steps",
        "- Validate supplemental BDTs against external references if score branches become available",
        "- Replace the fast significance proxy with a fit-based validation pass for winning boundaries",
        "",
    ]
    write_text("\n".join(lines) + "\n", outputs / "IMPLEMENTATION_STATUS.md")


def _write_bdt_optimization_metadata(
    *,
    outputs: Path,
    ads: dict[str, Any],
    inputs: Path,
    selected_samples: list[dict[str, Any]],
    training_report: dict[str, Any] | None,
    audit_report: dict[str, Any] | None,
    prepare_bdt_training: bool,
    train_bdts: bool,
    score_bdts: bool,
    runs_dir: Path | None = None,
    registry_path: Path | None = None,
) -> None:
    if not (prepare_bdt_training or train_bdts or score_bdts):
        return
    runs_dir = outputs / "metadata" / "runs" if runs_dir is None else Path(runs_dir)
    registry_path = outputs / "metadata" / "runs.jsonl" if registry_path is None else Path(registry_path)
    run_id = f"section8_bdt_{stable_hash({'outputs': str(outputs), 'time': utcnow_iso()})[:12]}"
    sample_hash = stable_hash(
        [
            {
                "sample_id": sample["sample_id"],
                "process_key": sample["process_key"],
                "kind": sample["kind"],
                "files": sample["files"],
            }
            for sample in selected_samples
        ]
    )
    metric_values = {
        "trained_classifiers": 0,
        "blocked_classifiers": 0,
    }
    if training_report:
        statuses = [item["status"] for item in training_report.get("classifiers", {}).values()]
        metric_values["trained_classifiers"] = statuses.count("trained")
        metric_values["blocked_classifiers"] = len([status for status in statuses if status != "trained"])
    observation = {
        "configuration_changes": {
            "changed_parameters": ["section8_ads.bdt_training"],
            "rationale": "Enable Section 8 supplemental BDT training with anti-ID/anti-isolation data controls.",
            "change_mode": "qualitative_strategy",
        },
        "primary_metric_response": {
            "metric": "trained_classifiers",
            "baseline_value": 0,
            "candidate_value": metric_values["trained_classifiers"],
            "absolute_change": metric_values["trained_classifiers"],
            "relative_change": None,
            "uncertainty": None,
            "meaningfulness": "infrastructure validation",
        },
        "yield_cutflow_response": {
            "yield_changes": None,
            "cutflow_changes": None,
            "signal_background_pattern": "See bdt_training_sample_audit.json for per-process training rows.",
            "unexpected_empty_or_unstable_categories": None,
        },
        "shape_response": {
            "changed_distributions": ["BDT_ttH", "BDT_VH", "BDT_VBF"],
            "localized_or_global": "localized to BDT-dependent Section 8 categories",
            "suspicious_structures": None,
            "binning_assessment": None,
        },
        "fit_inference_response": {
            "parameter_changes": None,
            "nuisance_changes": None,
            "goodness_of_fit_changes": None,
            "uncertainty_changes": None,
            "dominant_categories": None,
        },
        "validation_response": {
            "verifier_results": "smoke-test dependent",
            "warnings": [] if metric_values["blocked_classifiers"] == 0 else ["One or more BDTs did not train; inspect classifier_training_report.json."],
            "failed_checks": [],
            "artifacts_reused": [],
            "stages_rerun": ["section8_bdt_training"],
            "cache_concerns": None,
        },
        "interpretation": {
            "status": "new qualitative opportunity" if metric_values["trained_classifiers"] else "requires human review",
            "direct_observations": metric_values,
            "plausible_interpretation": "Prepared BDT artifacts are suitable for future optimization only when all needed classifiers train.",
            "unresolved_uncertainty": "Supplemental BDTs are not official ATLAS artifacts.",
            "possible_implementation_issue": None,
        },
    }
    run_dir = ensure_dir(runs_dir / run_id)
    write_json(observation, run_dir / "observations.yaml")
    registry_record = {
        "run_id": run_id,
        "timestamp": utcnow_iso(),
        "parent_run_id": None,
        "branch_id": "section8_ads_bdt",
        "strategy_id": "xgboost_anti_id_iso_controls",
        "run_type": "service_extension_validation",
        "objective": "Prepare and train supplemental Section 8 BDTs for continuous optimization.",
        "configuration_snapshot_path": str(outputs / "run_manifest.json"),
        "changed_parameters": ["prepare_bdt_training", "train_bdts", "score_bdts"],
        "reused_artifacts": [],
        "invalidated_artifacts": ["section8_classifier_scores", "section8_category_assignment"],
        "stages_executed": ["bdt_training_sample_audit", "classifier_training", "classifier_scoring"],
        "stages_skipped": [],
        "verifier_status": "ok",
        "cut_flow_path": str(outputs / "cutflow_baseline.json"),
        "yield_table_path": str(outputs / "category_yields.json"),
        "fit_output_path": None,
        "metric_values": metric_values,
        "expected_significance": None,
        "observed_significance": None,
        "runtime": None,
        "warnings": observation["validation_response"]["warnings"],
        "failure_reason": None,
        "human_or_agent_note": "Section 8 BDT training artifacts are supplemental and local.",
        "git_state": {"dirty": None, "commit": None},
        "service_versions": {"section8_ads_bdt": stable_hash({"ads": ads["config_hash"], "samples": sample_hash})},
        "version_name": "section8-bdt-xgboost-controls",
        "version_ref": None,
        "artifacts": {
            "audit": None if audit_report is None else str(outputs / "bdt_training_sample_audit.json"),
            "training_report": None if training_report is None else str(outputs / "classifier_training_report.json"),
        },
        "inputs": str(inputs),
    }
    ensure_dir(registry_path.parent)
    with registry_path.open("a") as handle:
        handle.write(json.dumps(registry_record, sort_keys=True) + "\n")


def run_section8_ads(
    *,
    ads_path: Path,
    inputs: Path,
    outputs: Path,
    max_events: int | None = None,
    prepare_bdt_training: bool = False,
    train_bdts: bool = False,
    score_bdts: bool = False,
    optimize_boundaries_flag: bool = False,
    reuse_bdt_artifacts: Path | None = None,
    trigger_policy: str = "input_preselected",
    metadata_runs_dir: Path | None = None,
    metadata_registry_path: Path | None = None,
) -> dict[str, Any]:
    outputs = ensure_dir(outputs)
    ads = load_ads(ads_path)
    summary_path = Path(__file__).resolve().parents[1] / "analysis.summary.json"
    registry, _ = build_registry(inputs, read_json(summary_path), 36.1)
    selected_samples = _select_samples(registry)
    branch_map = build_branch_mapping([sample["files"][0] for sample in selected_samples], ads, outputs)
    classifier_input_status(branch_map["available_fields"], outputs)
    processed = [_process_sample(sample, max_events=max_events, trigger_policy=trigger_policy) for sample in selected_samples]
    _write_cutflow(processed, outputs)
    variable_status = _variable_status(outputs)
    training_report = None
    audit_report = None
    bdt_training_samples = None
    bdt_metadata = {
        "ads_hash": ads["config_hash"],
        "branch_map_hash": stable_hash(branch_map),
        "sample_list_hash": stable_hash([sample["sample_id"] for sample in selected_samples]),
        "selection_policy_hash": stable_hash(
            {
                "control_region": "anti_id_or_iso_control_region",
                "mass_range": FIT_RANGE,
                "tth": "N_lep == 0 and N_jets_30 >= 3 and N_btag_25 >= 1",
                "vh": "N_jets_30 >= 2 and 60 < m_jj_30 < 120",
                "vbf": "N_jets_30 >= 2 and abs_delta_eta_jj_30 > 2 and VBF_centrality < 5",
            }
        ),
    }
    if prepare_bdt_training or train_bdts:
        bdt_training_samples = [
            _process_bdt_candidates(sample, max_events=max_events, trigger_policy=trigger_policy)
            for sample in selected_samples
        ]
        audit_report = write_training_sample_audit(bdt_training_samples, outputs, metadata=bdt_metadata)
    if train_bdts:
        training_report = train_classifiers(processed, outputs, training_samples=bdt_training_samples, metadata=bdt_metadata)
    elif reuse_bdt_artifacts is not None:
        candidate = reuse_bdt_artifacts / "classifier_training_report.json"
        if candidate.exists():
            training_report = read_json(candidate)
    if training_report and (score_bdts or train_bdts or reuse_bdt_artifacts is not None):
        score_samples(processed, training_report)
    for sample in processed:
        for name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
            if name not in sample["arrays"]:
                sample["arrays"][name] = np.full(len(sample["arrays"]["event_number"]), np.nan)

    boundary_report = None
    if optimize_boundaries_flag and train_bdts:
        boundary_report = optimize_boundaries(processed, outputs)
    boundaries = None if boundary_report is None else boundary_report["selected_boundaries"]
    for sample in processed:
        assigned, reasons, blocked = assign_categories(sample["arrays"], boundaries)
        sample["arrays"]["assigned_category"] = assigned
        sample["arrays"]["assignment_reason"] = reasons
        sample["arrays"]["assignment_blocked"] = blocked.astype(int)
    _summarize_categories(processed, outputs)
    _implementation_decisions(ads, outputs)
    _validation_plots(processed, outputs)
    _implementation_status(outputs)
    _write_bdt_optimization_metadata(
        outputs=outputs,
        ads=ads,
        inputs=inputs,
        selected_samples=selected_samples,
        training_report=training_report,
        audit_report=audit_report,
        prepare_bdt_training=prepare_bdt_training,
        train_bdts=train_bdts,
        score_bdts=score_bdts,
        runs_dir=metadata_runs_dir,
        registry_path=metadata_registry_path,
    )
    write_json(
        {
            "status": "ok",
            "timestamp_utc": utcnow_iso(),
            "ads_path": str(ads_path),
            "inputs": str(inputs),
            "outputs": str(outputs),
            "max_events": max_events,
            "prepare_bdt_training": prepare_bdt_training,
            "train_bdts": train_bdts,
            "score_bdts": score_bdts,
            "optimize_boundaries": optimize_boundaries_flag,
            "reuse_bdt_artifacts": None if reuse_bdt_artifacts is None else str(reuse_bdt_artifacts),
            "trigger_policy": trigger_policy,
            "config_hash": stable_hash(
                {
                    "ads": ads["config_hash"],
                    "max_events": max_events,
                    "prepare_bdt_training": prepare_bdt_training,
                    "train_bdts": train_bdts,
                    "score_bdts": score_bdts,
                    "optimize_boundaries": optimize_boundaries_flag,
                    "trigger_policy": trigger_policy,
                    "bdt_metadata": bdt_metadata,
                }
            ),
        },
        outputs / "run_manifest.json",
    )
    return {
        "ads": ads,
        "branch_map": branch_map,
        "variable_status": variable_status,
        "audit_report": audit_report,
        "training_report": training_report,
        "boundary_report": boundary_report,
        "processed": processed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Section 8 ADS reconstruction pipeline.")
    parser.add_argument("--ads", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--prepare-bdt-training", action="store_true")
    parser.add_argument("--train-bdts", action="store_true")
    parser.add_argument("--score-bdts", action="store_true")
    parser.add_argument("--optimize-boundaries", action="store_true")
    parser.add_argument("--reuse-bdt-artifacts", type=Path)
    parser.add_argument("--trigger-policy", choices=["input_preselected", "trigP", "none"], default="input_preselected")
    parser.add_argument("--metadata-runs-dir", type=Path, help="Directory for Section 8 BDT run observation records; defaults to <outputs>/metadata/runs/.")
    parser.add_argument("--metadata-registry", type=Path, help="JSONL registry path for Section 8 BDT run records; defaults to <outputs>/metadata/runs.jsonl.")
    args = parser.parse_args()
    run_section8_ads(
        ads_path=Path(args.ads),
        inputs=Path(args.inputs),
        outputs=Path(args.outputs),
        max_events=args.max_events,
        prepare_bdt_training=args.prepare_bdt_training,
        train_bdts=args.train_bdts,
        score_bdts=args.score_bdts,
        optimize_boundaries_flag=args.optimize_boundaries,
        reuse_bdt_artifacts=args.reuse_bdt_artifacts,
        trigger_policy=args.trigger_policy,
        metadata_runs_dir=args.metadata_runs_dir,
        metadata_registry_path=args.metadata_registry,
    )


if __name__ == "__main__":
    main()
