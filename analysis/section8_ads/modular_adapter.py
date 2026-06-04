from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common import ensure_dir, read_json
from analysis.section8_ads.categories import route_section8_categories
from analysis.section8_ads.classifiers import score_samples
from analysis.section8_ads.pipeline import _process_sample


SECTION8_CUTFLOW_MAP = {
    "all_events": "all_input_events",
    "two_photons": "at_least_two_photon_candidates",
    "pt_fraction": "subleading_photon_et_over_mgg",
    "mass_window": "diphoton_mass_window",
}


def _load_boundaries(section8_cfg: dict[str, Any]) -> dict[str, list[float]] | None:
    artifact_dir = Path(section8_cfg["bdt_artifacts_dir"])
    boundary_file = Path(section8_cfg.get("boundary_file", artifact_dir / "bdt_boundary_optimization.json"))
    if not boundary_file.exists():
        return None
    payload = read_json(boundary_file)
    return payload.get("selected_boundaries")


def _section8_cutflow(processed: dict[str, Any], arrays: dict[str, np.ndarray], assigned: np.ndarray, sample: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    out = {}
    for target, source in SECTION8_CUTFLOW_MAP.items():
        source_payload = processed["cutflow"][source]
        count = int(source_payload["after"])
        out[target] = {
            "weighted": float(source_payload.get("weighted_after", count)),
            "unweighted": count,
        }
    categorized = (assigned != "blocked_missing_input") & (assigned != "unassigned")
    out["categorized"] = {
        "weighted": float(np.sum(arrays["weight"][categorized])) if sample["kind"] != "data" else float(np.sum(categorized)),
        "unweighted": int(np.sum(categorized)),
    }
    return out


def process_sample_for_modular(
    sample: dict[str, Any],
    cfg: dict[str, Any],
    max_events: int | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    section8_cfg = cfg.get("section8_ads", {})
    if "bdt_artifacts_dir" not in section8_cfg:
        raise RuntimeError(
            "Section 8 modular adapter processing requires runtime_defaults.section8_ads.bdt_artifacts_dir "
            "for classifier scoring and categorization; the standalone hgg-section8 runner requires --ads separately."
        )
    artifact_dir = Path(section8_cfg["bdt_artifacts_dir"])
    training_report_path = artifact_dir / "classifier_training_report.json"
    if not training_report_path.exists():
        raise FileNotFoundError(f"Missing Section 8 classifier training report: {training_report_path}")

    trigger_policy = str(section8_cfg.get("trigger_policy", "input_preselected"))
    processed = _process_sample(sample, max_events=max_events, trigger_policy=trigger_policy)
    training_report = read_json(training_report_path)
    score_samples([processed], training_report)
    arrays = processed["arrays"]
    for score_name in ("BDT_ttH", "BDT_VH", "BDT_VBF"):
        if score_name not in arrays:
            arrays[score_name] = np.full(len(arrays["event_number"]), np.nan)
    routing_config = cfg.get("analysis_implementation", {}).get("routing_config")
    routing_result = route_section8_categories(arrays, _load_boundaries(section8_cfg), routing_config)
    assigned = routing_result.assigned_category
    reasons = routing_result.assignment_reason
    blocked = routing_result.assignment_blocked
    valid = (assigned != "blocked_missing_input") & (assigned != "unassigned")
    category_ids = assigned[valid].astype(str)

    events = defaultdict(list)
    field_map = {
        "category": category_ids,
        "mgg": arrays["m_gammagamma"][valid],
        "ptt": arrays["pTt_gammagamma"][valid],
        "delta_r": np.full(int(np.sum(valid)), np.nan),
        "lead_pt": arrays["lead_pt"][valid],
        "sublead_pt": arrays["sublead_pt"][valid],
        "weight": arrays["weight"][valid],
        "is_sideband": arrays["is_sideband"][valid].astype(bool),
        "is_signal_window": arrays["is_signal_window"][valid].astype(bool),
        "event_number": arrays["event_number"][valid],
        "run_number": arrays["run_number"][valid],
        "lead_eta": arrays["lead_eta"][valid],
        "sublead_eta": arrays["sublead_eta"][valid],
        "photon_multiplicity": np.full(int(np.sum(valid)), 2),
        "n_jets": arrays["N_jets_30"][valid],
        "mjj": arrays["m_jj_30"][valid],
        "delta_eta_jj": arrays["abs_delta_eta_jj_30"][valid],
        "N_jets_25": arrays["N_jets_25"][valid],
        "N_jets_30": arrays["N_jets_30"][valid],
        "N_central_jets_25": arrays["N_central_jets_25"][valid],
        "N_forward_jets_25": arrays["N_forward_jets_25"][valid],
        "N_btag_25": arrays["N_btag_25"][valid],
        "N_lep": arrays["N_lep"][valid],
        "m_ll": arrays["m_ll"][valid],
        "Z_ll_veto": arrays["Z_ll_veto"][valid],
        "m_e_gamma_veto": arrays["m_e_gamma_veto"][valid],
        "pT_lepton_plus_MET": arrays["pT_lepton_plus_MET"][valid],
        "MET": arrays["MET"][valid],
        "MET_significance": arrays["MET_significance"][valid],
        "leading_jet_pT_30": arrays["leading_jet_pT_30"][valid],
        "pT_gammagamma": arrays["pT_gammagamma"][valid],
        "max_abs_photon_eta": arrays["max_abs_photon_eta"][valid],
        "pT_Hjj_30": arrays["pT_Hjj_30"][valid],
        "VBF_centrality": arrays["VBF_centrality"][valid],
        "BDT_ttH": arrays["BDT_ttH"][valid],
        "BDT_VH": arrays["BDT_VH"][valid],
        "BDT_VBF": arrays["BDT_VBF"][valid],
    }
    for key, value in field_map.items():
        events[key] = np.asarray(value)

    output = {
        "sample_id": sample["sample_id"],
        "process_key": sample["process_key"],
        "kind": sample["kind"],
        "analysis_role": sample["analysis_role"],
        "cutflow": _section8_cutflow(processed, arrays, assigned, sample),
        "events": dict(events),
        "io_diagnostics": {"status": "section8_adapter"},
        "object_summary": {
            "selected_entries": int(len(events["mgg"])),
            "avg_good_jets": float(np.mean(events["n_jets"])) if len(events["n_jets"]) else 0.0,
            "avg_photon_multiplicity": 2.0 if len(events["mgg"]) else 0.0,
            "blocked_entries": int(np.sum(blocked)),
        },
        "section8_assignment_summary": {
            "valid_entries": int(np.sum(valid)),
            "blocked_entries": int(np.sum(blocked)),
            "unassigned_entries": int(np.sum(assigned == "unassigned")),
            "reason_examples": np.unique(reasons[~valid]).tolist()[:10] if np.any(~valid) else [],
        },
    }
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_path = cache_dir / f"{sample['sample_id']}.npz"
        np.savez_compressed(cache_path, **dict(events))
        output["cache_path"] = str(cache_path)
    return output
