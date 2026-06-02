from __future__ import annotations

from pathlib import Path
from typing import Any

import uproot

from analysis.common import write_json, write_text


BRANCH_CANDIDATES: dict[str, list[str]] = {
    "event_number": ["eventNumber"],
    "run_number": ["runNumber"],
    "mc_weight": ["mcWeight"],
    "photon_pt": ["photon_pt"],
    "photon_eta": ["photon_eta"],
    "photon_phi": ["photon_phi"],
    "photon_e": ["photon_e"],
    "photon_loose_id": ["photon_isLooseID"],
    "photon_tight_id": ["photon_isTightID"],
    "photon_loose_iso": ["photon_isLooseIso"],
    "photon_tight_iso": ["photon_isTightIso"],
    "photon_ptcone20": ["photon_ptcone20"],
    "photon_topoetcone40": ["photon_topoetcone40"],
    "jet_pt": ["jet_pt"],
    "jet_eta": ["jet_eta"],
    "jet_phi": ["jet_phi"],
    "jet_e": ["jet_e"],
    "jet_jvt": ["jet_jvt"],
    "jet_btag_quantile": ["jet_btag_quantile"],
    "lep_type": ["lep_type"],
    "lep_pt": ["lep_pt"],
    "lep_eta": ["lep_eta"],
    "lep_phi": ["lep_phi"],
    "lep_e": ["lep_e"],
    "lep_charge": ["lep_charge"],
    "lep_medium_id": ["lep_isMediumID"],
    "lep_loose_iso": ["lep_isLooseIso"],
    "lep_tight_iso": ["lep_isTightIso"],
    "lep_z0": ["lep_z0"],
    "lep_d0sig": ["lep_d0sig"],
    "met": ["met"],
    "met_phi": ["met_phi"],
    "trigger_diphoton_proxy": ["trigP"],
    "trigger_dilepton": ["TriggerMatch_DILEPTON"],
    "scale_factor_pileup": ["ScaleFactor_PILEUP"],
    "scale_factor_photon": ["ScaleFactor_PHOTON"],
    "scale_factor_jvt": ["ScaleFactor_JVT"],
    "scale_factor_ftag": ["ScaleFactor_FTAG", "ScaleFactor_BTAG"],
}


def inspect_available_fields(files: list[str], tree_name: str = "analysis", sample_limit: int = 3) -> list[str]:
    available: set[str] = set()
    for file_path in files[:sample_limit]:
        with uproot.open(file_path) as handle:
            available.update(handle[tree_name].keys())
    return sorted(available)


def build_branch_mapping(
    files: list[str],
    ads: dict[str, Any],
    outputs: Path,
    tree_name: str = "analysis",
) -> dict[str, Any]:
    available = inspect_available_fields(files, tree_name=tree_name)
    available_set = set(available)
    resolved = {}
    unresolved = []
    uncertain = []

    for logical_name, candidates in BRANCH_CANDIDATES.items():
        match = next((branch for branch in candidates if branch in available_set), None)
        if match is None:
            unresolved.append(
                {
                    "logical_name": logical_name,
                    "candidate_branches": candidates,
                    "status": "missing",
                }
            )
            continue
        resolved[logical_name] = {
            "branch": match,
            "status": "resolved",
        }

    btag_mapping = {
        "source_branch": "jet_btag_quantile" if "jet_btag_quantile" in available_set else None,
        "working_point_target": "70_percent_efficiency_proxy",
        "operational_choice": "jet_btag_quantile >= 4",
        "status": "approximate_with_local_evidence" if "jet_btag_quantile" in available_set else "missing",
        "confidence": "medium",
        "notes": [
            "Local open-data reference workflows in the adjacent analysis workspace use jet_btag_quantile >= 4 as the b-tag proxy.",
            "This is treated as a user-approved approximation for the ADS 70% working point.",
        ],
    }
    uncertain.append({"logical_name": "btag_working_point", **btag_mapping})

    missing_impacts = []
    if "sum_et" not in available_set:
        missing_impacts.append(
            {
                "missing_input": "sum_ET",
                "affected_variables": ["MET_significance"],
                "affected_categories": ["VH MET High", "VH MET Low", "VH lep Low"],
                "recovery": "Approximate MET_significance as MET/sqrt(HT).",
            }
        )
    if "BDT_ttH" not in available_set and "BDT_VH" not in available_set and "BDT_VBF" not in available_set:
        missing_impacts.append(
            {
                "missing_input": "official_classifier_score_branches",
                "affected_variables": ["BDT_ttH", "BDT_VH", "BDT_VBF"],
                "affected_categories": [
                    item["region_identifier"]
                    for item in ads["ordered_categories"]
                    if "BDT_" in " ".join(item["selection_requirements"])
                ],
                "recovery": "Train supplemental BDTs from ADS feature sets.",
            }
        )

    payload = {
        "status": "ok",
        "tree_name": tree_name,
        "available_fields": available,
        "resolved_mappings": resolved,
        "uncertain_mappings": uncertain,
        "unresolved_mappings": unresolved,
        "missing_input_impacts": missing_impacts,
    }
    write_json(payload, outputs / "branch_mapping_report.json")
    write_text(render_branch_mapping_markdown(payload), outputs / "branch_mapping_report.md")
    return payload


def render_branch_mapping_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Branch Mapping Report",
        "",
        f"- Tree name: `{payload['tree_name']}`",
        f"- Resolved mappings: `{len(payload['resolved_mappings'])}`",
        f"- Uncertain mappings: `{len(payload['uncertain_mappings'])}`",
        f"- Unresolved mappings: `{len(payload['unresolved_mappings'])}`",
        "",
        "## Resolved",
        "",
    ]
    for logical_name, mapping in sorted(payload["resolved_mappings"].items()):
        lines.append(f"- `{logical_name}` -> `{mapping['branch']}`")
    lines.extend(["", "## Uncertain", ""])
    for item in payload["uncertain_mappings"]:
        lines.append(
            f"- `{item['logical_name']}`: {item['operational_choice']} ({item['status']})"
        )
    lines.extend(["", "## Missing Impacts", ""])
    for item in payload["missing_input_impacts"]:
        lines.append(
            f"- `{item['missing_input']}` affects {', '.join(item['affected_categories'])}: {item['recovery']}"
        )
    return "\n".join(lines) + "\n"
