from __future__ import annotations

import copy
from typing import Any

import numpy as np

from analysis.routing.config import load_routing_config
from analysis.routing.router import route_categories


ORDERED_CATEGORIES = [
    "tH lep 0fwd",
    "tH lep 1fwd",
    "ttH lep",
    "ttH had BDT1",
    "ttH had BDT2",
    "ttH had BDT3",
    "ttH had BDT4",
    "tH had 4j1b",
    "tH had 4j2b",
    "VH dilep",
    "VH lep High",
    "VH lep Low",
    "VH MET High",
    "VH MET Low",
    "jet BSM",
    "VH had tight",
    "VH had loose",
    "VBF tight, high pT_Hjj",
    "VBF loose, high pT_Hjj",
    "VBF tight, low pT_Hjj",
    "VBF loose, low pT_Hjj",
    "ggH 2J BSM",
    "ggH 2J High",
    "ggH 2J Med",
    "ggH 2J Low",
    "ggH 1J BSM",
    "ggH 1J High",
    "ggH 1J Med",
    "ggH 1J Low",
    "ggH 0J Fwd",
    "ggH 0J Cen",
]

BDT_DEPENDENT_CATEGORIES = [
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
]

NON_BDT_CATEGORIES = [category for category in ORDERED_CATEGORIES if category not in BDT_DEPENDENT_CATEGORIES]


def _blocked_if_missing(values: dict[str, np.ndarray], required: list[str]) -> np.ndarray:
    blocked = np.zeros(len(values["event_number"]), dtype=bool)
    for name in required:
        array = values.get(name)
        if array is None:
            return np.ones(len(values["event_number"]), dtype=bool)
        blocked |= ~np.isfinite(array)
    return blocked


def legacy_assign_categories(values: dict[str, np.ndarray], boundaries: dict[str, list[float]] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = boundaries or {
        "BDT_ttH": [0.52, 0.79, 0.83, 0.92],
        "BDT_VH": [0.35, 0.78],
        "BDT_VBF_high": [-0.32, 0.47],
        "BDT_VBF_low": [0.26, 0.87],
    }
    n = len(values["event_number"])
    assigned = np.full(n, "unassigned", dtype=object)
    reasons = np.full(n, "", dtype=object)
    blocked = np.zeros(n, dtype=bool)
    active = values["baseline_selected"].astype(bool)

    tth_edges = boundaries["BDT_ttH"]
    vh_edges = boundaries["BDT_VH"]
    vbf_high = boundaries["BDT_VBF_high"]
    vbf_low = boundaries["BDT_VBF_low"]

    def claim(mask: np.ndarray, name: str, reason: str) -> None:
        claim_mask = active & ~blocked & (assigned == "unassigned") & mask
        assigned[claim_mask] = name
        reasons[claim_mask] = reason

    def block(mask: np.ndarray, reason: str) -> None:
        block_mask = active & (assigned == "unassigned") & mask
        blocked[block_mask] = True
        reasons[block_mask] = reason

    n_lep = values["N_lep"]
    n_jets25 = values["N_jets_25"]
    n_jets30 = values["N_jets_30"]
    n_central25 = values["N_central_jets_25"]
    n_forward25 = values["N_forward_jets_25"]
    n_b25 = values["N_btag_25"]
    mll = values["m_ll"]
    z_veto = values["Z_ll_veto"]
    megamma_veto = values["m_e_gamma_veto"]
    ptlepmet = values["pT_lepton_plus_MET"]
    met = values["MET"]
    met_sig = values["MET_significance"]
    lead_jet_pt30 = values["leading_jet_pT_30"]
    mjj30 = values["m_jj_30"]
    pthjj30 = values["pT_Hjj_30"]
    vbf_cent = values["VBF_centrality"]
    deta30 = values["abs_delta_eta_jj_30"]
    ptgg = values["pT_gammagamma"]
    abs_eta_max = values["max_abs_photon_eta"]
    bdt_tth = values.get("BDT_ttH")
    bdt_vh = values.get("BDT_VH")
    bdt_vbf = values.get("BDT_VBF")

    claim((n_lep == 1) & (n_central25 <= 3) & (n_b25 >= 1) & (n_forward25 == 0), "tH lep 0fwd", "single lepton top-associated with no forward jet")
    claim((n_lep == 1) & (n_central25 <= 4) & (n_b25 >= 1) & (n_forward25 >= 1), "tH lep 1fwd", "single lepton top-associated with forward jet")
    claim((n_lep >= 1) & (n_central25 >= 2) & (n_b25 >= 1) & z_veto.astype(bool), "ttH lep", "leptonic ttH topology with Z veto")

    tth_base = (n_lep == 0) & (n_jets30 >= 3) & (n_b25 >= 1)
    tth_block = tth_base & _blocked_if_missing(values, ["BDT_ttH"])
    block(tth_block, "BDT_ttH missing for ttH had categories")
    claim(tth_base & (bdt_tth > tth_edges[3]), "ttH had BDT1", "supplemental BDT_ttH highest-score bin")
    claim(tth_base & (bdt_tth > tth_edges[2]) & (bdt_tth <= tth_edges[3]), "ttH had BDT2", "supplemental BDT_ttH bin 2")
    claim(tth_base & (bdt_tth > tth_edges[1]) & (bdt_tth <= tth_edges[2]), "ttH had BDT3", "supplemental BDT_ttH bin 3")
    claim(tth_base & (bdt_tth > tth_edges[0]) & (bdt_tth <= tth_edges[1]), "ttH had BDT4", "supplemental BDT_ttH bin 4")

    claim((n_lep == 0) & (n_central25 == 4) & (n_b25 == 1), "tH had 4j1b", "hadronic tH 4j1b topology")
    claim((n_lep == 0) & (n_central25 == 4) & (n_b25 >= 2), "tH had 4j2b", "hadronic tH 4j2b topology")

    claim((n_lep >= 2) & np.isfinite(mll) & (mll >= 70.0) & (mll <= 110.0), "VH dilep", "same-flavor opposite-sign dilepton candidate")
    claim((n_lep == 1) & megamma_veto.astype(bool) & np.isfinite(ptlepmet) & (ptlepmet > 150.0), "VH lep High", "single-lepton VH high pT(l+MET)")
    claim((n_lep == 1) & megamma_veto.astype(bool) & np.isfinite(ptlepmet) & (ptlepmet <= 150.0) & np.isfinite(met_sig) & (met_sig > 1.0), "VH lep Low", "single-lepton VH low pT(l+MET)")
    claim((((met > 150.0) & (met < 250.0) & (met_sig > 9.0)) | (met > 250.0)), "VH MET High", "MET-enriched VH category")
    claim((met > 80.0) & (met < 150.0) & (met_sig > 8.0), "VH MET Low", "lower-MET VH category")
    claim(lead_jet_pt30 > 200.0, "jet BSM", "boosted leading jet category")

    vh_base = np.isfinite(mjj30) & (mjj30 > 60.0) & (mjj30 < 120.0)
    block(vh_base & _blocked_if_missing(values, ["BDT_VH"]), "BDT_VH missing for hadronic VH categories")
    claim(vh_base & (bdt_vh > vh_edges[1]), "VH had tight", "supplemental BDT_VH tight bin")
    claim(vh_base & (bdt_vh > vh_edges[0]) & (bdt_vh <= vh_edges[1]), "VH had loose", "supplemental BDT_VH loose bin")

    vbf_base = (n_jets30 >= 2) & (deta30 > 2.0) & (vbf_cent < 5.0) & np.isfinite(pthjj30)
    block(vbf_base & _blocked_if_missing(values, ["BDT_VBF"]), "BDT_VBF missing for VBF categories")
    high_hjj = vbf_base & (pthjj30 > 25.0)
    low_hjj = vbf_base & (pthjj30 <= 25.0)
    claim(high_hjj & (bdt_vbf > vbf_high[1]), "VBF tight, high pT_Hjj", "supplemental BDT_VBF high-pT tight bin")
    claim(high_hjj & (bdt_vbf > vbf_high[0]) & (bdt_vbf <= vbf_high[1]), "VBF loose, high pT_Hjj", "supplemental BDT_VBF high-pT loose bin")
    claim(low_hjj & (bdt_vbf > vbf_low[1]), "VBF tight, low pT_Hjj", "supplemental BDT_VBF low-pT tight bin")
    claim(low_hjj & (bdt_vbf > vbf_low[0]) & (bdt_vbf <= vbf_low[1]), "VBF loose, low pT_Hjj", "supplemental BDT_VBF low-pT loose bin")

    claim((n_jets30 >= 2) & (ptgg >= 200.0), "ggH 2J BSM", "untagged ggH 2-jet BSM bin")
    claim((n_jets30 >= 2) & (ptgg >= 120.0) & (ptgg < 200.0), "ggH 2J High", "untagged ggH 2-jet high pT")
    claim((n_jets30 >= 2) & (ptgg >= 60.0) & (ptgg < 120.0), "ggH 2J Med", "untagged ggH 2-jet medium pT")
    claim((n_jets30 >= 2) & (ptgg < 60.0), "ggH 2J Low", "untagged ggH 2-jet low pT")
    claim((n_jets30 == 1) & (ptgg >= 200.0), "ggH 1J BSM", "untagged ggH 1-jet BSM bin")
    claim((n_jets30 == 1) & (ptgg >= 120.0) & (ptgg < 200.0), "ggH 1J High", "untagged ggH 1-jet high pT")
    claim((n_jets30 == 1) & (ptgg >= 60.0) & (ptgg < 120.0), "ggH 1J Med", "untagged ggH 1-jet medium pT")
    claim((n_jets30 == 1) & (ptgg < 60.0), "ggH 1J Low", "untagged ggH 1-jet low pT")
    claim((n_jets30 == 0) & (abs_eta_max > 0.95), "ggH 0J Fwd", "untagged ggH forward-photon bin")
    claim((n_jets30 == 0) & (abs_eta_max <= 0.95), "ggH 0J Cen", "untagged ggH central-photon bin")

    final_blocked = active & (assigned == "unassigned") & blocked
    assigned[final_blocked] = "blocked_missing_input"
    reasons[final_blocked] = np.where(reasons[final_blocked] == "", "blocked by missing classifier or derived input", reasons[final_blocked])
    reasons[(assigned == "unassigned") & active] = "no category matched after full priority scan"
    return assigned.astype(str), reasons.astype(str), blocked.astype(bool)


def _replace_select_when(config, category_id: str, select_when: dict[str, Any]) -> None:
    for category in config.categories:
        if category.id == category_id:
            category.select_when.clear()
            category.select_when.update(copy.deepcopy(select_when))
            return
    raise KeyError(f"Unknown Section 8 routing category id: {category_id}")


def _section8_routing_config(boundaries: dict[str, list[float]] | None = None, routing_config: str | None = None):
    config = copy.deepcopy(load_routing_config(routing_config or "configs/routing/section8_ads_bdt.yaml"))
    if boundaries is None:
        return config
    tth_edges = boundaries["BDT_ttH"]
    vh_edges = boundaries["BDT_VH"]
    vbf_high = boundaries["BDT_VBF_high"]
    vbf_low = boundaries["BDT_VBF_low"]
    _replace_select_when(config, "ttH_had_BDT1", {"all": [{"field": "BDT_ttH", "op": ">", "value": tth_edges[3]}]})
    _replace_select_when(
        config,
        "ttH_had_BDT2",
        {"all": [{"field": "BDT_ttH", "op": ">", "value": tth_edges[2]}, {"field": "BDT_ttH", "op": "<=", "value": tth_edges[3]}]},
    )
    _replace_select_when(
        config,
        "ttH_had_BDT3",
        {"all": [{"field": "BDT_ttH", "op": ">", "value": tth_edges[1]}, {"field": "BDT_ttH", "op": "<=", "value": tth_edges[2]}]},
    )
    _replace_select_when(
        config,
        "ttH_had_BDT4",
        {"all": [{"field": "BDT_ttH", "op": ">", "value": tth_edges[0]}, {"field": "BDT_ttH", "op": "<=", "value": tth_edges[1]}]},
    )
    _replace_select_when(config, "VH_had_tight", {"all": [{"field": "BDT_VH", "op": ">", "value": vh_edges[1]}]})
    _replace_select_when(
        config,
        "VH_had_loose",
        {"all": [{"field": "BDT_VH", "op": ">", "value": vh_edges[0]}, {"field": "BDT_VH", "op": "<=", "value": vh_edges[1]}]},
    )
    _replace_select_when(config, "VBF_tight_high_pT_Hjj", {"all": [{"field": "BDT_VBF", "op": ">", "value": vbf_high[1]}]})
    _replace_select_when(
        config,
        "VBF_loose_high_pT_Hjj",
        {"all": [{"field": "BDT_VBF", "op": ">", "value": vbf_high[0]}, {"field": "BDT_VBF", "op": "<=", "value": vbf_high[1]}]},
    )
    _replace_select_when(config, "VBF_tight_low_pT_Hjj", {"all": [{"field": "BDT_VBF", "op": ">", "value": vbf_low[1]}]})
    _replace_select_when(
        config,
        "VBF_loose_low_pT_Hjj",
        {"all": [{"field": "BDT_VBF", "op": ">", "value": vbf_low[0]}, {"field": "BDT_VBF", "op": "<=", "value": vbf_low[1]}]},
    )
    return config


def route_section8_categories(values: dict[str, np.ndarray], boundaries: dict[str, list[float]] | None = None, routing_config: str | None = None):
    config = _section8_routing_config(boundaries, routing_config)
    return route_categories(values, config)


def assign_categories(values: dict[str, np.ndarray], boundaries: dict[str, list[float]] | None = None, routing_config: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = _section8_routing_config(boundaries, routing_config)
    result = route_categories(values, config)
    label_by_id = config.label_by_id
    assigned = np.asarray(
        [
            label_by_id.get(category, category)
            for category in result.assigned_category
        ],
        dtype=str,
    )
    return assigned, result.assignment_reason, result.assignment_blocked
