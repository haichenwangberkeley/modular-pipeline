from __future__ import annotations

import numpy as np

from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.config.versions import apply_analysis_version
from analysis.selections.engine import assign_categories, legacy_assign_categories
from analysis.section8_ads.categories import assign_categories as assign_section8_categories
from analysis.section8_ads.categories import legacy_assign_categories as legacy_assign_section8_categories


def test_generic_router_matches_legacy_five_category_assignments() -> None:
    features = {
        "diphoton_mass": np.full(5, 125.0),
        "ptt": np.asarray([40.0, 40.0, 80.0, 40.0, 80.0]),
        "delta_r": np.zeros(5),
        "lead_pt": np.zeros(5),
        "sublead_pt": np.zeros(5),
        "lead_eta": np.asarray([0.1, 0.1, 0.1, 1.1, 1.1]),
        "sublead_eta": np.asarray([0.1, 0.2, 0.2, 1.2, 1.2]),
        "photon_multiplicity": np.full(5, 2),
        "n_jets": np.asarray([2, 0, 0, 0, 0]),
        "mjj": np.asarray([450.0, 0.0, 0.0, 0.0, 0.0]),
        "delta_eta_jj": np.asarray([3.0, 0.0, 0.0, 0.0, 0.0]),
    }
    cfg = apply_analysis_version(DEFAULT_RUNTIME, version_name="round1_5cat")

    legacy = legacy_assign_categories(features, cfg)
    generic = assign_categories(features, cfg)

    assert generic.tolist() == legacy.tolist()
    assert generic.tolist() == [
        "two_jet_vbf_enriched",
        "central_low_ptt",
        "central_high_ptt",
        "rest_low_ptt",
        "rest_high_ptt",
    ]


def _base_section8_arrays(n: int) -> dict[str, np.ndarray]:
    return {
        "event_number": np.arange(n, dtype=np.int64),
        "baseline_selected": np.ones(n, dtype=int),
        "N_lep": np.zeros(n, dtype=float),
        "N_jets_25": np.zeros(n, dtype=float),
        "N_jets_30": np.zeros(n, dtype=float),
        "N_central_jets_25": np.zeros(n, dtype=float),
        "N_forward_jets_25": np.zeros(n, dtype=float),
        "N_btag_25": np.zeros(n, dtype=float),
        "m_ll": np.full(n, np.nan),
        "Z_ll_veto": np.ones(n, dtype=float),
        "m_e_gamma_veto": np.ones(n, dtype=float),
        "pT_lepton_plus_MET": np.full(n, np.nan),
        "MET": np.zeros(n, dtype=float),
        "MET_significance": np.zeros(n, dtype=float),
        "leading_jet_pT_30": np.zeros(n, dtype=float),
        "m_jj_30": np.full(n, np.nan),
        "pT_Hjj_30": np.full(n, np.nan),
        "VBF_centrality": np.full(n, np.nan),
        "abs_delta_eta_jj_30": np.full(n, np.nan),
        "pT_gammagamma": np.zeros(n, dtype=float),
        "max_abs_photon_eta": np.zeros(n, dtype=float),
        "BDT_ttH": np.full(n, np.nan),
        "BDT_VH": np.full(n, np.nan),
        "BDT_VBF": np.full(n, np.nan),
    }


def test_generic_router_matches_legacy_section8_assignments_for_all_categories() -> None:
    expected = [
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
        "blocked_missing_input",
        "unassigned",
    ]
    arrays = _base_section8_arrays(len(expected))

    arrays["N_lep"][0] = 1
    arrays["N_central_jets_25"][0] = 3
    arrays["N_btag_25"][0] = 1

    arrays["N_lep"][1] = 1
    arrays["N_central_jets_25"][1] = 4
    arrays["N_btag_25"][1] = 1
    arrays["N_forward_jets_25"][1] = 1

    arrays["N_lep"][2] = 1
    arrays["N_central_jets_25"][2] = 5
    arrays["N_btag_25"][2] = 1

    for idx, score in zip([3, 4, 5, 6], [0.93, 0.90, 0.80, 0.60]):
        arrays["N_jets_30"][idx] = 3
        arrays["N_btag_25"][idx] = 1
        arrays["BDT_ttH"][idx] = score

    arrays["N_central_jets_25"][7] = 4
    arrays["N_btag_25"][7] = 1
    arrays["N_jets_30"][7] = 2
    arrays["N_central_jets_25"][8] = 4
    arrays["N_btag_25"][8] = 2
    arrays["N_jets_30"][8] = 2

    arrays["N_lep"][9] = 2
    arrays["m_ll"][9] = 91.0
    arrays["N_lep"][10] = 1
    arrays["pT_lepton_plus_MET"][10] = 151.0
    arrays["N_lep"][11] = 1
    arrays["pT_lepton_plus_MET"][11] = 100.0
    arrays["MET_significance"][11] = 2.0
    arrays["MET"][12] = 200.0
    arrays["MET_significance"][12] = 10.0
    arrays["MET"][13] = 100.0
    arrays["MET_significance"][13] = 9.0
    arrays["N_jets_30"][14] = 1
    arrays["leading_jet_pT_30"][14] = 201.0

    arrays["m_jj_30"][15:17] = 90.0
    arrays["BDT_VH"][15] = 0.79
    arrays["BDT_VH"][16] = 0.50

    for idx, score, pth in [(17, 0.50, 30.0), (18, 0.00, 30.0), (19, 0.90, 20.0), (20, 0.50, 20.0)]:
        arrays["N_jets_30"][idx] = 2
        arrays["m_jj_30"][idx] = 130.0
        arrays["abs_delta_eta_jj_30"][idx] = 3.0
        arrays["VBF_centrality"][idx] = 1.0
        arrays["pT_Hjj_30"][idx] = pth
        arrays["BDT_VBF"][idx] = score

    for idx, pt in zip([21, 22, 23, 24], [220.0, 150.0, 100.0, 50.0]):
        arrays["N_jets_30"][idx] = 2
        arrays["pT_gammagamma"][idx] = pt
    for idx, pt in zip([25, 26, 27, 28], [220.0, 150.0, 100.0, 50.0]):
        arrays["N_jets_30"][idx] = 1
        arrays["pT_gammagamma"][idx] = pt
    arrays["max_abs_photon_eta"][29] = 1.1
    arrays["max_abs_photon_eta"][30] = 0.5
    arrays["N_jets_30"][31] = 3
    arrays["N_btag_25"][31] = 1
    arrays["pT_gammagamma"][31] = 250.0
    arrays["max_abs_photon_eta"][32] = np.nan

    legacy_category, legacy_reason, legacy_blocked = legacy_assign_section8_categories(arrays)
    generic_category, generic_reason, generic_blocked = assign_section8_categories(arrays)

    assert legacy_category.tolist() == expected
    assert generic_category.tolist() == legacy_category.tolist()
    assert generic_reason.tolist() == legacy_reason.tolist()
    assert generic_blocked.tolist() == legacy_blocked.tolist()
