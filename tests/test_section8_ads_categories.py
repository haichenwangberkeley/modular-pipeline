from __future__ import annotations

import numpy as np

from analysis.section8_ads.categories import assign_categories


def _base_arrays(n: int) -> dict[str, np.ndarray]:
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


def test_assign_categories_blocks_missing_bdt_before_later_match() -> None:
    arrays = _base_arrays(1)
    arrays["N_lep"][0] = 0
    arrays["N_jets_30"][0] = 3
    arrays["N_btag_25"][0] = 1
    arrays["pT_gammagamma"][0] = 250.0
    category, reason, blocked = assign_categories(arrays)
    assert category.tolist() == ["blocked_missing_input"]
    assert blocked.tolist() == [True]
    assert "BDT_ttH" in reason[0]


def test_assign_categories_uses_first_match_semantics() -> None:
    arrays = _base_arrays(1)
    arrays["N_jets_30"][0] = 2
    arrays["pT_gammagamma"][0] = 220.0
    arrays["max_abs_photon_eta"][0] = 1.2
    category, _, blocked = assign_categories(arrays)
    assert category.tolist() == ["ggH 2J BSM"]
    assert blocked.tolist() == [False]
