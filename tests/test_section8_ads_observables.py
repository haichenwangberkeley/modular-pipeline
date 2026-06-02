from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from analysis.section8_ads import pipeline


def _energy(pt: float, eta: float) -> float:
    return float(pt * np.cosh(eta))


def _synthetic_section8_batch() -> ak.Array:
    lead_pt = 80.0
    sublead_pt = 60.0
    lead_eta = 0.2
    sublead_eta = -0.3
    photon_pts = [[lead_pt, sublead_pt]] * 4
    photon_etas = [[lead_eta, sublead_eta]] * 4
    photon_phis = [[0.0, np.pi]] * 4
    photon_es = [[_energy(lead_pt, lead_eta), _energy(sublead_pt, sublead_eta)]] * 4
    return ak.Array(
        {
            "eventNumber": [101, 102, 103, 104],
            "runNumber": [1, 1, 2, 2],
            "mcWeight": [1.0, 1.0, 1.0, 1.0],
            "ScaleFactor_PILEUP": [1.0, 1.0, 1.0, 1.0],
            "ScaleFactor_PHOTON": [1.0, 1.0, 1.0, 1.0],
            "ScaleFactor_JVT": [1.0, 1.0, 1.0, 1.0],
            "ScaleFactor_FTAG": [1.0, 1.0, 1.0, 1.0],
            "trigP": [True, True, True, True],
            "photon_pt": photon_pts,
            "photon_eta": photon_etas,
            "photon_phi": photon_phis,
            "photon_e": photon_es,
            "photon_isLooseID": [[True, True], [True, True], [True, True], [True, True]],
            "photon_isTightID": [[True, True], [True, False], [True, True], [True, True]],
            "photon_ptcone20": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "photon_topoetcone40": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "photon_isLooseIso": [[True, True], [True, True], [True, True], [True, True]],
            "photon_isTightIso": [[True, True], [True, True], [True, True], [True, True]],
            "jet_pt": [[70.0, 60.0, 35.0], [45.0, 35.0], [], [50.0]],
            "jet_eta": [[3.0, -3.0, 0.5], [-2.5, 2.5], [], [0.3]],
            "jet_phi": [[1.0, -2.0, 0.8], [1.0, -2.0], [], [0.4]],
            "jet_e": [
                [_energy(70.0, 3.0), _energy(60.0, -3.0), _energy(35.0, 0.5)],
                [_energy(45.0, -2.5), _energy(35.0, 2.5)],
                [],
                [_energy(50.0, 0.3)],
            ],
            "jet_jvt": [[1.0, 1.0, 1.0], [1.0, 1.0], [], [1.0]],
            "jet_btag_quantile": [[4, 0, 0], [0, 0], [], [0]],
            "lep_type": [[11, 11, 11], [11], [], []],
            "lep_pt": [[45.0, 45.0, 25.3125], [5.0], [], []],
            "lep_eta": [[0.0, 0.0, 0.0], [0.1], [], []],
            "lep_phi": [[0.0, np.pi, np.pi], [0.2], [], []],
            "lep_e": [[45.0, 45.0, 25.3125], [5.1], [], []],
            "lep_charge": [[1, -1, 1], [1], [], []],
            "lep_isMediumID": [[True, True, True], [True], [], []],
            "lep_isLooseIso": [[True, True, True], [True], [], []],
            "lep_isTightIso": [[True, True, True], [True], [], []],
            "lep_z0": [[0.0, 0.0, 0.0], [0.0], [], []],
            "lep_d0sig": [[0.0, 0.0, 0.0], [0.0], [], []],
            "met": [40.0, 35.0, 20.0, 55.0],
            "met_phi": [0.1, -0.2, 1.2, -0.4],
        }
    )


def _sample() -> dict:
    return {
        "sample_id": "data_synthetic",
        "process_key": "data",
        "kind": "data",
        "analysis_role": "data",
        "files": ["synthetic.root"],
    }


def test_shared_observables_match_nominal_and_training_candidate_overlap(monkeypatch) -> None:
    batch = _synthetic_section8_batch()
    monkeypatch.setattr(pipeline, "_iterate_batches", lambda *args, **kwargs: iter([batch]))

    nominal = pipeline._process_sample(_sample(), trigger_policy="input_preselected")
    candidates = pipeline._process_bdt_candidates(_sample(), trigger_policy="input_preselected")

    nominal_arrays = nominal["arrays"]
    candidate_arrays = candidates["bdt_arrays"]
    assert nominal_arrays["event_number"].tolist() == [101, 103, 104]
    assert candidate_arrays["event_number"].tolist() == [101, 102, 103, 104]
    assert candidate_arrays["baseline_selected"].tolist() == [1, 0, 1, 1]
    assert candidate_arrays["photon_region"].tolist() == [
        "nominal_photon_region",
        "anti_id_or_iso_control_region",
        "nominal_photon_region",
        "nominal_photon_region",
    ]
    assert candidate_arrays["nominal_photon_region"].tolist() == [1, 0, 1, 1]
    assert candidate_arrays["anti_id_or_iso_control_region"].tolist() == [0, 1, 0, 0]
    assert "photon_region" not in nominal_arrays
    assert "bdt_subregion" not in nominal_arrays

    shared_fields = [
        "event_number",
        "run_number",
        "weight",
        "trigger_passed",
        "baseline_selected",
        "is_sideband",
        "is_signal_window",
        "m_gammagamma",
        "pT_gammagamma",
        "eta_gammagamma",
        "pTt_gammagamma",
        "lead_pt",
        "sublead_pt",
        "lead_pt_over_mgg",
        "sublead_pt_over_mgg",
        "lead_eta",
        "sublead_eta",
        "max_abs_photon_eta",
        "N_jets_25",
        "N_jets_30",
        "N_jets_25_jvt_diagnostic",
        "N_jets_30_jvt_diagnostic",
        "N_central_jets_25",
        "N_forward_jets_25",
        "N_btag_25",
        "N_lep",
        "MET",
        "MET_significance",
        "leading_jet_pT_30",
        "m_jj_30",
        "abs_delta_eta_jj_30",
        "pT_Hjj_30",
        "deltaR_min_gamma_j",
        "VBF_centrality",
        "H_T",
        "m_all_jets",
        "delta_y_gammagamma_jj",
        "cos_theta_star_gammagamma_jj",
        "abs_delta_phi_gammagamma_jj_capped",
        "training_mask_tth",
        "training_mask_vh",
        "training_mask_vbf",
    ]

    candidate_index_by_event = {
        int(event_number): idx for idx, event_number in enumerate(candidate_arrays["event_number"])
    }
    candidate_overlap = np.asarray(
        [candidate_index_by_event[int(event_number)] for event_number in nominal_arrays["event_number"]],
        dtype=int,
    )
    for field in shared_fields:
        nominal_values = np.asarray(nominal_arrays[field])
        candidate_values = np.asarray(candidate_arrays[field])[candidate_overlap]
        if np.issubdtype(nominal_values.dtype, np.floating):
            assert np.allclose(nominal_values, candidate_values, equal_nan=True), field
        else:
            assert np.array_equal(nominal_values, candidate_values), field

    candidate_first = candidate_index_by_event[101]
    intentional_candidate_placeholders = {
        "m_ll": np.nan,
        "Z_ll_veto": 1,
        "m_e_gamma_veto": 1,
        "pT_lepton_plus_MET": np.nan,
    }
    for field, placeholder in intentional_candidate_placeholders.items():
        assert field not in shared_fields
        if np.isnan(placeholder):
            assert np.isnan(candidate_arrays[field][candidate_first])
        else:
            assert candidate_arrays[field][candidate_first] == placeholder
    assert len(candidate_arrays["bdt_subregion"]) == 4
    assert candidate_arrays["bdt_subregion"].dtype.kind in {"O", "U"}


def test_section8_bdt_metadata_defaults_are_output_local(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    pipeline._write_bdt_optimization_metadata(
        outputs=outputs,
        ads={"config_hash": "ads-hash"},
        inputs=Path("input-data"),
        selected_samples=[
            {
                "sample_id": "sample",
                "process_key": "data",
                "kind": "data",
                "files": ["data.root"],
            }
        ],
        training_report={"classifiers": {"BDT_ttH": {"status": "trained"}}},
        audit_report={"status": "ok"},
        prepare_bdt_training=True,
        train_bdts=True,
        score_bdts=True,
    )

    registry_path = outputs / "metadata" / "runs.jsonl"
    runs_dir = outputs / "metadata" / "runs"
    assert registry_path.exists()
    assert len(list(runs_dir.glob("section8_bdt_*/observations.yaml"))) == 1
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "optimization_infra").exists()


def test_section8_bdt_metadata_paths_allow_explicit_central_overrides(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    runs_dir = tmp_path / "runs"
    registry_path = tmp_path / "optimization" / "runs.jsonl"
    pipeline._write_bdt_optimization_metadata(
        outputs=outputs,
        ads={"config_hash": "ads-hash"},
        inputs=Path("input-data"),
        selected_samples=[
            {
                "sample_id": "sample",
                "process_key": "data",
                "kind": "data",
                "files": ["data.root"],
            }
        ],
        training_report={"classifiers": {"BDT_ttH": {"status": "trained"}}},
        audit_report={"status": "ok"},
        prepare_bdt_training=True,
        train_bdts=True,
        score_bdts=True,
        runs_dir=runs_dir,
        registry_path=registry_path,
    )

    assert registry_path.exists()
    assert len(list(runs_dir.glob("section8_bdt_*/observations.yaml"))) == 1
    assert "/" + "Users" + "/" not in registry_path.read_text()
