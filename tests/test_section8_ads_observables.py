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
    return ak.Array(
        {
            "eventNumber": [101, 102],
            "runNumber": [1, 1],
            "mcWeight": [1.0, 1.0],
            "ScaleFactor_PILEUP": [1.0, 1.0],
            "ScaleFactor_PHOTON": [1.0, 1.0],
            "ScaleFactor_JVT": [1.0, 1.0],
            "ScaleFactor_FTAG": [1.0, 1.0],
            "trigP": [True, True],
            "photon_pt": [[lead_pt, sublead_pt], [lead_pt, sublead_pt]],
            "photon_eta": [[lead_eta, sublead_eta], [lead_eta, sublead_eta]],
            "photon_phi": [[0.0, np.pi], [0.0, np.pi]],
            "photon_e": [
                [_energy(lead_pt, lead_eta), _energy(sublead_pt, sublead_eta)],
                [_energy(lead_pt, lead_eta), _energy(sublead_pt, sublead_eta)],
            ],
            "photon_isLooseID": [[True, True], [True, True]],
            "photon_isTightID": [[True, True], [True, False]],
            "photon_ptcone20": [[0.0, 0.0], [0.0, 0.0]],
            "photon_topoetcone40": [[0.0, 0.0], [0.0, 0.0]],
            "photon_isLooseIso": [[True, True], [True, True]],
            "photon_isTightIso": [[True, True], [True, True]],
            "jet_pt": [[50.0, 45.0, 40.0], [45.0, 35.0]],
            "jet_eta": [[0.2, -1.0, 2.7], [-2.5, 2.5]],
            "jet_phi": [[0.4, 2.2, -1.4], [1.0, -2.0]],
            "jet_e": [[55.0, 70.0, 310.0], [280.0, 220.0]],
            "jet_jvt": [[1.0, 1.0, 1.0], [1.0, 1.0]],
            "jet_btag_quantile": [[4, 0, 0], [0, 0]],
            "lep_type": [[11], [11]],
            "lep_pt": [[5.0], [5.0]],
            "lep_eta": [[0.1], [0.1]],
            "lep_phi": [[0.2], [0.2]],
            "lep_e": [[5.1], [5.1]],
            "lep_charge": [[1], [1]],
            "lep_isMediumID": [[True], [True]],
            "lep_isLooseIso": [[True], [True]],
            "lep_isTightIso": [[True], [True]],
            "lep_z0": [[0.0], [0.0]],
            "lep_d0sig": [[0.0], [0.0]],
            "met": [40.0, 35.0],
            "met_phi": [0.1, -0.2],
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
    assert nominal_arrays["event_number"].tolist() == [101]
    assert candidate_arrays["event_number"].tolist() == [101, 102]
    assert candidate_arrays["baseline_selected"].tolist() == [1, 0]
    assert candidate_arrays["photon_region"].tolist() == [
        "nominal_photon_region",
        "anti_id_or_iso_control_region",
    ]

    shared_fields = [
        "m_gammagamma",
        "pT_gammagamma",
        "eta_gammagamma",
        "pTt_gammagamma",
        "lead_pt",
        "sublead_pt",
        "lead_pt_over_mgg",
        "sublead_pt_over_mgg",
        "N_jets_25",
        "N_jets_30",
        "N_jets_25_jvt_diagnostic",
        "N_jets_30_jvt_diagnostic",
        "N_btag_25",
        "MET",
        "MET_significance",
        "H_T",
        "training_mask_tth",
        "training_mask_vh",
        "training_mask_vbf",
    ]
    for field in shared_fields:
        assert np.allclose(nominal_arrays[field], candidate_arrays[field][:1], equal_nan=True), field

    assert np.isnan(candidate_arrays["m_ll"]).all()
    assert np.isnan(candidate_arrays["pT_lepton_plus_MET"]).all()
    assert len(candidate_arrays["bdt_subregion"]) == 2


def test_section8_bdt_metadata_paths_are_explicit_and_portable(tmp_path: Path) -> None:
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
