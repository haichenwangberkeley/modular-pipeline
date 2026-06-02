from __future__ import annotations

import numpy as np

from analysis.section8_ads.classifiers import (
    CONTINUUM_FALLBACK_TRANSFER_FACTOR,
    _balance_class_weights,
    _candidate_rows_for_spec,
    _peak_normalized_weights,
    deterministic_split,
    train_classifiers,
    write_training_sample_audit,
)


FEATURE_DEFAULTS = {
    "H_T": 100.0,
    "m_all_jets": 500.0,
    "N_jets_30": 3.0,
    "N_central_jets_25": 3.0,
    "N_btag_25": 1.0,
    "m_jj_30": 90.0,
    "pTt_gammagamma": 40.0,
    "delta_y_gammagamma_jj": 1.0,
    "cos_theta_star_gammagamma_jj": 0.2,
    "abs_delta_eta_jj_30": 3.0,
    "abs_delta_phi_gammagamma_jj_capped": 2.0,
    "deltaR_min_gamma_j": 1.0,
    "VBF_centrality": 1.0,
    "pT_Hjj_30": 20.0,
}


def _sample(sample_id: str, process_key: str, kind: str, photon_region: str, n: int, offset: int = 0) -> dict:
    arrays = {
        "event_number": np.arange(offset, offset + n, dtype=np.int64),
        "weight": np.ones(n, dtype=float),
        "m_gammagamma": np.linspace(105.0, 160.0, n, dtype=float),
        "photon_region": np.full(n, photon_region, dtype=object),
        "bdt_subregion": np.full(n, "inclusive", dtype=object),
        "training_mask_tth": np.ones(n, dtype=int),
        "training_mask_vh": np.ones(n, dtype=int),
        "training_mask_vbf": np.ones(n, dtype=int),
    }
    for key, value in FEATURE_DEFAULTS.items():
        arrays[key] = np.full(n, value, dtype=float)
    return {
        "sample_id": sample_id,
        "process_key": process_key,
        "kind": kind,
        "analysis_role": "signal_nominal" if kind != "data" else "data",
        "bdt_arrays": arrays,
    }


def test_deterministic_split_is_stable() -> None:
    sample_ids = np.asarray(["a", "a", "b"], dtype=object)
    events = np.asarray([1, 2, 1], dtype=np.int64)
    first = deterministic_split(sample_ids, events)
    second = deterministic_split(sample_ids, events)
    assert first.tolist() == second.tolist()
    assert np.all((first >= 0) & (first <= 9))


def test_training_audit_accepts_data_control_rows(tmp_path) -> None:
    samples = [
        _sample("data_control", "data", "data", "anti_id_or_iso_control_region", 5),
        _sample("tth", "tth", "mc", "nominal_photon_region", 5, offset=10),
    ]
    report = write_training_sample_audit(samples, tmp_path)
    assert report["classifiers"]["BDT_ttH"]["signal_rows"] == 5
    assert report["classifiers"]["BDT_ttH"]["background_rows"] == 5
    assert any(row["source_kind"] == "data_control" for row in report["rows"])


def test_train_classifiers_reports_precise_blocking_status(tmp_path) -> None:
    samples = [_sample("prompt", "prompt_diphoton", "mc", "nominal_photon_region", 20)]
    report = train_classifiers([], tmp_path, training_samples=samples)
    assert report["classifiers"]["BDT_VBF"]["status"] == "blocked_no_signal_rows"


def test_continuum_background_uses_peak_transfer_factor() -> None:
    from analysis.section8_ads.classifiers import CLASSIFIER_SPECS

    samples = [_sample("data_control", "data", "data", "anti_id_or_iso_control_region", 10)]
    payload = _candidate_rows_for_spec(CLASSIFIER_SPECS["BDT_ttH"], samples)
    weights, rows = _peak_normalized_weights("BDT_ttH", payload)
    assert rows[0]["component_type"] == "continuum_background"
    assert rows[0]["normalization_status"] == "fallback_shape_factor_low_statistics"
    assert np.isclose(np.sum(weights), 10.0 * CONTINUUM_FALLBACK_TRANSFER_FACTOR)


def test_class_balance_preserves_equal_train_weight() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    component_weights = np.asarray([2.0, 2.0, 1.0, 3.0], dtype=float)
    train_mask = np.ones(4, dtype=bool)
    final = _balance_class_weights(labels, component_weights, train_mask)
    assert np.isclose(np.sum(final[labels == 1]), 2.0)
    assert np.isclose(np.sum(final[labels == 0]), 2.0)
