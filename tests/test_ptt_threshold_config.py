from __future__ import annotations

import numpy as np

from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.selections.engine import assign_categories


def test_assign_categories_uses_independent_central_and_rest_ptt_boundaries() -> None:
    cfg = {
        **DEFAULT_RUNTIME,
        "category_ptt_boundaries_gev": {
            "central": 50.0,
            "rest": 70.0,
        },
    }
    features = {
        "diphoton_mass": np.array([125.0, 125.0, 125.0, 125.0]),
        "ptt": np.array([45.0, 55.0, 65.0, 75.0]),
        "delta_r": np.zeros(4),
        "lead_pt": np.zeros(4),
        "sublead_pt": np.zeros(4),
        "lead_eta": np.array([0.2, 0.2, 1.1, 1.1]),
        "sublead_eta": np.array([0.3, 0.3, 1.0, 1.0]),
        "photon_multiplicity": np.full(4, 2),
        "n_jets": np.array([0, 0, 0, 0]),
        "mjj": np.zeros(4),
        "delta_eta_jj": np.zeros(4),
    }

    categories = assign_categories(features, cfg)

    assert categories.tolist() == [
        "central_low_ptt",
        "central_high_ptt",
        "rest_low_ptt",
        "rest_high_ptt",
    ]


def test_assign_categories_supports_low_mid_high_ptt_splits() -> None:
    cfg = {
        **DEFAULT_RUNTIME,
        "category_ptt_boundaries_gev": {
            "central": [50.0, 80.0],
            "rest": [55.0, 75.0],
        },
    }
    features = {
        "diphoton_mass": np.full(6, 125.0),
        "ptt": np.array([45.0, 65.0, 90.0, 50.0, 65.0, 85.0]),
        "delta_r": np.zeros(6),
        "lead_pt": np.zeros(6),
        "sublead_pt": np.zeros(6),
        "lead_eta": np.array([0.2, 0.2, 0.2, 1.1, 1.1, 1.1]),
        "sublead_eta": np.array([0.3, 0.3, 0.3, 1.0, 1.0, 1.0]),
        "photon_multiplicity": np.full(6, 2),
        "n_jets": np.zeros(6, dtype=int),
        "mjj": np.zeros(6),
        "delta_eta_jj": np.zeros(6),
    }

    categories = assign_categories(features, cfg)

    assert categories.tolist() == [
        "central_low_ptt",
        "central_mid1_ptt",
        "central_high_ptt",
        "rest_low_ptt",
        "rest_mid1_ptt",
        "rest_high_ptt",
    ]
