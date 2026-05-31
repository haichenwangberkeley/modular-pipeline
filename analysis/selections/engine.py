from __future__ import annotations

import numpy as np


CATEGORY_ORDER = [
    "two_jet_vbf_enriched",
    "central_low_ptt",
    "central_high_ptt",
    "rest_low_ptt",
    "rest_high_ptt",
]


def assign_categories(features: dict) -> np.ndarray:
    n = len(features["diphoton_mass"])
    categories = np.full(n, "unassigned", dtype=object)
    two_jet = (features["n_jets"] >= 2) & (features["mjj"] > 400.0) & (features["delta_eta_jj"] > 2.8)
    central = (np.abs(features["lead_eta"]) < 0.75) & (np.abs(features["sublead_eta"]) < 0.75)
    high_ptt = features["ptt"] >= 60.0

    categories[two_jet] = "two_jet_vbf_enriched"
    categories[~two_jet & central & ~high_ptt] = "central_low_ptt"
    categories[~two_jet & central & high_ptt] = "central_high_ptt"
    categories[~two_jet & ~central & ~high_ptt] = "rest_low_ptt"
    categories[~two_jet & ~central & high_ptt] = "rest_high_ptt"
    return categories


def sideband_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 105.0) & (mass < 120.0) | (mass > 130.0) & (mass <= 160.0)


def signal_window_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 120.0) & (mass <= 130.0)
