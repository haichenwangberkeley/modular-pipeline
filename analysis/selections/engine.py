from __future__ import annotations

import re

import numpy as np


DEFAULT_CATEGORY_ORDER = [
    "two_jet_vbf_enriched",
    "central_low_ptt",
    "central_high_ptt",
    "rest_low_ptt",
    "rest_high_ptt",
]
CATEGORY_ORDER = DEFAULT_CATEGORY_ORDER


def selection_implementation(cfg: dict | None = None) -> str:
    if cfg is None:
        return "five_category_ptt"
    return cfg.get("analysis_implementation", {}).get("selection", "five_category_ptt")


def section8_category_id(label: str) -> str:
    cleaned = label.replace("γ", "gamma")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def section8_category_label(category_id: str) -> str:
    try:
        from analysis.section8_ads.categories import ORDERED_CATEGORIES
    except ModuleNotFoundError:
        return category_id
    mapping = {section8_category_id(label): label for label in ORDERED_CATEGORIES}
    return mapping.get(category_id, category_id)


def _as_sorted_boundaries(value: float | list[float] | tuple[float, ...], default: float) -> list[float]:
    if isinstance(value, (list, tuple)):
        boundaries = [float(item) for item in value]
    elif value is None:
        boundaries = [float(default)]
    else:
        boundaries = [float(value)]
    return sorted(boundaries)


def _ptt_boundaries(cfg: dict | None) -> dict[str, list[float]]:
    if cfg is None:
        return {
            "central": [60.0],
            "rest": [60.0],
        }
    boundaries = cfg.get("category_ptt_boundaries_gev", {})
    return {
        "central": _as_sorted_boundaries(boundaries.get("central", 60.0), 60.0),
        "rest": _as_sorted_boundaries(boundaries.get("rest", 60.0), 60.0),
    }


def category_order(cfg: dict | None = None) -> list[str]:
    if selection_implementation(cfg) == "section8_ads_bdt":
        from analysis.section8_ads.categories import ORDERED_CATEGORIES

        return [section8_category_id(label) for label in ORDERED_CATEGORIES]
    boundaries = _ptt_boundaries(cfg)
    order = ["two_jet_vbf_enriched"]
    for family in ("central", "rest"):
        family_boundaries = boundaries[family]
        if len(family_boundaries) <= 1:
            order.extend([f"{family}_low_ptt", f"{family}_high_ptt"])
            continue
        order.append(f"{family}_low_ptt")
        for idx in range(1, len(family_boundaries)):
            order.append(f"{family}_mid{idx}_ptt")
        order.append(f"{family}_high_ptt")
    return order


def region_id_for_category(category: str) -> str:
    if " " not in category and selection_implementation({"analysis_implementation": {"selection": "section8_ads_bdt"}}) == "section8_ads_bdt":
        return f"SR_{category.upper()}"
    return f"SR_{category.upper()}"


def selection_summary_for_category(category: str, cfg: dict | None = None) -> str:
    if selection_implementation(cfg) == "section8_ads_bdt":
        return f"Section 8 ADS first-match category `{section8_category_label(category)}`."
    boundaries = _ptt_boundaries(cfg)
    if category == "two_jet_vbf_enriched":
        return "At least two jets with m_jj > 400 GeV and delta_eta_jj > 2.8."

    family = "central" if category.startswith("central_") else "rest"
    family_boundaries = boundaries[family]
    topology = (
        "Two photons with |eta| < 0.75"
        if family == "central"
        else "Events not in central or 2-jet category"
    )

    if category.endswith("low_ptt"):
        return f"{topology} and pTt < {family_boundaries[0]:.0f} GeV."
    if category.endswith("high_ptt"):
        return f"{topology} and pTt >= {family_boundaries[-1]:.0f} GeV."
    if "_mid" in category:
        mid_index = int(category.split("_mid", 1)[1].split("_", 1)[0])
        lower = family_boundaries[mid_index - 1]
        upper = family_boundaries[mid_index]
        return f"{topology} and {lower:.0f} <= pTt < {upper:.0f} GeV."
    return f"{topology}."


def assign_categories(features: dict, cfg: dict | None = None) -> np.ndarray:
    n = len(features["diphoton_mass"])
    categories = np.full(n, "unassigned", dtype=object)
    two_jet = (features["n_jets"] >= 2) & (features["mjj"] > 400.0) & (features["delta_eta_jj"] > 2.8)
    central = (np.abs(features["lead_eta"]) < 0.75) & (np.abs(features["sublead_eta"]) < 0.75)

    categories[two_jet] = "two_jet_vbf_enriched"
    boundaries = _ptt_boundaries(cfg)

    for family, family_mask in (("central", ~two_jet & central), ("rest", ~two_jet & ~central)):
        family_boundaries = boundaries[family]
        low_mask = family_mask & (features["ptt"] < family_boundaries[0])
        categories[low_mask] = f"{family}_low_ptt"
        for idx in range(1, len(family_boundaries)):
            mid_mask = family_mask & (features["ptt"] >= family_boundaries[idx - 1]) & (features["ptt"] < family_boundaries[idx])
            categories[mid_mask] = f"{family}_mid{idx}_ptt"
        high_mask = family_mask & (features["ptt"] >= family_boundaries[-1])
        categories[high_mask] = f"{family}_high_ptt"
    return categories


def sideband_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 105.0) & (mass < 120.0) | (mass > 130.0) & (mass <= 160.0)


def signal_window_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 120.0) & (mass <= 130.0)
