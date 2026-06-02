from __future__ import annotations

import re

import numpy as np

from analysis.routing.config import load_routing_config, parse_routing_config, routing_config_path_from_runtime
from analysis.routing.router import route_categories


DEFAULT_CATEGORY_ORDER = [
    "two_jet_vbf_enriched",
    "central_low_ptt",
    "central_high_ptt",
    "rest_low_ptt",
    "rest_high_ptt",
]
CATEGORY_ORDER = DEFAULT_CATEGORY_ORDER
DEFAULT_FIVE_CATEGORY_ROUTING_CONFIG = "configs/routing/five_category_ptt.yaml"


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


def _five_category_payload(boundaries: dict[str, list[float]]) -> dict:
    categories = [
        {
            "id": "two_jet_vbf_enriched",
            "label": "Two-jet VBF enriched",
            "priority": 10,
            "required_inputs": ["n_jets", "mjj", "delta_eta_jj"],
            "select_when": {
                "all": [
                    {"field": "n_jets", "op": ">=", "value": 2},
                    {"field": "mjj", "op": ">", "value": 400.0},
                    {"field": "delta_eta_jj", "op": ">", "value": 2.8},
                ]
            },
            "reason": "at least two jets with high dijet mass and rapidity separation",
        }
    ]
    priority = 20
    for family in ("central", "rest"):
        family_boundaries = boundaries[family]
        if family == "central":
            family_predicate = {
                "all": [
                    {"field": "lead_eta_abs", "op": "<", "value": 0.75},
                    {"field": "sublead_eta_abs", "op": "<", "value": 0.75},
                ]
            }
            topology = "central photons"
        else:
            family_predicate = {
                "any": [
                    {"field": "lead_eta_abs", "op": ">=", "value": 0.75},
                    {"field": "sublead_eta_abs", "op": ">=", "value": 0.75},
                ]
            }
            topology = "non-central photons"
        categories.append(
            {
                "id": f"{family}_low_ptt",
                "label": f"{family} low pTt",
                "priority": priority,
                "required_inputs": ["lead_eta_abs", "sublead_eta_abs", "ptt"],
                "eligible_when": family_predicate,
                "select_when": {"all": [{"field": "ptt", "op": "<", "value": family_boundaries[0]}]},
                "reason": f"{topology} with low pTt",
            }
        )
        priority += 10
        for idx in range(1, len(family_boundaries)):
            categories.append(
                {
                    "id": f"{family}_mid{idx}_ptt",
                    "label": f"{family} mid {idx} pTt",
                    "priority": priority,
                    "required_inputs": ["lead_eta_abs", "sublead_eta_abs", "ptt"],
                    "eligible_when": family_predicate,
                    "select_when": {
                        "all": [
                            {"field": "ptt", "op": ">=", "value": family_boundaries[idx - 1]},
                            {"field": "ptt", "op": "<", "value": family_boundaries[idx]},
                        ]
                    },
                    "reason": f"{topology} with intermediate pTt",
                }
            )
            priority += 10
        categories.append(
            {
                "id": f"{family}_high_ptt",
                "label": f"{family} high pTt",
                "priority": priority,
                "required_inputs": ["lead_eta_abs", "sublead_eta_abs", "ptt"],
                "eligible_when": family_predicate,
                "select_when": {"all": [{"field": "ptt", "op": ">=", "value": family_boundaries[-1]}]},
                "reason": f"{topology} with high pTt",
            }
        )
        priority += 10
    return {"routing": {"mode": "ordered_first_match"}, "categories": categories}


def _five_category_routing_config(cfg: dict | None):
    boundaries = _ptt_boundaries(cfg)
    return parse_routing_config(_five_category_payload(boundaries))


def _load_runtime_routing_config(cfg: dict | None):
    if cfg is None:
        return _five_category_routing_config(cfg)
    routing_path = cfg.get("analysis_implementation", {}).get("routing_config")
    boundaries = _ptt_boundaries(cfg)
    default_boundaries = {"central": [60.0], "rest": [60.0]}
    if not routing_path:
        return _five_category_routing_config(cfg)
    if str(routing_path) == DEFAULT_FIVE_CATEGORY_ROUTING_CONFIG and boundaries != default_boundaries:
        return _five_category_routing_config(cfg)
    return load_routing_config(routing_config_path_from_runtime(cfg))


def category_order(cfg: dict | None = None) -> list[str]:
    if selection_implementation(cfg) == "section8_ads_bdt":
        try:
            return load_routing_config(routing_config_path_from_runtime(cfg)).category_ids
        except Exception:
            from analysis.section8_ads.categories import ORDERED_CATEGORIES

            return [section8_category_id(label) for label in ORDERED_CATEGORIES]
    return _load_runtime_routing_config(cfg).category_ids


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


def legacy_assign_categories(features: dict, cfg: dict | None = None) -> np.ndarray:
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


def assign_categories(features: dict, cfg: dict | None = None) -> np.ndarray:
    routing_features = dict(features)
    if "lead_eta_abs" not in routing_features and "lead_eta" in routing_features:
        routing_features["lead_eta_abs"] = np.abs(routing_features["lead_eta"])
    if "sublead_eta_abs" not in routing_features and "sublead_eta" in routing_features:
        routing_features["sublead_eta_abs"] = np.abs(routing_features["sublead_eta"])
    return route_categories(routing_features, _load_runtime_routing_config(cfg)).assigned_category


def sideband_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 105.0) & (mass < 120.0) | (mass > 130.0) & (mass <= 160.0)


def signal_window_mask(mass: np.ndarray) -> np.ndarray:
    return (mass >= 120.0) & (mass <= 130.0)
