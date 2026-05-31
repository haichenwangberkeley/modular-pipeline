from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from analysis.common import read_json, stable_hash, write_json
from analysis.config.summary_schema import validate_summary_schema


DEFAULT_RUNTIME = {
    "tree_name": "analysis",
    "target_lumi_fb": 36.1,
    "central_mc_lumi_fb": 36.1,
    "fit_mass_range_gev": [105.0, 160.0],
    "signal_window_gev": [120.0, 130.0],
    "sidebands_gev": [[105.0, 120.0], [130.0, 160.0]],
    "blinding": {
        "plot_signal_window": True,
        "observed_significance_allowed": False,
        "fit_uses_observed_data": False,
    },
    "photon_selection": {
        "pt_min_gev": 25.0,
        "abs_eta_max": 2.37,
        "eta_crack": [1.37, 1.52],
        "require_tight_id": True,
        "require_tight_iso": True,
        "leading_pt_over_mgg_min": 0.35,
        "subleading_pt_over_mgg_min": 0.25,
    },
    "jet_selection": {
        "pt_min_gev": 25.0,
        "abs_eta_max": 4.5,
    },
    "histogramming": {
        "mass_bin_width_gev": 1.0,
    },
    "background_model": {
        "candidates": ["exponential", "bernstein2", "bernstein3"],
        "selection_metric": "aic",
    },
    "smoothing_policy": {
        "required": True,
        "method": "TH1::Smooth",
        "scope": "prompt_diphoton_nominal_templates",
    },
    "systematics": {
        "mode": "placeholder",
        "entries": [
            "photon energy scale",
            "photon identification efficiency",
            "luminosity",
            "background modeling",
            "MC statistical uncertainty",
        ],
    },
}


CANONICAL_REGION_CATEGORY_IDS = {
    "SR_2JET": "two_jet_vbf_enriched",
}


def canonical_category_id(region_id: str) -> str:
    return CANONICAL_REGION_CATEGORY_IDS.get(region_id, region_id.removeprefix("SR_").lower())


def _categories(summary: dict[str, Any]) -> list[dict[str, Any]]:
    categories = []
    for region in summary.get("signal_regions", []):
        region_id = region["signal_region_id"]
        category_id = canonical_category_id(region_id)
        categories.append(
            {
                "category_id": category_id,
                "source_region_id": region_id,
                "selection_summary": region.get("selection_summary"),
                "fit_observable": region.get("fit_observable", "m_gammagamma"),
                "region_type": "signal",
            }
        )
    return categories


def normalize_summary(summary: dict[str, Any], summary_path: Path) -> tuple[dict[str, Any], list[str]]:
    errors = validate_summary_schema(summary)
    normalized = dict(summary)
    normalized["source_summary"] = str(summary_path)
    normalized["runtime_defaults"] = copy.deepcopy(DEFAULT_RUNTIME)
    normalized["categories"] = _categories(summary)
    normalized["overlap_policy"] = {
        "default_allow_overlap": False,
        "declared_exceptions": [
            {
                "control_region_id": "CR_BKG_VALIDATION",
                "signal_region_pattern": "SR_*",
                "allow_overlap": True,
                "reason": "validation control region is the sideband subset of the diphoton fit domain",
            }
        ],
    }
    normalized["fit_regions"] = {
        fit["fit_id"]: {
            "regions": fit.get("regions_included", []),
            "observable": "m_gammagamma",
            "mass_range_gev": DEFAULT_RUNTIME["fit_mass_range_gev"],
            "signal_window_gev": DEFAULT_RUNTIME["signal_window_gev"],
        }
        for fit in summary.get("fit_setup", [])
    }
    normalized["inventory"] = {
        "n_signal_regions": len(summary.get("signal_regions", [])),
        "n_control_regions": len(summary.get("control_regions", [])),
        "fit_ids": [fit.get("fit_id") for fit in summary.get("fit_setup", [])],
        "observables": sorted({region.get("fit_observable") for region in summary.get("signal_regions", [])}),
        "pois": sorted({poi for fit in summary.get("fit_setup", []) for poi in fit.get("parameters_of_interest", [])}),
    }
    normalized["config_hash"] = stable_hash(normalized)
    return normalized, errors


def write_regions_yaml(normalized: dict[str, Any], path: Path) -> Path:
    payload = {
        "categories": normalized["categories"],
        "runtime_defaults": normalized["runtime_defaults"],
        "overlap_policy": normalized["overlap_policy"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inventory-out")
    parser.add_argument("--diagnostics-out")
    parser.add_argument("--overlap-out")
    parser.add_argument("--regions-yaml", default="analysis/regions.yaml")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = read_json(summary_path)
    normalized, errors = normalize_summary(summary, summary_path)
    write_json(normalized, args.out)
    write_regions_yaml(normalized, Path(args.regions_yaml))
    if args.inventory_out:
        write_json(normalized["inventory"], args.inventory_out)
    if args.diagnostics_out:
        write_json({"status": "ok" if not errors else "failed", "errors": errors}, args.diagnostics_out)
    if args.overlap_out:
        write_json(normalized["overlap_policy"], args.overlap_out)

    print(
        "Inventory:",
        f"SR={normalized['inventory']['n_signal_regions']}",
        f"CR={normalized['inventory']['n_control_regions']}",
        f"fits={','.join(normalized['inventory']['fit_ids'])}",
        f"pois={','.join(normalized['inventory']['pois'])}",
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
