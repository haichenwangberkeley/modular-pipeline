from __future__ import annotations

import argparse
from pathlib import Path

from analysis.common import read_json, write_json
from analysis.config.load_summary import canonical_category_id


def build_partition(summary: dict) -> dict:
    runtime_defaults = summary.get("runtime_defaults", {})
    blinding = runtime_defaults.get("blinding", {})
    signal_window = runtime_defaults.get("signal_window_gev", [120.0, 130.0])
    signal_window_masked = bool(blinding.get("plot_signal_window", True))
    categories = []
    for region in summary["signal_regions"]:
        categories.append(
            {
                "category_id": canonical_category_id(region["signal_region_id"]),
                "region_id": region["signal_region_id"],
                "region_type": "signal",
                "selection_basis": "event_kinematics",
                "selection_definition": region["selection_summary"],
                "blinding_policy": {
                    "data_shown": not signal_window_masked,
                    "signal_window_gev": signal_window if signal_window_masked else None,
                    "observed_significance_allowed": bool(blinding.get("observed_significance_allowed", False)),
                },
                "mass_window": [105.0, 160.0],
            }
        )
    categories.append(
        {
            "category_id": "all_categories",
            "region_id": "CR_BKG_VALIDATION",
            "region_type": "control",
            "selection_basis": "fit_domain_sidebands",
            "selection_definition": summary["control_regions"][0]["selection_summary"],
            "blinding_policy": {
                "data_shown": True,
                "signal_window_masked": signal_window_masked,
                "signal_window_gev": signal_window if signal_window_masked else None,
            },
            "mass_window": [[105.0, 120.0], [130.0, 160.0]],
        }
    )
    return {
        "status": "ok",
        "category_list": [item["category_id"] for item in categories if item["region_type"] == "signal"],
        "region_list": [item["region_id"] for item in categories],
        "category_region_mapping": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="outputs/partition/partition_spec.json")
    args = parser.parse_args()
    summary = read_json(args.summary)
    write_json(build_partition(summary), args.out)
    print("ok")


if __name__ == "__main__":
    main()
