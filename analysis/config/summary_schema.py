from __future__ import annotations

from typing import Any

REQUIRED_TOP_LEVEL_KEYS = [
    "analysis_metadata",
    "analysis_objectives",
    "signal_signatures",
    "background_processes",
    "signal_regions",
    "control_regions",
    "fit_setup",
    "results",
]


def validate_summary_schema(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in summary:
            errors.append(f"missing top-level key: {key}")
    ids = set()
    for section, id_field in (("signal_regions", "signal_region_id"), ("control_regions", "control_region_id")):
        for item in summary.get(section, []):
            item_id = item.get(id_field)
            if item_id in ids:
                errors.append(f"duplicate region identifier: {item_id}")
            ids.add(item_id)
    signature_ids = {item.get("signature_id") for item in summary.get("signal_signatures", [])}
    for region in summary.get("signal_regions", []):
        for ref in region.get("associated_signature_ids", []):
            if ref not in signature_ids:
                errors.append(f"unknown signature reference: {ref}")
    fit_ids = {item.get("fit_id") for item in summary.get("fit_setup", [])}
    region_ids = ids
    for fit in summary.get("fit_setup", []):
        for region_id in fit.get("regions_included", []):
            if region_id not in region_ids:
                errors.append(f"fit {fit.get('fit_id')} references unknown region {region_id}")
    for result in summary.get("results", []):
        fit_id = result.get("associated_fit_id")
        if fit_id not in fit_ids:
            errors.append(f"result {result.get('result_id')} references unknown fit {fit_id}")
    return errors
