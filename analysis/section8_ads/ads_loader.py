from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.common import read_json, stable_hash


def load_ads(path: Path) -> dict[str, Any]:
    ads = read_json(path)
    categories = sorted(
        ads["section_4_event_selection_graph"]["exclusive_likelihood_categories"],
        key=lambda item: int(item["priority_order"]),
    )
    derived_variables = {
        item["variable_name"]: item for item in ads.get("section_5_derived_variable_library", [])
    }
    classifier_specs = {
        item["model_name"]: item for item in ads.get("section_6_multivariate_model_reconstruction", [])
    }
    payload = {
        "ads_path": str(path),
        "ads_instance_id": ads["ads_instance_id"],
        "ads_schema": ads["ads_schema"],
        "scope": ads["scope"],
        "analysis_metadata": ads["section_1_analysis_metadata"],
        "object_definitions": ads["section_3_analysis_object_definitions"],
        "event_selection_graph": ads["section_4_event_selection_graph"],
        "ordered_categories": categories,
        "derived_variables": derived_variables,
        "classifier_specs": classifier_specs,
        "sample_inventory": ads["section_7_sample_inventory"],
        "process_mapping": ads["section_8_process_to_sample_mapping"],
        "ambiguity_registry": ads["section_15_ambiguity_registry"],
        "missing_information_registry": ads["section_16_missing_information_registry"],
        "clarification_queue": ads["section_17_human_clarification_queue"],
        "readiness_audit": ads["section_18_ads_readiness_audit"],
    }
    payload["config_hash"] = stable_hash(payload)
    return payload
