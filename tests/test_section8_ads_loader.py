from __future__ import annotations

import json
from pathlib import Path

from analysis.section8_ads.ads_loader import load_ads
from analysis.section8_ads.categories import ORDERED_CATEGORIES


def _write_ads(path: Path) -> Path:
    categories = [
        {
            "priority_order": idx,
            "region_identifier": name,
            "selection_requirements": [],
        }
        for idx, name in enumerate(ORDERED_CATEGORIES, start=1)
    ]
    path.write_text(
        json.dumps(
            {
                "ads_instance_id": "synthetic",
                "ads_schema": "test",
                "scope": "unit",
                "section_1_analysis_metadata": {},
                "section_3_analysis_object_definitions": {},
                "section_4_event_selection_graph": {
                    "exclusive_likelihood_categories": list(reversed(categories)),
                },
                "section_5_derived_variable_library": [],
                "section_6_multivariate_model_reconstruction": [
                    {"model_name": "BDT_ttH"},
                    {"model_name": "BDT_VH"},
                    {"model_name": "BDT_VBF"},
                ],
                "section_7_sample_inventory": [],
                "section_8_process_to_sample_mapping": {},
                "section_15_ambiguity_registry": [],
                "section_16_missing_information_registry": [],
                "section_17_human_clarification_queue": [],
                "section_18_ads_readiness_audit": {},
            }
        )
    )
    return path


def test_load_ads_preserves_category_priority_order(tmp_path: Path) -> None:
    ads = load_ads(_write_ads(tmp_path / "ads.json"))
    names = [item["region_identifier"] for item in ads["ordered_categories"]]
    assert names == ORDERED_CATEGORIES


def test_load_ads_exposes_classifier_specs(tmp_path: Path) -> None:
    ads = load_ads(_write_ads(tmp_path / "ads.json"))
    assert "BDT_ttH" in ads["classifier_specs"]
    assert "BDT_VH" in ads["classifier_specs"]
    assert "BDT_VBF" in ads["classifier_specs"]
