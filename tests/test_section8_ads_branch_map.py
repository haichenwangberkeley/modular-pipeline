from __future__ import annotations

from pathlib import Path

from analysis.section8_ads import branch_map
from analysis.section8_ads.categories import ORDERED_CATEGORIES


def _ads() -> dict:
    return {
        "ordered_categories": [
            {
                "region_identifier": name,
                "selection_requirements": ["BDT_ttH"] if name.startswith("ttH had") else [],
            }
            for name in ORDERED_CATEGORIES
        ]
    }


def test_branch_mapping_resolves_core_objects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        branch_map,
        "inspect_available_fields",
        lambda files, tree_name="analysis", sample_limit=3: [
            "eventNumber",
            "photon_pt",
            "jet_btag_quantile",
        ],
    )
    payload = branch_map.build_branch_mapping(["synthetic.root"], _ads(), tmp_path)
    assert payload["resolved_mappings"]["photon_pt"]["branch"] == "photon_pt"
    assert payload["resolved_mappings"]["jet_btag_quantile"]["branch"] == "jet_btag_quantile"
    assert any(item["logical_name"] == "btag_working_point" for item in payload["uncertain_mappings"])
