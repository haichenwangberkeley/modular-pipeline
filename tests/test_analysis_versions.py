from __future__ import annotations

from pathlib import Path

import yaml

from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.config.versions import apply_analysis_version
from analysis.routing.config import load_routing_config, routing_config_path_from_runtime
from analysis.selections.engine import category_order, selection_summary_for_category


def test_round1_version_preserves_original_five_category_selection() -> None:
    cfg = apply_analysis_version(DEFAULT_RUNTIME, version_name="round1_5cat")

    assert cfg["analysis_implementation"]["selection"] == "five_category_ptt"
    routing_path = routing_config_path_from_runtime(cfg)
    assert routing_path.exists()
    assert load_routing_config(routing_path).category_ids == [
        "two_jet_vbf_enriched",
        "central_low_ptt",
        "central_high_ptt",
        "rest_low_ptt",
        "rest_high_ptt",
    ]
    assert category_order(cfg) == [
        "two_jet_vbf_enriched",
        "central_low_ptt",
        "central_high_ptt",
        "rest_low_ptt",
        "rest_high_ptt",
    ]


def test_round2_version_exposes_section8_categories_and_artifacts() -> None:
    cfg = apply_analysis_version(DEFAULT_RUNTIME, version_name="round2_section8_bdt")
    categories = category_order(cfg)

    assert cfg["analysis_implementation"]["selection"] == "section8_ads_bdt"
    routing_path = routing_config_path_from_runtime(cfg)
    assert routing_path.exists()
    assert len(load_routing_config(routing_path).categories) == 31
    assert len(categories) == 31
    assert categories[:5] == ["tH_lep_0fwd", "tH_lep_1fwd", "ttH_lep", "ttH_had_BDT1", "ttH_had_BDT2"]
    assert "ads_path" not in cfg["section8_ads"]
    assert "bdt_artifacts_dir" not in cfg["section8_ads"]
    assert cfg["section8_ads"]["standalone_required_external_inputs"] == ["ads_path"]
    assert cfg["section8_ads"]["modular_adapter_required_external_inputs"] == ["bdt_artifacts_dir"]
    assert "Section 8 ADS first-match" in selection_summary_for_category(categories[0], cfg)


def test_round2_section8_paths_are_explicit_cli_overrides() -> None:
    cfg = apply_analysis_version(
        DEFAULT_RUNTIME,
        version_name="round2_section8_bdt",
        section8_ads_path=Path("external/ads.json"),
        section8_bdt_artifacts=Path("outputs/section8_bdt"),
    )

    assert cfg["section8_ads"]["ads_path"] == "external/ads.json"
    assert cfg["section8_ads"]["bdt_artifacts_dir"] == "outputs/section8_bdt"


def test_analysis_version_routing_config_can_be_overridden(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        yaml.safe_dump(
            {
                "routing": {"mode": "ordered_first_match"},
                "categories": [
                    {
                        "id": "custom_inclusive",
                        "label": "Custom Inclusive",
                        "priority": 1,
                        "required_inputs": [],
                        "select_when": {"always": True},
                    }
                ],
            },
            sort_keys=False,
        )
    )

    cfg = apply_analysis_version(
        DEFAULT_RUNTIME,
        version_name="round1_5cat",
        routing_config=custom_config,
    )

    assert routing_config_path_from_runtime(cfg) == custom_config
    assert category_order(cfg) == ["custom_inclusive"]
    assert load_routing_config(custom_config).category_ids == ["custom_inclusive"]
