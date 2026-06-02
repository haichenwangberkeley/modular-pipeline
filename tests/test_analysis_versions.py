from __future__ import annotations

from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.config.versions import apply_analysis_version
from analysis.selections.engine import category_order, selection_summary_for_category


def test_round1_version_preserves_original_five_category_selection() -> None:
    cfg = apply_analysis_version(DEFAULT_RUNTIME, version_name="round1_5cat")

    assert cfg["analysis_implementation"]["selection"] == "five_category_ptt"
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
    assert len(categories) == 31
    assert categories[:5] == ["tH_lep_0fwd", "tH_lep_1fwd", "ttH_lep", "ttH_had_BDT1", "ttH_had_BDT2"]
    assert cfg["section8_ads"]["ads_path"].endswith("atlas_hgg_36fb_section8_ads.json")
    assert "Section 8 ADS first-match" in selection_summary_for_category(categories[0], cfg)
