from __future__ import annotations

from pathlib import Path

from analysis.section8_ads.ads_loader import load_ads
from analysis.section8_ads.categories import ORDERED_CATEGORIES


ADS_PATH = Path("/Users/haichenwang/Downloads/atlas_hgg_36fb_section8_ads.json")


def test_load_ads_preserves_category_priority_order() -> None:
    ads = load_ads(ADS_PATH)
    names = [item["region_identifier"] for item in ads["ordered_categories"]]
    assert names == ORDERED_CATEGORIES


def test_load_ads_exposes_classifier_specs() -> None:
    ads = load_ads(ADS_PATH)
    assert "BDT_ttH" in ads["classifier_specs"]
    assert "BDT_VH" in ads["classifier_specs"]
    assert "BDT_VBF" in ads["classifier_specs"]
