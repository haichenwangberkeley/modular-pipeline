from __future__ import annotations

from pathlib import Path

from analysis.section8_ads.ads_loader import load_ads
from analysis.section8_ads.branch_map import build_branch_mapping


ADS_PATH = Path("/Users/haichenwang/Downloads/atlas_hgg_36fb_section8_ads.json")
DATA_FILE = "/Users/haichenwang/Work/newpipeline/input-data/data/ODEO_FEB2025_v0_GamGam_data15_periodD.GamGam.root"


def test_branch_mapping_resolves_core_objects(tmp_path: Path) -> None:
    ads = load_ads(ADS_PATH)
    payload = build_branch_mapping([DATA_FILE], ads, tmp_path)
    assert payload["resolved_mappings"]["photon_pt"]["branch"] == "photon_pt"
    assert payload["resolved_mappings"]["jet_btag_quantile"]["branch"] == "jet_btag_quantile"
    assert any(item["logical_name"] == "btag_working_point" for item in payload["uncertain_mappings"])
