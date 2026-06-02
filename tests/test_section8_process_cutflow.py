from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.config.load_summary import DEFAULT_RUNTIME
from analysis.config.versions import apply_analysis_version
from analysis.hists.histmaker import CUT_STEPS
from analysis.report.artifacts import build_section8_process_cutflow_artifacts


def _cutflow(weight: float, count: int) -> dict[str, dict[str, float | int]]:
    return {
        step: {
            "weighted": weight,
            "unweighted": count,
        }
        for step in CUT_STEPS
    }


def _sample(
    *,
    sample_id: str,
    process_key: str,
    kind: str,
    analysis_role: str,
    masses: list[float],
    weights: list[float],
    categories: list[str],
) -> dict:
    return {
        "sample_id": sample_id,
        "process_key": process_key,
        "kind": kind,
        "analysis_role": analysis_role,
        "cutflow": _cutflow(sum(weights), len(weights)),
        "events": {
            "mgg": np.asarray(masses, dtype=float),
            "weight": np.asarray(weights, dtype=float),
            "category": np.asarray(categories, dtype=str),
            "is_sideband": np.asarray([mass < 120.0 or mass > 130.0 for mass in masses], dtype=bool),
            "is_signal_window": np.asarray([120.0 <= mass <= 130.0 for mass in masses], dtype=bool),
            "N_jets_30": np.zeros(len(masses)),
            "N_lep": np.zeros(len(masses)),
            "pT_gammagamma": np.full(len(masses), 100.0),
        },
    }


def test_section8_process_cutflow_writes_process_and_non_bdt_debug(tmp_path: Path) -> None:
    cfg = apply_analysis_version(DEFAULT_RUNTIME, version_name="round2_section8_bdt")
    processed = [
        _sample(
            sample_id="data_a",
            process_key="data",
            kind="data",
            analysis_role="data",
            masses=[110.0],
            weights=[1.0],
            categories=["ggH_0J_Cen"],
        ),
        _sample(
            sample_id="data_b",
            process_key="data",
            kind="data",
            analysis_role="data",
            masses=[125.0, 140.0],
            weights=[1.0, 1.0],
            categories=["ggH_0J_Cen", "ttH_had_BDT1"],
        ),
        _sample(
            sample_id="ggh",
            process_key="ggh",
            kind="signal",
            analysis_role="signal_nominal",
            masses=[124.0],
            weights=[2.5],
            categories=["ggH_0J_Cen"],
        ),
        _sample(
            sample_id="yy",
            process_key="prompt_diphoton",
            kind="background",
            analysis_role="background_nominal",
            masses=[118.0],
            weights=[4.0],
            categories=["ggH_0J_Cen"],
        ),
    ]

    payload = build_section8_process_cutflow_artifacts(processed, cfg, tmp_path)

    assert payload is not None
    assert payload["processes"]["data"]["steps"]["categorized"]["unweighted_events"] == 3
    assert payload["processes"]["ggh"]["steps"]["mass_window"]["normalized_yield_36fb"] == 2.5
    assert payload["category_yields"]["ggH_0J_Cen"]["processes"]["data"]["mgg"]["p50"] == 117.5
    assert payload["category_yields"]["ggH_0J_Cen"]["bdt_required"] is False
    assert payload["category_yields"]["ttH_had_BDT1"]["bdt_required"] is True
    assert (tmp_path / "report" / "section8_process_cutflow.csv").exists()
    assert (tmp_path / "report" / "section8_category_process_yields.csv").exists()
    saved = json.loads((tmp_path / "report" / "section8_process_cutflow.json").read_text())
    assert "ggH_0J_Cen" in saved["non_bdt_category_debug"]["recommended_first_debug_categories"]
