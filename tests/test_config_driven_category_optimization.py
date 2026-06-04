from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.routing.config import load_routing_config
from analysis.routing.router import route_categories


def _write_strategy(path: Path, *, tth_threshold: float) -> None:
    payload = {
        "routing": {"mode": "ordered_first_match"},
        "categories": [
            {
                "id": "boosted_tth",
                "label": "Boosted ttH",
                "priority": 10,
                "required_inputs": ["N_jets_30", "N_btag_25", "BDT_ttH"],
                "eligible_when": {
                    "all": [
                        {"field": "N_jets_30", "op": ">=", "value": 3},
                        {"field": "N_btag_25", "op": ">=", "value": 1},
                    ]
                },
                "select_when": {"all": [{"field": "BDT_ttH", "op": ">", "value": tth_threshold}]},
                "block_if_missing": ["BDT_ttH"],
                "reason": "boosted ttH score bin",
                "block_reason": "BDT_ttH missing for boosted ttH",
            },
            {
                "id": "high_pt_ggh",
                "label": "High-pT ggH",
                "priority": 20,
                "required_inputs": ["pT_gammagamma"],
                "select_when": {"all": [{"field": "pT_gammagamma", "op": ">", "value": 200.0}]},
                "reason": "high pT fallback",
            },
            {
                "id": "inclusive",
                "label": "Inclusive",
                "priority": 999,
                "required_inputs": [],
                "select_when": {"always": True},
                "reason": "inclusive fallback",
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _observables() -> dict[str, np.ndarray]:
    return {
        "N_jets_30": np.asarray([3, 3, 1, 0], dtype=float),
        "N_btag_25": np.asarray([1, 1, 0, 0], dtype=float),
        "BDT_ttH": np.asarray([0.93, np.nan, 0.20, 0.10], dtype=float),
        "pT_gammagamma": np.asarray([250.0, 260.0, 240.0, 80.0], dtype=float),
    }


def test_category_design_changes_by_editing_config_only(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed.yaml"
    _write_strategy(config_path, tth_threshold=0.90)

    first = route_categories(_observables(), load_routing_config(config_path))

    assert first.assigned_category.tolist() == [
        "boosted_tth",
        "blocked_missing_input",
        "high_pt_ggh",
        "inclusive",
    ]
    assert first.assignment_blocked.tolist() == [False, True, False, False]
    assert "BDT_ttH" in first.assignment_reason[1]

    _write_strategy(config_path, tth_threshold=0.95)
    second = route_categories(_observables(), load_routing_config(config_path))

    assert second.assigned_category.tolist() == [
        "high_pt_ggh",
        "blocked_missing_input",
        "high_pt_ggh",
        "inclusive",
    ]
    assert second.assignment_reason[0] == "high pT fallback"


def test_route_npz_command_reroutes_materialized_observables(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed.yaml"
    events_path = tmp_path / "events.npz"
    outputs = tmp_path / "routed"
    _write_strategy(config_path, tth_threshold=0.90)
    np.savez_compressed(events_path, **_observables())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.routing.route_npz",
            "--events",
            str(events_path),
            "--routing-config",
            str(config_path),
            "--outputs",
            str(outputs),
        ],
        cwd=Path.cwd(),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "routed_categories.npz" in result.stdout
    with np.load(outputs / "routed_categories.npz", allow_pickle=False) as payload:
        assert payload["assigned_category"].tolist() == [
            "boosted_tth",
            "blocked_missing_input",
            "high_pt_ggh",
            "inclusive",
        ]
        assert payload["assignment_blocked"].tolist() == [0, 1, 0, 0]
